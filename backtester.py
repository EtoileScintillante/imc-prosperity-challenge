from typing import Dict, List, Any
import pandas as pd
import json
from collections import defaultdict
from datamodel import TradingState, Listing, OrderDepth, Trade, Order, UserId
import time
import numpy as np

class ConversionObservation:
    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sugarPrice: float, sunlightIndex: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex


class Observation:
    def __init__(self, plainValueObservations: Dict[str, float], conversionObservations: Dict[str, ConversionObservation]):
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations


class Backtester:
    def __init__(self, trader, listings: Dict[str, Listing], position_limit: Dict[str, int], fair_marks,
                 market_data: pd.DataFrame, trade_history: pd.DataFrame, observation_data=None, file_name: str = None):
        self.trader = trader
        self.listings = listings
        self.market_data = market_data
        self.position_limit = position_limit
        self.fair_marks = fair_marks
        self.trade_history = trade_history.sort_values(by=['timestamp', 'symbol'])
        self.observation_data = observation_data
        self.file_name = file_name

        self.observations = []

        self.current_position = {product: 0 for product in self.listings.keys()}
        self.pnl_history = [] 
        self.pnl = {product: 0 for product in self.listings.keys()}
        self.cash = {product: 0 for product in self.listings.keys()}
        self.trades = []
        self.sandbox_logs = []
        self.run_times = []

        self.macaron_conversion_count = 0
        self.macaron_position = 0
        self.pnl_over_time = [] # PnL over time (needed for plotting)
        self.unrealized_pnl = 0 # PnL that could be made when converting MACARONS
        self.capital_history = [] # Needed to create a graph
        self.timestamp_history = [] # Needed to create a graph

    def run(self):
        traderData = ""

        timestamp_group_md = self.market_data.groupby('timestamp')
        timestamp_group_th = self.trade_history.groupby('timestamp')

        trade_history_dict = {}
        for timestamp, group in timestamp_group_th:
            trades = []
            for _, row in group.iterrows():
                symbol = row['symbol']
                price = row['price']
                quantity = row['quantity']
                buyer = row['buyer'] if pd.notnull(row['buyer']) else ""
                seller = row['seller'] if pd.notnull(row['seller']) else ""

                trade = Trade(symbol, int(price), int(quantity), buyer, seller, timestamp)
                trades.append(trade)
            trade_history_dict[timestamp] = trades

        for timestamp, group in timestamp_group_md:
            own_trades = defaultdict(list)
            market_trades = defaultdict(list)
            pnl_product = defaultdict(float)

            order_depths = self._construct_order_depths(group)
            order_depths_matching = self._construct_order_depths(group)
            order_depths_pnl = self._construct_order_depths(group)

            products_in_market = group['product'].unique().tolist()

            # Get observation data for current timestamp if available
            if self.observation_data is not None:
                obs_row = self.observation_data[self.observation_data['timestamp'] == timestamp]
                if not obs_row.empty:
                    row = obs_row.iloc[0]
                    conversion_obs = {
                        "MAGNIFICENT_MACARONS": ConversionObservation(
                            bidPrice=row["bidPrice"],
                            askPrice=row["askPrice"],
                            transportFees=row["transportFees"],
                            exportTariff=row["exportTariff"],
                            importTariff=row["importTariff"],
                            sugarPrice=row["sugarPrice"],
                            sunlightIndex=row["sunlightIndex"]
                        )
                    }
                else:
                    conversion_obs = {}
            else:
                conversion_obs = {}

            observation = Observation({}, conversion_obs)

            state = self._construct_trading_state(traderData, timestamp, self.listings, order_depths,
                                                  dict(own_trades), dict(market_trades), self.current_position,
                                                  observation)

            start_time = time.time()
            orders, conversions, traderData = self.trader.run(state)
            end_time = time.time()
            self.run_times.append(end_time - start_time)

            sandboxLog = ""
            trades_at_timestamp = trade_history_dict.get(timestamp, [])

            # Execute conversions FIRST before any trades
            if "MAGNIFICENT_MACARONS" in observation.conversionObservations:
                self._execute_conversion(conversions, order_depths_matching, self.current_position, self.cash,
                                        observation.conversionObservations["MAGNIFICENT_MACARONS"])
            # Now execute orders
            for product in products_in_market:
                new_trades = []

                for order in orders.get(product, []):
                    executed_orders = self._execute_order(timestamp, order, order_depths_matching,
                                                          self.current_position, self.cash, trade_history_dict,
                                                          sandboxLog)
                    if len(executed_orders) > 0:
                        trades_done, sandboxLog = executed_orders
                        new_trades.extend(trades_done)
                if len(new_trades) > 0:
                    own_trades[product] = new_trades

            self.sandbox_logs.append({"sandboxLog": sandboxLog, "lambdaLog": "", "timestamp": timestamp})

            trades_at_timestamp = trade_history_dict.get(timestamp, [])
            if trades_at_timestamp:
                for trade in trades_at_timestamp:
                    symbol = trade.symbol
                    market_trades[symbol].append(trade)
            else:
                for product in products_in_market:
                    market_trades[product] = []

            for product in products_in_market:
                self._mark_pnl(self.cash, self.current_position, order_depths_pnl, self.pnl, product)
                self.pnl_history.append(self.pnl[product])

            self.capital_history.append(sum(self.cash.values()))
            self.timestamp_history.append(timestamp)
            # Capture PnL snapshot at this timestamp
            self.pnl_over_time.append(sum(v for v in self.pnl.values() if v is not None))
            self._add_trades(own_trades, market_trades)
            if np.mean(self.run_times) * 1000 > 900:
                print(f"Mean Run time: {np.mean(self.run_times) * 1000} ms")

        self.macaron_position = self.current_position.get("MAGNIFICENT_MACARONS", 0)
        return self._log_trades(self.file_name)
    
    def _log_trades(self, filename: str = None):
        if filename is None:
            return 

        self.market_data['profit_and_loss'] = self.pnl_history

        output = ""
        output += "Sandbox logs:\n"
        for i in self.sandbox_logs:
            output += json.dumps(i, indent=2) + "\n"

        output += "\n\n\n\nActivities log:\n"
        market_data_csv = self.market_data.to_csv(index=False, sep=";")
        market_data_csv = market_data_csv.replace("\r\n", "\n")
        output += market_data_csv

        output += "\n\n\n\nTrade History:\n"
        output += json.dumps(self.trades, indent=2)

        with open(filename, 'w') as file:
            file.write(output)

            
    def _add_trades(self, own_trades: Dict[str, List[Trade]], market_trades: Dict[str, List[Trade]]):
        products = set(own_trades.keys()) | set(market_trades.keys())
        for product in products:
            self.trades.extend([self._trade_to_dict(trade) for trade in own_trades.get(product, [])])
        for product in products:
            self.trades.extend([self._trade_to_dict(trade) for trade in market_trades.get(product, [])])

    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        return {
            "timestamp": trade.timestamp,
            "buyer": trade.buyer,
            "seller": trade.seller,
            "symbol": trade.symbol,
            "currency": "SEASHELLS",
            "price": trade.price,
            "quantity": trade.quantity,
        }
        
    def _construct_trading_state(self, traderData, timestamp, listings, order_depths, 
                             own_trades, market_trades, position, observation_obj):
        """
        Wrap the conversion observation into a full Observation instance,
        as expected by the TradingState.
        """
        observations = Observation(
            plainValueObservations={},  # could be extended later if needed
            conversionObservations=observation_obj.conversionObservations
        )

        state = TradingState(
            traderData,
            timestamp,
            listings,
            order_depths,
            own_trades,
            market_trades,
            position,
            observations
        )
        return state
    
        
    def _construct_order_depths(self, group):
        order_depths = {}
        for idx, row in group.iterrows():
            product = row['product']
            order_depth = OrderDepth()
            for i in range(1, 4):
                if f'bid_price_{i}' in row and f'bid_volume_{i}' in row:
                    bid_price = row[f'bid_price_{i}']
                    bid_volume = row[f'bid_volume_{i}']
                    if not pd.isna(bid_price) and not pd.isna(bid_volume):
                        order_depth.buy_orders[int(bid_price)] = int(bid_volume)
                if f'ask_price_{i}' in row and f'ask_volume_{i}' in row:
                    ask_price = row[f'ask_price_{i}']
                    ask_volume = row[f'ask_volume_{i}']
                    if not pd.isna(ask_price) and not pd.isna(ask_volume):
                        order_depth.sell_orders[int(ask_price)] = -int(ask_volume)
            order_depths[product] = order_depth
        return order_depths
    
        
        
    def _execute_buy_order(self, timestamp, order, order_depths, position, cash, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]

        # Execute against sell order book
        for price, volume in list(order_depth.sell_orders.items()):
            if price > order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(trade_volume + position[order.symbol]) <= int(self.position_limit[order.symbol]):
                trades.append(Trade(order.symbol, price, trade_volume, "SUBMISSION", "", timestamp))
                position[order.symbol] += trade_volume
                self.cash[order.symbol] -= price * trade_volume
                order_depth.sell_orders[price] += trade_volume
                order.quantity -= trade_volume
            else:
                sandboxLog += f"\nOrders for {order.symbol} exceeded limit {self.position_limit[order.symbol]}"

            if order_depth.sell_orders[price] == 0:
                del order_depth.sell_orders[price]

        # Execute against historical trades
        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol and trade.price < order.price:
                trade_volume = min(abs(order.quantity), abs(trade.quantity))

                if abs(position[order.symbol] + trade_volume) > self.position_limit[order.symbol]:
                    #sandboxLog += f"\nSkipped historical trade for {order.symbol} — would exceed limit."
                    new_trades_at_timestamp.append(trade)
                    continue

                trades.append(Trade(order.symbol, order.price, trade_volume, "SUBMISSION", "", timestamp))
                order.quantity -= trade_volume
                position[order.symbol] += trade_volume
                self.cash[order.symbol] -= order.price * trade_volume

                if trade_volume < abs(trade.quantity):
                    new_quantity = trade.quantity - trade_volume
                    new_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
            else:
                new_trades_at_timestamp.append(trade)

        if new_trades_at_timestamp:
            trade_history_dict[timestamp] = new_trades_at_timestamp

        return trades, sandboxLog
        
        
        
    def _execute_sell_order(self, timestamp, order, order_depths, position, cash, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]

        # Execute against buy order book
        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if price < order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(position[order.symbol] - trade_volume) <= int(self.position_limit[order.symbol]):
                trades.append(Trade(order.symbol, price, trade_volume, "", "SUBMISSION", timestamp))
                position[order.symbol] -= trade_volume
                self.cash[order.symbol] += price * abs(trade_volume)
                order_depth.buy_orders[price] -= abs(trade_volume)
                order.quantity += trade_volume
            else:
                sandboxLog += f"\nOrders for {order.symbol} exceeded limit {self.position_limit[order.symbol]}"

            if order_depth.buy_orders[price] == 0:
                del order_depth.buy_orders[price]

        # Execute against historical trades
        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol and trade.price > order.price:
                trade_volume = min(abs(order.quantity), abs(trade.quantity))

                if abs(position[order.symbol] - trade_volume) > self.position_limit[order.symbol]:
                    #sandboxLog += f"\nSkipped historical trade for {order.symbol} — would exceed limit."
                    new_trades_at_timestamp.append(trade)
                    continue

                trades.append(Trade(order.symbol, order.price, trade_volume, "", "SUBMISSION", timestamp))
                order.quantity += trade_volume
                position[order.symbol] -= trade_volume
                self.cash[order.symbol] += order.price * trade_volume

                if trade_volume < abs(trade.quantity):
                    new_quantity = trade.quantity - trade_volume
                    new_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
            else:
                new_trades_at_timestamp.append(trade)

        if new_trades_at_timestamp:
            trade_history_dict[timestamp] = new_trades_at_timestamp

        return trades, sandboxLog
        
        
        
    def _execute_order(self, timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog):
        if order.quantity == 0:
            return []
        order_depth = order_depths[order.symbol]
        if order.quantity > 0:
            return self._execute_buy_order(timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog)
        else:
            return self._execute_sell_order(timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog)
    
    def _execute_conversion(self, conversions, order_depths, position, cash, observation):
        if conversions == 0 or conversions is None:
            return

        product = "MAGNIFICENT_MACARONS"
        pos = position[product]
        limit = self.position_limit[product]

        # Enforce IMC: convert only what you have (or can absorb), max 10
        max_convertible = min(abs(pos), 10) if conversions < 0 else min(10, limit - pos)

        if abs(conversions) > max_convertible:
            return

        obs = observation

        transport = obs.transportFees
        import_tariff = obs.importTariff
        export_tariff = obs.exportTariff

        if conversions > 0:
            # Buying from Pristine Cuisine
            total_cost = (obs.askPrice + transport + import_tariff) * conversions
            position[product] += conversions
            cash[product] -= total_cost
            self.macaron_conversion_count += conversions
        else:
            # Selling to Pristine Cuisine
            total_revenue = (obs.bidPrice - transport - export_tariff) * abs(conversions)
            position[product] += conversions  # conversions < 0
            cash[product] += total_revenue
            self.macaron_conversion_count += abs(conversions)


    def _mark_pnl(self, cash, position, order_depths, pnl, product):
        order_depth = order_depths.get(product)
        pos = position[product]

        # Default fallback: no order book
        if not order_depth:
            pnl[product] = cash[product]
            return

        # Special handling for MAGNIFICENT_MACARONS
        if product == "MAGNIFICENT_MACARONS":
            if self.observation_data is not None:
                current_obs_row = self.observation_data[
                    self.observation_data['timestamp'] == self.market_data[self.market_data['product'] == product]['timestamp'].iloc[0]
                ]

                if not current_obs_row.empty:
                    obs = current_obs_row.iloc[0]

                    bid_val = obs["bidPrice"] - obs["transportFees"] - obs["exportTariff"]
                    ask_val = obs["askPrice"] + obs["transportFees"] + obs["importTariff"]

                    if pos > 0:
                        unrealized = pos * bid_val
                        storage_cost = 0.1 * pos 
                    elif pos < 0:
                        unrealized = pos * ask_val
                        storage_cost = 0
                    else:
                        unrealized = 0
                        storage_cost = 0

                    self.unrealized_pnl = unrealized
                    pnl[product] = cash[product] + unrealized - storage_cost
                    return

            # fallback when no observation data is available
            pnl[product] = cash[product]
            return

        # Default handling for all other products
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        mid = None
        if best_bid is not None and best_ask is not None:
            mid = (best_ask + best_bid) / 2

        fair = mid
        if product in self.fair_marks:
            get_fair = self.fair_marks[product]
            fair = get_fair(order_depth)

        if fair is None:
            pnl[product] = None
            return

        pnl[product] = cash[product] + (pos * fair)
