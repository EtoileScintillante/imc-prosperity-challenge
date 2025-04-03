# Run underneath command to visualize the results of the algo
# Command to use in terminal: prosperity3bt vis.py 0 --vis 
# (or some other number than 0, depends on what is available)
# Credits: https://jmerle.github.io/imc-prosperity-3-visualizer/?/assets/index-q2HCHXPI.js:133:987
# Must have installed: https://github.com/jmerle/imc-prosperity-3-backtester

import json
from typing import Any, List
import jsonpickle
import numpy as np
import math

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        
        sell_candidates = [price for price in order_depth.sell_orders.keys() if price > fair_value]
        baaf = min(sell_candidates) if sell_candidates else fair_value + 1

        buy_candidates = [price for price in order_depth.buy_orders.keys() if price < fair_value]
        bbbf = max(buy_candidates) if buy_candidates else fair_value - 1

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", int(best_ask), quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", int(best_bid), -quantity))
                    sell_order_volume += quantity
        
        buy_order_volume, sell_order_volume = self.clear_position_order(orders, 
                order_depth, position, position_limit, "RAINFOREST_RESIN", 
                buy_order_volume, sell_order_volume, fair_value
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(bbbf + 1), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(baaf - 1), -sell_quantity))  # Sell order

        return orders
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float) -> List[Order]:    
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        
    
        return buy_order_volume, sell_order_volume
    
    def kelp_orders(self, order_depth: OrderDepth, timespan: int, kelp_take_width: float, position: int, position_limit: int, min_vol_threshold: int,vwap: bool = True) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders  # Can't trade without a book

        # Filter valid levels based on min volume threshold
        valid_asks = [p for p, v in order_depth.sell_orders.items() if abs(v) >= min_vol_threshold]
        valid_bids = [p for p, v in order_depth.buy_orders.items() if abs(v) >= min_vol_threshold]

        # Fallback to best available if filters fail
        best_ask = min(order_depth.sell_orders)
        best_bid = max(order_depth.buy_orders)
        best_valid_ask = min(valid_asks) if valid_asks else best_ask
        best_valid_bid = max(valid_bids) if valid_bids else best_bid

        # Midpoint (from filtered or fallback)
        filtered_mid = (best_valid_ask + best_valid_bid) / 2
        self.kelp_prices.append(filtered_mid)

        # VWAP calc from top levels
        volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
        if volume != 0:
            top_vwap = (
                best_bid * order_depth.buy_orders[best_bid] + 
                best_ask * -order_depth.sell_orders[best_ask]
            ) / volume
        else:
            top_vwap = filtered_mid  # fallback

        self.kelp_vwap.append({"vol": volume, "vwap": top_vwap})

        if len(self.kelp_vwap) > timespan:
            self.kelp_vwap.pop(0)
        if len(self.kelp_prices) > timespan:
            self.kelp_prices.pop(0)

        # Final fair value
        if vwap:
            total_volume = sum(x["vol"] for x in self.kelp_vwap)
            if total_volume > 0:
                fair_value = sum(x["vwap"] * x["vol"] for x in self.kelp_vwap) / total_volume
            else:
                fair_value = filtered_mid
        else:
            fair_value = filtered_mid

        # Market-taking logic
        if best_valid_ask <= fair_value - kelp_take_width:
            ask_volume = -order_depth.sell_orders[best_valid_ask]
            quantity = min(ask_volume, position_limit - position)
            if quantity > 0:
                orders.append(Order("KELP", int(best_valid_ask), quantity))
                buy_order_volume += quantity

        if best_valid_bid >= fair_value + kelp_take_width:
            bid_volume = order_depth.buy_orders[best_valid_bid]
            quantity = min(bid_volume, position_limit + position)
            if quantity > 0:
                orders.append(Order("KELP", int(best_valid_bid), -quantity))
                sell_order_volume += quantity

        # Position clearing
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "KELP",
            buy_order_volume, sell_order_volume, fair_value
        )

        # Passive orders (liquidity provision)
        # We post passive limit orders to provide liquidity
        # i.e. placing orders that are unlikely to execute immediately
        # but may fill later at favorable prices
        passive_asks = [p for p in order_depth.sell_orders if p > fair_value + 1]
        passive_bids = [p for p in order_depth.buy_orders if p < fair_value - 1]
        passive_ask = min(passive_asks) if passive_asks else fair_value + 2
        passive_bid = max(passive_bids) if passive_bids else fair_value - 2

        buy_qty = position_limit - (position + buy_order_volume)
        if buy_qty > 0:
            orders.append(Order("KELP", int(passive_bid + 1), buy_qty))

        sell_qty = position_limit + (position - sell_order_volume)
        if sell_qty > 0:
            orders.append(Order("KELP", int(passive_ask - 1), -sell_qty))

        return orders
    

    def run(self, state: TradingState):
        result = {}

        resin_fair_value = 10000
        resin_position_limit = 50

        # Market-taking threshold: if price deviates this far from fair value, we hit the market to take liquidity
        kelp_take_width = 1
        kelp_position_limit = 50

        # VWAP smoothing window: how many recent ticks of price/volume history we consider for VWAP fair value calculation in KELP
        kelp_timemspan = 9

        # Only consider order book levels with at least this volume when calculating valid bids/asks to avoid reacting to low-liquidity "fake" orders
        kelp_min_vol_thresh = 15

        # If False: midpoint strategy will be used
        kelp_vwap = True 

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            resin_orders = self.resin_orders(state.order_depths["RAINFOREST_RESIN"], resin_fair_value, resin_position, resin_position_limit)
            result["RAINFOREST_RESIN"] = resin_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position["KELP"] if "KELP" in state.position else 0
            kelp_orders = self.kelp_orders(state.order_depths["KELP"], kelp_timemspan, kelp_take_width, kelp_position, kelp_position_limit, kelp_min_vol_thresh, kelp_vwap)
            result["KELP"] = kelp_orders

        
        traderData = jsonpickle.encode( { "kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap})
        conversions = 1
        logger.flush(state, result, conversions, traderData) # THIS LINE MUST BE HERE
        return result, conversions, traderData