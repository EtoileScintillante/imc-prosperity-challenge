# ====================== #
# CODE USED IN TUTORIAL  #
# ====================== #


from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import math

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        """
        Generate orders for RAINFOREST_RESIN based on a fixed fair value strategy.

        Args:
            order_depth (OrderDepth): The current order book for RAINFOREST_RESIN.
            fair_value (int): Assumed static fair value for the product.
            position (int): Current position in the product.
            position_limit (int): Maximum allowable position.

        Returns:
            List[Order]: A list of buy/sell orders.
        """
        orders = [] # Initialize orders list
        buy_order_volume = 0 # To keep track of the buy and sell volumes
        sell_order_volume = 0 # Needed to flatten our positions (see flatten() function)

        # Can't trade without a book
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders 

        # Since the value of RESIN does not fluctuate that much, we check for potential 
        # buy and sell opportunities by checking prices above and below the fair value
        # First we check for buy options
        if len(order_depth.sell_orders) != 0:
            # get best ask price and its amount
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value: # Lower than our set fair value?
                quantity = min(best_ask_amount, position_limit - position) # Max amount we can buy
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", int(best_ask), quantity)) 
                    buy_order_volume += quantity

        # Now check for sell options
        if len(order_depth.buy_orders) != 0:
            # get best bid price and its amount
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value: # Higher than fair value?
                quantity = min(best_bid_amount, position_limit + position) # max amount we can sell
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", int(best_bid), -quantity))
                    sell_order_volume += quantity
        
        # Flatten positions (updates buy and sell volumes)
        buy_order_volume, sell_order_volume = self.flatten(orders, 
                order_depth, position, position_limit, "RAINFOREST_RESIN", 
                buy_order_volume, sell_order_volume, fair_value
        )

        # Market Making/Providing liquidity 
        # we filter for sell prices above fair value (potentially overpriced)
        # and buy prices below fair value (potentially underpriced), then:
        # - we place a buy order at (best_below_bid_fallback + 1) to outbid the current best bid by 1
        # - we place a sell order at (best_above_ask_fallback - 1) to undercut the current best ask by 1
        sell_candidates = [price for price in order_depth.sell_orders.keys() if price > fair_value]
        best_above_ask_fallback = min(sell_candidates) if sell_candidates else fair_value + 1
        buy_candidates = [price for price in order_depth.buy_orders.keys() if price < fair_value]
        best_below_bid_fallback = max(buy_candidates) if buy_candidates else fair_value - 1

        buy_quantity = position_limit - (position + buy_order_volume) # Calculate amount we can buy
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(best_below_bid_fallback + 1), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume) # Calculate amount we can sell
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(best_above_ask_fallback - 1), -sell_quantity))  # Sell order

        return orders
    
    def flatten(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float) -> List[Order]:    
        """
        Place position-flattening orders near the fair value to reduce inventory risk.
        This function aims to bring the net position closer to zero (flattening);
        it tries to clear excess exposure using the available liquidity at or near the fair value.

        Args:
            orders (List[Order]): Current list of orders to append to.
            order_depth (OrderDepth): Order book for the product.
            position (int): Current position.
            position_limit (int): Maximum allowable position.
            product (str): Name of the product being traded.
            buy_order_volume (int): Quantity bought so far.
            sell_order_volume (int): Quantity sold so far.
            fair_value (float): Fair value to target for flattening.

        Returns:
            List[Order]: Buy and sell volumes.
        """
        # This function aims to bring the net position closer to zero;
        # It tries to clear excess exposure using the available liquidity at or near the fair value
        # First calculate net position, i.e. adding our buy volume and
        # subtracting sell volume to/from our current position
        net_position = position + buy_order_volume - sell_order_volume

        # Then we calculate fair prices for selling and buying
        good_bid_price = math.floor(fair_value)
        good_ask_price = math.ceil(fair_value)

        # Calculate the maximum amounts we could sell and buy
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # Create orders:
        # If our net position is positive (i.e. we are long), we sell
        # Check if a bot has a buy order out for the price we set
        if net_position > 0:
            if good_ask_price in order_depth.buy_orders.keys():
                flatten_amount = min(order_depth.buy_orders[good_ask_price], net_position) # Max amount possible for flatten
                actual_amount = min(sell_quantity, flatten_amount) # Amount we can actually sell
                orders.append(Order(product, int(good_ask_price), -abs(actual_amount)))
                sell_order_volume += abs(actual_amount)

        # If our net position is negative (short), we can buy
        # Check if a bot has a sell order out for the price we set
        if net_position < 0:
            if good_bid_price in order_depth.sell_orders.keys():
                flatten_amount = min(abs(order_depth.sell_orders[good_bid_price]), abs(net_position)) # Max amount possible for flatten
                actual_amount = min(buy_quantity, flatten_amount) # Amount we can actually buy
                orders.append(Order(product, int(good_bid_price), abs(actual_amount)))
                buy_order_volume += abs(actual_amount)
        
        return buy_order_volume, sell_order_volume
    
    def kelp_orders(self, order_depth: OrderDepth, timespan: int, take_width: float, position: int, position_limit: int, min_vol_threshold: int,vwap: bool = True) -> List[Order]:
        """
        Generate orders for KELP based on filtered midpoint or VWAP.

        Args:
            order_depth (OrderDepth): The current order book for KELP.
            timespan (int): The lookback window for VWAP smoothing.
            width (float): The width used for flattening logic.
            kelp_take_width (float): Minimum edge to trigger market-taking.
            position (int): Current position in KELP.
            position_limit (int): Maximum allowable position.
            min_vol_threshold (int): Minimum order book volume to consider a level valid.
            vwap (bool): Whether to use VWAP or filtered midpoint as fair value.

        Returns:
            List[Order]: A list of KELP orders.
        """
        orders = [] # Initialize orders list
        buy_order_volume = 0 # To keep track of the buy and sell volumes
        sell_order_volume = 0 # Needed to flatten our positions (see flatten() function)

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders  # Can't trade without a book

        # In this function we calculate the VWAP if flagged true
        # Otherwise the estimated fair value for KELP will be based on a midpoint:
        # best_valid_ask + best_valid_bid divided by 2
        # So first we filter the valid asks and bids (valid meaning enough volume)
        valid_asks = [price for price, volume in order_depth.sell_orders.items() if abs(volume) >= min_vol_threshold]
        valid_bids = [price for price, volume in order_depth.buy_orders.items() if abs(volume) >= min_vol_threshold]

        # If valid_asks/valid_bids is empty, we just take the best available, ignoring the volume
        best_ask = min(order_depth.sell_orders)
        best_bid = max(order_depth.buy_orders)
        best_valid_ask = min(valid_asks) if valid_asks else best_ask
        best_valid_bid = max(valid_bids) if valid_bids else best_bid

        # Midpoint
        midpoint = (best_valid_ask + best_valid_bid) / 2
        self.kelp_prices.append(midpoint)

        # Now calculate VWAP
        # Note: here we calculate the 'spot' VWAP, we use only the best bid and best ask at *this moment* in time
        # Later we calculate the rolling VWAP, using historical spot VWAPs over a time window (timespan)
        # This should smooth out noise, dampening quick spikes and helping avoid overreacting to short-lived moves
        volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
        if volume != 0:
            spot_vwap = (
                best_bid * order_depth.buy_orders[best_bid] + 
                best_ask * -order_depth.sell_orders[best_ask]
            ) / volume
        else:
            spot_vwap = midpoint  # fallback

        # Store spot vwap
        self.kelp_vwap.append({"vol": volume, "vwap": spot_vwap})

        # If old data exceeds the timespan, remove oldest entries (more memory efficient)
        # We won't use that data anyway
        if len(self.kelp_vwap) > timespan:
            self.kelp_vwap.pop(0)
        if len(self.kelp_prices) > timespan:
            self.kelp_prices.pop(0)

        # Now calculate the final fair value, i.e. the one based on rolling VWAP
        if vwap:
            # For this we need the total volume
            total_volume = sum(x["vol"] for x in self.kelp_vwap)
            if total_volume > 0:
                fair_value = sum(x["vwap"] * x["vol"] for x in self.kelp_vwap) / total_volume
            else:
                fair_value = midpoint
        else:
            fair_value = midpoint

        # Market-taking logic
        if best_valid_ask <= fair_value - take_width: # Note the use of take_width!
            ask_volume = -order_depth.sell_orders[best_valid_ask] # Amount available
            quantity = min(ask_volume, position_limit - position) # Amount we can buy
            if quantity > 0:
                orders.append(Order("KELP", int(best_valid_ask), quantity))
                buy_order_volume += quantity

        if best_valid_bid >= fair_value + take_width: # Note the use of take_width!
            bid_volume = order_depth.buy_orders[best_valid_bid] # Amount available
            quantity = min(bid_volume, position_limit + position) # Amount we can sell
            if quantity > 0:
                orders.append(Order("KELP", int(best_valid_bid), -quantity))
                sell_order_volume += quantity

        # Flatten
        buy_order_volume, sell_order_volume = self.flatten(
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

        buy_quantity = position_limit - (position + buy_order_volume) # Amount we can buy
        if buy_quantity > 0:
            orders.append(Order("KELP", int(passive_bid + 1), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume) # Amount we can sell
        if sell_quantity > 0:
            orders.append(Order("KELP", int(passive_ask - 1), -sell_quantity))

        return orders
    

    def run(self, state: TradingState):
        result = {}

        # Resin related variables
        resin_fair_value = 10000
        resin_position_limit = 50

        # Kelp related variables
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
            resin_position = state.position.get("RAINFOREST_RESIN", 0) 
            resin_orders = self.resin_orders(state.order_depths["RAINFOREST_RESIN"], resin_fair_value, resin_position, resin_position_limit)
            result["RAINFOREST_RESIN"] = resin_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0) 
            kelp_orders = self.kelp_orders(state.order_depths["KELP"], kelp_timemspan, kelp_take_width, kelp_position, kelp_position_limit, kelp_min_vol_thresh, kelp_vwap)
            result["KELP"] = kelp_orders

        
        traderData = jsonpickle.encode( { "kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap})
        conversions = 1
        return result, conversions, traderData