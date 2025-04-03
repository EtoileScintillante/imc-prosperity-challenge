# ============================================
# STRATEGY OVERVIEW
# ============================================
# This trading bot operates on two products: RAINFOREST_RESIN and KELP.
# RAINFOREST_RESIN is stable, KELP is volatile.

# For each product, the bot attempts to:
# 1. Take advantage of mispriced orders via aggressive (market-taking) trades.
# 2. Provide passive liquidity using limit orders just inside the spread.
# 3. Clear position if needed near fair value to stay within limits.

# ============================================
# RAINFOREST_RESIN STRATEGY
# ============================================
# - Assumes a fixed fair value (e.g. 10000).
# - Takes any best ask that is under fair value (buy low).
# - Takes any best bid that is over fair value (sell high).
# - Posts passive limit orders just inside the spread:
#     • Buys at (max bid below fair value) + 1
#     • Sells at (min ask above fair value) - 1
# - Clears position when it deviates from zero using market orders near fair value.

# ============================================
# KELP STRATEGY
# ============================================
# - Computes fair value using volume-weighted average price (VWAP) over a time window.
#     • Top-level VWAP is computed from best bid and ask.
#     • If VWAP is enabled, rolling VWAP is used as fair value.
#     • Otherwise, filtered midpoint is used.
# - Filters bids and asks based on volume threshold to avoid reacting to "dust".
# - Takes any best ask under (fair value - kelp_take_width).
# - Takes any best bid above (fair value + kelp_take_width).
# - Clears position when necessary using near-fair-value market orders.
# - Posts passive liquidity at prices unlikely to execute immediately:
#     • Buys at (max bid below fair) + 1
#     • Sells at (min ask above fair) - 1

# ============================================
# RISK MANAGEMENT
# ============================================
# - Both products have position limits (e.g. ±50).
# - Orders are sized to never exceed position limit.
# - clear_position_order() is used to flatten positions near fair value
#   using natural liquidity when current position is not neutral.
#   (Flattening means reducing an open position back toward zero, minimizing exposure.
#   This helps lock in profits or limit losses, especially when price is close to fair value)

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
        return result, conversions, traderData