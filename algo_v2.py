from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import math

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_timestamps = []

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
        best_above_ask_fallback = min(sell_candidates) if sell_candidates else fair_value + 1

        buy_candidates = [price for price in order_depth.buy_orders.keys() if price < fair_value]
        best_below_bid_fallback = max(buy_candidates) if buy_candidates else fair_value - 1

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
            orders.append(Order("RAINFOREST_RESIN", int(best_below_bid_fallback + 1), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(best_above_ask_fallback - 1), -sell_quantity))  # Sell order

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

    def kelp_orders(self, order_depth: OrderDepth, timestamp: int, position: int, position_limit: int) -> List[Order]:
        """
        Regression-based strategy for KELP.
        Uses linear regression on recent price history to predict the next price and trade accordingly.
        """
        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        self.kelp_prices.append(mid_price)
        self.kelp_timestamps.append(timestamp)

        # Check if enough data to perform regression
        if len(self.kelp_prices) > 20:
            self.kelp_prices.pop(0)
            self.kelp_timestamps.pop(0)

        if len(self.kelp_prices) < 20:
            return orders  # Not enough data yet

        # Linear regression to predict next mid_price
        X = np.array(self.kelp_timestamps)
        y = np.array(self.kelp_prices)
        A = np.vstack([X, np.ones(len(X))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        predicted_price = m * (timestamp + 100) + c  # Predict next timestamp mid

        # Market-taking if prices are favorable
        if best_ask < predicted_price:
            volume = min(position_limit - position, -order_depth.sell_orders[best_ask])
            if volume > 0:
                orders.append(Order("KELP", int(best_ask), volume))

        if best_bid > predicted_price:
            volume = min(position_limit + position, order_depth.buy_orders[best_bid])
            if volume > 0:
                orders.append(Order("KELP", int(best_bid), -volume))

        # Flattening
        buy_volume, sell_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "KELP", buy_volume, sell_volume, predicted_price
        )

        # Light market-making
        buy_price = int(predicted_price - 1)
        sell_price = int(predicted_price + 1)

        buy_vol = min(position_limit - (position + buy_volume), 15)
        sell_vol = min(position_limit + (position - sell_volume), 15)

        if buy_vol > 0:
            orders.append(Order("KELP", buy_price, buy_vol))
        if sell_vol > 0:
            orders.append(Order("KELP", sell_price, -sell_vol))

        return orders

    def run(self, state: TradingState):
        result = {}

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders = self.resin_orders(state.order_depths["RAINFOREST_RESIN"], fair_value=10000, position=resin_position, position_limit=50)
            result["RAINFOREST_RESIN"] = resin_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(state.order_depths["KELP"], timestamp=state.timestamp, position=kelp_position, position_limit=50)
            result["KELP"] = kelp_orders

        traderData = jsonpickle.encode({"kelp_prices": self.kelp_prices})
        conversions = 1
        return result, conversions, traderData
