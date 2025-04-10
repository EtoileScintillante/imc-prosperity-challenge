# ===================== #
# CODE USED IN ROUND 1 #
# ===================== #

from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import math
import pandas as pd


class Trader:
    def __init__(self):
        # Initialize lists to store historical data
        self.kelp_prices = []
        self.kelp_vwap = []
        self.squid_log_prices = []

    def resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        take_width: float,
        position: int,
        position_limit: int,
    ) -> List[Order]:
        """
        Generate orders for RAINFOREST_RESIN based on a fixed fair value strategy.

        Args:
            order_depth (OrderDepth): The current order book for RAINFOREST_RESIN.
            fair_value (int): Assumed static fair value for the product.
            take_width (float): How far price must deviate from fair value to take.
            position (int): Current position in the product.
            position_limit (int): Maximum allowable position.

        Returns:
            List[Order]: A list of buy/sell orders.
        """
        orders = []  # Initialize orders list
        buy_order_volume = 0  # To keep track of the buy and sell volumes
        sell_order_volume = (
            0  # Needed to flatten our positions (see flatten() function)
        )

        # Can't trade without a book
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        # Since the value of RESIN does not fluctuate that much, we check for potential
        # buy and sell opportunities by checking prices above and below the fair value
        # First we check for buy options
        if len(order_depth.sell_orders) != 0:
            # get best ask price and its amount
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders.get(best_ask, 0)
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", int(best_ask), quantity))
                    buy_order_volume += quantity

        # Now check for sell options
        if len(order_depth.buy_orders) != 0:
            # get best bid price and its amount
            best_bid = max(order_depth.buy_orders.keys())
            
            # Use .get for safety
            best_bid_amount = order_depth.buy_orders.get(best_bid, 0)
            #!! Could buy more than just the best one ?
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", int(best_bid), -quantity))
                    sell_order_volume += quantity

        # Flatten positions (updates buy and sell volumes)
        buy_order_volume, sell_order_volume = self.flatten(
            orders,
            order_depth,
            position,
            position_limit,
            "RAINFOREST_RESIN",
            buy_order_volume,
            sell_order_volume,
            fair_value,
        )

        # Market Making/Providing liquidity
        # we filter for sell prices above fair value (potentially overpriced)
        # and buy prices below fair value (potentially underpriced), then:
        # - we place a buy order at (best_below_bid_fallback + 1) to outbid the current best bid by 1
        # - we place a sell order at (best_above_ask_fallback - 1) to undercut the current best ask by 1
        sell_candidates = [
            price for price in order_depth.sell_orders.keys() if price > fair_value
        ]
        best_above_ask_fallback = (
            min(sell_candidates) if sell_candidates else fair_value + 1
        )
        buy_candidates = [
            price for price in order_depth.buy_orders.keys() if price < fair_value
        ]
        best_below_bid_fallback = (
            max(buy_candidates) if buy_candidates else fair_value - 1
        )

        buy_quantity = position_limit - (
            position + buy_order_volume
        )  # Calculate amount we can buy
        if buy_quantity > 0:
            orders.append(
                Order(
                    "RAINFOREST_RESIN", int(best_below_bid_fallback + 1), buy_quantity
                )
            )  # Buy order

        sell_quantity = position_limit + (
            position - sell_order_volume
        )  # Calculate amount we can sell
        if sell_quantity > 0:
            orders.append(
                Order(
                    "RAINFOREST_RESIN", int(best_above_ask_fallback - 1), -sell_quantity
                )
            )  # Sell order

        return orders

    def flatten(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
    ) -> List[Order]:
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

        # Calculate the maximum amounts we could sell and buy
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # NOTE!
        # So this logic has been changed a bit
        # Instead of just looking if our estimated 'good enough' price is in the orderbook
        # We check if any price equal or better than our minimum buy/sell is in there
        # Sell side flattening 
        if net_position > 0:
            # Loop over bid prices
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):  # Highest bid first
                if price >= fair_value:
                    volume = order_depth.buy_orders[price]
                    flatten_amount = min(volume, net_position)
                    actual_amount = min(sell_quantity, flatten_amount)
                    if actual_amount > 0:
                        orders.append(Order(product, int(price), -actual_amount))
                        sell_order_volume += actual_amount
                        net_position -= actual_amount
                        if net_position <= 0: # Done flattening
                            break

        # Buy side flattening 
        if net_position < 0:
            # Loop over ask prices
            for price in sorted(order_depth.sell_orders.keys()):  # Lowest ask first
                if price <= fair_value:
                    volume = -order_depth.sell_orders[price] 
                    flatten_amount = min(volume, abs(net_position))
                    actual_amount = min(buy_quantity, flatten_amount)
                    if actual_amount > 0:
                        orders.append(Order(product, int(price), actual_amount))
                        buy_order_volume += actual_amount
                        net_position += actual_amount
                        if net_position >= 0: # Done flattening
                            break

        return buy_order_volume, sell_order_volume


    def kelp_orders(
        self,
        order_depth: OrderDepth,
        timespan: int,
        take_width: float,
        position: int,
        position_limit: int,
        min_vol_threshold: int,
        vwap: bool,
        prevent_adverse: bool,
        adverse_volume: int,
    ) -> List[Order]:
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
        orders = []  # Initialize orders list
        buy_order_volume = 0  # To keep track of the buy and sell volumes
        sell_order_volume = (
            0  # Needed to flatten our positions (see flatten() function)
        )

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders  # Can't trade without a book

        # In this function we calculate the VWAP if flagged true
        # Otherwise the estimated fair value for KELP will be based on a midpoint:
        # best_valid_ask + best_valid_bid divided by 2
        # So first we filter the valid asks and bids (valid meaning enough volume)
        valid_asks = [
            price
            for price, volume in order_depth.sell_orders.items()
            if abs(volume) >= min_vol_threshold
        ]
        valid_bids = [
            price
            for price, volume in order_depth.buy_orders.items()
            if abs(volume) >= min_vol_threshold
        ]

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
                best_bid * order_depth.buy_orders[best_bid]
                + best_ask * -order_depth.sell_orders[best_ask]
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
                fair_value = (
                    sum(x["vwap"] * x["vol"] for x in self.kelp_vwap) / total_volume
                )
            else:
                fair_value = midpoint
        else:
            fair_value = midpoint

        # Market-taking logic
        if best_valid_ask in order_depth.sell_orders:
            ask_volume = -order_depth.sell_orders[best_valid_ask]
            if best_valid_ask <= fair_value - take_width and (
                not prevent_adverse or ask_volume <= adverse_volume
            ):
                quantity = min(ask_volume, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", int(best_valid_ask), quantity))
                    buy_order_volume += quantity
        if best_valid_bid in order_depth.buy_orders:
            bid_volume = order_depth.buy_orders[best_valid_bid]
            if best_valid_bid >= fair_value + take_width and (
                not prevent_adverse or bid_volume <= adverse_volume
            ):
                quantity = min(bid_volume, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", int(best_valid_bid), -quantity))
                    sell_order_volume += quantity

        # Flatten
        buy_order_volume, sell_order_volume = self.flatten(
            orders,
            order_depth,
            position,
            position_limit,
            "KELP",
            buy_order_volume,
            sell_order_volume,
            fair_value,
        )

        # Passive orders (liquidity provision)
        # We post passive limit orders to provide liquidity
        # i.e. placing orders that are unlikely to execute immediately
        # but may fill later at favorable prices
        passive_asks = [p for p in order_depth.sell_orders if p > fair_value + 1]
        passive_bids = [p for p in order_depth.buy_orders if p < fair_value - 1]
        passive_ask = min(passive_asks) if passive_asks else fair_value + 2
        passive_bid = max(passive_bids) if passive_bids else fair_value - 2

        buy_quantity = position_limit - (
            position + buy_order_volume
        )  # Amount we can buy
        if buy_quantity > 0:
            orders.append(Order("KELP", int(passive_bid + 1), buy_quantity))

        sell_quantity = position_limit + (
            position - sell_order_volume
        )  # Amount we can sell
        if sell_quantity > 0:
            orders.append(Order("KELP", int(passive_ask - 1), -sell_quantity))

        return orders

    def squid_orders(
        self,
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        timespan: int = 30,  # Increased default timespan for volatility calc
        base_take_width: float = 2.0,  # Base deviation for taking
        volatility_take_factor: float = 60.0,  # Factor scaling take_width with volatility
        base_passive_spread: int = 1,  # Minimum passive spread in ticks
        volatility_spread_factor: float = 60.0,  # Factor scaling passive spread with volatility
        volatility_size_factor: float = 60.0,  # Factor scaling passive size with volatility (inverse)
        min_passive_volume: int = 15,  # Minimum size for passive orders
    ) -> List[Order]:
        """
        Strategy for trading SQUID_INK, a highly volatile asset, using insights from the Rough Fractional Stochastic Volatility (RFSV) model.

        Main Strategy:
        - Adaptive market-taking and market-making using real-time estimates of volatility.
        - The strategy adjusts aggressiveness and order sizing dynamically based on estimated short-term volatility from log returns.

        Paper Insights Used:
        - The RFSV model (Gatheral et al., 2014) shows that log-volatility follows a fractional Brownian motion with a very low Hurst exponent (H ≈ 0.1),
        meaning volatility is highly irregular and "rough" at all timescales.
        - Because volatility spikes and mean-reverts rapidly, trading strategies should:
            * Be cautious with large volumes during high volatility.
            * Quote wider spreads in volatile periods to avoid adverse selection.
            * Scale order sizes inversely with volatility.
        - The function uses log midprice returns over a rolling window to estimate local rough volatility and dynamically scale:
            * The market-taking threshold (take_width)
            * The passive spread (spread)
            * The maximum quote volume (max_passive_volume)
        """
        # Initialize order list and volume trackers for this round
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders  # Can't trade without a book

        # Get best bid and ask prices and calculate midpoint price
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        midpoint = (best_ask + best_bid) / 2

        # We use the midpoint as the simple fair value estimate for this strategy
        fair_value = midpoint

        # Append the log of the current midpoint to our historical list
        # Using log prices helps stabilize variance for volatility calculation
        self.squid_log_prices.append(np.log(midpoint))

        # Maintain a rolling window of log prices
        if len(self.squid_log_prices) > timespan:
            self.squid_log_prices.pop(0)

        # Calculate volatility if we have enough data points (at least 2 for np.diff)
        if len(self.squid_log_prices) >= 2:
            # Calculate log returns (difference between consecutive log prices)
            log_returns = np.diff(self.squid_log_prices)
            # Calculate standard deviation of log returns as our volatility measure
            volatility = np.std(log_returns)
        else:
            # Use a default non-zero volatility if history is too short
            volatility = 0.005  # Can test with other values

        # Calculate dynamic take width --> increases with volatility
        # Requires larger deviation from fair value to take aggressively in volatile markets
        take_width = base_take_width + (volatility * volatility_take_factor)

        # Calculate dynamic passive spread --> also increases with volatility
        # Quote wider in volatile markets to reduce adverse selection risk
        # Ensure spread is at least the base minimum
        spread = max(
            base_passive_spread,
            round(base_passive_spread + volatility * volatility_spread_factor),
        )

        # Calculate dynamic passive volume size --> decreases with volatility (inverse scaling)
        # Quote smaller size when risk (volatility) is high
        # we use max() to ensure a minimum quoting size and add 1 to denominator to prevent division by zero if vola is near zero
        max_passive_volume = max(
            min_passive_volume,
            round(position_limit / (volatility * volatility_size_factor + 1)),
        )

        # Market Taking:
        # Buy if best ask is significantly below fair value
        if best_ask <= fair_value - take_width:
            # Get volume available at best ask
            ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
            if ask_volume > 0:
                quantity_to_buy = min(ask_volume, position_limit - position)
                if quantity_to_buy > 0:
                    orders.append(Order("SQUID_INK", int(best_ask), quantity_to_buy))
                    buy_order_volume += quantity_to_buy

        # Sell if best bid is significantly above fair value
        if best_bid >= fair_value + take_width:
            # Get volume available at best bid
            bid_volume = order_depth.buy_orders.get(best_bid, 0)
            if bid_volume > 0:
                quantity_to_sell = min(bid_volume, position_limit + position)
                if quantity_to_sell > 0:
                    orders.append(Order("SQUID_INK", int(best_bid), -quantity_to_sell))
                    sell_order_volume += quantity_to_sell

        # Flatten
        buy_order_volume, sell_order_volume = self.flatten(
            orders,
            order_depth,
            position,
            position_limit,
            "SQUID_INK",
            buy_order_volume,
            sell_order_volume,
            fair_value,
        )

        # Passive Market Making:
        # Calculate buy and sell prices using the dynamic spread
        # Possible addition: inventory skew (if LONG sell more easily, and if SHORT buy more easily)???
        buy_price = int(fair_value - spread)
        sell_price = int(fair_value + spread)

        # Make sure sell price is at least buy price + 1 (minimum spread)
        if sell_price <= buy_price:
            sell_price = buy_price + 1

        # Calculate remaining capacity after taking and flattening trades
        remaining_buy_capacity = position_limit - (position + buy_order_volume)
        remaining_sell_capacity = position_limit + (position - sell_order_volume)

        # Determine final passive order volume, using dynamic max size and respecting capacity
        buy_volume_passive = min(max_passive_volume, remaining_buy_capacity)
        buy_volume_passive = max(
            0, buy_volume_passive
        )  # Make surte this is non-negative

        sell_volume_passive = min(max_passive_volume, remaining_sell_capacity)
        sell_volume_passive = max(
            0, sell_volume_passive
        )  # Make surte this is non-negative

        # Buy order
        if buy_volume_passive > 0:
            orders.append(Order("SQUID_INK", buy_price, buy_volume_passive))

        # Sell order
        if sell_volume_passive > 0:
            orders.append(Order("SQUID_INK", sell_price, -sell_volume_passive))

        return orders

    def run(self, state: TradingState):
        result = {}


        # --- Define Parameters ---
        # Resin
        resin_fair_value = 10000
        resin_take_width = 0.5
        resin_position_limit = 50

        # Kelp
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timemspan = 9
        kelp_min_vol_thresh = 15
        kelp_vwap = True
        kelp_prevent_adverse = True
        kelp_adverse_volume = 20

        # Squid Ink Parameters (including new ones)
        squid_position_limit = 50
        squid_volatility_timespan = 20  # Lookback window for volatility calc
        squid_base_take_width = 2.0  # Minimum deviation to take
        squid_volatility_take_factor = 50.0  # How much take_width increases with vol
        squid_base_passive_spread = 1  # Minimum passive spread
        squid_volatility_spread_factor = 50.0  # How much passive spread increases with vol
        squid_volatility_size_factor = 50.0  # How much passive size decreases with vol
        squid_min_passive_volume = 5  # Minimum size for passive orders


        if "RAINFOREST_RESIN" in state.order_depths:
             resin_position = state.position.get("RAINFOREST_RESIN", 0)
             resin_orders = self.resin_orders(
                 state.order_depths["RAINFOREST_RESIN"],
                 resin_fair_value,
                 resin_take_width,
                 resin_position,
                 resin_position_limit,
             )
             result["RAINFOREST_RESIN"] = resin_orders


        if "KELP" in state.order_depths:
             kelp_position = state.position.get("KELP", 0)
             kelp_orders = self.kelp_orders(
                 order_depth=state.order_depths["KELP"],
                 timespan=kelp_timemspan,
                 take_width=kelp_take_width,
                 position=kelp_position,
                 position_limit=kelp_position_limit,
                 min_vol_threshold=kelp_min_vol_thresh,
                 vwap=kelp_vwap,
                 prevent_adverse=kelp_prevent_adverse,
                 adverse_volume=kelp_adverse_volume,
             )
             result["KELP"] = kelp_orders


        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_orders = self.squid_orders(
                order_depth=state.order_depths["SQUID_INK"],
                position=squid_position,
                position_limit=squid_position_limit,
            )
            result["SQUID_INK"] = squid_orders

        traderData = jsonpickle.encode(
            {
                "kelp_prices": self.kelp_prices,
                "kelp_vwap": self.kelp_vwap,
                "squid_mid_prices": self.squid_log_prices
            }
        )
        conversions = 0
        return result, conversions, traderData
