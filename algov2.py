# ===================== #
# CODE USED IN ROUND 2  #
# ===================== #

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple
import string
import jsonpickle
import collections
import numpy as np
import math

# In this file:
# Same functions for KELP, SQUID, RESIN
# Function for basket1 and basket2
# New function for JAMS: similar to SQUID function
# JAMS and SQUID have a pretty high correlation so I thought maybe we could use the same function
# Maybe we can do something similar for CROISSANTS? That has a correlation of 0.89 with JAMS
# I haven't tried anything for DJEMBES yet...

# I also noticed that when trying out different param values for functions, that some values might lead
# to a really good PnL in the  backtester, but to a significantly worse PnL on the website
# So yeah we should be careful not to focus only on the results of the backtest haha

# Btw on website: with this algo up until timestep 60k we make zero profit, super weird
# then in the last 40k steps it increases to 12.9k. So weird

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"


BASKET1_WEIGHTS = {
    Product.JAMS: 3,
    Product.CROISSANTS: 6,
    Product.DJEMBES: 1,
}

BASKET2_WEIGHTS = {
    Product.JAMS: 4,
    Product.CROISSANTS: 2,
}


class Trader:
    def __init__(self):
        self.b1_prices = []
        self.b2_prices = []
        self.squid_log_prices = []
        self.jams_log_prices = []
        self.kelp_prices = []
        self.kelp_vwap = []
        self.basket1_std = 122 

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.JAMS: 350,
            Product.CROISSANTS: 250,
            Product.DJEMBES: 60,
        }

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
                    orders.append(Order(Product.KELP, int(best_valid_ask), quantity))
                    buy_order_volume += quantity
        if best_valid_bid in order_depth.buy_orders:
            bid_volume = order_depth.buy_orders[best_valid_bid]
            if best_valid_bid >= fair_value + take_width and (
                not prevent_adverse or bid_volume <= adverse_volume
            ):
                quantity = min(bid_volume, position_limit + position)
                if quantity > 0:
                    orders.append(Order(Product.KELP, int(best_valid_bid), -quantity))
                    sell_order_volume += quantity

        # Flatten
        buy_order_volume, sell_order_volume = self.flatten(
            orders,
            order_depth,
            position,
            position_limit,
            Product.KELP,
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
            orders.append(Order(Product.KELP, int(passive_bid + 1), buy_quantity))

        sell_quantity = position_limit + (
            position - sell_order_volume
        )  # Amount we can sell
        if sell_quantity > 0:
            orders.append(Order(Product.KELP, int(passive_ask - 1), -sell_quantity))

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
        if len(self.squid_log_prices) >= timespan:
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
                    orders.append(Order(Product.SQUID_INK, int(best_ask), quantity_to_buy))
                    buy_order_volume += quantity_to_buy

        # Sell if best bid is significantly above fair value
        if best_bid >= fair_value + take_width:
            # Get volume available at best bid
            bid_volume = order_depth.buy_orders.get(best_bid, 0)
            if bid_volume > 0:
                quantity_to_sell = min(bid_volume, position_limit + position)
                if quantity_to_sell > 0:
                    orders.append(Order(Product.SQUID_INK, int(best_bid), -quantity_to_sell))
                    sell_order_volume += quantity_to_sell

        # Flatten
        buy_order_volume, sell_order_volume = self.flatten(
            orders,
            order_depth,
            position,
            position_limit,
            Product.SQUID_INK,
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
            orders.append(Order(Product.SQUID_INK, buy_price, buy_volume_passive))

        # Sell order
        if sell_volume_passive > 0:
            orders.append(Order(Product.SQUID_INK, sell_price, -sell_volume_passive))

        return orders
    
    def resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        take_width: float,
        position: int,
        position_limit: int,
    ) -> List[Order]:
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
                    orders.append(Order(Product.RAINFOREST_RESIN, int(best_ask), quantity))
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
                    orders.append(Order(Product.RAINFOREST_RESIN, int(best_bid), -quantity))
                    sell_order_volume += quantity

        # Flatten positions (updates buy and sell volumes)
        buy_order_volume, sell_order_volume = self.flatten(
            orders,
            order_depth,
            position,
            position_limit,
            Product.RAINFOREST_RESIN,
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
                    Product.RAINFOREST_RESIN, int(best_below_bid_fallback + 1), buy_quantity
                )
            )  # Buy order

        sell_quantity = position_limit + (
            position - sell_order_volume
        )  # Calculate amount we can sell
        if sell_quantity > 0:
            orders.append(
                Order(
                    Product.RAINFOREST_RESIN, int(best_above_ask_fallback - 1), -sell_quantity
                )
            )  # Sell order

        return orders

    def basket_orders(self, order_depth, position_b1, position_b2, corr_window=100):
        # This version uses basket1 synthetic signal to trade both baskets
        # It also uses correlation to determine the direction of B2
        # As long as the correlation is high, we trade B2 in the same direction as B1
        # Otherwise we trade B2 in the opposite direction
        # It works pretty well on the 'training data'
        orders_b1 = []
        orders_b2 = []

        # Dicts to store osell, obuy: orderbooks sorted, best_*: best prices, worst_*: worst prices and mid_price
        prods = ['JAMS', 'CROISSANTS', 'DJEMBES', 'PICNIC_BASKET1', 'PICNIC_BASKET2']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}

        # Extract prices
        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))
            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2

        # Save prices for correlation
        self.b1_prices.append(mid_price['PICNIC_BASKET1'])
        self.b2_prices.append(mid_price['PICNIC_BASKET2'])

        # Keep window size fixed
        if len(self.b1_prices) > corr_window:
            self.b1_prices.pop(0)
            self.b2_prices.pop(0)

        # Default to high correlation until we have enough data
        correlation = 1.0
        if len(self.b1_prices) == corr_window:
            correlation = np.corrcoef(self.b1_prices, self.b2_prices)[0, 1]

        # Compute spread between B1 and its synthetic value
        spread = mid_price['PICNIC_BASKET1'] - (
            3 * mid_price['JAMS'] + 6 * mid_price['CROISSANTS'] + mid_price['DJEMBES']
        )

        # Set a threshold for how big the mispricing must be before trading
        trade_at = self.basket1_std * 0.5 # This 0.5 is also from the 2023 code, haven't tried other values yet

        # Determine trading direction for B2 based on correlation
        same_direction = correlation > 0.3 # started with 0.7, just trying random values

        # SELL signal
        if spread > trade_at:
            vol_b1 = self.LIMIT['PICNIC_BASKET1'] + position_b1
            if vol_b1 > 0:
                orders_b1.append(Order('PICNIC_BASKET1', worst_buy['PICNIC_BASKET1'], -vol_b1))

            vol_b2 = self.LIMIT['PICNIC_BASKET2'] + position_b2
            if vol_b2 > 0:
                action = -vol_b2 if same_direction else vol_b2
                price = worst_buy['PICNIC_BASKET2'] if same_direction else worst_sell['PICNIC_BASKET2']
                orders_b2.append(Order('PICNIC_BASKET2', price, action))

        # BUY signal
        elif spread < -trade_at:
            vol_b1 = self.LIMIT['PICNIC_BASKET1'] - position_b1
            if vol_b1 > 0:
                orders_b1.append(Order('PICNIC_BASKET1', worst_sell['PICNIC_BASKET1'], vol_b1))

            vol_b2 = self.LIMIT['PICNIC_BASKET2'] - position_b2
            if vol_b2 > 0:
                action = vol_b2 if same_direction else -vol_b2
                price = worst_sell['PICNIC_BASKET2'] if same_direction else worst_buy['PICNIC_BASKET2']
                orders_b2.append(Order('PICNIC_BASKET2', price, action))


        return orders_b1, orders_b2
    
    def jams_orders(
        self,
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        timespan: int = 30,  # Increased default timespan for volatility calc
        base_take_width: float = 1.0,  # Base deviation for taking
        volatility_take_factor: float = 60.0,  # Factor scaling take_width with volatility
        base_passive_spread: int = 1.0,  # Minimum passive spread in ticks
        volatility_spread_factor: float = 60.0,  # Factor scaling passive spread with volatility
        volatility_size_factor: float = 60.0,  # Factor scaling passive size with volatility (inverse)
        min_passive_volume: int = 200,  # Minimum size for passive orders (needs to be higher than for SQUID)
    ) -> List[Order]:
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
        self.jams_log_prices.append(np.log(midpoint))

        # Maintain a rolling window of log prices
        if len(self.jams_log_prices) > timespan:
            self.jams_log_prices.pop(0)

        # Calculate volatility if we have enough data points 
        if len(self.jams_log_prices) >= timespan:
            # Calculate log returns (difference between consecutive log prices)
            log_returns = np.diff(self.jams_log_prices)
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
                    orders.append(Order(Product.JAMS, int(best_ask), quantity_to_buy))
                    buy_order_volume += quantity_to_buy

        # Sell if best bid is significantly above fair value
        if best_bid >= fair_value + take_width:
            # Get volume available at best bid
            bid_volume = order_depth.buy_orders.get(best_bid, 0)
            if bid_volume > 0:
                quantity_to_sell = min(bid_volume, position_limit + position)
                if quantity_to_sell > 0:
                    orders.append(Order(Product.JAMS, int(best_bid), -quantity_to_sell))
                    sell_order_volume += quantity_to_sell

        # Flatten
        buy_order_volume, sell_order_volume = self.flatten(
            orders,
            order_depth,
            position,
            position_limit,
            Product.JAMS,
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
            orders.append(Order(Product.JAMS, buy_price, buy_volume_passive))

        # Sell order
        if sell_volume_passive > 0:
            orders.append(Order(Product.JAMS, sell_price, -sell_volume_passive))

        return orders
    
    def run(self, state: TradingState):
        result = {}

        # Resin
        resin_fair_value = 10000
        resin_take_width = 0.5

        # Kelp
        kelp_take_width = 1
        kelp_timemspan = 9
        kelp_min_vol_thresh = 15
        kelp_vwap = True
        kelp_prevent_adverse = True
        kelp_adverse_volume = 20
        
        if Product.PICNIC_BASKET1 in state.order_depths:
            position_b1 = state.position.get(Product.PICNIC_BASKET1, 0)
            position_b2 = state.position.get(Product.PICNIC_BASKET2, 0)
            orders_b1, orders_b2 = self.basket_orders(
                order_depth=state.order_depths,
                position_b1=position_b1,
                position_b2=position_b2,
                corr_window=300 
            )
            result[Product.PICNIC_BASKET1] = orders_b1
            result[Product.PICNIC_BASKET2] = orders_b2
        
        
        if "RAINFOREST_RESIN" in state.order_depths:
             resin_position = state.position.get("RAINFOREST_RESIN", 0)
             resin_orders = self.resin_orders(
                 state.order_depths[Product.RAINFOREST_RESIN],
                 resin_fair_value,
                 resin_take_width,
                 resin_position,
                 self.LIMIT["RAINFOREST_RESIN"],
             )
             result["RAINFOREST_RESIN"] = resin_orders


        if "KELP" in state.order_depths:
             kelp_position = state.position.get("KELP", 0)
             kelp_orders = self.kelp_orders(
                 order_depth=state.order_depths["KELP"],
                 timespan=kelp_timemspan,
                 take_width=kelp_take_width,
                 position=kelp_position,
                 position_limit=self.LIMIT["KELP"],
                 min_vol_threshold=kelp_min_vol_thresh,
                 vwap=kelp_vwap,
                 prevent_adverse=kelp_prevent_adverse,
                 adverse_volume=kelp_adverse_volume)
             result["KELP"] = kelp_orders

        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_orders = self.squid_orders(
                order_depth=state.order_depths["SQUID_INK"],
                position=squid_position,
                position_limit=self.LIMIT["SQUID_INK"],
            )
            result["SQUID_INK"] = squid_orders
        

        if "JAMS" in state.order_depths:
            jams_position = state.position.get("JAMS", 0)
            jams_orders = self.jams_orders(
                order_depth=state.order_depths["JAMS"],
                position=jams_position,
                position_limit=self.LIMIT["JAMS"],
            )
            result["JAMS"] = jams_orders


        conversions = 0
        traderData = jsonpickle.encode({"b1_prices": self.b1_prices,
                                        "b2_prices": self.b2_prices,
                                        "squid_log_prices": self.squid_log_prices,
                                        "jams_log_prices": self.jams_log_prices,
                                        "kelp_prices": self.kelp_prices,
                                        "kelp_vwap": self.kelp_vwap,
                                        })
        return result, conversions, traderData