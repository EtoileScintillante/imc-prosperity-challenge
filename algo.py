# ===================== #
# CODE USED IN ROUND 3  #
# ===================== #

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple
import string
import jsonpickle
import collections
import statistics as stat
from collections import defaultdict
import numpy as np
import math

# In this file: only trading vouchers! 

# https://math.stackexchange.com/questions/97/how-to-accurately-calculate-the-error-function-operatornameerfx-with-a-co
# this is what the other team used
def cdf(x):
    """Approximation of the cumulative distribution function for the standard normal distribution."""
    # Constants for the approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x)

    return 0.5 * (1.0 + sign * y)

VOUCHERS = [
    "VOLCANIC_ROCK_VOUCHER_9500",
    "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000",
    "VOLCANIC_ROCK_VOUCHER_10250",
    "VOLCANIC_ROCK_VOUCHER_10500"]

# From analysis.ipynb
STD_DEV = {
    "VOLCANIC_ROCK_VOUCHER_9500": 3.939477454016052,
    "VOLCANIC_ROCK_VOUCHER_9750": 12.599919393329856,
    "VOLCANIC_ROCK_VOUCHER_10000": 23.080535216209032,
    "VOLCANIC_ROCK_VOUCHER_10250": 12.813832915870327,
    "VOLCANIC_ROCK_VOUCHER_10500": 17.22436913225563
}
class Trader:
    def __init__(self):
        self.cached_midprices = defaultdict(list)
        self.voucher_log_prices = defaultdict(list)
        self.sigma = 0.0039733772549522567 # analysis.ipynb
        self.log_limit = 300 # Can be adjusted, but the higher the slower the algo runs
        self.r = 0.0 # Risk-free rate set to zero (as recommended by one of the moderators)
        self.LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "JAMS": 350,
            "CROISSANTS": 250,
            "DJEMBES": 60,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }
    
    def update_volatility(self, historical_prices):
        # Use Exponentially Weighted Moving Average (EWMA) to update volatility
        lambda_ = 0.94  # Decay factor for EWMA, common choice in finance
        if len(historical_prices) > 1:
            returns = np.diff(historical_prices) / historical_prices[:-1]
            var = np.var(returns)
            if hasattr(self, 'sigma'):
                self.sigma = np.sqrt(lambda_ * self.sigma**2 + (1 - lambda_) * var) * np.sqrt(252)
                self.sigma = min(self.sigma, 5.0) # we cap sigma
            else:
                self.sigma = np.sqrt(var) * np.sqrt(252)
                self.sigma = min(self.sigma, 5.0) # we cap sigma

    def black_scholes_price(self, current_price, strike_price, time_to_maturity, premium = 0):
        if current_price <= 0 or strike_price <= 0 or time_to_maturity <= 0 or self.sigma <= 0:
            return 0
        S = current_price
        K = strike_price
        T = time_to_maturity 
        r = self.r
        sigma = self.sigma
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = (S * cdf(d1) - (K + premium) * np.exp(-r * T) * cdf(d2))
        
        return call_price

    def trade(self, trading_state: TradingState, product: str, DTE: int, position: int):
        current_price = self.get_current_price(trading_state, product)
        self.cached_midprices[product].append(current_price)
        if self.cached_midprices[product] and len(self.cached_midprices[product]) > self.log_limit:
            self.cached_midprices[product].pop(0)

        # Vola update
        self.update_volatility(self.cached_midprices[product]) 

        # Premium calculation
        order_depth = trading_state.order_depths["VOLCANIC_ROCK"]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        rock_mid_price = (best_bid + best_ask) / 2
        strike = int(product.split("_")[-1])
        intrinsic_value = max(0, rock_mid_price - strike)
        premium = self.calculate_dynamic_premium(current_price, intrinsic_value)  # Calculate premium dynamically
        
        # BS value calculation
        theoretical_price = self.black_scholes_price(current_price, strike, DTE, premium)
        order_depth = trading_state.order_depths[product]

        # Place orders
        orders = self.execute_trading_logic(order_depth,theoretical_price, product, position)
        return orders

    def calculate_dynamic_premium(self, voucher_mid_price: int, intrinsic_value: int):
        base_premium = np.mean(voucher_mid_price - intrinsic_value)
        return base_premium

    def execute_trading_logic(self, order_depth, theoretical_price, product, position):
        orders = []
        max_pos = self.LIMITS[product]

        best_asks = sorted(order_depth.sell_orders.keys())
        best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

        # Buying logic
        for ask_price in best_asks:
            if ask_price < theoretical_price:
                ask_volume = order_depth.sell_orders[ask_price]
                if position + ask_volume <= max_pos:
                    orders.append(Order(product, ask_price, ask_volume))
                else:
                    allowed_volume = max_pos - position
                    if allowed_volume > 0:
                        orders.append(Order(product, ask_price, allowed_volume))
                break  # Exit after first executable level

        # Selling logic
        for bid_price in best_bids:
            if bid_price > theoretical_price:
                bid_volume = order_depth.buy_orders[bid_price]
                if position - bid_volume >= -max_pos:
                    orders.append(Order(product, bid_price, -bid_volume))

                else:
                    allowed_volume = position + max_pos
                    if allowed_volume > 0:
                        orders.append(Order(product, bid_price, -allowed_volume))
                break  # Exit after first executable level

        return orders

    def get_current_price(self, trading_state, product):
        # Retrieve the current price from order depth
        order_depth = trading_state.order_depths[product]
        if order_depth.sell_orders:
            return min(order_depth.sell_orders.keys()) # This team does not use mid price
        return float('inf')  # High price if no sell orders are available
    
    
    def run(self, state: TradingState):
        result = {}
        
        DTE = ((8 - (state.timestamp//(3*1000000)))/250) - (state.timestamp/1000000)/250 # from Discord 
        for v in VOUCHERS:
            if v in state.order_depths:
                v_pos = state.position.get(v, 0)
                v_orders = self.trade(state, v, DTE, v_pos)
                if v_orders:
                    result[v] = v_orders
            
        conversions = 0
        traderData = jsonpickle.encode({"cashed_midprices": self.cached_midprices, 
                                        "sigma": self.sigma, 
                                        "voucher_log_prices": self.voucher_log_prices})
        return result, conversions, traderData