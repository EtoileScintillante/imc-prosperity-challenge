# ===================== #
# CODE USED IN ROUND 3  #
# ===================== #

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple
import string
import jsonpickle
import collections
import numpy as np
import math

class Trader:
    def __init__(self):
        raise NotImplementedError("")
    
    def run(self, state: TradingState):
        result = {}

        conversions = 0
        traderData = jsonpickle.encode({})
        return result, conversions, traderData