"""
Trading strategies for the back tester.
"""

from .buy_and_hold import BuyAndHoldStrategy
from .moving_average import MovingAverageStrategy

__all__ = ['BuyAndHoldStrategy', 'MovingAverageStrategy'] 