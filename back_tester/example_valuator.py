"""
Example valuator implementation for testing.
"""

import random
from typing import List, Tuple
from valuator import Valuator


class ExampleValuator(Valuator):
    """
    Example valuator that generates random stock prices.
    
    This is for testing purposes only. In a real implementation,
    you would connect to actual stock data sources.
    """
    
    def __init__(self, base_prices: dict = None):
        """
        Initialize the example valuator.
        
        Args:
            base_prices: Dictionary of base prices for stocks
        """
        self.base_prices = base_prices or {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'AMZN': 3300.0,
            'TSLA': 700.0,
            'NVDA': 500.0,
            'META': 350.0,
            'NFLX': 500.0,
            'ADBE': 500.0,
            'CRM': 250.0
        }
        self.price_history = {}
    
    def calculate_values(self, stocks: List[str], date: str) -> List[Tuple[str, float]]:
        """
        Calculate stock values for a given date.
        
        Args:
            stocks: List of stock symbols to value
            date: Date for valuation (YYYY-MM-DD format)
            
        Returns:
            List of tuples containing (stock_symbol, current_price)
        """
        if not self.validate_date(date):
            raise ValueError(f"Invalid date format: {date}")
        
        values = []
        for stock in stocks:
            price = self._get_stock_price(stock, date)
            if price > 0:
                values.append((stock, price))
        
        return values
    
    def _get_stock_price(self, stock: str, date: str) -> float:
        """Get the price for a stock on a given date."""
        # Initialize price history for this stock if needed
        if stock not in self.price_history:
            base_price = self.base_prices.get(stock, 100.0)
            self.price_history[stock] = {date: base_price}
        
        # If we don't have a price for this date, generate one
        if date not in self.price_history[stock]:
            # Get the last known price or base price
            last_price = self._get_last_price(stock, date)
            
            # Add some random variation (±5%)
            variation = random.uniform(-0.05, 0.05)
            new_price = last_price * (1 + variation)
            
            # Ensure price doesn't go below $1
            new_price = max(new_price, 1.0)
            
            self.price_history[stock][date] = new_price
        
        return self.price_history[stock][date]
    
    def _get_last_price(self, stock: str, date: str) -> float:
        """Get the last known price for a stock before the given date."""
        if stock not in self.price_history:
            return self.base_prices.get(stock, 100.0)
        
        # Find the most recent date before the given date
        dates = sorted(self.price_history[stock].keys())
        for d in reversed(dates):
            if d < date:
                return self.price_history[stock][d]
        
        # If no previous date found, use base price
        return self.base_prices.get(stock, 100.0) 