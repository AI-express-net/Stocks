"""
Mock Valuator Implementation

This module provides a mock valuator for testing that simulates real stock data
without depending on the actual stock infrastructure.
"""

import logging
from datetime import date, datetime
from typing import List, Tuple, Optional

from back_tester.valuator import Valuator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockValuator(Valuator):
    """
    Mock valuator that simulates real stock prices for testing.
    
    This valuator provides realistic-looking stock prices without requiring
    the actual stock data infrastructure, making it suitable for testing.
    """
    
    def __init__(self):
        """Initialize the mock valuator."""
        super().__init__()
        self._price_cache = {}  # Cache prices for consistency
        self._base_prices = {
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 300.0,
            "AMZN": 3300.0,
            "TSLA": 700.0,
            "META": 350.0,
            "NVDA": 500.0,
            "NFLX": 500.0,
            "JPM": 150.0,
            "JNJ": 170.0
        }
        
    def calculate_values(self, stocks: List[str], target_date: date) -> List[Tuple[str, float]]:
        """
        Calculate mock stock values for the given stocks on the target date.
        
        Args:
            stocks: List of stock symbols
            target_date: Date for which to get stock prices
            
        Returns:
            List of tuples containing (stock_symbol, price) pairs
            
        Raises:
            ValueError: If target_date is invalid
        """
        self.validate_date(target_date)
        
        results = []
        date_str = target_date.strftime("%Y-%m-%d")
        
        for stock_symbol in stocks:
            try:
                price = self._get_mock_price(stock_symbol, target_date)
                if price is not None:
                    results.append((stock_symbol, price))
                    logger.debug(f"Got mock price for {stock_symbol} on {date_str}: ${price:.2f}")
                else:
                    logger.warning(f"No mock price available for {stock_symbol} on {date_str}")
            except Exception as e:
                logger.error(f"Error getting mock price for {stock_symbol} on {date_str}: {str(e)}")
                continue
                
        return results
    
    def _get_mock_price(self, stock_symbol: str, target_date: date) -> Optional[float]:
        """
        Get a mock stock price for a specific date.
        
        Args:
            stock_symbol: Stock symbol (e.g., 'AAPL')
            target_date: Date for which to get the price
            
        Returns:
            Mock stock price as float, or None if not available
        """
        # Get base price for this stock
        base_price = self._base_prices.get(stock_symbol, 100.0)
        
        # Create a cache key
        cache_key = f"{stock_symbol}_{target_date.isoformat()}"
        
        # Check if we have a cached price
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        # Generate a realistic price with some variation
        import random
        import math
        
        # Use the date as a seed for consistent prices
        random.seed(hash(target_date) + hash(stock_symbol))
        
        # Add some time-based trend and random variation
        days_since_epoch = (target_date - date(1970, 1, 1)).days
        
        # Create a trend (slight upward trend over time)
        trend_factor = 1.0 + (days_since_epoch * 0.0001)
        
        # Add some cyclical variation (market cycles)
        cycle_factor = 1.0 + 0.1 * math.sin(days_since_epoch * 0.01)
        
        # Add random daily variation
        daily_variation = 1.0 + random.uniform(-0.05, 0.05)
        
        # Calculate final price
        final_price = base_price * trend_factor * cycle_factor * daily_variation
        
        # Ensure price is positive and reasonable
        final_price = max(final_price, 1.0)
        
        # Cache the result
        self._price_cache[cache_key] = final_price
        
        return final_price
    
    def get_available_dates(self, stock_symbol: str) -> List[date]:
        """
        Get list of available dates for a stock.
        
        Args:
            stock_symbol: Stock symbol
            
        Returns:
            List of available dates (simulated)
        """
        # Return a range of dates for testing
        start_date = date(2020, 1, 1)
        end_date = date(2024, 12, 31)
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date = current_date.replace(day=current_date.day + 1)
        
        return dates
    
    def get_price_range(self, stock_symbol: str, start_date: date, end_date: date) -> List[Tuple[date, float]]:
        """
        Get price range for a stock between two dates.
        
        Args:
            stock_symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of (date, price) tuples
        """
        results = []
        current_date = start_date
        
        while current_date <= end_date:
            price = self._get_mock_price(stock_symbol, current_date)
            if price is not None:
                results.append((current_date, price))
            current_date = current_date.replace(day=current_date.day + 1)
        
        return results
    
    def clear_cache(self):
        """Clear the internal cache."""
        self._price_cache.clear()
        logger.info("MockValuator cache cleared") 