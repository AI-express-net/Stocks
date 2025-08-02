"""
Real Valuator Implementation

This module provides a real valuator that integrates with the existing stock data
infrastructure to fetch actual historical stock prices.
"""

import sys
import os
import logging
from datetime import date, datetime
from typing import List, Tuple, Optional

# Add the parent directory to the path to import from stocks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valuator import Valuator
from models.transaction import Transaction, TransactionType

# Import the existing stock infrastructure
from stocks.fmp_stock import Stock
from stocks.data_names import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealValuator(Valuator):
    """
    Real valuator that fetches actual stock prices from the existing infrastructure.
    
    This valuator integrates with the existing Stock class and FMP API to provide
    real historical stock prices for back testing.
    """
    
    def __init__(self):
        """Initialize the real valuator."""
        super().__init__()
        self._stock_cache = {}  # Cache Stock objects to avoid recreating them
        self._price_cache = {}  # Cache historical prices for performance
        
    def calculate_values(self, stocks: List[str], target_date: date) -> List[Tuple[str, float]]:
        """
        Calculate stock values for the given stocks on the target date.
        
        Args:
            stocks: List of stock symbols
            target_date: Date for which to get stock prices
            
        Returns:
            List of tuples containing (stock_symbol, price) pairs
            
        Raises:
            ValueError: If target_date is invalid
            Exception: If there are issues fetching stock data
        """
        self.validate_date(target_date)
        
        results = []
        date_str = target_date.strftime("%Y-%m-%d")
        
        for stock_symbol in stocks:
            try:
                price = self._get_stock_price(stock_symbol, target_date)
                if price is not None:
                    results.append((stock_symbol, price))
                    logger.debug(f"Got price for {stock_symbol} on {date_str}: ${price:.2f}")
                else:
                    logger.warning(f"No price data available for {stock_symbol} on {date_str}")
            except Exception as e:
                logger.error(f"Error getting price for {stock_symbol} on {date_str}: {str(e)}")
                # Continue with other stocks even if one fails
                continue
                
        return results
    
    def _get_stock_price(self, stock_symbol: str, target_date: date) -> Optional[float]:
        """
        Get the stock price for a specific date.
        
        Args:
            stock_symbol: Stock symbol (e.g., 'AAPL')
            target_date: Date for which to get the price
            
        Returns:
            Stock price as float, or None if not available
        """
        # Get or create Stock object
        stock = self._get_stock_object(stock_symbol)
        if stock is None:
            return None
            
        # Get historical prices
        historical_prices = self._get_historical_prices(stock)
        if not historical_prices:
            return None
            
        # Find the closest date (exact match or closest previous date)
        target_date_str = target_date.strftime("%Y-%m-%d")
        price_data = self._find_closest_price(historical_prices, target_date_str)
        
        if price_data:
            return float(price_data.get('close', 0))
        
        return None
    
    def _get_stock_object(self, stock_symbol: str):
        """
        Get or create a Stock object for the given symbol.
        
        Args:
            stock_symbol: Stock symbol
            
        Returns:
            Stock object or None if creation fails
        """
        if stock_symbol in self._stock_cache:
            return self._stock_cache[stock_symbol]
            
        try:
            stock = Stock(stock_symbol)
            self._stock_cache[stock_symbol] = stock
            return stock
        except Exception as e:
            logger.error(f"Error creating Stock object for {stock_symbol}: {str(e)}")
            return None
    
    def _get_historical_prices(self, stock: Stock) -> List[dict]:
        """
        Get historical prices for a stock.
        
        Args:
            stock: Stock object
            
        Returns:
            List of historical price data
        """
        try:
            # Fetch historical prices data
            stock.fetch_stock_data('historical_prices')
            
            # Get the data from the historical prices entity
            historical_prices_entity = stock.historical_prices
            data = historical_prices_entity.get_data()
            
            if not data or 'historical' not in data:
                logger.warning(f"No historical price data available for {stock.name}")
                return []
                
            return data['historical']
            
        except Exception as e:
            logger.error(f"Error fetching historical prices for {stock.name}: {str(e)}")
            return []
    
    def _find_closest_price(self, historical_prices: List[dict], target_date_str: str) -> Optional[dict]:
        """
        Find the closest price data for the target date.
        
        Args:
            historical_prices: List of historical price data
            target_date_str: Target date as string (YYYY-MM-DD)
            
        Returns:
            Price data dict or None if not found
        """
        if not historical_prices:
            return None
            
        # Sort by date (newest first, as that's typically how API returns data)
        sorted_prices = sorted(historical_prices, key=lambda x: x.get('date', ''), reverse=True)
        
        # Try to find exact match first
        for price_data in sorted_prices:
            if price_data.get('date') == target_date_str:
                return price_data
        
        # If no exact match, find the closest previous date
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        
        for price_data in sorted_prices:
            try:
                price_date_str = price_data.get('date')
                if price_date_str:
                    price_date = datetime.strptime(price_date_str, "%Y-%m-%d").date()
                    if price_date <= target_date:
                        return price_data
            except (ValueError, TypeError):
                continue
                
        return None
    
    def get_available_dates(self, stock_symbol: str) -> List[date]:
        """
        Get list of available dates for a stock.
        
        Args:
            stock_symbol: Stock symbol
            
        Returns:
            List of available dates
        """
        stock = self._get_stock_object(stock_symbol)
        if stock is None:
            return []
            
        historical_prices = self._get_historical_prices(stock)
        if not historical_prices:
            return []
            
        dates = []
        for price_data in historical_prices:
            try:
                date_str = price_data.get('date')
                if date_str:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                    dates.append(date_obj)
            except (ValueError, TypeError):
                continue
                
        return sorted(dates)
    
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
        stock = self._get_stock_object(stock_symbol)
        if stock is None:
            return []
            
        historical_prices = self._get_historical_prices(stock)
        if not historical_prices:
            return []
            
        results = []
        for price_data in historical_prices:
            try:
                date_str = price_data.get('date')
                if date_str:
                    price_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if start_date <= price_date <= end_date:
                        price = float(price_data.get('close', 0))
                        results.append((price_date, price))
            except (ValueError, TypeError):
                continue
                
        return sorted(results, key=lambda x: x[0])
    
    def clear_cache(self):
        """Clear the internal caches."""
        self._stock_cache.clear()
        self._price_cache.clear()
        logger.info("RealValuator cache cleared") 