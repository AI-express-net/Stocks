"""
Abstract interface for stock valuation.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class Valuator(ABC):
    """
    Abstract base class for stock valuators.
    
    A valuator calculates the current market value of stocks for a given date.
    """
    
    @abstractmethod
    def calculate_values(self, stocks: List[str], date: str) -> List[Tuple[str, float]]:
        """
        Calculate stock values for a given date.
        
        Args:
            stocks: List of stock symbols to value
            date: Date for valuation (YYYY-MM-DD format)
            
        Returns:
            List of tuples containing (stock_symbol, current_price)
            
        Raises:
            ValueError: If date format is invalid
            Exception: If unable to fetch stock data
        """
        pass
    
    def get_stock_value(self, stock: str, date: str) -> float:
        """
        Get the value of a single stock for a given date.
        
        Args:
            stock: Stock symbol
            date: Date for valuation (YYYY-MM-DD format)
            
        Returns:
            Current price of the stock
            
        Raises:
            ValueError: If stock not found or date invalid
        """
        values = self.calculate_values([stock], date)
        if not values:
            raise ValueError(f"No value found for stock {stock} on {date}")
        return values[0][1]
    
    def validate_date(self, date) -> bool:
        """
        Validate date format.
        
        Args:
            date: Date to validate (string or date object)
            
        Returns:
            True if date is in valid format (YYYY-MM-DD)
        """
        try:
            from datetime import datetime, date as date_type
            # Handle both string and date objects
            if isinstance(date, date_type):
                return True  # date objects are always valid
            elif isinstance(date, str):
                datetime.strptime(date, '%Y-%m-%d')
                return True
            else:
                return False
        except ValueError:
            return False 