"""
Abstract interface for trading strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from models.transaction import Transaction, TransactionType


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    A strategy analyzes the current portfolio and available stock values
    to generate buy/sell transactions.
    """
    
    @abstractmethod
    def generate_transactions(self, portfolio_items: List, stock_values: List[Tuple[str, float]], 
                            date: str, available_cash: float) -> List[Transaction]:
        """
        Generate buy/sell transactions based on current portfolio and market conditions.
        
        Args:
            portfolio_items: List of current portfolio positions
            stock_values: List of (stock_symbol, current_price) tuples
            date: Current date (YYYY-MM-DD format)
            available_cash: Available cash for trading
            
        Returns:
            List of Transaction objects representing buy/sell orders
            
        Raises:
            ValueError: If date format is invalid or insufficient data
        """
        pass
    
    def validate_inputs(self, portfolio_items: List, stock_values: List[Tuple[str, float]], 
                       date: str, available_cash: float) -> bool:
        """
        Validate strategy inputs.
        
        Args:
            portfolio_items: List of portfolio items
            stock_values: List of stock values
            date: Date string
            available_cash: Available cash amount
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate date format
        try:
            from datetime import datetime, date as date_type
            # Handle both string and date objects
            if isinstance(date, date_type):
                # Convert date object to string for validation
                date_str = date.strftime('%Y-%m-%d')
                datetime.strptime(date_str, '%Y-%m-%d')
            elif isinstance(date, str):
                datetime.strptime(date, '%Y-%m-%d')
            else:
                raise ValueError("Date must be string or date object")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        
        # Validate cash amount
        if available_cash < 0:
            raise ValueError("Available cash cannot be negative")
        
        # Validate stock values
        if not stock_values:
            raise ValueError("Stock values list cannot be empty")
        
        for stock, price in stock_values:
            if not stock or not stock.strip():
                raise ValueError("Stock symbol cannot be empty")
            if price <= 0:
                raise ValueError(f"Price for {stock} must be positive")
        
        return True
    
    def get_portfolio_value(self, portfolio_items: List, stock_values: List[Tuple[str, float]]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            portfolio_items: List of portfolio items
            stock_values: List of (stock_symbol, current_price) tuples
            
        Returns:
            Total portfolio value
        """
        # Create a dictionary for quick stock price lookup
        price_dict = {stock: price for stock, price in stock_values}
        
        total_value = 0.0
        for item in portfolio_items:
            if item.name in price_dict:
                item.update_current_value(price_dict[item.name])
                total_value += item.current_value
        
        return total_value
    
    def get_portfolio_summary(self, portfolio_items: List, stock_values: List[Tuple[str, float]]) -> dict:
        """
        Get a summary of the current portfolio.
        
        Args:
            portfolio_items: List of portfolio items
            stock_values: List of (stock_symbol, current_price) tuples
            
        Returns:
            Dictionary with portfolio summary
        """
        price_dict = {stock: price for stock, price in stock_values}
        
        summary = {
            'total_value': 0.0,
            'total_cost': 0.0,
            'unrealized_gain_loss': 0.0,
            'positions': []
        }
        
        for item in portfolio_items:
            if item.name in price_dict:
                item.update_current_value(price_dict[item.name])
                summary['total_value'] += item.current_value
                summary['total_cost'] += item.shares * item.average_price
                summary['unrealized_gain_loss'] += item.get_unrealized_gain_loss()
                summary['positions'].append(item.to_dict())
        
        return summary 