"""
Portfolio item data model for the back tester.
"""

from datetime import datetime
from typing import Dict, Any


class PortfolioItem:
    """Represents a stock position in the portfolio."""
    
    def __init__(self, name: str, shares: int = 0, average_price: float = 0.0,
                 current_value: float = 0.0, date_added: str = None, 
                 last_modified: str = None):
        """
        Initialize a portfolio item.
        
        Args:
            name: Stock symbol
            shares: Number of shares owned
            average_price: Average purchase price per share
            current_value: Current market value of the position
            date_added: Date when stock was first added to portfolio
            last_modified: Date when position was last modified
        """
        self.name = name
        self.shares = shares
        self.average_price = average_price
        self.current_value = current_value
        self.date_added = date_added or datetime.now().strftime('%Y-%m-%d')
        self.last_modified = last_modified or datetime.now().strftime('%Y-%m-%d')
        
        # Validate inputs
        self._validate()
    
    def _validate(self):
        """Validate portfolio item data."""
        if not self.name or not self.name.strip():
            raise ValueError("Stock name cannot be empty")
        
        if self.shares < 0:
            raise ValueError("Shares cannot be negative")
        
        if self.average_price < 0:
            raise ValueError("Average price cannot be negative")
        
        if self.current_value < 0:
            raise ValueError("Current value cannot be negative")
        
        # Validate dates
        for date_str in [self.date_added, self.last_modified]:
            if date_str:
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Date must be in YYYY-MM-DD format")
    
    def update_position(self, shares_change: int, price: float, date: str):
        """
        Update the position based on a transaction.
        
        Args:
            shares_change: Change in shares (positive for buy, negative for sell)
            price: Transaction price per share
            date: Transaction date
        """
        if shares_change == 0:
            return
        
        # Calculate new total shares and average price
        total_cost = self.shares * self.average_price
        new_shares = self.shares + shares_change
        
        if new_shares < 0:
            raise ValueError(f"Insufficient shares to sell. Have {self.shares}, trying to sell {abs(shares_change)}")
        
        if new_shares == 0:
            # All shares sold, reset average price
            self.average_price = 0.0
        else:
            # Calculate new average price
            transaction_cost = shares_change * price
            new_total_cost = total_cost + transaction_cost
            self.average_price = new_total_cost / new_shares
        
        self.shares = new_shares
        self.last_modified = date
    
    def update_current_value(self, current_price: float):
        """Update the current market value."""
        self.current_value = self.shares * current_price
    
    def get_unrealized_gain_loss(self) -> float:
        """Calculate unrealized gain/loss."""
        if self.shares == 0:
            return 0.0
        return self.current_value - (self.shares * self.average_price)
    
    def get_unrealized_gain_loss_percentage(self) -> float:
        """Calculate unrealized gain/loss percentage."""
        if self.shares == 0 or self.average_price == 0:
            return 0.0
        return ((self.current_value / (self.shares * self.average_price)) - 1) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio item to dictionary."""
        return {
            'name': self.name,
            'shares': self.shares,
            'average_price': self.average_price,
            'current_value': self.current_value,
            'date_added': self.date_added,
            'last_modified': self.last_modified,
            'unrealized_gain_loss': self.get_unrealized_gain_loss(),
            'unrealized_gain_loss_percentage': self.get_unrealized_gain_loss_percentage()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioItem':
        """Create portfolio item from dictionary."""
        return cls(
            name=data['name'],
            shares=data.get('shares', 0),
            average_price=data.get('average_price', 0.0),
            current_value=data.get('current_value', 0.0),
            date_added=data.get('date_added'),
            last_modified=data.get('last_modified')
        )
    
    def __str__(self) -> str:
        """String representation of portfolio item."""
        return f"{self.name}: {self.shares} shares at ${self.average_price:.2f} avg (${self.current_value:.2f} current)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"PortfolioItem(name='{self.name}', shares={self.shares}, avg_price={self.average_price}, current_value={self.current_value})" 