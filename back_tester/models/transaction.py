"""
Transaction data model for the back tester.
"""

from datetime import datetime
from typing import Dict, Any
from enum import Enum


class TransactionType(Enum):
    """Transaction types."""
    BUY = "buy"
    SELL = "sell"


class Transaction:
    """Represents a stock transaction."""
    
    def __init__(self, stock: str, date: str, price: float, shares: int, 
                 transaction_type: TransactionType, transaction_id: str = None):
        """
        Initialize a transaction.
        
        Args:
            stock: Stock symbol
            date: Transaction date (YYYY-MM-DD format)
            price: Price per share
            shares: Number of shares
            transaction_type: BUY or SELL
            transaction_id: Optional unique identifier
        """
        self.stock = stock
        self.date = date
        self.price = price
        self.shares = shares
        self.transaction_type = transaction_type
        self.transaction_id = transaction_id or self._generate_id()
        
        # Validate inputs
        self._validate()
    
    def _validate(self):
        """Validate transaction data."""
        if not self.stock or not self.stock.strip():
            raise ValueError("Stock symbol cannot be empty")
        
        if self.price <= 0:
            raise ValueError("Price must be positive")
        
        if self.shares <= 0:
            raise ValueError("Shares must be positive")
        
        try:
            datetime.strptime(self.date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
    
    def _generate_id(self) -> str:
        """Generate a unique transaction ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"{self.stock}_{self.transaction_type.value}_{timestamp}"
    
    def get_total_value(self) -> float:
        """Calculate total transaction value."""
        return self.price * self.shares
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            'stock': self.stock,
            'date': self.date,
            'price': self.price,
            'shares': self.shares,
            'type': self.transaction_type.value,
            'transaction_id': self.transaction_id,
            'total_value': self.get_total_value()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary."""
        return cls(
            stock=data['stock'],
            date=data['date'],
            price=data['price'],
            shares=data['shares'],
            transaction_type=TransactionType(data['type']),
            transaction_id=data.get('transaction_id')
        )
    
    def __str__(self) -> str:
        """String representation of transaction."""
        return f"{self.transaction_type.value.upper()} {self.shares} shares of {self.stock} at ${self.price:.2f} on {self.date}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Transaction(stock='{self.stock}', date='{self.date}', price={self.price}, shares={self.shares}, type={self.transaction_type.value})" 