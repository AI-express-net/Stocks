"""
Portfolio management for the back tester.
"""

import json
import os
from typing import List, Dict, Any
from models.portfolio_item import PortfolioItem
from models.transaction import Transaction, TransactionType


class Portfolio:
    """Manages portfolio positions and cash balance."""
    
    def __init__(self, cash: float = 0.0, portfolio_items: List[PortfolioItem] = None):
        """
        Initialize portfolio.
        
        Args:
            cash: Initial cash balance
            portfolio_items: List of portfolio items
        """
        self.cash = cash
        self.portfolio_items = portfolio_items or []
        self._create_item_lookup()
    
    @property
    def total_positions(self) -> int:
        """Get the total number of positions in the portfolio."""
        return len(self.portfolio_items)
    
    @property
    def active_positions(self) -> int:
        """Get the number of positions with shares > 0."""
        return sum(1 for item in self.portfolio_items if item.shares > 0)
    
    @property
    def stock_symbols(self) -> List[str]:
        """Get list of stock symbols in the portfolio."""
        return [item.name for item in self.portfolio_items]
    
    def _create_item_lookup(self):
        """Create a lookup dictionary for portfolio items."""
        self._item_lookup = {item.name: item for item in self.portfolio_items}
    
    def add_cash(self, amount: float):
        """Add cash to portfolio."""
        if amount < 0:
            raise ValueError("Cannot add negative cash amount")
        self.cash += amount
    
    def remove_cash(self, amount: float):
        """Remove cash from portfolio."""
        if amount < 0:
            raise ValueError("Cannot remove negative cash amount")
        if self.cash < amount:
            raise ValueError(f"Insufficient cash. Have ${self.cash:.2f}, need ${amount:.2f}")
        self.cash -= amount
    
    def get_portfolio_item(self, stock_name: str) -> PortfolioItem:
        """Get portfolio item for a stock."""
        return self._item_lookup.get(stock_name)
    
    def add_portfolio_item(self, item: PortfolioItem):
        """Add a portfolio item."""
        if item.name in self._item_lookup:
            raise ValueError(f"Portfolio item for {item.name} already exists")
        
        self.portfolio_items.append(item)
        self._item_lookup[item.name] = item
    
    def update_portfolio_item(self, item: PortfolioItem):
        """Update an existing portfolio item."""
        if item.name not in self._item_lookup:
            raise ValueError(f"Portfolio item for {item.name} does not exist")
        
        # Find and replace the item
        for i, existing_item in enumerate(self.portfolio_items):
            if existing_item.name == item.name:
                self.portfolio_items[i] = item
                self._item_lookup[item.name] = item
                break
    
    def remove_portfolio_item(self, stock_name: str):
        """Remove a portfolio item."""
        if stock_name not in self._item_lookup:
            return
        
        self.portfolio_items = [item for item in self.portfolio_items if item.name != stock_name]
        del self._item_lookup[stock_name]
    
    def execute_transaction(self, transaction: Transaction):
        """
        Execute a transaction and update portfolio.
        
        Args:
            transaction: Transaction to execute
            
        Raises:
            ValueError: If transaction cannot be executed
        """
        if transaction.transaction_type == TransactionType.BUY:
            self._execute_buy_transaction(transaction)
        elif transaction.transaction_type == TransactionType.SELL:
            self._execute_sell_transaction(transaction)
        else:
            raise ValueError(f"Unknown transaction type: {transaction.transaction_type}")
    
    def _execute_buy_transaction(self, transaction: Transaction):
        """Execute a buy transaction."""
        # Check if we have enough cash
        total_cost = transaction.get_total_value()
        if self.cash < total_cost:
            raise ValueError(f"Insufficient cash for buy transaction. Need ${total_cost:.2f}, have ${self.cash:.2f}")
        
        # Remove cash
        self.remove_cash(total_cost)
        
        # Get or create portfolio item
        portfolio_item = self.get_portfolio_item(transaction.stock)
        if portfolio_item is None:
            # Create new portfolio item
            portfolio_item = PortfolioItem(
                name=transaction.stock,
                shares=transaction.shares,
                average_price=transaction.price,
                current_value=transaction.get_total_value(),
                date_added=transaction.date,
                last_modified=transaction.date
            )
            self.add_portfolio_item(portfolio_item)
        else:
            # Update existing portfolio item
            portfolio_item.update_position(transaction.shares, transaction.price, transaction.date)
            self.update_portfolio_item(portfolio_item)
    
    def _execute_sell_transaction(self, transaction: Transaction):
        """Execute a sell transaction."""
        portfolio_item = self.get_portfolio_item(transaction.stock)
        if portfolio_item is None:
            raise ValueError(f"Cannot sell {transaction.stock}: not in portfolio")
        
        if portfolio_item.shares < transaction.shares:
            raise ValueError(f"Insufficient shares to sell. Have {portfolio_item.shares}, trying to sell {transaction.shares}")
        
        # Update portfolio item
        portfolio_item.update_position(-transaction.shares, transaction.price, transaction.date)
        
        # Add cash from sale
        self.add_cash(transaction.get_total_value())
        
        # Update or remove portfolio item
        if portfolio_item.shares == 0:
            self.remove_portfolio_item(transaction.stock)
        else:
            self.update_portfolio_item(portfolio_item)
    
    def get_total_value(self, stock_values: List[tuple] = None) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            stock_values: List of (stock_name, current_price) tuples
            
        Returns:
            Total portfolio value
        """
        total_value = self.cash
        
        if stock_values:
            price_dict = {stock: price for stock, price in stock_values}
            for item in self.portfolio_items:
                if item.name in price_dict:
                    item.update_current_value(price_dict[item.name])
                    total_value += item.current_value
        else:
            # Use current values if no new prices provided
            for item in self.portfolio_items:
                total_value += item.current_value
        
        return total_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary."""
        return {
            'cash': self.cash,
            'stocks': [item.to_dict() for item in self.portfolio_items]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create portfolio from dictionary."""
        cash = data.get('cash', 0.0)
        portfolio_items = []
        
        for stock_data in data.get('stocks', []):
            portfolio_items.append(PortfolioItem.from_dict(stock_data))
        
        return cls(cash=cash, portfolio_items=portfolio_items)
    
    def save_to_file(self, file_path: str):
        """Save portfolio to JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save portfolio: {e}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'Portfolio':
        """Load portfolio from JSON file."""
        try:
            if not os.path.exists(file_path):
                return cls()  # Return empty portfolio if file doesn't exist
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            raise ValueError(f"Failed to load portfolio: {e}")
    
    def __str__(self) -> str:
        """String representation of portfolio."""
        return f"Portfolio(cash=${self.cash:.2f}, positions={self.total_positions})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Portfolio(cash={self.cash}, items={self.portfolio_items})" 