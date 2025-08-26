"""
Data model tests using pytest framework.
"""

import pytest

from back_tester.models.transaction import Transaction, TransactionType
from back_tester.models.portfolio_item import PortfolioItem


class TestDataModels:
    """Test data models."""
    
    def test_transaction_creation(self):
        """Test transaction creation and validation."""
        transaction = Transaction(
            stock="AAPL",
            date="2025-01-01",
            price=150.0,
            shares=10,
            transaction_type=TransactionType.BUY
        )
        assert transaction.stock == "AAPL"
        assert transaction.price == 150.0
        assert transaction.shares == 10
        assert transaction.get_total_value() == 1500.0
    
    def test_transaction_validation(self):
        """Test transaction validation."""
        # Test valid transaction
        transaction = Transaction(
            stock="AAPL",
            date="2025-01-01",
            price=150.0,
            shares=10,
            transaction_type=TransactionType.BUY
        )
        assert transaction is not None
        
        # Test invalid price
        with pytest.raises(ValueError):
            Transaction(
                stock="AAPL",
                date="2025-01-01",
                price=-150.0,
                shares=10,
                transaction_type=TransactionType.BUY
            )
    
    def test_portfolio_item_creation(self):
        """Test portfolio item creation."""
        item = PortfolioItem(
            name="AAPL",
            shares=100,
            average_price=150.0,
            current_value=15500.0
        )
        assert item.name == "AAPL"
        assert item.shares == 100
        assert item.average_price == 150.0
        assert item.get_unrealized_gain_loss() == 500.0
    
    def test_portfolio_item_validation(self):
        """Test portfolio item validation."""
        # Test valid item
        item = PortfolioItem("AAPL", 100, 150.0, 15500.0)
        assert item is not None
        
        # Test invalid shares
        with pytest.raises(ValueError):
            PortfolioItem("AAPL", -100, 150.0, 15500.0)
