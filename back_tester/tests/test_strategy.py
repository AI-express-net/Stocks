"""
Strategy tests using pytest framework.
"""

import pytest

from back_tester.strategies.buy_and_hold import BuyAndHoldStrategy
from back_tester.models.transaction import Transaction, TransactionType


class TestStrategy:
    """Test strategy interface."""
    
    def test_strategy_creation(self):
        """Test strategy creation."""
        strategy = BuyAndHoldStrategy(target_stocks=["AAPL", "GOOGL"])
        assert strategy is not None
    
    def test_generate_transactions(self):
        """Test transaction generation."""
        strategy = BuyAndHoldStrategy(target_stocks=["AAPL", "GOOGL"])
        portfolio_items = []
        stock_values = [("AAPL", 150.0), ("GOOGL", 2800.0)]
        
        transactions = strategy.generate_transactions(
            portfolio_items, stock_values, "2025-01-01", 10000.0
        )
        
        assert len(transactions) > 0
        assert all(isinstance(t, Transaction) for t in transactions)
    
    def test_input_validation(self):
        """Test strategy input validation."""
        strategy = BuyAndHoldStrategy()
        portfolio_items = []
        stock_values = [("AAPL", 150.0)]
        
        # Test valid inputs
        assert strategy.validate_inputs(portfolio_items, stock_values, "2025-01-01", 10000.0) == True
        
        # Test invalid date
        with pytest.raises(ValueError):
            strategy.validate_inputs(portfolio_items, stock_values, "invalid-date", 10000.0)
