"""
Portfolio tests using pytest framework.
"""

import pytest

from back_tester.portfolio import Portfolio
from back_tester.models.transaction import Transaction, TransactionType
from back_tester.models.portfolio_item import PortfolioItem


class TestPortfolio:
    """Test portfolio management."""
    
    def test_portfolio_creation(self):
        """Test portfolio creation."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.cash == 10000.0
        assert len(portfolio.portfolio_items) == 0
    
    def test_cash_management(self):
        """Test cash management."""
        portfolio = Portfolio(cash=10000.0)
        
        # Test adding cash
        portfolio.add_cash(1000.0)
        assert portfolio.cash == 11000.0
        
        # Test removing cash
        portfolio.remove_cash(500.0)
        assert portfolio.cash == 10500.0
        
        # Test removing too much cash
        with pytest.raises(ValueError):
            portfolio.remove_cash(20000.0)
    
    def test_portfolio_item_management(self):
        """Test portfolio item management."""
        portfolio = Portfolio(cash=10000.0)
        
        # Test adding portfolio item
        item = PortfolioItem("AAPL", 100, 150.0, 15500.0)
        portfolio.add_portfolio_item(item)
        assert len(portfolio.portfolio_items) == 1
        assert portfolio.get_portfolio_item("AAPL") is not None
    
    def test_transaction_execution(self):
        """Test transaction execution."""
        portfolio = Portfolio(cash=10000.0)
        
        # Test buy transaction
        transaction = Transaction("AAPL", "2025-01-01", 155.0, 10, TransactionType.BUY)
        portfolio.execute_transaction(transaction)
        assert portfolio.cash == 8450.0  # 10000 - (155 * 10)
        assert len(portfolio.portfolio_items) == 1
        
        # Test sell transaction
        sell_transaction = Transaction("AAPL", "2025-01-02", 160.0, 5, TransactionType.SELL)
        portfolio.execute_transaction(sell_transaction)
        assert portfolio.cash == 9250.0  # 8450 + (160 * 5)
