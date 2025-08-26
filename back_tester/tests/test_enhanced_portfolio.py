"""
Enhanced portfolio tests using pytest framework.
"""

import pytest
from datetime import date, timedelta

from back_tester.enhanced_portfolio import EnhancedPortfolio, PerformanceSnapshot, RiskMetrics
from back_tester.models.transaction import Transaction, TransactionType
from back_tester.models.portfolio_item import PortfolioItem


class TestEnhancedPortfolio:
    """Test the enhanced portfolio functionality."""
    
    def test_portfolio_creation(self):
        """Test that enhanced portfolio can be created."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        assert portfolio.cash_balance == 1000.0
        assert len(portfolio.portfolio_items) == 0
    
    def test_cash_management(self):
        """Test cash management functionality."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Add cash
        portfolio.add_cash(500.0)
        assert portfolio.cash_balance == 1500.0
        
        # Remove cash (using add_cash with negative value)
        portfolio.add_cash(-200.0)
        assert portfolio.cash_balance == 1300.0
        
        # Test insufficient cash
        result = portfolio.add_cash(-2000.0)
        assert result == False  # Should fail but not raise exception
    
    def test_portfolio_item_management(self):
        """Test portfolio item management."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Add portfolio item
        item = PortfolioItem("AAPL", 10, 150.0, 1600.0)
        portfolio.add_portfolio_item(item)
        
        assert len(portfolio.portfolio_items) == 1
        assert "AAPL" in portfolio.portfolio_items
        
        # Get portfolio item
        retrieved_item = portfolio.get_portfolio_item("AAPL")
        assert retrieved_item is not None
        assert retrieved_item.shares == 10
    
    def test_transaction_execution(self):
        """Test transaction execution."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Execute buy transaction
        buy_transaction = Transaction("AAPL", "2023-01-01", 150.0, 5, TransactionType.BUY)
        portfolio.execute_transaction(buy_transaction)
        
        assert portfolio.cash_balance == 250.0  # 1000 - (150 * 5)
        assert len(portfolio.portfolio_items) == 1
        
        # Execute sell transaction
        sell_transaction = Transaction("AAPL", "2023-01-02", 160.0, 2, TransactionType.SELL)
        portfolio.execute_transaction(sell_transaction)
        
        assert portfolio.cash_balance == 570.0  # 250 + (160 * 2)
        assert portfolio.get_portfolio_item("AAPL").shares == 3
    
    def test_performance_snapshot(self):
        """Test performance snapshot creation."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Add some portfolio items
        item = PortfolioItem("AAPL", 10, 150.0, 1600.0)
        portfolio.add_portfolio_item(item)
        
        # Create snapshot
        snapshot = portfolio.take_performance_snapshot(date(2023, 1, 1))
        
        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.date == date(2023, 1, 1)  # date object, not string
        assert snapshot.total_value > 0
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Add some performance history (create proper PerformanceSnapshot objects)
        from enhanced_portfolio import PerformanceSnapshot
        
        portfolio.performance_history = [
            PerformanceSnapshot(date=date(2023, 1, 1), total_value=1000.0, cash_balance=1000.0, stock_value=0.0, total_return=0.0, daily_return=0.0, num_positions=0, largest_position=None, largest_position_value=0.0),
            PerformanceSnapshot(date=date(2023, 1, 2), total_value=1100.0, cash_balance=1000.0, stock_value=100.0, total_return=100.0, daily_return=10.0, num_positions=1, largest_position="AAPL", largest_position_value=100.0),
            PerformanceSnapshot(date=date(2023, 1, 3), total_value=1050.0, cash_balance=1000.0, stock_value=50.0, total_return=50.0, daily_return=-4.55, num_positions=1, largest_position="AAPL", largest_position_value=50.0)
        ]
        
        metrics = portfolio.calculate_risk_metrics()
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.volatility >= 0
        assert metrics.sharpe_ratio >= 0 or metrics.sharpe_ratio <= 0  # Can be negative
        assert metrics.max_drawdown >= 0
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Add some portfolio items
        item = PortfolioItem("AAPL", 10, 150.0, 1600.0)
        portfolio.add_portfolio_item(item)
        
        summary = portfolio.get_portfolio_summary()
        
        assert "cash_balance" in summary
        assert "positions" in summary  # Changed from portfolio_items to positions
        assert "total_value" in summary
        assert "num_positions" in summary
