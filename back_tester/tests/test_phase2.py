"""
Phase 2 Test Suite

This module tests the enhanced back tester functionality including:
- Real data integration
- Enhanced portfolio management
- Performance tracking
- Risk metrics calculation
"""

import sys
import os
import pytest
from datetime import date, timedelta

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BackTesterConfig
from enhanced_portfolio import EnhancedPortfolio, PerformanceSnapshot, RiskMetrics
from mock_valuator import MockValuator
from enhanced_back_tester import EnhancedBackTester
from strategies.buy_and_hold import BuyAndHoldStrategy
from models.transaction import Transaction, TransactionType
from models.portfolio_item import PortfolioItem


class TestMockValuator:
    """Test the mock valuator functionality."""
    
    def test_valuator_creation(self):
        """Test that the mock valuator can be created."""
        valuator = MockValuator()
        assert valuator is not None
        assert hasattr(valuator, 'calculate_values')
    
    def test_date_validation(self):
        """Test date validation in mock valuator."""
        valuator = MockValuator()
        
        # Valid date
        valid_date = date(2023, 1, 1)
        valuator.validate_date(valid_date)
        
        # Invalid date
        assert valuator.validate_date(None) == False
    
    def test_calculate_values(self):
        """Test calculate_values with mock data."""
        valuator = MockValuator()
        
        # Test with mock data
        stocks = ["AAPL", "GOOGL"]
        target_date = date(2023, 1, 1)
        
        result = valuator.calculate_values(stocks, target_date)
        
        # Should return a list of tuples
        assert isinstance(result, list)
        assert len(result) == 2
        
        # Check that we got prices for both stocks
        stock_symbols = [item[0] for item in result]
        assert "AAPL" in stock_symbols
        assert "GOOGL" in stock_symbols
        
        # Check that prices are reasonable
        for stock_symbol, price in result:
            assert price > 0
            assert isinstance(price, float)
    
    def test_price_consistency(self):
        """Test that prices are consistent for the same date and stock."""
        valuator = MockValuator()
        
        stocks = ["AAPL"]
        target_date = date(2023, 1, 1)
        
        # Get prices twice
        result1 = valuator.calculate_values(stocks, target_date)
        result2 = valuator.calculate_values(stocks, target_date)
        
        # Prices should be the same (due to caching)
        assert result1 == result2


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
        assert portfolio.add_cash(500.0) == True
        assert portfolio.get_cash_balance() == 1500.0
        
        # Remove cash
        assert portfolio.add_cash(-200.0) == True
        assert portfolio.get_cash_balance() == 1300.0
        
        # Try to remove too much
        assert portfolio.add_cash(-2000.0) == False
        assert portfolio.get_cash_balance() == 1300.0  # Should remain unchanged
    
    def test_portfolio_item_management(self):
        """Test portfolio item management."""
        portfolio = EnhancedPortfolio()
        
        # Create a portfolio item
        item = PortfolioItem(
            name="AAPL",
            shares=10,
            average_price=150.0,
            current_value=1500.0,
            date_added=date.today(),
            last_modified=date.today()
        )
        
        portfolio.add_portfolio_item(item)
        assert "AAPL" in portfolio.portfolio_items
        assert portfolio.get_portfolio_item("AAPL") == item
        
        # Remove item
        assert portfolio.remove_portfolio_item("AAPL") == True
        assert "AAPL" not in portfolio.portfolio_items
    
    def test_transaction_execution(self):
        """Test transaction execution."""
        portfolio = EnhancedPortfolio(initial_cash=10000.0)
        
        # Test buy transaction
        buy_transaction = Transaction(
            stock="AAPL",
            date=date.today(),
            price=150.0,
            shares=10,
            transaction_type=TransactionType.BUY
        )
        
        assert portfolio.execute_transaction(buy_transaction) == True
        assert portfolio.get_cash_balance() == 8500.0  # 10000 - (150 * 10)
        assert "AAPL" in portfolio.portfolio_items
        
        # Test sell transaction
        sell_transaction = Transaction(
            stock="AAPL",
            date=date.today(),
            price=160.0,
            shares=5,
            transaction_type=TransactionType.SELL
        )
        
        assert portfolio.execute_transaction(sell_transaction) == True
        assert portfolio.get_cash_balance() == 9300.0  # 8500 + (160 * 5)
        assert portfolio.portfolio_items["AAPL"].shares == 5
    
    def test_performance_snapshot(self):
        """Test performance snapshot functionality."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Add a position
        item = PortfolioItem(
            name="AAPL",
            shares=10,
            average_price=150.0,
            current_value=1600.0,
            date_added=date.today(),
            last_modified=date.today()
        )
        portfolio.add_portfolio_item(item)
        
        # Take snapshot
        snapshot = portfolio.take_performance_snapshot(date.today())
        
        assert snapshot.total_value == 2600.0  # 1000 cash + 1600 stock
        assert snapshot.cash_balance == 1000.0
        assert snapshot.stock_value == 1600.0
        assert snapshot.num_positions == 1
        assert snapshot.largest_position == "AAPL"
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Create some performance history
        dates = [date.today() + timedelta(days=i) for i in range(5)]
        values = [1000.0, 1050.0, 1100.0, 1080.0, 1120.0]
        
        for i, (d, v) in enumerate(zip(dates, values)):
            # Simulate portfolio value
            portfolio.cash_balance = v
            snapshot = portfolio.take_performance_snapshot(d)
        
        # Calculate risk metrics
        risk_metrics = portfolio.calculate_risk_metrics()
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert hasattr(risk_metrics, 'volatility')
        assert hasattr(risk_metrics, 'sharpe_ratio')
        assert hasattr(risk_metrics, 'max_drawdown')
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        portfolio = EnhancedPortfolio(initial_cash=1000.0)
        
        # Add some positions
        item1 = PortfolioItem(
            name="AAPL",
            shares=10,
            average_price=150.0,
            current_value=1600.0,
            date_added=date.today(),
            last_modified=date.today()
        )
        item2 = PortfolioItem(
            name="GOOGL",
            shares=5,
            average_price=2800.0,
            current_value=14000.0,
            date_added=date.today(),
            last_modified=date.today()
        )
        
        portfolio.add_portfolio_item(item1)
        portfolio.add_portfolio_item(item2)
        
        summary = portfolio.get_portfolio_summary()
        
        assert summary["cash_balance"] == 1000.0
        assert summary["total_value"] == 16600.0  # 1000 + 1600 + 14000
        assert summary["num_positions"] == 2
        assert len(summary["positions"]) == 2
        
        # Check that positions are sorted by value (largest first)
        assert summary["positions"][0]["stock"] == "GOOGL"  # Higher value
        assert summary["positions"][1]["stock"] == "AAPL"


class TestEnhancedBackTester:
    """Test the enhanced back tester functionality."""
    
    def test_back_tester_creation(self):
        """Test that enhanced back tester can be created."""
        config = BackTesterConfig()
        back_tester = EnhancedBackTester(config)
        
        assert back_tester.config == config
        assert back_tester.portfolio is not None
        assert back_tester.valuator is not None
        assert back_tester.strategy is None  # Not set yet
    
    def test_strategy_setting(self):
        """Test strategy setting functionality."""
        config = BackTesterConfig()
        back_tester = EnhancedBackTester(config)
        
        strategy = BuyAndHoldStrategy()
        back_tester.set_strategy(strategy)
        
        assert back_tester.strategy == strategy
    
    def test_portfolio_status(self):
        """Test portfolio status retrieval."""
        config = BackTesterConfig()
        back_tester = EnhancedBackTester(config)
        
        status = back_tester.get_portfolio_status()
        
        assert "current_date" in status
        assert "cash_balance" in status
        assert "total_value" in status
        assert "num_positions" in status
        assert "portfolio_summary" in status
    
    def test_reset_functionality(self):
        """Test back tester reset functionality."""
        config = BackTesterConfig()
        back_tester = EnhancedBackTester(config)
        
        # Add some state
        strategy = BuyAndHoldStrategy()
        back_tester.set_strategy(strategy)
        
        # Reset
        back_tester.reset()
        
        assert back_tester.strategy is None
        assert len(back_tester.performance_snapshots) == 0
        assert len(back_tester.transaction_log) == 0


class TestIntegration:
    """Integration tests for Phase 2 components."""
    
    def test_component_integration(self):
        """Test that all Phase 2 components work together."""
        # Create components
        config = BackTesterConfig()
        portfolio = EnhancedPortfolio(initial_cash=10000.0)
        valuator = MockValuator()
        strategy = BuyAndHoldStrategy()
        
        # Test that all components can be created
        assert config is not None
        assert portfolio is not None
        assert valuator is not None
        assert strategy is not None
        
        # Test portfolio operations
        assert portfolio.get_cash_balance() == 10000.0
        assert portfolio.get_total_value() == 10000.0
        
        # Test valuator operations
        test_date = date(2023, 1, 1)
        valuator.validate_date(test_date)
        
        # Test strategy operations
        portfolio_items = []
        stock_values = [("AAPL", 150.0), ("GOOGL", 2800.0)]
        transactions = strategy.generate_transactions(
            portfolio_items, stock_values, test_date, 10000.0
        )
        
        assert isinstance(transactions, list)
    
    def test_file_operations(self):
        """Test file operations for portfolio and transactions."""
        # Test portfolio file operations
        portfolio = EnhancedPortfolio(initial_cash=1000.0, portfolio_file="test_portfolio.json")
        
        # Add a position
        item = PortfolioItem(
            name="AAPL",
            shares=10,
            average_price=150.0,
            current_value=1600.0,
            date_added=date.today(),
            last_modified=date.today()
        )
        portfolio.add_portfolio_item(item)
        
        # Save and load
        portfolio.save_to_file()
        
        # Create new portfolio and load
        new_portfolio = EnhancedPortfolio(initial_cash=0.0, portfolio_file="test_portfolio.json")
        
        assert new_portfolio.get_cash_balance() == 1000.0
        assert "AAPL" in new_portfolio.portfolio_items
        
        # Clean up
        import os
        if os.path.exists("test_portfolio.json"):
            os.remove("test_portfolio.json")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 