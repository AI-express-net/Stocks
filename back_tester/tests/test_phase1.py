"""
Phase 1 tests using pytest framework.
"""

import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BackTesterConfig
from portfolio import Portfolio
from example_valuator import ExampleValuator
from strategies.buy_and_hold import BuyAndHoldStrategy
from models.transaction import Transaction, TransactionType
from models.portfolio_item import PortfolioItem


class TestConfiguration:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BackTesterConfig()
        assert config.get('start_cash') == 0.0
        assert config.get('add_amount') == 0.0
        assert config.get('start_date') == '1970-01-01'
        assert config.validate() == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_config = BackTesterConfig({
            'start_cash': 10000.0,
            'add_amount': 1000.0,
            'start_date': '2025-01-01',
            'end_date': '2025-01-31'
        })
        assert custom_config.get('start_cash') == 10000.0
        assert custom_config.get('add_amount') == 1000.0
        assert custom_config.validate() == True
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = BackTesterConfig()
        assert config.validate() == True
        
        # Test invalid date
        invalid_config = BackTesterConfig({'start_date': 'invalid-date'})
        assert invalid_config.validate() == False


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


class TestValuator:
    """Test valuator interface."""
    
    def test_valuator_creation(self):
        """Test valuator creation."""
        valuator = ExampleValuator()
        assert valuator is not None
    
    def test_calculate_values(self):
        """Test stock value calculation."""
        valuator = ExampleValuator()
        values = valuator.calculate_values(["AAPL", "GOOGL"], "2025-01-01")
        
        assert len(values) == 2
        assert all(isinstance(v, tuple) and len(v) == 2 for v in values)
        assert all(isinstance(v[0], str) and isinstance(v[1], float) for v in values)
    
    def test_date_validation(self):
        """Test date validation."""
        valuator = ExampleValuator()
        assert valuator.validate_date("2025-01-01") == True
        assert valuator.validate_date("invalid-date") == False


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


class TestIntegration:
    """Test basic integration."""
    
    def test_component_creation(self):
        """Test that all components can be created."""
        config = BackTesterConfig({
            'start_cash': 10000.0,
            'add_amount': 0.0,
            'start_date': '2025-01-01',
            'end_date': '2025-01-02',
            'test_frequency_days': 1
        })
        
        valuator = ExampleValuator()
        strategy = BuyAndHoldStrategy(target_stocks=["AAPL", "GOOGL"])
        
        assert config is not None
        assert valuator is not None
        assert strategy is not None
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        portfolio = Portfolio(cash=10000.0)
        item = PortfolioItem("AAPL", 100, 150.0, 15500.0)
        portfolio.add_portfolio_item(item)
        
        stock_values = [("AAPL", 160.0)]
        total_value = portfolio.get_total_value(stock_values)
        
        assert total_value == 26000.0  # 10000 cash + (100 * 160) 