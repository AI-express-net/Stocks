"""
Test script for Phase 1 implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import BackTesterConfig
from portfolio import Portfolio
from example_valuator import ExampleValuator
from strategies.buy_and_hold import BuyAndHoldStrategy
from models.transaction import Transaction, TransactionType
from models.portfolio_item import PortfolioItem


def test_config():
    """Test configuration system."""
    print("Testing configuration system...")
    
    # Test default config
    config = BackTesterConfig()
    assert config.start_cash == 10000.0
    assert config.add_amount == 0.0
    assert config.add_amount_frequency_days == 30
    assert config.validate() == True
    print("✓ Configuration system working")
    
    # Test custom config
    custom_config = BackTesterConfig(
        start_cash=10000.0,
        add_amount=1000.0,
        add_amount_frequency_days=15,
        start_date='2025-01-01',
        end_date='2025-01-31'
    )
    assert custom_config.start_cash == 10000.0
    assert custom_config.add_amount_frequency_days == 15
    assert custom_config.validate() == True
    print("✓ Custom configuration working")


def test_data_models():
    """Test data models."""
    print("\nTesting data models...")
    
    # Test Transaction
    transaction = Transaction(
        stock="AAPL",
        date="2025-01-01",
        price=150.0,
        shares=10,
        transaction_type=TransactionType.BUY
    )
    assert transaction.stock == "AAPL"
    assert transaction.get_total_value() == 1500.0
    print("✓ Transaction model working")
    
    # Test PortfolioItem
    item = PortfolioItem(
        name="AAPL",
        shares=100,
        average_price=150.0,
        current_value=15500.0
    )
    assert item.name == "AAPL"
    assert item.get_unrealized_gain_loss() == 500.0
    print("✓ PortfolioItem model working")


def test_portfolio():
    """Test portfolio management."""
    print("\nTesting portfolio management...")
    
    portfolio = Portfolio(cash=10000.0)
    
    # Test adding cash
    portfolio.add_cash(1000.0)
    assert portfolio.cash == 11000.0
    
    # Test removing cash
    portfolio.remove_cash(500.0)
    assert portfolio.cash == 10500.0
    
    # Test adding portfolio item
    item = PortfolioItem("AAPL", 100, 150.0, 15500.0)
    portfolio.add_portfolio_item(item)
    assert len(portfolio.portfolio_items) == 1
    
    # Test executing transaction
    transaction = Transaction("AAPL", "2025-01-01", 155.0, 10, TransactionType.BUY)
    portfolio.execute_transaction(transaction)
    assert portfolio.cash == 8950.0  # 10500 - (155 * 10)
    
    print("✓ Portfolio management working")


def test_valuator():
    """Test valuator interface."""
    print("\nTesting valuator...")
    
    valuator = ExampleValuator()
    values = valuator.calculate_values(["AAPL", "GOOGL"], "2025-01-01")
    
    assert len(values) == 2
    assert all(isinstance(v, tuple) and len(v) == 2 for v in values)
    assert all(isinstance(v[0], str) and isinstance(v[1], float) for v in values)
    
    print("✓ Valuator working")


def test_strategy():
    """Test strategy interface."""
    print("\nTesting strategy...")
    
    strategy = BuyAndHoldStrategy(target_stocks=["AAPL", "GOOGL"])
    portfolio_items = []
    stock_values = [("AAPL", 150.0), ("GOOGL", 2800.0)]
    
    transactions = strategy.generate_transactions(
        portfolio_items, stock_values, "2025-01-01", 10000.0
    )
    
    assert len(transactions) > 0
    assert all(isinstance(t, Transaction) for t in transactions)
    
    print("✓ Strategy working")


def test_integration():
    """Test basic integration."""
    print("\nTesting basic integration...")
    
    # Create components
    config = BackTesterConfig({
        'start_cash': 10000.0,
        'add_amount': 0.0,
        'start_date': '2025-01-01',
        'end_date': '2025-01-02',
        'test_frequency_days': 1
    })
    
    valuator = ExampleValuator()
    strategy = BuyAndHoldStrategy(target_stocks=["AAPL", "GOOGL"])
    
    # Test that components can be created without errors
    assert config is not None
    assert valuator is not None
    assert strategy is not None
    
    print("✓ Basic integration working")


def main():
    """Run all tests."""
    print("Running Phase 1 tests...\n")
    
    try:
        test_config()
        test_data_models()
        test_portfolio()
        test_valuator()
        test_strategy()
        test_integration()
        
        print("\n🎉 All Phase 1 tests passed!")
        print("Phase 1 implementation is complete and working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 