"""
Enhanced back tester tests using pytest framework.
"""

import pytest
from datetime import date, timedelta

from back_tester.config import BackTesterConfig
from back_tester.enhanced_back_tester import EnhancedBackTester
from back_tester.strategies.buy_and_hold import BuyAndHoldStrategy


class TestEnhancedBackTester:
    """Test the enhanced back tester functionality."""
    
    def test_back_tester_creation(self):
        """Test that enhanced back tester can be created."""
        config = BackTesterConfig(
            start_cash=1000.0,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        back_tester = EnhancedBackTester(config)
        assert back_tester is not None
        assert back_tester.config == config
    
    def test_strategy_setting(self):
        """Test strategy setting functionality."""
        config = BackTesterConfig(
            start_cash=1000.0,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        back_tester = EnhancedBackTester(config)
        strategy = BuyAndHoldStrategy()
        
        back_tester.set_strategy(strategy)
        assert back_tester.strategy is not None
    
    def test_portfolio_status(self):
        """Test portfolio status checking."""
        config = BackTesterConfig(
            start_cash=1000.0,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        back_tester = EnhancedBackTester(config)
        
        # Check initial status
        assert back_tester.portfolio.cash_balance == 1000.0
        assert len(back_tester.portfolio.portfolio_items) == 0
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        config = BackTesterConfig(
            start_cash=1000.0,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        back_tester = EnhancedBackTester(config)
        
        # Modify portfolio
        back_tester.portfolio.add_cash(500.0)
        
        # Reset
        back_tester.reset()
        
        # Check reset
        assert back_tester.portfolio.cash_balance == 1000.0
        assert len(back_tester.portfolio.portfolio_items) == 0
