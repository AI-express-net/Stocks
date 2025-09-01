"""
Enhanced integration tests using pytest framework.
"""

import pytest
import os

from back_tester.config import BackTesterConfig
from back_tester.enhanced_back_tester import EnhancedBackTester
from back_tester.strategies.buy_and_hold import BuyAndHoldStrategy


class TestIntegration:
    """Test enhanced integration functionality."""
    
    def test_component_integration(self):
        """Test that all enhanced components work together."""
        config = BackTesterConfig(
            start_cash=1000.0,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        back_tester = EnhancedBackTester(config)
        strategy = BuyAndHoldStrategy()
        
        back_tester.set_strategy(strategy)
        
        assert back_tester.config is not None
        assert back_tester.strategy is not None
        assert back_tester.portfolio is not None
        assert back_tester.benchmark_portfolio is not None
    
    def test_file_operations(self):
        """Test file operations (save/load)."""
        config = BackTesterConfig(
            start_cash=1000.0,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        back_tester = EnhancedBackTester(config)
        
        # Test save operations
        try:
            back_tester.portfolio.save_to_file()  # No arguments needed
            assert os.path.exists(back_tester.portfolio.portfolio_file)
            
            # Clean up
            os.remove(back_tester.portfolio.portfolio_file)
        except Exception as e:
            pytest.fail(f"File operations failed: {e}")
