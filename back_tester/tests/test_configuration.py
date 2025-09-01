"""
Configuration tests using pytest framework.
"""

import os
import sys

# Add the back_tester directory to the Python path for imports
back_tester_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if back_tester_dir not in sys.path:
    sys.path.insert(0, back_tester_dir)

from back_tester.config import BackTesterConfig


class TestConfiguration:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BackTesterConfig()
        assert config.start_cash == 10000.0  # Updated to match new default
        assert config.add_amount == 0.0
        assert config.add_amount_frequency_days == 30
        assert config.start_date == '1970-01-01'
        assert config.validate() == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_config = BackTesterConfig(
            start_cash=10000.0,
            add_amount=1000.0,
            add_amount_frequency_days=15,
            start_date='2025-01-01',
            end_date='2025-01-31'
        )
        assert custom_config.start_cash == 10000.0
        assert custom_config.add_amount == 1000.0
        assert custom_config.add_amount_frequency_days == 15
        assert custom_config.validate() == True
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = BackTesterConfig()
        assert config.validate() == True
        
        # Test invalid date
        invalid_config = BackTesterConfig(start_date='invalid-date')
        assert invalid_config.validate() == False
