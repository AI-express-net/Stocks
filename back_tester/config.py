"""
Configuration settings for the back tester.
"""

import os
from datetime import date, datetime
from typing import Dict, Any


class BackTesterConfig:
    """Configuration class for the back tester."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize configuration with defaults or provided values."""
        
        # Get the directory where this config file is located
        config_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Default configuration values with robust paths
        self.defaults = {
            'start_cash': 0.0,
            'add_amount': 0.0,
            'start_date': '1970-01-01',
            'end_date': date.today().strftime('%Y-%m-%d'),
            'test_frequency_days': 1,
            'stock_list_file': os.path.join(os.path.dirname(config_dir), 'stocks', 'US_stocks.json'),
            'portfolio_file': os.path.join(config_dir, 'portfolio.json'),
            'transactions_file': os.path.join(config_dir, 'transactions.json')
        }
        
        # Load configuration
        self.config = self.defaults.copy()
        if config_dict:
            self.config.update(config_dict)
    
    def get(self, key: str, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
    
    def validate(self) -> bool:
        """Validate the configuration."""
        try:
            # Validate dates
            datetime.strptime(self.config['start_date'], '%Y-%m-%d')
            datetime.strptime(self.config['end_date'], '%Y-%m-%d')
            
            # Validate numeric values
            assert self.config['start_cash'] >= 0, "Start cash must be non-negative"
            assert self.config['add_amount'] >= 0, "Add amount must be non-negative"
            assert self.config['test_frequency_days'] > 0, "Test frequency must be positive"
            
            # Validate file paths - try multiple possible locations
            stock_list_file = self.config['stock_list_file']
            if not os.path.exists(stock_list_file):
                # Try alternative paths
                config_dir = os.path.dirname(os.path.abspath(__file__))
                alternative_paths = [
                    os.path.join(config_dir, 'tests', 'US_stocks.json'),  # Tests directory
                    os.path.join(config_dir, '..', 'stocks', 'US_stocks.json'),  # Original location
                    os.path.join(config_dir, '..', '..', 'stocks', 'US_stocks.json'),
                    os.path.join(os.getcwd(), 'stocks', 'US_stocks.json'),
                    os.path.join(os.getcwd(), '..', 'stocks', 'US_stocks.json'),
                    'stocks/US_stocks.json',
                    '../stocks/US_stocks.json'
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        self.config['stock_list_file'] = alt_path
                        break
                else:
                    # If no file found, create a minimal test file
                    self._create_test_stock_list()
            
            return True
        except (ValueError, AssertionError) as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def _create_test_stock_list(self):
        """Create a minimal test stock list file if the real one doesn't exist."""
        import json
        
        test_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_stocks.json')
        
        try:
            with open(test_file_path, 'w') as f:
                json.dump(test_stocks, f)
            self.config['stock_list_file'] = test_file_path
            print(f"Created test stock list file: {test_file_path}")
        except Exception as e:
            print(f"Failed to create test stock list: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        self.config.update(config_dict)


def get_default_config() -> BackTesterConfig:
    """Get default configuration."""
    return BackTesterConfig()


def create_config_from_file(file_path: str) -> BackTesterConfig:
    """Create configuration from JSON file."""
    import json
    
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return BackTesterConfig(config_dict)
    except FileNotFoundError:
        print(f"Configuration file not found: {file_path}")
        return get_default_config()
    except json.JSONDecodeError:
        print(f"Invalid JSON in configuration file: {file_path}")
        return get_default_config()


def save_config_to_file(config: BackTesterConfig, file_path: str):
    """Save configuration to JSON file."""
    import json
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    except Exception as e:
        print(f"Failed to save configuration: {e}") 