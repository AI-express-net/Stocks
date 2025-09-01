"""
Configuration settings for the back tester.
"""

import os
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Any


@dataclass
class BackTesterConfig:
    """Configuration class for the back tester."""
    
    start_cash: float = 10000.0
    add_amount: float = 0.0
    add_amount_frequency_days: int = 30  # Monthly default
    start_date: str = '1970-01-01'
    end_date: str = field(default_factory=lambda: date.today().strftime('%Y-%m-%d'))
    test_frequency_days: int = 1
    stock_list_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'back_tester/tests', 'SP500_stocks.json'
    ))
    portfolio_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results', 'portfolio.json'
    ))
    transactions_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results', 'transactions.json'
    ))
    strategy: str = 'moving_average'
    valuator: str = 'real_valuator'
    benchmark_instrument: str = 'SP500'
    benchmark_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results', 'benchmark_portfolio.json'
    ))
    dividend_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'sample_dividends.json'
    ))
    

    
    def get(self, key: str, default=None):
        """Get a configuration value."""
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(f"Configuration has no attribute '{key}'")
    
    def validate(self) -> bool:
        """Validate the configuration."""
        try:
            # Validate dates
            datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.strptime(self.end_date, '%Y-%m-%d')
            
            # Validate numeric values
            assert self.start_cash >= 0, "Start cash must be non-negative"
            assert self.add_amount >= 0, "Add amount must be non-negative"
            assert self.test_frequency_days > 0, "Test frequency must be positive"
            assert self.add_amount_frequency_days > 0, "Add amount frequency must be positive"
            
            # Validate file paths - try multiple possible locations
            if not os.path.exists(self.stock_list_file):
                # Check if this is a user-provided path (not the default)
                config_dir = os.path.dirname(os.path.abspath(__file__))
                default_path = os.path.join(os.path.dirname(config_dir), 'stocks', 'US_stocks.json')
                
                # If it's not the default path, don't override user's choice
                if self.stock_list_file != default_path:
                    print(f"Warning: Stock list file not found: {self.stock_list_file}")
                    return False
                
                # Try alternative paths only for default file
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
                        self.stock_list_file = alt_path
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
        test_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_stocks.json')
        
        try:
            with open(test_file_path, 'w') as f:
                json.dump(test_stocks, f, indent=2)
            self.stock_list_file = test_file_path
            print(f"Created test stock list: {test_file_path}")
        except Exception as e:
            print(f"Failed to create test stock list: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'start_cash': self.start_cash,
            'add_amount': self.add_amount,
            'add_amount_frequency_days': self.add_amount_frequency_days,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'test_frequency_days': self.test_frequency_days,
            'stock_list_file': self.stock_list_file,
            'portfolio_file': self.portfolio_file,
            'transactions_file': self.transactions_file,
            'strategy': self.strategy,
            'valuator': self.valuator
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BackTesterConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


def get_default_config() -> BackTesterConfig:
    """Get default configuration."""
    return BackTesterConfig()


def create_config_from_file(file_path: str) -> BackTesterConfig:
    """Create configuration from JSON file."""
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return BackTesterConfig.from_dict(config_dict)
    except Exception as e:
        print(f"Failed to load configuration from {file_path}: {e}")
        return get_default_config()


def save_config_to_file(config: BackTesterConfig, file_path: str):
    """Save configuration to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to save configuration: {e}") 