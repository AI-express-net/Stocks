"""
Mock valuator tests using pytest framework.
"""

import pytest
from datetime import date, timedelta

from back_tester.tests.mock_valuator import MockValuator


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
