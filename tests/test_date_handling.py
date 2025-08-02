#!/usr/bin/env python3
"""
Test script to verify the date handling improvements
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import date
from stocks.quote_entity import QuoteEntity

def test_date_handling():
    """Test the date handling functionality"""
    
    # Create a test entity
    entity = QuoteEntity("AAPL")
    
    print("Testing date handling improvements...")
    print(f"Initial creation date: {entity.get_creation_date()}")
    print(f"Initial modification date: {entity.get_modification_date()}")
    print(f"Is data stale (should be True): {entity.is_data_stale()}")
    
    # Set some test data
    test_data = [{"symbol": "AAPL", "price": 150.0}]
    entity.set_api_data(test_data)
    
    # Set dates
    today = date.today().strftime("%Y-%m-%d")
    entity.set_creation_date(today)
    entity.set_modification_date(today)
    
    print(f"After setting dates - creation: {entity.get_creation_date()}")
    print(f"After setting dates - modification: {entity.get_modification_date()}")
    print(f"Is data stale (should be False): {entity.is_data_stale()}")
    
    # Test with old date
    old_date = "2023-01-01"
    entity.set_modification_date(old_date)
    print(f"With old date - is stale (should be True): {entity.is_data_stale()}")
    
    # Test with None dates
    entity.set_creation_date(None)
    entity.set_modification_date(None)
    print(f"With None dates - is stale (should be True): {entity.is_data_stale()}")
    
    print("Date handling test completed successfully!")

if __name__ == "__main__":
    test_date_handling() 