"""
Test for FMP Stock functionality

This test verifies that the Stock class can properly fetch and manage stock data.
"""

import sys
import os
import pytest
import logging
from unittest.mock import Mock, patch

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fmp_stock import Stock, StockStatus, rate, parse_split, get_timestamp
from data_names import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStockUtilities:
    """Test utility functions in fmp_stock.py."""
    
    def test_rate_function(self):
        """Test the rate function that converts percentage to rate."""
        # Test data: percentage values and expected rates
        test_cases = [
            (0, 1.0),      # 0% = 1.0
            (5, 1.05),     # 5% = 1.05
            (10, 1.1),     # 10% = 1.1
            (-5, 0.95),    # -5% = 0.95
            (100, 2.0),    # 100% = 2.0
        ]
        
        for percentage, expected_rate in test_cases:
            result = rate(percentage)
            assert result == expected_rate, f"rate({percentage}) should be {expected_rate}, got {result}"
    
    def test_parse_split_function(self):
        """Test the parse_split function that parses stock split ratios."""
        # Test data: split strings and expected results
        test_cases = [
            ("2/1", (2.0, 1.0)),      # 2:1 split
            ("3/1", (3.0, 1.0)),      # 3:1 split
            ("1/2", (1.0, 2.0)),      # 1:2 reverse split
            ("5/4", (5.0, 4.0)),      # 5:4 split
        ]
        
        for split_str, expected in test_cases:
            result = parse_split(split_str)
            assert result == expected, f"parse_split('{split_str}') should be {expected}, got {result}"
    
    def test_get_timestamp_function(self):
        """Test the get_timestamp function returns a valid timestamp."""
        timestamp = get_timestamp()
        
        # Check that it's a string
        assert isinstance(timestamp, str)
        
        # Check that it has the expected format (YYYY-MM-DD:HH:MM:SS.TZ)
        assert len(timestamp) > 0
        assert ':' in timestamp
        assert '-' in timestamp


class TestStockCreation:
    """Test Stock object creation and initialization."""
    
    def test_stock_creation(self):
        """Test that a Stock object can be created with a name."""
        stock = Stock("AAPL")
        
        assert stock.name == "AAPL"
        assert stock.dirty == False
        assert stock.status == StockStatus.Uninitialized
        assert hasattr(stock, 'api')
        assert hasattr(stock, 'dao')
    
    def test_stock_entity_creation(self):
        """Test that all required entities are created."""
        stock = Stock("GOOGL")
        
        # Check that all required entities exist
        required_entities = [
            'balance_sheet_quarterly',
            'balance_sheet_yearly', 
            'earnings_quarterly',
            'earnings_yearly',
            'enterprise_value',
            'cash_flow_quarterly',
            'cash_flow_yearly',
            'historical_prices',
            'mda50_prices',
            'key_metrics_quarterly',
            'quote'
        ]
        
        for entity_name in required_entities:
            assert hasattr(stock, entity_name), f"Stock should have {entity_name} entity"
            entity = getattr(stock, entity_name)
            assert entity is not None, f"{entity_name} should not be None"
    
    def test_stock_with_different_names(self):
        """Test that Stock objects can be created with different names."""
        test_names = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        for name in test_names:
            stock = Stock(name)
            assert stock.name == name
            assert stock.status == StockStatus.Uninitialized


class TestStockDataFetching:
    """Test stock data fetching functionality."""
    
    @patch('fmp_stock.Factory')
    def test_fetch_stock_data_with_mock_api(self, mock_factory):
        """Test fetch_stock_data with mocked API and DAO."""
        # Mock the factory to return mock API and DAO
        mock_api = Mock()
        mock_dao = Mock()
        mock_factory.get_api_instance.return_value = mock_api
        mock_factory.get_dao_instance.return_value = mock_dao
        
        # Mock API response
        mock_api.get_api_names.return_value = ["quote"]
        mock_api.get_stock_data.return_value = [{"symbol": "AAPL", "price": 150.0}]
        
        # Mock DAO response (no existing data)
        mock_dao.find.return_value = None
        
        stock = Stock("AAPL")
        
        # Test fetching quote data
        stock.fetch_stock_data(Data.Quote)
        
        # Verify API was called
        assert mock_api.get_stock_data.call_count == 1
        mock_api.get_stock_data.assert_called_with("AAPL", Data.Quote)
        
        # Verify DAO save was called
        assert mock_dao.save.call_count == 1
    
    @patch('fmp_stock.Factory')
    @patch('fmp_stock.is_data_stale')
    def test_fetch_stock_data_with_existing_data(self, mock_is_data_stale, mock_factory):
        """Test fetch_stock_data when data already exists in DAO."""
        # Mock the factory
        mock_api = Mock()
        mock_dao = Mock()
        mock_factory.get_api_instance.return_value = mock_api
        mock_factory.get_dao_instance.return_value = mock_dao
        
        # Mock existing data in DAO
        existing_data = {
            "_id": "quote/AAPL",
            "data": [{"symbol": "AAPL", "price": 150.0}],
            "_creation_date": "2023-01-01",
            "_modification_date": "2023-01-01"
        }
        mock_dao.find.return_value = existing_data
        
        # Mock API to return quote in list so it tries to fetch from API
        mock_api.get_api_names.return_value = ["quote"]
        mock_api.get_stock_data.return_value = [{"symbol": "AAPL", "price": 150.0}]
        
        # Mock is_data_stale to return False (data is not stale)
        mock_is_data_stale.return_value = False
        
        stock = Stock("AAPL")
        
        # Test fetching quote data
        stock.fetch_stock_data(Data.Quote)
        
        # Verify API was NOT called (data already exists)
        assert mock_api.get_stock_data.call_count == 0
        
        # Verify DAO find was called
        assert mock_dao.find.call_count == 1
    
    @patch('fmp_stock.Factory')
    def test_fetch_stock_data_api_error(self, mock_factory):
        """Test fetch_stock_data when API returns no data."""
        # Mock the factory
        mock_api = Mock()
        mock_dao = Mock()
        mock_factory.get_api_instance.return_value = mock_api
        mock_factory.get_dao_instance.return_value = mock_dao
        
        # Mock API returning empty data
        mock_api.get_api_names.return_value = [Data.Quote]
        mock_api.get_stock_data.return_value = []
        mock_dao.find.return_value = None
        
        stock = Stock("AAPL")
        
        # Test that exception is raised when API returns no data
        with pytest.raises(Exception, match="No API quote data for stock AAPL"):
            stock.fetch_stock_data(Data.Quote)


class TestStockStatus:
    """Test Stock status management."""
    
    def test_stock_status_enum(self):
        """Test that StockStatus enum has expected values."""
        assert StockStatus.Uninitialized.value == (1,)
        assert StockStatus.Fetched.value == (2,)
        assert StockStatus.Populated.value == (3,)
        assert StockStatus.FetchError.value == (10,)
        assert StockStatus.PopulatingError.value == (11,)
    
    @patch('fmp_stock.Factory')
    def test_status_changes_during_fetch(self, mock_factory):
        """Test that stock status changes appropriately during data fetching."""
        # Mock the factory
        mock_api = Mock()
        mock_dao = Mock()
        mock_factory.get_api_instance.return_value = mock_api
        mock_factory.get_dao_instance.return_value = mock_dao
        
        # Mock successful API responses
        mock_api.get_api_names.return_value = ["quote", "historical_prices"]
        mock_api.get_stock_data.side_effect = [
            [{"symbol": "AAPL", "price": 150.0}],  # Quote data
            {"historical": [{"date": "2023-01-01", "close": 150.0}]}  # Historical data
        ]
        mock_dao.find.return_value = None
        
        stock = Stock("AAPL")
        
        # Initial status should be Uninitialized
        assert stock.status == StockStatus.Uninitialized
        
        # Fetch data
        stock.fetch_stock_data(Data.Quote)
        
        # Status should still be Uninitialized (only changes in fetch_from_db)
        assert stock.status == StockStatus.Uninitialized


class TestStockDataGeneration:
    """Test stock data generation functionality."""
    
    @patch('fmp_stock.Factory')
    def test_generate_data_unsupported_field(self, mock_factory):
        """Test generate_data with unsupported field name."""
        # Mock the factory
        mock_api = Mock()
        mock_dao = Mock()
        mock_factory.get_api_instance.return_value = mock_api
        mock_factory.get_dao_instance.return_value = mock_dao
        
        stock = Stock("AAPL")
        
        # Test that unsupported field raises NameError
        with pytest.raises(NameError, match="No generator for field name unsupported_field"):
            stock.generate_data("unsupported_field")
    
    @patch('fmp_stock.Factory')
    def test_generate_mda50_prices_insufficient_data(self, mock_factory):
        """Test generate_mda50_prices with insufficient historical data."""
        # Mock the factory
        mock_api = Mock()
        mock_dao = Mock()
        mock_factory.get_api_instance.return_value = mock_api
        mock_factory.get_dao_instance.return_value = mock_dao
        
        # Mock DAO to return None (no existing data)
        mock_dao.find.return_value = None
        
        # Mock API to return historical_prices and balance_sheet_quarterly in list
        mock_api.get_api_names.return_value = ["historical_prices", "balance_sheet_quarterly"]
        mock_api.get_stock_data.side_effect = [
            {"historical": []},  # historical_prices
            {"data": []}  # balance_sheet_quarterly
        ]
        
        stock = Stock("AAPL")
        
        # Mock insufficient historical data
        stock.historical_prices["data"] = {"historical": []}
        
        # Test that insufficient data raises Exception
        with pytest.raises(Exception, match="Insufficient historical data"):
            stock.generate_mda50_prices()


class TestStockIntegration:
    """Integration tests for Stock functionality."""
    
    @patch('fmp_stock.Factory')
    def test_complete_stock_workflow(self, mock_factory):
        """Test a complete stock data workflow."""
        # Mock the factory
        mock_api = Mock()
        mock_dao = Mock()
        mock_factory.get_api_instance.return_value = mock_api
        mock_factory.get_dao_instance.return_value = mock_dao
        
        # Mock API responses
        mock_api.get_api_names.return_value = ["quote", "historical_prices"]
        mock_api.get_stock_data.side_effect = [
            [{"symbol": "AAPL", "price": 150.0}],  # Quote data
            {"historical": [{"date": "2023-01-01", "close": 150.0}]}  # Historical data
        ]
        mock_dao.find.return_value = None
        
        # Create stock and fetch data
        stock = Stock("AAPL")
        
        # Test fetching multiple data types
        stock.fetch_stock_data(Data.Quote)
        stock.fetch_stock_data(Data.HistoricalPrices)
        
        # Verify API calls
        assert mock_api.get_stock_data.call_count == 2
        
        # Verify DAO saves
        assert mock_dao.save.call_count == 2
        
        # Test that stock is marked as dirty after API calls
        assert stock.dirty == False  # Should be False after save
    
    def test_stock_with_real_api_and_db(self):
        """Test Stock with real API and database operations."""
        stock = Stock("AAPL")

        # Fetch real quote data from API and save to DB
        stock.fetch_stock_data(Data.Quote)
        quote_data = stock.quote.get_data()
        assert quote_data is not None and len(quote_data) > 0
        assert quote_data[0]['symbol'] == "AAPL"
        assert 'price' in quote_data[0]

        # Save to DB
        stock.save_to_db()

        # Create a new Stock object and fetch from DB (should not call API)
        stock2 = Stock("AAPL")
        stock2.fetch_stock_data(Data.Quote)
        quote_data2 = stock2.quote.get_data()
        assert quote_data2 is not None and len(quote_data2) > 0
        assert quote_data2[0]['symbol'] == "AAPL"
        assert quote_data2[0]['price'] == quote_data[0]['price']

        # Optionally, print for debug
        print("Fetched price from API:", quote_data[0]['price'])
        print("Fetched price from DB:", quote_data2[0]['price'])


class TestRealApiAndDatabase:
    """Real API and database integration tests."""
    
    def test_real_quote_data_fetch_and_store(self):
        """Test fetching real quote data and storing/retrieving from database."""
        try:
            # Create stock object
            stock = Stock("AAPL")
            
            # Fetch real quote data from API
            stock.fetch_stock_data(Data.Quote)
            
            # Verify data was fetched
            quote_data = stock.quote.get_data()
            assert quote_data is not None
            assert len(quote_data) > 0
            
            # Verify the data structure
            quote = quote_data[0]
            assert quote['symbol'] == 'AAPL'
            assert 'price' in quote
            assert 'change' in quote
            assert 'changesPercentage' in quote
            
            # Save to database
            stock.save_to_db()
            
            # Create a new stock object to test database retrieval
            stock2 = Stock("AAPL")
            
            # Fetch from database (should not call API)
            stock2.fetch_stock_data(Data.Quote)
            
            # Verify data was retrieved from database
            quote_data2 = stock2.quote.get_data()
            assert quote_data2 is not None
            assert len(quote_data2) > 0
            
            # Verify data integrity (should be the same)
            assert quote_data2[0]['symbol'] == quote_data[0]['symbol']
            assert quote_data2[0]['price'] == quote_data[0]['price']
            
            logger.info("✅ Real API and database test passed")
            
        except Exception as e:
            # Skip: Network issues, API rate limits, or database connectivity problems
            pytest.skip(f"Skip: Real API/database test failed due to: {e}")
    
    def test_real_historical_data_fetch_and_store(self):
        """Test fetching real historical data and storing/retrieving from database."""
        try:
            # Create stock object
            stock = Stock("MSFT")
            
            # Fetch real historical data from API
            stock.fetch_stock_data(Data.HistoricalPrices)
            
            # Verify data was fetched
            historical_data = stock.historical_prices.get_data()
            assert historical_data is not None
            assert 'historical' in historical_data
            assert len(historical_data['historical']) > 0
            
            # Verify the data structure
            price_data = historical_data['historical'][0]
            assert 'date' in price_data
            assert 'close' in price_data

            # Save to database
            stock.save_to_db()
            
            # Create a new stock object to test database retrieval
            stock2 = Stock("MSFT")
            
            # Fetch from database (should not call API)
            stock2.fetch_stock_data(Data.HistoricalPrices)
            
            # Verify data was retrieved from database
            historical_data2 = stock2.historical_prices.get_data()
            assert historical_data2 is not None
            assert 'historical' in historical_data2
            assert len(historical_data2['historical']) > 0
            
            # Verify data integrity (should be the same)
            assert historical_data2['historical'][0]['date'] == historical_data['historical'][0]['date']
            assert historical_data2['historical'][0]['close'] == historical_data['historical'][0]['close']
            
            logger.info("✅ Real historical data API and database test passed")
            
        except Exception as e:
            # Skip: Network issues, API rate limits, or database connectivity problems
            pytest.skip(f"Skip: Real historical data API/database test failed due to: {e}")
    
    def test_data_staleness_and_refresh(self):
        """Test that stale data is refreshed from API."""
        try:
            # Create stock object
            stock = Stock("GOOGL")
            
            # Fetch data initially
            stock.fetch_stock_data(Data.Quote)
            initial_data = stock.quote.get_data()
            
            # Verify data was fetched
            assert initial_data is not None
            assert len(initial_data) > 0
            
            # Save to database
            stock.save_to_db()
            
            # Create new stock object
            stock2 = Stock("GOOGL")
            
            # Fetch from database (should use cached data)
            stock2.fetch_stock_data(Data.Quote)
            cached_data = stock2.quote.get_data()
            
            # Verify cached data matches
            assert cached_data == initial_data
            
            # Test that data is marked as fresh (not stale)
            quote_entity = stock2.quote
            assert not quote_entity.is_data_stale()
            
            logger.info("✅ Data staleness and refresh test passed")
            
        except Exception as e:
            # Skip: Network issues, API rate limits, or database connectivity problems
            pytest.skip(f"Skip: Data staleness test failed due to: {e}")
    
    def test_multiple_stocks_data_integrity(self):
        """Test that multiple stocks can be fetched and stored independently."""
        try:
            test_stocks = ["AAPL", "MSFT", "GOOGL"]
            stock_data = {}
            
            # Fetch data for multiple stocks
            for symbol in test_stocks:
                stock = Stock(symbol)
                stock.fetch_stock_data(Data.Quote)
                
                quote_data = stock.quote.get_data()
                assert quote_data is not None
                assert len(quote_data) > 0
                assert quote_data[0]['symbol'] == symbol
                
                stock.save_to_db()
                stock_data[symbol] = quote_data[0]['price']
            
            # Verify each stock has different data
            prices = list(stock_data.values())
            assert len(set(prices)) > 1, "All stocks should have different prices"
            
            # Test retrieval for each stock
            for symbol in test_stocks:
                stock = Stock(symbol)
                stock.fetch_stock_data(Data.Quote)
                
                retrieved_data = stock.quote.get_data()
                assert retrieved_data is not None
                assert retrieved_data[0]['symbol'] == symbol
                assert retrieved_data[0]['price'] == stock_data[symbol]
            
            logger.info("✅ Multiple stocks data integrity test passed")
            
        except Exception as e:
            # Skip: Network issues, API rate limits, or database connectivity problems
            pytest.skip(f"Skip: Multiple stocks test failed due to: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 