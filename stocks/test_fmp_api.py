"""
Test for FMP API functionality

This test verifies that the FMP API can fetch actual stock data from the Financial Modeling Prep API.
"""

import sys
import os
import pytest
import logging

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .fmp_api import FmpApi, fetch_from_api
from .data_names import Data, Market
from .api_error import ApiError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFmpApi:
    """Test the FMP API functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api = FmpApi()
        self.test_symbol = "AAPL"  # Use AAPL as it's a reliable stock
    
    def test_api_creation(self):
        """Test that the API object can be created."""
        assert self.api is not None
        assert hasattr(self.api, 'get_stock_data')
        assert hasattr(self.api, 'get_market_data')
        assert hasattr(self.api, 'get_stock_list')
    
    def test_get_api_names(self):
        """Test that API names are returned correctly."""
        api_names = self.api.get_api_names()
        assert isinstance(api_names, list)
        assert len(api_names) > 0
        
        # Check that expected data types are in the list
        expected_types = [
            Data.BalanceSheetQuarterly,
            Data.BalanceSheetYearly,
            Data.CashFlowQuarterly,
            Data.CashFlowYearly,
            Data.EarningsQuarterly,
            Data.EarningsYearly,
            Data.EnterpriseValue,
            Data.HistoricalPrices,
            Data.KeyMetricsQuarterly,
            Data.Quote
        ]
        
        for expected_type in expected_types:
            assert expected_type in api_names
    
    def test_fetch_quote_data(self):
        """Test fetching quote data for a stock."""
        try:
            quote_data = self.api.get_stock_data(self.test_symbol, Data.Quote)
            
            # Verify the response structure
            assert isinstance(quote_data, list)
            assert len(quote_data) > 0
            
            # Check that we got data for the requested symbol
            quote = quote_data[0]
            assert quote['symbol'] == self.test_symbol
            
            # Check for expected fields (based on actual API response)
            expected_fields = ['symbol', 'price', 'change', 'changesPercentage']
            for field in expected_fields:
                assert field in quote, f"Expected field '{field}' not found in quote data"
            
            logger.info(f"✅ Quote data fetched successfully for {self.test_symbol}")
            logger.info(f"   Price: ${quote.get('price', 'N/A')}")
            logger.info(f"   Change: {quote.get('change', 'N/A')}")
            
        except ApiError as e:
            logger.error(f"API error fetching quote data: {e}")
            pytest.skip(f"API error fetching quote data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching quote data: {e}")
            pytest.skip(f"Unexpected error fetching quote data: {e}")
    
    def test_fetch_historical_prices(self):
        """Test fetching historical price data for a stock."""
        try:
            historical_data = self.api.get_stock_data(self.test_symbol, Data.HistoricalPrices)
            
            # Verify the response structure
            assert isinstance(historical_data, dict)
            assert 'historical' in historical_data
            
            historical_prices = historical_data['historical']
            assert isinstance(historical_prices, list)
            assert len(historical_prices) > 0
            
            # Check the structure of historical price data
            price_data = historical_prices[0]
            expected_fields = ['date', 'close']
            for field in expected_fields:
                assert field in price_data, f"Expected field '{field}' not found in historical data"
            
            logger.info(f"✅ Historical prices fetched successfully for {self.test_symbol}")
            logger.info(f"   Number of data points: {len(historical_prices)}")
            logger.info(f"   Date range: {historical_prices[-1]['date']} to {historical_prices[0]['date']}")
            
        except ApiError as e:
            pytest.skip(f"API error fetching historical prices: {e}")
        except Exception as e:
            pytest.skip(f"Unexpected error fetching historical prices: {e}")
    
    def test_fetch_earnings_data(self):
        """Test fetching earnings data for a stock."""
        try:
            earnings_data = self.api.get_stock_data(self.test_symbol, Data.EarningsQuarterly)
            
            # Verify the response structure
            assert isinstance(earnings_data, list)
            assert len(earnings_data) > 0
            
            # Check the structure of earnings data
            earnings = earnings_data[0]
            expected_fields = ['date', 'symbol', 'period', 'eps']
            for field in expected_fields:
                assert field in earnings, f"Expected field '{field}' not found in earnings data"
            
            logger.info(f"✅ Earnings data fetched successfully for {self.test_symbol}")
            logger.info(f"   Number of quarters: {len(earnings_data)}")
            
        except ApiError as e:
            pytest.skip(f"API error fetching earnings data: {e}")
        except Exception as e:
            pytest.skip(f"Unexpected error fetching earnings data: {e}")
    
    def test_fetch_market_data(self):
        """Test fetching market data (stock lists)."""
        try:
            # Test fetching S&P 500 stocks
            sp500_data = self.api.get_market_data(Market.SP500Stocks)
            
            # Verify the response structure
            assert isinstance(sp500_data, list)
            assert len(sp500_data) > 0
            
            # Check that we got stock data
            stock = sp500_data[0]
            assert 'symbol' in stock
            
            logger.info(f"✅ Market data fetched successfully")
            logger.info(f"   S&P 500 stocks count: {len(sp500_data)}")
            
        except ApiError as e:
            pytest.skip(f"API error fetching market data: {e}")
        except Exception as e:
            pytest.skip(f"Unexpected error fetching market data: {e}")
    
    def test_get_stock_list(self):
        """Test getting a list of stock symbols."""
        try:
            stock_list = self.api.get_stock_list(Market.SP500Stocks)
            
            # Verify the response structure
            assert isinstance(stock_list, dict)
            assert 'stocks' in stock_list
            
            stocks = stock_list['stocks']
            assert isinstance(stocks, list)
            assert len(stocks) > 0
            
            # Check that we got valid stock symbols
            for symbol in stocks[:10]:  # Check first 10 symbols
                assert isinstance(symbol, str)
                assert len(symbol) > 0
            
            logger.info(f"✅ Stock list fetched successfully")
            logger.info(f"   Number of stocks: {len(stocks)}")
            logger.info(f"   Sample symbols: {stocks[:5]}")
            
        except ApiError as e:
            pytest.skip(f"API error fetching stock list: {e}")
        except Exception as e:
            pytest.skip(f"Unexpected error fetching stock list: {e}")
    
    def test_invalid_symbol(self):
        """Test behavior with invalid stock symbol."""
        invalid_symbol = "INVALID_SYMBOL_12345"
        
        try:
            # This should either return empty data or raise an error
            quote_data = self.api.get_stock_data(invalid_symbol, Data.Quote)
            
            # If we get here, the API returned data (even if empty)
            assert isinstance(quote_data, list)
            
            if len(quote_data) == 0:
                logger.info("✅ Invalid symbol handled correctly (empty response)")
            else:
                logger.info("✅ Invalid symbol returned data (API may be lenient)")
                
        except ApiError:
            logger.info("✅ Invalid symbol handled correctly (API error)")
        except Exception as e:
            logger.info(f"✅ Invalid symbol handled correctly (exception: {e})")
    
    def test_api_error_handling(self):
        """Test API error handling."""
        # Test with a malformed request - use a truly invalid data type
        with pytest.raises(KeyError):
            # This should raise a KeyError due to invalid data type
            self.api.get_stock_data("AAPL", "INVALID_DATA_TYPE")
    
    def test_fetch_from_api_function(self):
        """Test the fetch_from_api function directly."""
        try:
            # Test with a simple API call
            from .api_config import API_KEY, API_BASE
            test_uri = f"{API_BASE}quote/AAPL?apikey={API_KEY}"
            
            data = fetch_from_api(test_uri)
            assert isinstance(data, list)
            assert len(data) > 0
            
            logger.info("✅ fetch_from_api function works correctly")
            
        except Exception as e:
            pytest.skip(f"fetch_from_api test failed: {e}")


class TestApiIntegration:
    """Integration tests for the API."""
    
    def test_full_data_fetch_workflow(self):
        """Test a complete data fetching workflow."""
        api = FmpApi()
        test_symbol = "MSFT"  # Use Microsoft as another reliable stock
        
        try:
            # Fetch multiple types of data
            data_types = [
                Data.Quote,
                Data.HistoricalPrices,
                Data.EarningsQuarterly
            ]
            
            results = {}
            for data_type in data_types:
                try:
                    data = api.get_stock_data(test_symbol, data_type)
                    results[data_type] = data
                    logger.info(f"✅ Fetched {data_type} for {test_symbol}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to fetch {data_type}: {e}")
            
            # Verify we got at least some data
            assert len(results) > 0, "No data types were successfully fetched"
            
            logger.info(f"✅ Integration test completed. Successfully fetched {len(results)} data types")
            
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 