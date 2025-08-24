"""
Copyright 2021 - Mark Boon, All rights reserved.

Data provided by Financial Modeling Prep
"""
import json
import logging
import requests

from datetime import date
from dateutil.relativedelta import relativedelta

from stocks.api_error import ApiError
from stocks.api_config import API_KEY, API_BASE
from stocks.data_names import Data
from stocks.data_names import Market

logging.basicConfig(format='%(asctime)s [%(levelname)s] "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"', level=logging.INFO)

"""
This file defines the API end-points used to fetch data about a stock.
The various APIs are mapped to classes that represent the fields in the stock class in the stock_data_uri_map
"""

# Stock information APIs
DCF_API = f"{API_BASE}historical-daily-discounted-cash-flow/{{}}&apikey={API_KEY}"
CASH_FLOW_QUARTERLY_API = f"{API_BASE}cash-flow-statement/{{}}?period=quarter&limit={{}}&apikey={API_KEY}"
CASH_FLOW_YEARLY_API = f"{API_BASE}cash-flow-statement/{{}}?period=year&limit={{}}&apikey={API_KEY}"
EARNINGS_QUARTERLY_API = f"{API_BASE}income-statement/{{}}?period=quarter&limit={{}}&apikey={API_KEY}"
EARNINGS_YEARLY_API = f"{API_BASE}income-statement/{{}}?period=year&limit={{}}&apikey={API_KEY}"
ENTERPRISE_VALUE_API = f"{API_BASE}enterprise-values/{{}}?period=quarter&limit={{}}&apikey={API_KEY}"
BALANCE_SHEET_QUARTERLY_API = f"{API_BASE}balance-sheet-statement/{{}}?period=quarter&limit={{}}&apikey={API_KEY}"
BALANCE_SHEET_YEARLY_API = f"{API_BASE}balance-sheet-statement/{{}}?period=year&limit={{}}&apikey={API_KEY}"
HISTORICAL_PRICES_API = f"{API_BASE}historical-price-full/{{}}?from={{}}&to={{}}&serietype=line&apikey={API_KEY}"
KEY_METRICS_API = f"{API_BASE}key-metrics/{{}}?period=quarter&limit={{}}&apikey={API_KEY}"
STOCK_SPLITS_API = f"{API_BASE}historical-price-full/stock_split/{{}}?apikey={API_KEY}"
QUOTE_API = f"{API_BASE}quote/{{}}?apikey={API_KEY}"

# Market information APIs
STOCK_LIST_API = f"{API_BASE}stock/list?apikey={API_KEY}"
DOW_STOCKS_API = f"{API_BASE}dowjones_constituent?apikey={API_KEY}"
SP500_STOCKS_API = f"{API_BASE}sp500_constituent?apikey={API_KEY}"
NASDAQ_STOCKS_API = f"{API_BASE}nasdaq_constituent?apikey={API_KEY}"

# The maximum number of quarters or years to fetch
QUARTER_LIMIT = 100
YEAR_LIMIT = 25

# The mapping of the data names to the API end-points
stock_data_uri_map = {
    Data.BalanceSheetQuarterly: BALANCE_SHEET_QUARTERLY_API,
    Data.BalanceSheetYearly: BALANCE_SHEET_YEARLY_API,
    Data.CashFlowQuarterly: CASH_FLOW_QUARTERLY_API,
    Data.CashFlowYearly: CASH_FLOW_YEARLY_API,
    Data.EarningsQuarterly: EARNINGS_QUARTERLY_API,
    Data.EarningsYearly: EARNINGS_YEARLY_API,
    Data.EnterpriseValue: ENTERPRISE_VALUE_API,
    Data.HistoricalPrices: HISTORICAL_PRICES_API,
    Data.KeyMetricsQuarterly: KEY_METRICS_API,
    Data.Quote: QUOTE_API,
    #    Data.StockSplits: STOCK_SPLITS_API,
}

# The mapping of the market data names to the API end-points
market_data_uri_map = {
    Market.DowJonesStocks: DOW_STOCKS_API,
    Market.SP500Stocks: SP500_STOCKS_API,
    Market.NasdaqStocks: NASDAQ_STOCKS_API,
}

def fetch_from_api(uri):
    logging.info(uri)
    data = json.loads(requests.get(uri).text)
    # if len(data) > 10:
    #     logging.debug(uri + "\n" + pprint.pformat(data[0], indent=4))
    return data


class FmpApi:

    def get_api_names(self):
        return list(stock_data_uri_map.keys())

    def get_stock_data(self, stock_symbol, data_name):
        uri_format = stock_data_uri_map[data_name]
        try:
            if "quarter" in uri_format:
                uri = uri_format.format(stock_symbol, QUARTER_LIMIT)
            elif "year" in uri_format:
                uri = uri_format.format(stock_symbol, YEAR_LIMIT)
            elif "to=" in uri_format and "from=" in uri_format:
                to_date = date.today()
                to_date_str = to_date.strftime("%Y-%m-%d")
                from_date = to_date - relativedelta(years=YEAR_LIMIT)
                from_date_str = from_date.strftime("%Y-%m-%d")
                uri = uri_format.format(stock_symbol, from_date_str, to_date_str)
            else:
                uri = uri_format.format(stock_symbol)
            return fetch_from_api(uri)
        except Exception as exception:
            raise ApiError(uri_format, str(exception))

    def get_market_data(self, data_name):
        try:
            uri = market_data_uri_map[data_name]
            return fetch_from_api(uri)
        except Exception as exception:
            raise ApiError(uri, str(exception))

    def get_stock_list(self, data_name):
        stock_list = self.get_market_data(data_name)
        
        # Handle error responses from API
        if isinstance(stock_list, dict) and "Error Message" in stock_list:
            logging.warning(f"API Error: {stock_list['Error Message']}")
            return {"stocks": []}
        
        # Handle case where stock_list is not a list
        if not isinstance(stock_list, list):
            logging.error(f"Unexpected response format: {type(stock_list)}")
            return {"stocks": []}
        
        symbol_list = []
        for s in stock_list:
            if isinstance(s, dict) and "symbol" in s:
                symbol_list.append(s["symbol"])
            elif isinstance(s, str):
                # Handle case where API returns list of strings directly
                symbol_list.append(s)
            else:
                logging.warning(f"Skipping invalid stock item: {s}")
        
        stocks = {"stocks": symbol_list}
        return stocks

    def get_dcf(self, stock_symbol):
        return fetch_from_api(DCF_API.format(stock_symbol))


if __name__ == "__main__":
    test_symbol = "IBM"
    api = FmpApi()
    json = api.get_stock_data(test_symbol, Data.HistoricalPrices)
    logging.info(json)

