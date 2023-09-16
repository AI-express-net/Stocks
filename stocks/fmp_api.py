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
from stocks.data_names import Data
from stocks.data_names import Market

logging.basicConfig(format='%(asctime)s [%(levelname)s] "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"', level=logging.INFO)

API_KEY = "97e4449ce1e26a08b7025a5c01492796"

API = "https://financialmodelingprep.com/api/v3/"
# Stock information APIs
DCF_URI = API + "historical-daily-discounted-cash-flow/{}&apikey=" + API_KEY
CASH_FLOW_QUARTERLY_URI = API + "cash-flow-statement/{}?period=quarter&limit={}&apikey=" + API_KEY
CASH_FLOW_YEARLY_URI = API + "cash-flow-statement/{}?period=year&limit={}&apikey=" + API_KEY
EARNINGS_QUARTERLY_URI = API + "income-statement/{}?period=quarter&limit={}&apikey=" + API_KEY
EARNINGS_YEARLY_URI = API + "income-statement/{}?period=year&limit={}&apikey=" + API_KEY
ENTERPRISE_VALUE_URI = API + "enterprise-values/{}?period=quarter&limit={}&apikey=" + API_KEY
BALANCE_SHEET_QUARTERLY_URI = API + "balance-sheet-statement/{}?period=quarter&limit={}&apikey=" + API_KEY
BALANCE_SHEET_YEARLY_URI = API + "balance-sheet-statement/{}?period=year&limit={}&apikey=" + API_KEY
HISTORICAL_PRICES_URI = API + "historical-price-full/{}?from={}&to={}&serietype=line&apikey=" + API_KEY
KEY_METRICS_URI = API + "key-metrics/{}?period=quarter&limit={}&apikey=" + API_KEY
STOCK_SPLITS_URI = API + "historical-price-full/stock_split/{}?apikey=" + API_KEY
QUOTE_URI = API + "quote/{}?apikey=" + API_KEY

# Market information APIs
STOCK_LIST_URI = API + "stock/list?apikey=" + API_KEY
DOW_STOCKS_URI = API + "dowjones_constituent?apikey=" + API_KEY
SP500_STOCKS_URI = API + "sp500_constituent?apikey=" + API_KEY
NASDAQ_STOCKS_URI = API + "nasdaq_constituent?apikey=" + API_KEY

QUARTER_LIMIT = 100
YEAR_LIMIT = 25

stock_data_uri_map = {
    Data.BalanceSheetQuarterly: BALANCE_SHEET_QUARTERLY_URI,
    Data.BalanceSheetYearly: BALANCE_SHEET_YEARLY_URI,
    Data.CashFlowQuarterly: CASH_FLOW_QUARTERLY_URI,
    Data.CashFlowYearly: CASH_FLOW_YEARLY_URI,
    Data.EarningsQuarterly: EARNINGS_QUARTERLY_URI,
    Data.EarningsYearly: EARNINGS_YEARLY_URI,
    Data.EnterpriseValue: ENTERPRISE_VALUE_URI,
    Data.HistoricalPrices: HISTORICAL_PRICES_URI,
    Data.KeyMetricsQuarterly: KEY_METRICS_URI,
    Data.Quote: QUOTE_URI,
    #    Data.StockSplits: STOCK_SPLITS_API,
}

market_data_uri_map = {
    Market.DowJonesStocks: DOW_STOCKS_URI,
    Market.SP500Stocks: SP500_STOCKS_URI,
    Market.NasdaqStocks: NASDAQ_STOCKS_URI,
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
        symbol_list = []
        for s in stock_list:
            symbol_list.append(s["symbol"])
        stocks = {"stocks": symbol_list}
        return stocks

    def get_dcf(self, stock_symbol):
        return fetch_from_api(DCF_URI.format(stock_symbol))


if __name__ == "__main__":
    test_symbol = "AAPL"
    api = FmpApi()
    json = api.get_stock_data(test_symbol, Data.Quote)
    logging.info(json)

