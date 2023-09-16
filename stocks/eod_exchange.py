import pymongo
import requests
import json
import logging
from stocks.eod_stock import Stock

logging.basicConfig(format='%(asctime)s [%(levelname)s] "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"', level=logging.INFO)

API_KEY = "5fd52ea92b3786.45774921"

EXCHANGES_API = "https://eodhistoricaldata.com/api/exchanges-list/?fmt=json&api_token=" + API_KEY
TICKERS_API = "https://eodhistoricaldata.com/api/exchange-symbol-list/{}?fmt=json&api_token=" + API_KEY

EXCHANGES = {"NYSE ARCA", "NYSE MKT", "NYSE", "AMEX", "BATS", "NASDAQ"}


def get_exchanges():
    uri = EXCHANGES_API
    json_str = requests.get(uri).text
    exchanges = json.loads(json_str)
    for i in range(100):
        logging.info(str(list(exchanges)[i]))
    print(exchanges)


class Exchange:

    def __init__(self, name):
        self.name = name
        self.symbols = None
        self.db_client = pymongo.MongoClient("mongodb://localhost:27017/")

    def fetch_stock_symbols_from_api(self):
        logging.info("Fetching stock symbols from {}".format(self.name))
        uri = TICKERS_API.format(self.name)
        symbols = json.loads(requests.get(uri).text)
        stock_symbols = []
        for s in symbols:
            if s["Exchange"] in EXCHANGES and s["Type"] == "Common Stock":
                stock_symbols.append(s["Code"])
        for i in range(100):
            logging.info(str(list(stock_symbols)[i]))
        print(len(stock_symbols))

        stocks = {"stocks": stock_symbols}
        with open(self.name + "_stocks.json", 'w') as outfile:
            json.dump(stocks, outfile)

        return stock_symbols

    def delete_all_stocks(self):
        logging.info("Deleting all stock symbols of {} from DB".format(self.name))
        db = self.db_client.stocks
        fundamentals = db.fundamentals
        fundamentals.delete_many({})
        eod_prices = db.eod_prices
        eod_prices.delete_many({})
        splits = db.splits
        splits.delete_many({})

    def fetch_and_save_all(self):
        symbols = self.fetch_stock_symbols_from_api()
        for symbol in symbols:
            code = symbol["Code"] + "." + self.name
            stock = Stock(code)
            stock.fetch_from_db()
            print("Fetched {}".format(code))
            if stock.valid:
                stock.plot_iv_vs_price()


if __name__ == "__main__":
    #get_exchanges()
    exchange = Exchange('US')
    stocks = exchange.fetch_stock_symbols_from_api()
    #exchange.fetch_and_save_all()
    #exchange.delete_all_stocks()