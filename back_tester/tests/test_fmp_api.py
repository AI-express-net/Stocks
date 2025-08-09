from stocks.data_names import Market
from stocks.factory import Factory


def test_stock_list():
    api = Factory.get_api_instance()
    stock_list = api.get_stock_list(Market.SP500Stocks)
    print(str(stock_list))
    print("")
