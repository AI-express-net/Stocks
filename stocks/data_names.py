class Data:
    # Stock field names
    BalanceSheetQuarterly = "balance_sheet_quarterly"
    BalanceSheetYearly = "balance_sheet_yearly"
    CashFlowQuarterly = "cash_flow_quarterly"
    CashFlowYearly = "cash_flow_yearly"
    EarningsQuarterly = "earnings_quarterly"
    EarningsYearly = "earnings_yearly"
    EarningsYearly = "earnings_yearly"
    EnterpriseValue = "enterprise_value"
    HistoricalPrices = "historical_prices"
    Mda50Prices = "mda50_prices"
    KeyMetricsQuarterly = "key_metrics_quarterly"
    Quote = "quote"
#    StockSplits = "stock_splits"

class Market:
    # General market, exchange and index names
    USStocks = "us_stocks"
    DowJonesStocks = "dow_jones"
    SP500Stocks = "sp500"
    NasdaqStocks = "nasdaq100"

name_list = [getattr(Data, x) for x in dir(Data) if not x.startswith("__")]


def get_name_list():
    return name_list
