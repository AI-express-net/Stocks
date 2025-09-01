import enum
import logging

from datetime import date, datetime
from dateutil import parser as date_parser
from dateutil import relativedelta

from stocks.balance_sheet_entity import BalanceSheetQuarterlyEntity
from stocks.balance_sheet_entity import BalanceSheetYearlyEntity
from stocks.cash_flow_entity import CashFlowQuarterlyEntity
from stocks.cash_flow_entity import CashFlowYearlyEntity
from stocks.earnings_entity import EarningsQuarterlyEntity
from stocks.earnings_entity import EarningsYearlyEntity
from stocks.ev_value_entity import EnterpriseValueEntity
from stocks.historical_prices_entity import HistoricalPricesEntity
from stocks.historical_dividends_entity import HistoricalDividendsEntity
from stocks.key_metrics_entity import KeyMetricsQuarterlyEntity
from stocks.mda50_prices_entity import Mda50PricesEntity
from stocks.quote_entity import QuoteEntity

from stocks.data_names import Data
from stocks.data_names import get_name_list
from stocks.factory import Factory
from stocks.financial_data import AnalysisModel

"""
This file defines the Stock class, which is used to fetch and store the data about a stock.
"""

logging.basicConfig(format='%(asctime)s [%(levelname)s] "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"', level=logging.INFO)

MIN_RATE = 1.03


def rate(percentage):
    return (100.0 + percentage) / 100.0


def parse_split(split_str):
    split_numbers = split_str.split('/')
    return float(split_numbers[0]), float(split_numbers[1])


def get_timestamp():
    return date.today().strftime("%Y-%m-%d:%H:%M:%S.%Z")

def is_data_stale(modification_date):
    """
    Check if the data is more than a day old.
    Returns True if data should be refreshed, False otherwise.
    """
    if modification_date is None:
        return True

    try:
        # Parse the modification date
        if isinstance(modification_date, str):
            mod_date = datetime.strptime(modification_date, "%Y-%m-%d").date()
        else:
            mod_date = modification_date

        # Check if it's more than a month old
        today = date.today()
        return (today - mod_date).days > 30
    except (ValueError, TypeError):
        # If we can't parse the date, consider it stale
        return True


class StockStatus(enum.Enum):
    Uninitialized = 1,
    Fetched = 2,
    Populated = 3,
    # Errors
    FetchError = 10,
    PopulatingError = 11,


class Stock:

    def __init__(self, name):
        self.name = name
        self.balance_sheet_quarterly = BalanceSheetQuarterlyEntity(name)
        self.balance_sheet_yearly = BalanceSheetYearlyEntity(name)
        self.earnings_quarterly = EarningsQuarterlyEntity(name)
        self.earnings_yearly = EarningsYearlyEntity(name)
        self.enterprise_value = EnterpriseValueEntity(name)
        self.cash_flow_quarterly = CashFlowQuarterlyEntity(name)
        self.cash_flow_yearly = CashFlowYearlyEntity(name)
        self.historical_prices = HistoricalPricesEntity(name)
        self.historical_dividends = HistoricalDividendsEntity(name)
        self.mda50_prices = Mda50PricesEntity(name)
        self.key_metrics_quarterly = KeyMetricsQuarterlyEntity(name)
        self.quote = QuoteEntity(name)
#        self.stock_splits = StockSplitsEntity(name)
        self.dirty = False
        self.status = StockStatus.Uninitialized
        self.api = Factory.get_api_instance()
        self.dao = Factory.get_dao_instance()
        self.analysis_models = {}

    def fetch_stock_data(self, field_name):
        field_value = getattr(self, field_name)
        saved_value = self.dao.find(field_value.get_id())
        
        # Check if we need to fetch from API
        should_fetch_from_api = (
            saved_value is None or 
            saved_value["data"] is None or 
            len(saved_value["data"]) == 0 or
            is_data_stale(saved_value.get("_modification_date"))
        )
        
        if should_fetch_from_api:
            # Fetch data from API
            if field_name in self.api.get_api_names():
                api_data = self.api.get_stock_data(self.name, field_name)
            else:
                api_data = self.generate_data(field_name)
            
            if api_data is None or len(api_data) == 0:
                raise Exception("No API {} data for stock {}".format(field_name, self.name))
            
            # Set the data
            field_value.set_api_data(api_data)
            
            # Set creation and modification dates
            today = date.today().strftime("%Y-%m-%d")
            if field_value.get_creation_date() is None:
                field_value.set_creation_date(today)
            field_value.set_modification_date(today)
            
            # Save to database
            self.dao.save(field_value)
        else:
            # Use existing data from database
            field_value.set_db_data(saved_value)


    def generate_data(self, field_name):
        if field_name == Data.Mda50Prices:
            return self.generate_mda50_prices()
        raise NameError("No generator for field name {}".format(field_name))

    def fetch_from_api(self):
        for field_name in get_name_list():
            field_value = getattr(self, field_name)
            if field_name in self.api.get_api_names():
                field_value.set_api_data(self.api.get_stock_data(self.name, field_name))
        self.dirty = True

    def fetch_from_db(self):
        try:
            for field_name in get_name_list():
                self.fetch_stock_data(field_name)
                self.status = StockStatus.Fetched
            min_quarters_required = 16
            if len(self.balance_sheet_quarterly["data"]) < min_quarters_required or \
                    len(self.cash_flow_quarterly["data"]) < min_quarters_required or \
                    len(self.earnings_quarterly["data"]) < min_quarters_required:
                self.status = StockStatus.InsufficientData
                logging.warning("Insufficient financial data for {}".format(self.name))
                return
        except Exception as exception:
            logging.warning("Unable to fetch data for {}. Reason: {}".format(self.name, str(exception)))
            self.status = StockStatus.FetchError
            return
#        self.populate_analysis_models()

    def save_to_db(self):
        if self.dirty:
            logging.info("Saving {} to DB".format(self.name))
            for field_name in get_name_list():
                field_value = getattr(self, field_name)
                self.dao.save(field_value)
            self.dirty = False

    def generate_mda50_prices(self):
        if len(self.historical_prices["data"]["historical"]) == 0:
            self.fetch_stock_data(Data.HistoricalPrices)
        if len(self.balance_sheet_quarterly["data"]) == 0:
            self.fetch_stock_data(Data.BalanceSheetQuarterly)
        n = 50
        historical_list = self.historical_prices["data"]["historical"]
        if len(historical_list) < 50:
            raise Exception("Insufficient historical data")
        historical_list.reverse()
        mda50_list = []
        date_list = []
        moving_average = 0
        self.get_shares_outstanding()
        self.balance_sheet_quarterly["data"]

        for i in range(n):
            moving_average += historical_list[i]["close"]
            mda50_list.append(moving_average / (i + 1))
            date_list.append(historical_list[i]["date"])

        for i in range(n, len(historical_list)):
            moving_average -= historical_list[i-n]["close"]
            moving_average += historical_list[i]["close"]
            mda50_list.append(moving_average / n)
            date_list.append(historical_list[i]["date"])

        mda50_list.reverse()
        date_list.reverse()
        historical_list.reverse()
        self.mda50_prices["data"]["prices"] = mda50_list
        self.mda50_prices["data"]["dates"] = date_list
        return self.mda50_prices["data"]

    def populate_analysis_models(self, populate_fd=False):
        try:
            nr_shares = self.get_shares_outstanding()
            key_metrics = self.key_metrics_quarterly["data"]
            for km in key_metrics:
                model = AnalysisModel(self.name, km["date"], nr_shares)
                model.populate_key_metrics(km, nr_shares)
                self.analysis_models[model.date] = model

            length = len(key_metrics)
            if populate_fd:
                balance_sheets = self.balance_sheet_quarterly["data"]
                for bs in balance_sheets:
                    model = self.analysis_models[self.get_closest_date(bs["date"])]
                    model.populate_financial_data(bs)

                cash_flows = self.cash_flow_quarterly["data"]
                for cf in cash_flows:
                    model = self.analysis_models[self.get_closest_date(cf["date"])]
                    model.populate_financial_data(cf)

                earnings = self.earnings_quarterly["data"]
                for e in earnings:
                    model = self.analysis_models[self.get_closest_date(e["date"])]
                    model.populate_financial_data(e)
                length = min(length, len(earnings), len(balance_sheets), len(cash_flows))

            # Now same for 1y, 2y and 3y past data
            start = 0
            start_1y = 0
            start_3y = 0
            start_5y = 0
            for i in range(length):
                try:
                    current_date = key_metrics[i]["date"]
                    current_model = self.analysis_models[current_date]
                    mda50, start = self.get_mda50_price(current_date, start)
                    current_model.mda50 = round(mda50, 3)

                    for y in range(1, len(current_model.key_metrics)):
                        if i < length - 4 * y:
                            y_date = key_metrics[i + 4 * y]["date"]
                            y_model = self.analysis_models[y_date]
                            current_model.populate_relative_metrics(current_model.key_metrics[0],
                                                                 y_model.key_metrics[0],
                                                                 current_model.key_metrics[y])
                    if populate_fd:
                        for y in range(1, len(current_model.financial_data)):
                            if i < length - 4*y:
                                y_date = key_metrics[i+4*y]["date"]
                                y_model = self.analysis_models[y_date]
                                current_model.populate_relative_data(current_model.financial_data[0],
                                                                     y_model.financial_data[0],
                                                                     current_model.financial_data[y])
                    # Populate future 50-day average market-values 1y, 3y and 5y out
                    if i >= 4:
                        y1_date = key_metrics[i-4]["date"]
                        mda50, start_1y = self.get_best_mda50_price(current_date, y1_date, start_1y)
                        current_model.mda50_1y = round(mda50, 3)
                    if i >= 12:
                        y3_date = key_metrics[i-12]["date"]
                        mda50, start_3y = self.get_best_mda50_price(current_date, y3_date, start_3y)
                        current_model.mda50_3y = round(mda50, 3)
                    if i >= 20:
                        y5_date = key_metrics[i-20]["date"]
                        mda50, start_5y = self.get_best_mda50_price(current_date, y5_date, start_5y)
                        current_model.mda50_5y = round(mda50, 3)
                except Exception as exception:
                    if i > 24:
                        logging.info("Stop populating analytical data beyond {}".format(key_metrics[i]["date"]))
                        break
                    else:
                        logging.info("Insufficient data. Reason: {}".format(str(exception)))
                        raise exception
            self.status = StockStatus.Populated
        except Exception as exception:
            self.status = StockStatus.PopulatingError
            logging.warning("Couldn't populate analytical data for {}. Reason: {}".format(self.name, str(exception)))

    def get_closest_date(self, date_str):
        if date_str in self.analysis_models.keys():
            return date_str

        min_str = None
        date_value = date_parser.parse(date_str)
        future_date_value = date_parser.parse("2100-01-01")
        min_diff = future_date_value - date_value
        for key in self.analysis_models.keys():
            dv = date_parser.parse(key)
            if abs(date_value - dv) < min_diff:
                min_diff = abs(date_value - dv)
                min_str = key
        return min_str

    def get_mda50_price(self, date_formatted, start=0):
        nr_items = len(self.mda50_prices["data"]["dates"])
        if start+50 < nr_items and self.mda50_prices["data"]["dates"][start+50] > date_formatted:
            start += 50
        for i in range(start, nr_items):
            if self.mda50_prices["data"]["dates"][i] <= date_formatted:
                return self.mda50_prices["data"]["prices"][i], i

        first_date = date_parser.parse(self.mda50_prices["data"]["dates"][0])
        last_date = date_parser.parse(self.mda50_prices["data"]["dates"][-1])
        if relativedelta.relativedelta(last_date, first_date).years < 3:
            raise Exception("Couldn't find mda50 for date {} for stock {}".format(date_formatted, self.name))
        return self.mda50_prices["data"]["prices"][nr_items-1], nr_items-1


    def get_best_mda50_price(self, start_date, future_date, start=0):
        nr_items = len(self.mda50_prices["data"]["dates"])
        best = 0
        new_start = 0
        if start + 50 < nr_items and self.mda50_prices["data"]["dates"][start + 50] > future_date:
            start += 50
        for i in range(start, nr_items):
            if self.mda50_prices["data"]["dates"][i] <= future_date:
                best = self.mda50_prices["data"]["prices"][i]
                new_start = i
                break

        j = new_start
        while j < nr_items and self.mda50_prices["data"]["dates"][j] > start_date:
            best = max(best, self.mda50_prices["data"]["prices"][j])
            j += 50

        return best, new_start

    def get_shares_outstanding(self):
        if len(self.quote["data"]) == 0:
            evs = self.enterprise_value["data"]
            if len(evs) > 0:
                return evs[0]["numberOfShares"]
            logging.error("No outstanding shares found in quote for {}".format(self.name))
        return self.quote["data"][0]["sharesOutstanding"]


if __name__ == "__main__":
    test_stock = Stock("INTC")
    test_stock.fetch_from_db()
    print(test_stock.status)
    # test_stock.save_to_db()
#    test_stock.fetch_splits_from_api()

    # test_stock.fetch_fundamentals_from_db()
    # test_stock.get_reported_dates()
    # date = test_stock.get_most_recent_quarter_date()
    #
    # mrd = test_stock.get_most_recent_quarter_date()
    # print("FCF = " + str(test_stock.get_free_cash_flow(mrd)))
    # print("DCF = " + str(test_stock.get_dcf(mrd, 10, 3, 3, 3)))
    # change_rates = test_stock.get_rate_changes(mrd)
    # print("Extrapolated DCF = " + str(test_stock.get_extrapolated_dcf(mrd, change_rates)))
    #
    # print("First q: " + apple.get_first_quarter_date())
    # #test_stock.fetch_prices_from_api()
    # #test_stock.save_to_db()
    # start_date = "1970-01-01"
    # test_stock.plot_iv_vs_price()
    # test_stock.plot_prices()
    # test_stock.save_to_db()
