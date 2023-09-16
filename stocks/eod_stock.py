import pymongo
import requests
import json
import numpy as np
import matplotlib.dates as plt_dates
import matplotlib.pyplot as plt
import enum
import logging
from dateutil import parser as date_parser
from dateutil import relativedelta
from datetime import date
from datetime import datetime

logging.basicConfig(format='%(asctime)s [%(levelname)s] "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"', level=logging.INFO)

#API_KEY = "5fd52ea92b3786.45774921"
API_KEY = "97e4449ce1e26a08b7025a5c01492796"

#FUNDAMENTALS_URI = "https://eodhistoricaldata.com/api/fundamentals/{}?api_token=" + API_KEY
FUNDAMENTALS_URI = "https://financialmodelingprep.com/api/v3/fundamentals/{}?api_token=" + API_KEY
EOD_URI = "https://eodhistoricaldata.com/api/eod/{}?api_token=" + API_KEY +"&period=d&fmt=json&from=2005-09-30&to=2020-09-30"
SPLITS_URI = "https://eodhistoricaldata.com/api/splits/{}?api_token=" + API_KEY + "&fmt=json"

MIN_RATE = 1.03

def rate(percentage):
    return (100.0 + percentage) / 100.0

def parse_split(split_str):
    split_numbers = split_str.split('/')
    return float(split_numbers[0]), float(split_numbers[1])

def get_timestamp():
    return date.today().strftime("%Y-%m-%d:%H:%M:%S.%Z")

class Stock:
    class Section(enum.Enum):
        General = 1,
        Highlights = 2,
        Valuation = 3,
        SharesStats = 4,
        Technicals = 5,
        SplitsDividends = 6,
        AnalystRatings = 7,
        Holders = 8,
        ESGScores = 9,
        outstandingShares = 10,
        Earnings = 11,
        Financials = 12

    def __init__(self, name):
        self.name = name
        self.key = {"_id": name}
        self.fundamental_data = None
        self.historical_prices = None
        self.adjusted_prices = None
        self.splits = None
        self.db_client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.db_client.Stocks
        self.dirty = False
        self.valid = False

    def fetch_fundamentals_from_db(self):
        db = self.db_client.Stocks
        fundamentals = db.fundamentals
        self.fundamental_data = fundamentals.find_one(self.key)
        if self.fundamental_data is None:
            self.fetch_fundamentals_from_api()
            self.save_fundamentals_to_db()
        self.valid = self.is_valid()
        #print(self.data)

    # Only valid if we have a minimum of 3 years and 6 quarters of fundamental data.
    def is_valid(self):
        return len(self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["yearly"]) > 5 and \
               len(self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["quarterly"]) > 20

    def fetch_historical_prices_from_db(self):
        eod_prices = self.db.eod_prices
        self.historical_prices = eod_prices.find_one(self.key)
        if self.historical_prices is None:
            self.fetch_historical_prices_from_api()
            self.save_historical_prices_to_db()
        #print(self.data)

    def fetch_splits_from_db(self):
        splits = self.db.splits
        self.splits = splits.find_one(self.key)
        if self.splits is None:
            self.fetch_splits_from_api()
            self.save_splits_to_db()
        #print(self.data)

    def fetch_from_db(self):
        self.fetch_fundamentals_from_db()
        self.fetch_historical_prices_from_db()
        self.fetch_splits_from_db()

    def save_to_db(self):
        if self.dirty:
            logging.info("Saving {} to DB".format(self.name))
            self.save_fundamentals_to_db()
            self.save_historical_prices_to_db()
            self.save_splits_to_db()
            self.dirty = False

    def save_fundamentals_to_db(self):
        logging.info("Saving fundamentals for {} to DB".format(self.name))
        self.fundamental_data["_id"] = self.name
        self.fundamental_data["_last_save"] = get_timestamp()
        db = self.db_client.Stocks
        if db.fundamentals.find_one(self.key) is not None:
            db.fundamentals.delete_one(self.key)
        db.fundamentals.insert_one(self.fundamental_data)

    def save_historical_prices_to_db(self):
        self.historical_prices["_last_save"] = get_timestamp()
        db = self.db_client.stocks
        if db.eod_prices.find_one(self.key) is not None:
            db.eod_prices.delete_one(self.key)
        db.eod_prices.insert_one(self.historical_prices)

    def save_splits_to_db(self):
        self.splits["_last_save"] = get_timestamp()
        db = self.db_client.stocks
        if db.splits.find_one(self.key) is not None:
            db.splits.delete_one(self.key)
        db.splits.insert_one(self.splits)

    def fetch_fundamentals_from_api(self):
        logging.info("Fetching {} from API".format(self.name))
        self.fundamental_data = json.loads(requests.get(FUNDAMENTALS_URI.format(self.name)).text)
        self.dirty = True
        #print(self.data)

    def fetch_historical_prices_from_api(self):
        eod_prices = json.loads(requests.get(EOD_URI.format(self.name)).text)
        self.historical_prices = {"_id": self.name, "eod_prices": eod_prices}
        self.dirty = True
        #print(eod_prices)

    def fetch_splits_from_api(self):
        splits = json.loads(requests.get(SPLITS_URI.format(self.name)).text)
        self.splits = {"_id": self.name, "splits": splits}
        self.dirty = True
        #print(splits)

    def get_most_recent_quarter_date(self):
        return list(self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["quarterly"].keys())[0]
        # date_list = list(self.data[Stock.Section.outstandingShares.name]["quarterly"].values())
        # while len(date_list) > 0:
        #     date_str = date_list.pop(0)["dateFormatted"]
        #     d = dateutil.parser.parse(date_str)
        #     if d.date() <= today:
        #         break
        # return date_str

    def get_first_quarter_date(self):
        # earliest_stock_number_date = list(self.fundamental_data[Stock.Section.outstandingShares.name]["quarterly"].values()).pop()["dateFormatted"]
        earliest_stock_number_date = "1970-01-01"
        q_list = list(self.fundamental_data[Stock.Section.Financials.name]["Balance_Sheet"]["quarterly"].values())
        while len(q_list) > 4:
            q = q_list.pop()
            if q["date"] >= earliest_stock_number_date and q["commonStockSharesOutstanding"]:
                return q["date"]
        return earliest_stock_number_date

    def get_reported_dates(self):
        return list(self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["quarterly"].keys())
        # qs = list(self.fundamental_data[Stock.Section.outstandingShares.name]["quarterly"].values())
        # date_list = []
        # for q in qs:
        #     date_list.append(q["dateFormatted"])
        # return date_list

    def get_year_index(self, date_formatted):
        start_index = 0
        all_cf_quarters = self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["yearly"]
        date_list = all_cf_quarters.keys()
        for d in date_list:
            if date_formatted >= d:
                return start_index
            start_index += 1

    def get_quarter_index(self, date_formatted):
        start_index = 0
        all_cf_quarters = self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["quarterly"]
        date_list = all_cf_quarters.keys()
        for d in date_list:
            if date_formatted >= d:
                return start_index
            start_index += 1

    def adjust_prices(self):
        if self.historical_prices is None:
            self.fetch_historical_prices_from_db()
        if self.splits is None:
            self.fetch_splits_from_db()

    def get_free_cash_flow(self, date_formatted):
        end_date = date_parser.parse(date_formatted)
        running_index = 0
        all_cf_quarters = self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["quarterly"]
        # all_income_quarters = self.fundamental_data[Stock.Section.Financials.name]["Income_Statement"]["quarterly"]
        # income_quarters = []
        fcf = 0
        cf_q_list = list(all_cf_quarters.values())
        # income_q_list = list(all_income_quarters.values())

        date_list = all_cf_quarters.keys()
        for d in date_list:
            if d == date_formatted:
                break
            running_index += 1

        while len(cf_q_list) > running_index and \
                relativedelta.relativedelta(end_date, date_parser.parse(cf_q_list[running_index]["date"])).years < 1:
            fcf += float(cf_q_list[running_index]["totalCashFromOperatingActivities"] or 0.0)
            fcf -= float(cf_q_list[running_index]["capitalExpenditures"] or 0.0)
            #            fcf += float(income_quarters[running_index]["interestExpense"])
            running_index += 1

        # cfq = cf_q_list[start_index]
        # if len(cf_q_list) >= start_index + 4:
        #     for i in range(4):
        #         cash_flow_quarters.append(cf_q_list[start_index + i])
        #         #income_quarters.append(income_q_list[start_index + i])
        #         fcf += float(cash_flow_quarters[i]["totalCashFromOperatingActivities"] or 0.0)
        #         fcf -= float(cash_flow_quarters[i]["capitalExpenditures"] or 0.0)
        #     #            fcf += float(income_quarters[i]["interestExpense"])
        return max(fcf, 1)

    def get_shares_outstanding(self, date_formatted):
        shares = int(self.fundamental_data[Stock.Section.SharesStats.name]["SharesOutstanding"])
        result = self.fundamental_data[Stock.Section.Financials.name]["Balance_Sheet"]["quarterly"][date_formatted]["commonStockSharesOutstanding"]
        if result is None or float(result) < shares/100:
            recent = 0.0
            found = False
            counter = 1
            for data in list(self.fundamental_data[Stock.Section.Financials.name]["Balance_Sheet"]["quarterly"].values()):
                if data["date"] == date_formatted:
                    found = True
                    counter = 1
                if found:
                    if data["commonStockSharesOutstanding"] is not None:
                        current = float(data["commonStockSharesOutstanding"])
                        if recent == 0.0:
                            return current
                        return recent + (recent-current) / counter
                elif data["commonStockSharesOutstanding"] is not None:
                    recent = float(data["commonStockSharesOutstanding"])
                counter += 1
            return 0.0
        return float(result)

    def has_reported_quarter(self, date_formatted):
        all_cf_quarters = self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["quarterly"]
        return date_formatted in all_cf_quarters.keys()

    # Set the intrinsic value for a quarter.
    # Return True if the value is different from what was stored already, False otherwise.
    def set_cached_iv(self, iv, date_formatted):
        all_cf_quarters = self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["quarterly"]
        cf = all_cf_quarters[date_formatted]
        cached_iv = None
        if "dcfIntrinsicValue" in cf:
            cached_iv = cf["dcfIntrinsicValue"]
        if iv != cached_iv:
            cf["dcfIntrinsicValue"] = iv
            self.dirty = True

    def get_cached_iv(self, date_formatted):
        all_cf_quarters = self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["quarterly"]
        cf = all_cf_quarters[date_formatted]
        cached_iv = None
        if "dcfIntrinsicValue" in cf:
            cached_iv = cf["dcfIntrinsicValue"]
            #print("Used cashed IV {} for {}".format(cached_iv, date_formatted))
        return cached_iv

    def get_dcf(self, date_formatted, discount_rate=10, growth_5_years=3, growth_10_years=3, growth_forever=3, dilution=0):
        shares_outstanding = self.get_shares_outstanding(date_formatted)
        fcf = self.get_free_cash_flow(date_formatted)
        future_cash_flows = [fcf / shares_outstanding]
        pv_future_cash_flow = 0

        for i in range(1, 6):
            cf = future_cash_flows[i - 1] * rate(growth_5_years)
            future_cash_flows.append(cf)
            pv_future_cash_flow += cf / pow(rate(discount_rate), i)
        for i in range(6, 11):
            cf = future_cash_flows[i - 1] * rate(growth_10_years)
            future_cash_flows.append(cf)
            pv_future_cash_flow += cf / pow(rate(discount_rate), i)

        capitalization_rate = 0.09
        terminal_value = future_cash_flows[10] * rate(growth_forever) * (1.0 / capitalization_rate)
        discounted_value = terminal_value / pow(rate(discount_rate), 10)
        intrinsic_value = pv_future_cash_flow + discounted_value
        return intrinsic_value / pow(rate(dilution), 10)

    def get_extrapolated_dcf(self, date_formatted, growth_rates, discount_rate=10, dilution=0):
        if date_formatted == '2006-09-30':
            print("Calculate DCF for {} on {}".format(self.name, date_formatted))
        shares_outstanding = self.get_shares_outstanding(date_formatted)
        if shares_outstanding == 0:
            return 0.0
        fcf = self.get_free_cash_flow(date_formatted)
        future_cash_flows = [fcf / shares_outstanding]
        pv_future_cash_flow = 0

        for i in range(1, 11):
            cf = future_cash_flows[i - 1] * growth_rates[i - 1]
            future_cash_flows.append(cf)
            pv_future_cash_flow += cf / pow(rate(discount_rate), i)

        capitalization_rate = 0.09
        terminal_value = future_cash_flows[10] * growth_rates[10] * (1.0 / capitalization_rate)
        discounted_value = terminal_value / pow(rate(discount_rate), 10)
        intrinsic_value = pv_future_cash_flow + discounted_value
        return max(0.0, intrinsic_value / pow(rate(dilution), 10))

    def get_rate_changes(self, date_formatted):
        end_index = self.get_year_index(date_formatted)
        all_cf_years = self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["yearly"]
        cf_y_list = list(all_cf_years.values())[end_index:]
        years = []
        cf_list = []
        max_value = 0
        min_value = 10000
        prev_outstanding = 1
        for i in range(len(cf_y_list) + 20):
            years.append(i)

        for i in range(len(cf_y_list)):
            cf = float(cf_y_list[i]["totalCashFromOperatingActivities"] or 0.0)
            capex = float(cf_y_list[i]["capitalExpenditures"] or 0.0)
            shares_outstanding = max(int(self.get_shares_outstanding(cf_y_list[i]["date"] or prev_outstanding)), prev_outstanding)
            if shares_outstanding != 0:
                fcf = max(cf - capex, 1) / shares_outstanding   # Don't let FCF go negative
                prev_outstanding = shares_outstanding
                max_value = max(fcf, max_value)
                min_value = min(fcf, min_value)
                cf_list.append(fcf)

        cf_list.reverse()

        try:
            z = np.polyfit(years[0:len(cf_list)], cf_list, 2)
            f = np.poly1d(z)
        except:
            z = np.polyfit(years[0:len(cf_list)], cf_list, 1)
            f = np.poly1d(z)

        for x1 in np.linspace(len(cf_list), len(cf_list) + 10, 10):
            cf_list.append(f(x1))

        terminal = cf_list[-1] * rate(3)
        for i in range(10):
            cf_list.append(terminal)

        years = years[-len(cf_list):]
        z1 = np.polyfit(years, cf_list, 3)
        f1 = np.poly1d(z1)

        extrapolated = []
        for x1 in np.linspace(0, len(cf_list), len(cf_list)):
            f1x1 = f1(x1)
            extrapolated.append(f1x1)
            max_value = max(max_value, f1x1)

        rate_changes = [0.0, extrapolated[1] / extrapolated[0]]
        for i in range(2, len(cf_list) - 1):
            rate_changes.append(max(extrapolated[i] / extrapolated[i - 1], MIN_RATE))

        return rate_changes[-20:]

    def plot_cash_flow_trend(self, date_formatted):
        end_index = self.get_year_index(date_formatted)
        all_cf_years = self.fundamental_data[Stock.Section.Financials.name]["Cash_Flow"]["yearly"]
        cf_y_list = list(all_cf_years.values())[end_index:]
        years = []
        cf_list = []
        index = 0
        max_value = 0
        min_value = 10000000
        for i in range(len(cf_y_list) + 20):
            years.append(i)

        for i in range(len(cf_y_list)):
            cf = float(cf_y_list[i]["totalCashFromOperatingActivities"] or 0.0)
            capex = float(cf_y_list[i]["capitalExpenditures"] or 0.0)
            fcf = (cf - capex) / int(self.get_shares_outstanding(cf_y_list[i]["date"] or 1))
            max_value = max(fcf, max_value)
            min_value = min(fcf, min_value)
            cf_list.append(fcf)

        cf_list.reverse()

        for x, y in zip(years, cf_list):
            plt.plot(x, y, 'ro')

        z = np.polyfit(years[0:len(cf_list)], cf_list, 2)
        f = np.poly1d(z)

        for x1 in np.linspace(len(cf_list), len(cf_list) + 10, 10):
            cf_list.append(f(x1))

        terminal = cf_list[len(cf_list)-1] * rate(3)
        for i in range(10):
            cf_list.append(terminal)
            index = i

        z1 = np.polyfit(years, cf_list, 3)
        f1 = np.poly1d(z1)

        extrapolated = []
        rate_changes = [0]
        for x1 in np.linspace(0, len(cf_list), len(cf_list)):
            rate_changes.append(cf_list[index] / cf_list[index-1])
            extrapolated.append(f1(x1))
            f1x1 = f1(x1)
            max_value = max(max_value, f1x1)
            plt.plot(x1, f1x1, 'g.')

        rate_changes = [0.0, extrapolated[1] / extrapolated[0]]
        for i in range(2, len(cf_list) - 1):
            rate_changes.append(extrapolated[i] / extrapolated[i-1])

        plt.axis([0, len(cf_list) - 1, min_value, max_value*1.1])

        return rate_changes[-20:], cf_list

    def get_historical_prices(self):
        if self.historical_prices is None:
            self.fetch_historical_prices_from_db()
        return self.historical_prices["eod_prices"]

    def get_adjusted_prices(self):
        adjusted_prices = []
        eod_prices = self.get_historical_prices()
        eod_prices.reverse()
        split_list = self.get_splits()
        split_list.reverse()
        share_multiplier = 1.0
        split_index = 0
        len_split_list = len(split_list)
        split_date = None
        if len_split_list > 0:
            split_date = date_parser.parse(split_list[0]["date"])

        for price in eod_prices:
            day = date_parser.parse(price["date"])

            # Adjust share_multiplier for split
            if split_date is not None and day < split_date:
                split_str = split_list[split_index]["split"]
                multiplier, divider = parse_split(split_str)
                share_multiplier *= multiplier
                share_multiplier /= divider
                split_index += 1
                if len_split_list > split_index:
                    split_date = date_parser.parse(split_list[split_index]["date"])
                else:
                    split_date = None

            adjusted_price = float(price["close"]) / share_multiplier
            adjusted_price = {"date": day, "close": adjusted_price}
            adjusted_prices.append(adjusted_price)
            adjusted_prices.reverse()
        return adjusted_prices

    def plot_prices(self):
        prices = self.get_adjusted_prices()
        for price in prices:
            plt.plot(price["date"], price["close"], 'r.')
        plt.show()

    def get_splits(self):
        if self.splits is None:
            self.fetch_splits_from_db()
        return self.splits["splits"]

    def plot_dcf_vs_price(self, not_before_str="1970-01-01"):
        print("Start plot for " + self.name)
        not_before = date_parser.parse(not_before_str)
        eod_prices = self.get_historical_prices()
        split_list = self.get_splits()
        multiplier = 1
        divider = 1
        share_multiplier = 1.0
        split_index = 0
        split_date = None
        len_split_list = len(split_list)

        if len_split_list > 0:
            split_date = date_parser.parse(split_list[split_index]["date"])
            split_str = split_list[split_index]["split"]
            multiplier, divider = parse_split(split_str)
        start_date = date_parser.parse(eod_prices[0]["date"])
        while split_date is not None and split_date < start_date:
            split_index += 1
            if len_split_list > split_index:
                split_date = date_parser.parse(split_list[split_index]["date"])
                split_str = split_list[split_index]["split"]
                multiplier, divider = parse_split(split_str)
            else:
                split_date = None

        for p in eod_prices:
            d = date_parser.parse(p["date"])
            if d >= not_before:
                if split_date is not None and d >= split_date:
                    share_multiplier *= multiplier
                    share_multiplier /= divider
                    split_index += 1
                    if len_split_list > split_index:
                        split_date = date_parser.parse(split_list[split_index]["date"])
                        split_str = split_list[split_index]["split"]
                        multiplier, divider = parse_split(split_str)
                    else:
                        split_date = None
                price = float(p["close"]) * share_multiplier
                plt.plot(d, price, 'g.')

        dates = self.get_reported_dates()
        now = datetime.now()
        for d in dates:
            date_value = date_parser.parse(d)
            if now >= date_value >= not_before:
                try:
                    rates = self.get_rate_changes(d)
                    iv = self.get_extrapolated_dcf(d, rates)
                    plt.plot(date_value, iv * share_multiplier, 'bo')
                except:
                    rates = [MIN_RATE] * 20
                    iv = self.get_extrapolated_dcf(d, rates)
                    plt.plot(date_value, iv * share_multiplier, 'ro')

        plt.show()
        print("Finished plot for " + self.name)

    def plot_iv_vs_price(self, not_before_str=None):
        if not not_before_str:
            not_before_str = self.get_first_quarter_date()
            # shares = self.get_shares_outstanding(not_before_str)
            # print(shares)
        print("Start plot for {} starting from {}".format(self.name, not_before_str))
        not_before = date_parser.parse(not_before_str)
        # eod_prices = self.get_adjusted_prices()
        eod_prices = self.historical_prices["eod_prices"]

        plt.axes().xaxis.set_major_formatter(plt_dates.DateFormatter("%y-%b"))

        for p in eod_prices:
            price = float(p["close"])
            day = date_parser.parse(p["date"])
            plt.plot(day, price, 'g.')

        dates = self.get_reported_dates()
        now = datetime.now()
        for d in dates:
            date_value = date_parser.parse(d)
            if now >= date_value >= not_before and self.has_reported_quarter(d):
                try:
                    iv = None
                    # iv = self.get_cached_iv(d)
                    if iv is None or iv < 0:
                        rates = self.get_rate_changes(d)
                        iv = self.get_extrapolated_dcf(d, rates)
                        # print(iv)
                        plt.plot(date_value, iv, 'bo')
                except:
                    rates = [MIN_RATE] * 20
                    iv = self.get_extrapolated_dcf(d, rates)
                    plt.plot(date_value, iv, 'ro')
                if iv is not None:
                    self.set_cached_iv(iv, d)

        plt.title(self.name)
        plt.show()
        self.save_to_db()
        print("Finished plot for " + self.name)


if __name__ == "__main__":
    test_stock = Stock("AAPL.US")
    test_stock.fetch_from_db()
    # test_stock.save_to_db()
#    test_stock.fetch_splits_from_api()

    # test_stock.fetch_fundamentals_from_db()
    # test_stock.get_reported_dates()
    # date = test_stock.get_most_recent_quarter_date()
    #
    mrd = test_stock.get_most_recent_quarter_date()
    print("FCF = " + str(test_stock.get_free_cash_flow(mrd)))
    print("DCF = " + str(test_stock.get_dcf(mrd, 10, 3, 3, 3)))
    change_rates = test_stock.get_rate_changes(mrd)
    print("Extrapolated DCF = " + str(test_stock.get_extrapolated_dcf(mrd, change_rates)))
    #
    # print("First q: " + apple.get_first_quarter_date())
    # #test_stock.fetch_prices_from_api()
    # #test_stock.save_to_db()
    # start_date = "1970-01-01"
    test_stock.plot_iv_vs_price()
    # test_stock.plot_prices()
    # test_stock.save_to_db()
