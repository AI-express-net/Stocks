import json
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import threading
import unittest

from dateutil import parser as date_parser
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from stocks.data_names import Data
from stocks.data_names import Market
from stocks.factory import Factory
from stocks.fmp_stock import Stock
from stocks.fmp_stock import StockStatus
from stocks.financial_data import KEY_METRICS_FIELD_LIST

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

threadLock = threading.Lock()
threads = []
thread_count = multiprocessing.cpu_count() * 2


class PopulatingThread (threading.Thread):

    def __init__(self, stock_list, start_index, end_index, populated_stocks):
        threading.Thread.__init__(self)
        self.stock_list = stock_list
        self.start_index = start_index
        self.end_index = end_index
        self.populated_stocks = populated_stocks

    def run(self):
        logging.info("Start populating {} to {}.\n"
                     "=============================".format(self.start_index+1, self.end_index))
        for i in range(self.start_index, self.end_index):
            stock = self.stock_list[i]
            logging.info("Populating {}".format(stock.name))
            stock.populate_analysis_models()
            if stock.status == StockStatus.Populated:
                threadLock.acquire()
                self.populated_stocks.append(stock)
                threadLock.release()
            logging.info("Finished populating {}".format(stock.name))
        logging.info("Finished populating {} to {}.\n"
                     "=============================".format(self.start_index+1, self.end_index))


class StockTestCase(unittest.TestCase):
    def test_balance_sheet(self):
        stock = Stock("AAPL")
        balance_sheet_quarterly = stock.api.get_stock_data(stock.name, Data.BalanceSheetQuarterly)
        stock.balance_sheet_quarterly.set_api_data(balance_sheet_quarterly)
        stock.dao.save(stock.balance_sheet_quarterly)
        self.assertEqual(balance_sheet_quarterly, stock.balance_sheet_quarterly.get_stock_data())

        stock.fetch_stock_data(Data.BalanceSheetQuarterly)
        self.assertEqual(balance_sheet_quarterly, stock.balance_sheet_quarterly.get_stock_data())

    # @unittest.skip("Avoid multiple API calls for now.")
    def test_stock(self):
        stock = Stock("AAPL")
        stock.fetch_from_api()

        balance_sheet_quarterly = stock.balance_sheet_quarterly.get_stock_data()
        balance_sheet_yearly = stock.balance_sheet_yearly.get_stock_data()
        cash_flow_quarterly = stock.cash_flow_quarterly.get_stock_data()
        cash_flow_yearly = stock.cash_flow_yearly.get_stock_data()
        earnings_quarterly = stock.earnings_quarterly.get_stock_data()
        earnings_yearly = stock.earnings_yearly.get_stock_data()
        historical_prices = stock.historical_prices.get_stock_data()
#        stock_splits = stock.stock_splits.get_data()
        stock.dirty = True
        stock.save_to_db()
        stock.fetch_from_db()

        self.assertEqual(balance_sheet_quarterly, stock.balance_sheet_quarterly.get_stock_data())
        self.assertEqual(balance_sheet_yearly, stock.balance_sheet_yearly.get_stock_data())
        self.assertEqual(cash_flow_quarterly, stock.cash_flow_quarterly.get_stock_data())
        self.assertEqual(cash_flow_yearly, stock.cash_flow_yearly.get_stock_data())
        self.assertEqual(earnings_quarterly, stock.earnings_quarterly.get_stock_data())
        self.assertEqual(earnings_yearly, stock.earnings_yearly.get_stock_data())
        self.assertEqual(historical_prices, stock.historical_prices.get_stock_data())
#        self.assertEqual(stock_splits, stock.stock_splits.get_data())

#        modification_date = stock.earnings_yearly.get_modification_date()
        stock.save_to_db()
#        self.assertNotEqual(modification_date, stock.earnings_yearly.get_modification_date())

    def test_fetch_stock_data(self):
        stock = Stock("AAPL")
        stock.fetch_stock_data(Data.HistoricalPrices)
        nr_prices = len(stock.historical_prices["data"]["historical"])
        self.assertGreater(nr_prices, 250*25)
        stock.fetch_stock_data(Data.BalanceSheetQuarterly)

    def test_plot_historical_prices(self):
        stock = Stock("AAPL")
        stock.fetch_stock_data(Data.HistoricalPrices)
        for price_data in stock.historical_prices["data"]["historical"]:
            price = float(price_data["close"])
            date_value = date_parser.parse(price_data["date"])
            plt.plot(date_value, price, 'g.')
        plt.show()

    def test_plot_mda50_prices(self):
        stock = Stock("AAPL")
        stock.fetch_stock_data(Data.HistoricalPrices)
        stock.generate_data(Data.Mda50Prices)
        for i in range(len(stock.mda50_prices["data"]["prices"])):
            price = float(stock.mda50_prices["data"]["prices"][i])
            date_value = date_parser.parse(stock.mda50_prices["data"]["dates"][i])
            plt.plot(date_value, price, 'g.')
        plt.show()

    def test_plot_mda50_value(self):
        stock = Stock("AAPL")
        stock.fetch_stock_data(Data.HistoricalPrices)
        stock.generate_data(Data.Mda50Prices)
        nr_shares = stock.get_shares_outstanding()
        for i in range(len(stock.mda50_prices["data"]["prices"])):
            value = float(stock.mda50_prices["data"]["prices"][i]) * nr_shares
            date_value = date_parser.parse(stock.mda50_prices["data"]["dates"][i])
            plt.plot(date_value, value, 'g.')
        plt.show()

    def test_analysis_models(self):
        stock = Stock("ADIL")
        stock.fetch_from_db()
        print(stock.status)
#        stock.generate_mda50_prices()
        stock.populate_analysis_models()

    def test_stock_list(self):
        api = Factory.get_api_instance()
        stock_list = api.get_market_data(Market.SP500Stocks)
        stocks = {}
        list = []
        for s in stock_list:
            list.append(s["symbol"])
        # print(list)
        stocks = {"stocks": str(list)}
        print(stocks)
        print("")

    # def test_clean_db(self):
    #     from stocks.factory import Factory
    #     dao = Factory.get_dao_instance()
    #     dao.db_client.drop_database("Stocks")

    def read_all_stocks(self):
        with open('US_stocks.json') as json_file:
            stocks = json.load(json_file)
            stock_list = list(stocks["stocks"])
        try:
            with open('bad_US_stocks.json') as json_file:
                bad_stocks_json = json.load(json_file)
                bad_stocks = list(bad_stocks_json["bad_stocks"])
        except Exception:
            bad_stocks = []
        return stock_list, bad_stocks

    def populate_stocks(self, symbol_list, bad_stocks=[]):
        stock_list = []
        for symbol in symbol_list:
            if symbol not in bad_stocks:
                logging.info("Loading {}...".format(symbol))
                stock = Stock(symbol)
                stock.fetch_from_db()

                logging.info("Populating {}...".format(symbol))
                stock.populate_analysis_models()
                if stock.status == StockStatus.Populated:
                    stock_list.append(stock)
                else:
                    bad_stocks.append(symbol)

                logging.info("Finished Loading.\n=============================".format(symbol))
        logging.info("Loaded {} stocks.".format(len(symbol_list) - len(bad_stocks)))
        return stock_list

    def write_to_csv(self, stock_list, filename, before=None, after=None):
        with open(filename, 'w') as outfile:
            # Write column names
            outfile.write('symbol, date, 3y_gain, intrinsic_value')
            for field in KEY_METRICS_FIELD_LIST:
                outfile.write("," + field)
            outfile.write('\n')
            for stock in stock_list:
                for model in stock.analysis_models.values():
                    date_value = date_parser.parse(model.date)
                    if (before is None or date_value < before) and (after is None or date_value >= after):
                        if model.is_valid():
                            gain = round(100 * ((model.mda50_3y - model.mda50) / model.mda50), 2)
                            field_list = [stock.name, model.date, gain]
                            field_list = field_list + model.get_key_metrics()
                            outfile.write(field_list[0])
                            for i in range(1, len(field_list)):
                                outfile.write(',')
                                outfile.write(str(field_list[i]))
                            outfile.write('\n')

    def write_deltas_to_csv(self, stock_list, filename):
        with open(filename, 'w') as outfile:
            # Write column names
            outfile.write('symbol, date, 3y_gain, intrinsic_value')
            for i in range(4):
               for field in KEY_METRICS_FIELD_LIST:
                    outfile.write("," + field+"-"+str(i))
            outfile.write('\n')
            for stock in stock_list:
                for model in stock.analysis_models.values():
                    if model.is_all_valid():
                        outfile.write(model.symbol)
                        outfile.write(',')
                        outfile.write(model.date)
                        outfile.write(',')
                        gain = int(100 * ((model.mda50_3y - model.mda50) / model.mda50))
                        outfile.write(str(gain))
                        outfile.write(',')
                        outfile.write(str(int(model.mda50_3y)))
                        for i in range(4):
                            for field in KEY_METRICS_FIELD_LIST:
                                field_value = getattr(model.key_metrics[i], field)
                                outfile.write("," + str(field_value))
                        outfile.write('\n')

    def get_columns(self):
        columns = ["intrinsic_value"] + KEY_METRICS_FIELD_LIST
        return columns

    def get_delta_columns(self):
        columns = ["intrinsic_value"]
        for i in range(4):
            for field in KEY_METRICS_FIELD_LIST:
                columns.append(field + "_" + str(i))
        logging.debug("list: "+str(columns))

    def get_model_memory_usage(self, batch_size, model):
        import numpy as np
        from tensorflow.keras import backend as K

        shapes_mem_count = 0
        for l in model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes

    def create_mlp(self, dim, n, regress=False):
        # define our MLP network
        model = Sequential()
        model.add(Dense(64 * n, input_dim=dim, activation="relu"))
        model.add(Dense(128 * n, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(256 * n, activation="relu"))
        # model.add(Dropout(0.5))
        # model.add(Dense(512, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(256 * n, activation="relu"))
        model.add(Dense(256 * n, activation="relu"))
        model.add(Dense(256 * n, activation="relu"))
        model.add(Dense(256 * n, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(128 * n, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(64 * n, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(32, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(16, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(8, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(4, activation="relu"))

        # check to see if the regression node should be added
        if regress:
            model.add(Dense(1, activation="linear"))

#        logging.info("Memory usage:{}Gb".format(self.get_model_memory_usage(128, model)))
        # return our model
        return model

    def load_csv_data(self, filename):
        df = pd.read_csv(filename, sep=",", header=None, names=self.get_columns(), skiprows=1, low_memory=False)
        return df

    def load_delta_csv_data(self, filename):
        df = pd.read_csv(filename, sep=",", header=None, names=self.get_delta_columns(), skiprows=1, low_memory=False)
        return df

    def test_read_all_stocks(self):
        symbol_list, bad_stocks = self.read_all_stocks()
        stock_list = self.populate_stocks(symbol_list, bad_stocks)
        logging.info("Loaded {} stocks.".format(len(stock_list)))
        bad_stocks_json = {"bad_stocks": bad_stocks}
        with open("bad_US_stocks.json", 'w') as outfile:
            json.dump(bad_stocks_json, outfile)

    @unittest.skip("Obsolete.")
    def test_populate_all_stocks(self):
        stock_list, bad_stocks = self.read_all_stocks()
        valid_stocks = []
        populated_stocks = []
        valid_stock_names = []
        for symbol in stock_list:
            if symbol not in bad_stocks:
                logging.info("Loading {}...".format(symbol))
                stock = Stock(symbol)
                stock.fetch_from_db()
                logging.info("Finished Loading.\n=============================".format(symbol))
                if stock.status == StockStatus.Fetched:
                    if len(valid_stocks)%10 == 0:
                        logging.info("Save {} stocks".format(len(valid_stocks)))
                    valid_stocks.append(stock)
                else:
                    bad_stocks.append(symbol)

        logging.info("Populating using {} CPUs...\n=============================".format(thread_count))

        count = int(len(valid_stocks) / thread_count)
        for t in range(thread_count + 1):
            start_index = t * count
            end_index = min(start_index + count, len(valid_stocks))
            thread = PopulatingThread(valid_stocks, start_index, end_index, populated_stocks)
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

        logging.info("Populated {} stocks.".format(len(valid_stocks)))

        threads.clear()

        populated_stock_names = []
        for s in populated_stocks:
            populated_stock_names.append(s.name)

        for s in valid_stock_names:
            if s not in populated_stock_names:
                bad_stocks.append(s.name)

        bad_stocks_json = {"bad_stocks": bad_stocks}
        with open("bad_US_stocks.json", 'w') as outfile:
            json.dump(bad_stocks_json, outfile)

        valid_stocks_json = {"valid_stocks": valid_stock_names}
        with open("valid_US_stocks.json", 'w') as outfile:
            json.dump(valid_stocks_json, outfile)

    def test_create_csv(self):
        symbol_list, bad_stocks = self.read_all_stocks()
        stock_list = self.populate_stocks(symbol_list, bad_stocks)
        logging.info("Loaded {} stocks.".format(len(stock_list)))
        self.write_to_csv(stock_list, "stocks_financial.csv")

    def test_create_csv_dow(self):
        with open('Dow_stocks.json') as json_file:
            stocks = json.load(json_file)
            symbol_list = list(stocks["stocks"])
        stock_list = self.populate_stocks(symbol_list)
        logging.info("Loaded {} stocks.".format(len(stock_list)))
        cutoff_date = date_parser.parse('2013-01-01')
        self.write_to_csv(stock_list, "train_dow_financial.csv", before=cutoff_date)
        self.write_to_csv(stock_list, "test_dow_financial.csv", after=cutoff_date)

    def test_create_csv_sp500(self):
        with open('SP500_stocks.json') as json_file:
            stocks = json.load(json_file)
            symbol_list = list(stocks["stocks"])
        stock_list = self.populate_stocks(symbol_list)
        logging.info("Loaded {} stocks.".format(len(stock_list)))
        with open('SP500_test_stocks.json') as json_file:
            stocks = json.load(json_file)
            test_stocks = list(stocks["stocks"])
        test_list = []
        train_list = []
        for stock in stock_list:
            if stock.name in test_stocks:
                test_list.append(stock)
            else:
                train_list.append(stock)
        self.write_to_csv(test_list, "test_sp500_financial.csv")
        self.write_to_csv(train_list, "train_sp500_financial.csv")

    def test_create_delta_csv_sp500(self):
        with open('SP500_stocks.json') as json_file:
            stocks = json.load(json_file)
            symbol_list = list(stocks["stocks"])
        stock_list = self.populate_stocks(symbol_list)
        logging.info("Loaded {} stocks.".format(len(stock_list)))
        with open('SP500_test_stocks.json') as json_file:
            stocks = json.load(json_file)
            test_stocks = list(stocks["stocks"])
        test_list = []
        train_list = []
        for stock in stock_list:
            if stock.name in test_stocks:
                test_list.append(stock)
            else:
                train_list.append(stock)
        self.write_deltas_to_csv(test_list, "test_sp500_financial.csv")
        self.write_deltas_to_csv(train_list, "train_sp500_financial.csv")

    def prepare_split_test_set(self, filename):
        df = self.load_csv_data(filename)
        logging.info("Processing data...")
        split = train_test_split(df, test_size=0.25, random_state=42)
        (trainAttrX, testAttrX) = split
        # for d in trainAttrX["intrinsic_value"]:
        #     logging.info("Value " + str(d))
        max_value = trainAttrX["intrinsic_value"].max()
        trainY = trainAttrX["intrinsic_value"] / max_value
        testY = testAttrX["intrinsic_value"] / max_value
        return trainAttrX, trainY, testAttrX, testY

    def prepare_test_set(self, filename, max_value):
        df = self.load_delta_csv_data(filename)
        logging.info("Processing data...")
        # for d in trainAttrX["intrinsic_value"]:
        #     logging.info("Value " + str(d))
#        max_value = trainAttrX["intrinsic_value"].max()
        trainY = df["intrinsic_value"] / max_value
        return df, trainY

    def prepare_delta_test_set(self, filename):
        for i in range(4):
            training_sets = []
            df = self.load_csv_data(filename+"_"+i.csv)
            logging.info("Processing data...")
            split = train_test_split(df, test_size=0.25, random_state=42)
            (trainAttrX, testAttrX) = split
            # for d in trainAttrX["intrinsic_value"]:
            #     logging.info("Value " + str(d))
            max_value = trainAttrX["intrinsic_value"].max()
            trainY = trainAttrX["intrinsic_value"] / max_value
            testY = testAttrX["intrinsic_value"] / max_value
            training_sets.append(trainAttrX, trainY, testAttrX, testY)
            return training_sets

    def prepare_model(self, trainAttrX, n=1):
        mlp = self.create_mlp(trainAttrX.shape[1], n, regress=False)
        # if np.any(np.isnan(mlp.output)):
        #     logging.warning("NaN in input")
        #        combinedInput = concatenate([mlp.output, mlp.output])
        #        x = Dense(4, activation="relu")(mlp.output)
        x = Dense(1, activation="linear")(mlp.output)
        model = Model(inputs=[mlp.input], outputs=x)
        opt = Adam(lr=0.01, decay=0.01 / 200)
        #        model.compile(loss="mse", optimizer=opt)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
        return model

    def test_start_training_dow(self):
        trainAttrX, trainY = self.prepare_test_set("train_dow_financial.csv", 1000)
        testAttrX, testY = self.prepare_test_set("test_dow_financial.csv", 1000)
        model = self.prepare_model(trainAttrX, 4)
        self.train_model(model, testAttrX, testY, trainAttrX, trainY, "test_models/blind-model.h5")

    def test_start_training_sp500(self):
        logging.info("Preparing test data...")
        trainAttrX, trainY = self.prepare_test_set("train_sp500_financial.csv", 1000)
        testAttrX, testY = self.prepare_test_set("test_sp500_financial.csv", 1000)
        logging.info("Preparing model...")
        model = self.prepare_model(trainAttrX, 1)
        self.train_model(model, testAttrX, testY, trainAttrX, trainY, "test_models/sp500-model.h5")

    def test_start_training_sp500_delta(self):
        logging.info("Preparing test data...")
        trainAttrX, trainY = self.prepare_test_set("train_sp500_financial.csv", 4000)
        testAttrX, testY = self.prepare_test_set("test_sp500_financial.csv", 4000)
        logging.info("Preparing model...")
        model = self.prepare_model(trainAttrX, 4)
        self.train_model(model, testAttrX, testY, trainAttrX, trainY, "test_models/sp500-model.h5")

    def train_model(self, model, testAttrX, testY, trainAttrX, trainY, model_filename, epochs=10, batch_size=8):
        logging.info("Training set has {} entries".format(trainAttrX.shape[0]))
        logging.info("Model uses {} attributes".format(trainAttrX.shape[1]))

        best = 1000.0
        for i in range(1000):
            # train the model
            logging.info("Training model...")
            model.fit(
                x=[trainAttrX], y=trainY,
                #validation_data=([testAttrX], testY),
                epochs=epochs, batch_size=batch_size)
            model.save(model_filename)

            # make predictions on the testing data
            logging.info("Predicting stock prices...")
            preds = model.predict([testAttrX])

            # compute the difference between the *predicted* stock values and the
            # *actual* stock values, then compute the percentage difference and
            # the absolute percentage difference
            diff = preds.flatten() - testY
            percentDiff = (diff / testY) * 100
            absPercentDiff = np.abs(percentDiff)

            # logging.info("Error {}% after {} batches".format(absPercentDiff, i))
            # compute the mean and standard deviation of the absolute percentage
            # difference
            mean = np.mean(absPercentDiff)
            std = np.std(absPercentDiff)
            if mean < best:
                best = mean
                model.save("sp_"+str(round(best, 2))+"_"+model_filename)
            print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))

    def test_resume_dow_training(self):
        trainAttrX, trainY = self.prepare_test_set("train_dow_financial.csv", 1000)
        testAttrX, testY = self.prepare_test_set("test_dow_financial.csv", 1000)
        # trainAttrX, trainY, testAttrX, testY = self.prepare_test_set("dow_financial.csv")
        model = load_model('test_models/blind-model.h5')

        self.train_model(model, testAttrX, testY, trainAttrX, trainY, 'test_models/blind-model.h5')

    def test_resume_sp500_training(self):
        trainAttrX, trainY = self.prepare_test_set("train_sp500_financial.csv", 1000)
        testAttrX, testY = self.prepare_test_set("test_sp500_financial.csv", 1000)
        # trainAttrX, trainY, testAttrX, testY = self.prepare_test_set("dow_financial.csv")
        model = load_model('test_models/sp500-model.h5')

        self.train_model(model, testAttrX, testY, trainAttrX, trainY, 'test_models/sp500-model.h5')

    def test_create_delta_csv_dow(self):
        with open('Dow_stocks.json') as json_file:
            stocks = json.load(json_file)
            symbol_list = list(stocks["stocks"])
        stock_list = self.populate_stocks(symbol_list)
        logging.info("Loaded {} stocks.".format(len(stock_list)))
        self.write_deltas_to_csv(stock_list, "dow_deltas.csv")

    def test_predictions(self):
        with open('Dow_stocks.json') as json_file:
            stocks = json.load(json_file)
            symbol_list = list(stocks["stocks"])

        # trainAttrX, trainY = self.prepare_test_set("train_dow_financial.csv", 1000)
        # testAttrX, testY = self.prepare_test_set("test_dow_financial.csv", 1000)
        neural_model = load_model('test_models/blind-model.h5')

        start_date = '2017-12-30'
        stock_list = self.populate_stocks(symbol_list)
        for stock in stock_list:
            models = stock.analysis_models
            closest_date = stock.get_closest_date(start_date)
            model = models[closest_date]
            data = model.get_key_metrics()
            predicted_value = neural_model.predict([data])[0][0]
            predicted_value = round(float(predicted_value * 1000.0), 2)
            print("{} on {} is predicted to be valued ${} in 3 years".format(stock.name, closest_date, predicted_value))


if __name__ == '__main__':
    unittest.main()
