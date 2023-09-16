from stocks.mongodb_dao import MongoDbDao
from stocks.fmp_api import FmpApi

dao_instance = MongoDbDao("Stocks")
api_instance = FmpApi()


class Factory:

    @staticmethod
    def get_dao_instance():
        return dao_instance

    @staticmethod
    def get_api_instance():
        return api_instance
