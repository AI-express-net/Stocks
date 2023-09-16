from stocks.be import BaseEntity
from stocks.data_names import Data


class Mda50PricesEntity(BaseEntity):

    def __init__(self, name):
        super().__init__(name)

    def get_table_name(self):
        return Data.Mda50Prices
