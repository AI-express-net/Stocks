from stocks.be import BaseEntity
from stocks.data_names import Data


class EnterpriseValueEntity(BaseEntity):

    def __init__(self, name):
        super().__init__(name)
        self.period = BaseEntity.Period.Quarterly

    def get_table_name(self):
        return Data.KeyMetricsQuarterly
