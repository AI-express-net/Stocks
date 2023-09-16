import unittest
from stocks.data_names import Data
from stocks.data_names import get_name_list


class DataNamesTestCase(unittest.TestCase):

    def test_data_names(self):
        field_name = Data.BalanceSheetQuarterly
        self.assertEqual("balance_sheet_quarterly", field_name)
        name_list = get_name_list()
        self.assertIn(field_name, name_list)
