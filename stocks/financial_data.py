import logging

from stocks.be import BaseEntity

logging.basicConfig(format='%(asctime)s [%(levelname)s] "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"', level=logging.INFO)

class FinancialData(BaseEntity):

    FINANCIAL_DATA_FIELDS = {
#        "fillingDate": None,
        "commonStock": None,
        # Balance Sheet fields
        "accountPayables": None,
        "accumulatedOtherComprehensiveIncomeLoss": None,
        "cashAndCashEquivalents": None,
        "cashAndShortTermInvestments": None,
        "deferredRevenue": None,
        "deferredRevenueNonCurrent": None,
        "deferredTaxLiabilitiesNonCurrent": None,
        "goodwill": None,
        "goodwillAndIntangibleAssets": None,
        "intangibleAssets": None,
        "inventory": None,
        "longTermDebt": None,
        "longTermInvestments": None,
        "netDebt": None,
        "netReceivables": None,
        "otherAssets": None,
        "otherCurrentAssets": None,
        "otherCurrentLiabilities": None,
        "otherLiabilities": None,
        "otherNonCurrentAssets": None,
        "otherNonCurrentLiabilities": None,
        "othertotalStockholdersEquity": None,
        "propertyPlantEquipmentNet": None,
        "retainedEarnings": None,
        "shortTermDebt": None,
        "shortTermInvestments": None,
        "taxAssets": None,
        "taxPayables": None,
        "totalAssets": None,
        "totalCurrentAssets": None,
        "totalCurrentLiabilities": None,
        "totalDebt": None,
        "totalInvestments": None,
        "totalLiabilities": None,
        "totalLiabilitiesAndStockholdersEquity": None,
        "totalNonCurrentAssets": None,
        "totalNonCurrentLiabilities": None,
        "totalStockholdersEquity": None,
#    }

#    CASH_FLOW_FIELDS = {
        "accountsPayables": None,
        "accountsReceivables": None,
        "acquisitionsNet": None,
        "capitalExpenditure": None,
        "cashAtBeginningOfPeriod": None,
        "cashAtEndOfPeriod": None,
        "changeInWorkingCapital": None,
        "commonStockIssued": None,
        "commonStockRepurchased": None,
        "debtRepayment": None,
        "deferredIncomeTax": None,
#        "depreciationAndAmortization": None,
        "dividendsPaid": None,
        "effectOfForexChangesOnCash": None,
        "freeCashFlow": None,
        "inventoryChange": None,  # changed from "inventory"
        "investmentsInPropertyPlantAndEquipment": None,
        "netCashProvidedByOperatingActivities": None,
        "netCashUsedForInvestingActivites": None,
        "netCashUsedProvidedByFinancingActivities": None,
        "netChangeInCash": None,
#        "netIncome": None,
        "operatingCashFlow": None,
        "otherFinancingActivites": None,
        "otherInvestingActivites": None,
        "otherNonCashItems": None,
        "otherWorkingCapital": None,
        "purchasesOfInvestments": None,
        "salesMaturitiesOfInvestments": None,
        "stockBasedCompensation": None,
#    }

#    EARNINGS_FIELDS = {
        "costAndExpenses": None,
        "costOfRevenue": None,
        "depreciationAndAmortization": None,
        "ebitda": None,
        "ebitdaratio": None,
        "eps": None,
        "epsdiluted": None,
        "generalAndAdministrativeExpenses": None,
        "grossProfit": None,
        "grossProfitRatio": None,
        "incomeBeforeTax": None,
        "incomeBeforeTaxRatio": None,
        "incomeTaxExpense": None,
        "interestExpense": None,
        "netIncome": None,
        "netIncomeRatio": None,
        "operatingExpenses": None,
        "operatingIncome": None,
        "operatingIncomeRatio": None,
        "otherExpenses": None,
        "researchAndDevelopmentExpenses": None,
        "revenue": None,
        "sellingAndMarketingExpenses": None,
        "totalOtherIncomeExpensesNet": None,
        "weightedAverageShsOut": None,
        "weightedAverageShsOutDil": None,
    }

#    ALL_FIELDS = BALANCE_SHEET_FIELDS + CASH_FLOW_FIELDS + EARNINGS_FIELDS

    def __init__(self, symbol, date_formatted, index):
        # PK
        self.symbol = symbol
        self.date = date_formatted
        self.index = index

        # Common fields
        self.fillingDate = None
        self.commonStock = None
        # Balance Sheet fields
        self.accountPayables = 0.0
        self.accumulatedOtherComprehensiveIncomeLoss = 0.0
        self.cashAndCashEquivalents = 0.0
        self.cashAndShortTermInvestments = 0.0
        self.deferredRevenue = 0.0
        self.deferredRevenueNonCurrent = 0.0
        self.deferredTaxLiabilitiesNonCurrent = 0.0
        self.goodwill = 0.0
        self.goodwillAndIntangibleAssets = 0.0
        self.intangibleAssets = 0.0
        self.inventory = 0.0
        self.longTermDebt = 0.0
        self.longTermInvestments = 0.0
        self.netDebt = 0.0
        self.netReceivables = 0.0
        self.otherAssets = 0.0
        self.otherCurrentAssets = 0.0
        self.otherCurrentLiabilities = 0.0
        self.otherLiabilities = 0.0
        self.otherNonCurrentAssets = 0.0
        self.otherNonCurrentLiabilities = 0.0
        self.othertotalStockholdersEquity = 0.0
        self.propertyPlantEquipmentNet = 0.0
        self.retainedEarnings = 0.0
        self.shortTermDebt = 0.0
        self.shortTermInvestments = 0.0
        self.taxAssets = 0.0
        self.taxPayables = 0.0
        self.totalAssets = 0.0
        self.totalCurrentAssets = 0.0
        self.totalCurrentLiabilities = 0.0
        self.totalDebt = 0.0
        self.totalInvestments = 0.0
        self.totalLiabilities = 0.0
        self.totalLiabilitiesAndStockholdersEquity = 0.0
        self.totalNonCurrentAssets = 0.0
        self.totalNonCurrentLiabilities = 0.0
        self.totalStockholdersEquity = 0.0
        # Cash Flow fields
        self.accountsPayables = 0.0
        self.accountsReceivables = 0.0
        self.acquisitionsNet = 0.0
        self.capitalExpenditure = 0.0
        self.cashAtBeginningOfPeriod = 0.0
        self.cashAtEndOfPeriod = 0.0
        self.changeInWorkingCapital = 0.0
        self.commonStockIssued = 0.0
        self.commonStockRepurchased = 0.0
        self.debtRepayment = 0.0
        self.deferredIncomeTax = 0.0
        self.depreciationAndAmortization = 0.0
        self.dividendsPaid = 0.0
        self.effectOfForexChangesOnCash = 0.0
        self.freeCashFlow = 0.0
        self.inventory = 0.0  # change
        self.investmentsInPropertyPlantAndEquipment = 0.0
        self.netCashProvidedByOperatingActivities = 0.0
        self.netCashUsedForInvestingActivites = 0.0
        self.netCashUsedProvidedByFinancingActivities = 0.0
        self.netChangeInCash = 0.0
        self.netIncome = 0.0
        self.operatingCashFlow = 0.0
        self.otherFinancingActivites = 0.0
        self.otherInvestingActivites = 0.0
        self.otherNonCashItems = 0.0
        self.otherWorkingCapital = 0.0
        self.purchasesOfInvestments = 0.0
        self.salesMaturitiesOfInvestments = 0.0
        self.stockBasedCompensation = 0.0
        # Earnings fields
        self.costAndExpenses = 0.0
        self.costOfRevenue = 0.0
        self.depreciationAndAmortization = 0.0
        self.ebitda = 0.0
        self.ebitdaratio = 0.0
        self.eps = 0.0
        self.epsdiluted = 0.0
        self.generalAndAdministrativeExpenses = 0.0
        self.grossProfit = 0.0
        self.grossProfitRatio = 0.0
        self.incomeBeforeTax = 0.0
        self.incomeBeforeTaxRatio = 0.0
        self.incomeTaxExpense = 0.0
        self.interestExpense = 0.0
        self.netIncome = 0.0
        self.netIncomeRatio = 0.0
        self.operatingExpenses = 0.0
        self.operatingIncome = 0.0
        self.operatingIncomeRatio = 0.0
        self.otherExpenses = 0.0
        self.researchAndDevelopmentExpenses = 0.0
        self.revenue = 0.0
        self.sellingAndMarketingExpenses = 0.0
        self.totalOtherIncomeExpensesNet = 0.0
        self.weightedAverageShsOut = 0.0
        self.weightedAverageShsOutDil = 0.0


FINANCIAL_DATA_FIELD_LIST = []
KEY_METRICS_FIELD_LIST = []


def populate_financial_data_fields():
    if len(FINANCIAL_DATA_FIELD_LIST) == 0:
        for key in FinancialData.FINANCIAL_DATA_FIELDS:
            fd_key = FinancialData.FINANCIAL_DATA_FIELDS[key]
            if fd_key is None:
                fd_key = key
            FINANCIAL_DATA_FIELD_LIST.append(fd_key)

def populate_key_metrics_fields():
    if len(KEY_METRICS_FIELD_LIST) == 0:
        for key in KeyMetricsData.KEY_METRICS_FIELDS:
            fd_key = KeyMetricsData.KEY_METRICS_FIELDS[key]
            if fd_key is None:
                fd_key = key
            KEY_METRICS_FIELD_LIST.append(fd_key)

class KeyMetricsData(BaseEntity):
    KEY_METRICS_FIELDS = {
        "revenuePerShare": None,
        "netIncomePerShare": None,
        "operatingCashFlowPerShare": None,
        "freeCashFlowPerShare": None,
        "cashPerShare": None,
        "bookValuePerShare": None,
        "tangibleBookValuePerShare": None,
        "shareholdersEquityPerShare": None,
        "interestDebtPerShare": None,
        "marketCap": None,
        "enterpriseValue": None,
        "peRatio": None,
        "priceToSalesRatio": None,
        "pocfratio": None,
        "pfcfRatio": None,
        "pbRatio":None,
        "ptbRatio": None,
        "evToSales": None,
        "enterpriseValueOverEBITDA": None,
        "evToOperatingCashFlow": None,
        "evToFreeCashFlow": None,
        "earningsYield": None,
        "freeCashFlowYield": None,
        "debtToEquity": None,
        "debtToAssets": None,
        "netDebtToEBITDA": None,
        "currentRatio": None,
        "interestCoverage": None,
        "incomeQuality": None,
        "dividendYield": None,
        "payoutRatio": None,
        "salesGeneralAndAdministrativeToRevenue": None,
        "researchAndDdevelopementToRevenue": None,
        "intangiblesToTotalAssets": None,
        "capexToOperatingCashFlow": None,
        "capexToRevenue": None,
        "capexToDepreciation": None,
        "stockBasedCompensationToRevenue": None,
        "grahamNumber": None,
        "roic": None,
        "returnOnTangibleAssets": None,
        "grahamNetNet": None,
        "workingCapital": None,
        "tangibleAssetValue": None,
        "netCurrentAssetValue": None,
        "investedCapital": None,
        "averageReceivables": None,
        "averagePayables": None,
        "averageInventory": None,
        "daysSalesOutstanding": None,
        "daysPayablesOutstanding": None,
        "daysOfInventoryOnHand": None,
        "receivablesTurnover": None,
        "payablesTurnover": None,
        "inventoryTurnover": None,
        "roe": None,
        "capexPerShare": None
    }
    def __init__(self, symbol, date_formatted, index):
        # PK
        self.symbol = symbol
        self.date = date_formatted
        self.index = index

        # Key Metrics
        self.revenuePerShare = 0.0
        self.netIncomePerShare = 0.0
        self.operatingCashFlowPerShare = 0.0
        self.freeCashFlowPerShare = 0.0
        self.cashPerShare = 0.0
        self.bookValuePerShare = 0.0
        self.tangibleBookValuePerShare = 0.0
        self.shareholdersEquityPerShare = 0.0
        self.interestDebtPerShare = 0.0
        self.marketCap = 0.0
        self.enterpriseValue = 0.0
        self.peRatio = 0.0
        self.priceToSalesRatio = 0.0
        self.pocfratio = 0.0
        self.pfcfRatio = 0.0
        self.pbRatio = 0.0
        self.ptbRatio = 0.0
        self.evToSales = 0.0
        self.enterpriseValueOverEBITDA = 0.0
        self.evToOperatingCashFlow = 0.0
        self.evToFreeCashFlow = 0.0
        self.earningsYield = 0.0
        self.freeCashFlowYield = 0.0
        self.debtToEquity = 0.0
        self.debtToAssets = 0.0
        self.netDebtToEBITDA = 0.0
        self.currentRatio = 0.0
        self.interestCoverage = 0.0
        self.incomeQuality = 0.0
        self.dividendYield = 0.0
        self.payoutRatio = 0.0
        self.salesGeneralAndAdministrativeToRevenue = 0.0
        self.researchAndDdevelopementToRevenue = 0.0
        self.intangiblesToTotalAssets = 0.0
        self.capexToOperatingCashFlow = 0.0
        self.capexToRevenue = 0.0
        self.capexToDepreciation = 0.0
        self.stockBasedCompensationToRevenue = 0.0
        self.grahamNumber = 0.0
        self.roic = 0.0
        self.returnOnTangibleAssets = 0.0
        self.grahamNetNet = 0.0
        self.workingCapital = 0.0
        self.tangibleAssetValue = 0.0
        self.netCurrentAssetValue = 0.0
        self.investedCapital = 0.0
        self.averageReceivables = 0.0
        self.averagePayables = 0.0
        self.averageInventory = 0.0
        self.daysSalesOutstanding = 0.0
        self.daysPayablesOutstanding = 0.0
        self.daysOfInventoryOnHand = 0.0
        self.receivablesTurnover = 0.0
        self.payablesTurnover = 0.0
        self.inventoryTurnover = 0.0
        self.roe = 0.0
        self.capexPerShare = 0.0

class AnalysisModel:

    def __init__(self, symbol, date_formatted, shares_outstanding):
        self.symbol = symbol
        self.date = date_formatted
        self.shares_outstanding = shares_outstanding
        self.mda50 = 0.0
        self.mda50_1y = 0.0
        self.mda50_3y = 0.0
        self.mda50_5y = 0.0
        self.financial_data = [FinancialData(symbol, date_formatted, 0), FinancialData(symbol, date_formatted, 1),
                               FinancialData(symbol, date_formatted, 2), FinancialData(symbol, date_formatted, 3)]
        self.key_metrics = [KeyMetricsData(symbol, date_formatted, 0), KeyMetricsData(symbol, date_formatted, 1),
                               KeyMetricsData(symbol, date_formatted, 2), KeyMetricsData(symbol, date_formatted, 3)]

    def is_valid(self):
        km = self.key_metrics[0]
        # TODO: '+' should be '*'?
        key_data = km.revenuePerShare + km.netIncomePerShare + \
                   km.operatingCashFlowPerShare + km.freeCashFlowPerShare
        return self.mda50_3y * key_data != 0

    def is_all_valid(self):
        for i in range(4):
            km = self.key_metrics[i]
            key_data = km.revenuePerShare + km.netIncomePerShare + \
                       km.operatingCashFlowPerShare + km.freeCashFlowPerShare
            if key_data == 0:
                return False
        return self.mda50_3y != 0

    def populate_financial_data(self, json_data):
        for key in json_data.keys():
            if key in FinancialData.FINANCIAL_DATA_FIELDS.keys():
                value = float(json_data[key] or 0.0)
                if key == "inventory" and getattr(self.financial_data[0], key) is not None:
                    setattr(self.financial_data[0], "inventoryChange", value)
                else:
                    setattr(self.financial_data[0], key, value)

    def populate_relative_data(self, current_data, past_data, destination):
        for field in FINANCIAL_DATA_FIELD_LIST:
            current_value = getattr(current_data, field)
            past_value = getattr(past_data, field)
            value = 0 if current_value == past_value else \
                1.0 if past_value == 0 else round((current_value / past_value) - 1.0, 3)
            setattr(destination, field, value)

    def populate_relative_metrics(self, current_data, past_data, destination):
        for field in KEY_METRICS_FIELD_LIST:
            current_value = round(getattr(current_data, field), 3)
            past_value = round(getattr(past_data, field), 3)
            value = 0 if current_value == past_value else \
                1.0 if past_value == 0.0 else round((current_value / past_value) - 1.0, 3)
            setattr(destination, field, value)

    def populate_key_metrics(self, json_data, nr_shares):
        for key in json_data.keys():
            if key in KeyMetricsData.KEY_METRICS_FIELDS.keys():
                value = float(json_data[key] or 0.0)
                if abs(value) > 10000:
                    value = value / nr_shares
                setattr(self.key_metrics[0], key, value)

    def get_key_metrics(self):
        # field_list = [self.symbol, self.date]
        # gain = round(100 * ((self.mda50_3y - self.mda50) / self.mda50), 2)
        # field_list.append(gain)
        field_list = []
        field_list.append(round(self.mda50_3y, 2))
        for field in KEY_METRICS_FIELD_LIST:
            field_value = getattr(self.key_metrics[0], field)
            field_list.append(field_value)
        return field_list


populate_financial_data_fields()
populate_key_metrics_fields()
