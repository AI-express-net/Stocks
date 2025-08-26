#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from fmp_api import FmpApi
from data_names import Data

def test_spy_api():
    api = FmpApi()
    print('Testing SPY historical data fetch...')
    try:
        sp500_historical_data = api.get_stock_data('SPY', Data.HistoricalPrices)
        print('Successfully fetched SPY data')
        print(f'Data type: {type(sp500_historical_data)}')
        if isinstance(sp500_historical_data, dict):
            print(f'Symbol: {sp500_historical_data.get("symbol", "N/A")}')
            historical = sp500_historical_data.get('historical', [])
            print(f'Number of data points: {len(historical)}')
            if historical:
                latest = historical[0]
                print(f'Latest date: {latest.get("date", "N/A")}')
                print(f'Latest close: ${latest.get("close", "N/A")}')
                
                # Show a few more recent entries
                print('\nRecent SPY prices:')
                for i, entry in enumerate(historical[:5]):
                    print(f'  {entry.get("date", "N/A")}: ${entry.get("close", "N/A")}')
        else:
            print(f'Response: {sp500_historical_data}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    test_spy_api()
