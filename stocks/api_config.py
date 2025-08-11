"""
Configuration settings for the Stocks application API.
"""

import os

# Financial Modeling Prep API configuration
API_KEY = os.getenv("FMP_API_KEY", "{API_KEY}")
API_BASE = "https://financialmodelingprep.com/api/v3/"

# Add other API configuration settings here as needed
