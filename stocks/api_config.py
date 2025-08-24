"""
Configuration settings for the Stocks application API.
"""

import os

# Load environment variables from api.env file if it exists
def load_env_file():
    """Load environment variables from api.env file."""
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load the environment file
load_env_file()

# Financial Modeling Prep API configuration
API_KEY = os.getenv("FMP_API_KEY", "{API_KEY}")
API_BASE = "https://financialmodelingprep.com/api/v3/"

# Add other API configuration settings here as needed
