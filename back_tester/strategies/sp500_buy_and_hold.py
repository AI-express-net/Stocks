from typing import List, Tuple
from datetime import datetime

from back_tester.strategy import Strategy
from back_tester.models.transaction import Transaction, TransactionType
from back_tester.models.portfolio_item import PortfolioItem


class SP500BuyAndHoldStrategy(Strategy):
    """Buy-and-hold strategy for SP500 (SPX) benchmark."""
    
    def __init__(self, max_position_size: float = 1.0):
        """
        Initialize SP500 buy and hold strategy.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio (default 1.0 = 100%)
        """
        self.max_position_size = max_position_size
        self.sp500_symbol = 'SPY'  # SP500 ETF symbol
        self.initial_investment_made = False
    
    def generate_transactions(self, portfolio_items: List[PortfolioItem], 
                            stock_values: List[Tuple[str, float]], 
                            date: str, 
                            available_cash: float) -> List[Transaction]:
        """
        Generate buy-and-hold transactions for SP500.
        
        Strategy:
        1. On first call, invest all available cash in SP500
        2. On subsequent calls with additional cash, invest that cash in SP500
        """
        transactions = []
        
        # Get SP500 price from stock values (convert list to dict for lookup)
        stock_values_dict = dict(stock_values)
        sp500_price = stock_values_dict.get(self.sp500_symbol, 0.0)
        
        if sp500_price <= 0:
            # No valid SP500 price available
            return transactions
        
        # Calculate how much to invest
        if not self.initial_investment_made:
            # First time - invest all available cash
            investment_amount = available_cash * self.max_position_size
            self.initial_investment_made = True
        else:
            # Subsequent calls - invest any additional cash
            investment_amount = available_cash * self.max_position_size
        
        if investment_amount > 0 and sp500_price > 0:
            # Calculate number of shares to buy
            shares_to_buy = investment_amount / sp500_price
            
            if shares_to_buy > 0:
                transaction = Transaction(
                    stock=self.sp500_symbol,
                    transaction_type=TransactionType.BUY,
                    shares=shares_to_buy,
                    price=sp500_price,
                    date=date
                )
                transactions.append(transaction)
        
        return transactions
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "SP500_Buy_And_Hold"
    
    def get_strategy_description(self) -> str:
        """Get strategy description."""
        return f"Buy and hold SP500 with max position size {self.max_position_size * 100}%"
