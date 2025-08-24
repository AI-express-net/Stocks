"""
Buy and Hold strategy implementation.
"""

from typing import List, Tuple
from strategy import Strategy
from models.transaction import Transaction, TransactionType


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy and hold strategy.
    
    This strategy buys stocks when they are not in the portfolio and holds them.
    It does not sell existing positions.
    
    ===== BUY/SELL CRITERIA =====
    
    BUY DECISIONS:
    - Stock is not currently in the portfolio
    - Stock is in the target_stocks list (if specified) or any available stock
    - Sufficient cash available to purchase at least 1 share
    - Position size would not exceed max_position_size (default 10% of portfolio)
    
    SELL DECISIONS:
    - NONE - This strategy never sells existing positions
    
    HOLDING CRITERIA:
    - Once a stock is purchased, it is held indefinitely
    - No rebalancing or position adjustment occurs
    - Portfolio grows through new purchases only, not by selling existing positions
    
    ===== STRATEGY LOGIC =====
    This is a passive investment strategy that assumes stocks will appreciate over time.
    It's designed for long-term investors who believe in the overall growth of the market
    and don't want to actively trade or time the market.
    """
    
    def __init__(self, target_stocks: List[str] = None, max_position_size: float = 0.1):
        """
        Initialize the buy and hold strategy.
        
        Args:
            target_stocks: List of stocks to buy and hold (if None, uses all available)
            max_position_size: Maximum percentage of portfolio per stock (0.1 = 10%)
        """
        self.target_stocks = target_stocks
        self.max_position_size = max_position_size
    
    def generate_transactions(self, portfolio_items: List, stock_values: List[Tuple[str, float]], 
                            date: str, available_cash: float) -> List[Transaction]:
        """
        Generate buy transactions for stocks not in portfolio.
        
        Args:
            portfolio_items: Current portfolio positions
            stock_values: Available stock prices
            date: Current date
            available_cash: Available cash for trading
            
        Returns:
            List of buy transactions
        """
        # Validate inputs
        self.validate_inputs(portfolio_items, stock_values, date, available_cash)
        
        transactions = []
        
        # Get current portfolio value
        portfolio_value = self.get_portfolio_value(portfolio_items, stock_values)
        total_portfolio_value = portfolio_value + available_cash
        
        # Create a set of stocks already in portfolio
        portfolio_stocks = {item.name for item in portfolio_items}
        
        # Determine which stocks to consider
        stocks_to_consider = self.target_stocks if self.target_stocks else [stock for stock, _ in stock_values]
        
        for stock, price in stock_values:
            if stock not in stocks_to_consider:
                continue
                
            # Skip if already in portfolio
            if stock in portfolio_stocks:
                continue
            
            # Calculate maximum position size
            max_position_value = total_portfolio_value * self.max_position_size
            
            # Calculate how many shares we can buy
            max_shares = int(max_position_value / price)
            affordable_shares = int(available_cash / price)
            shares_to_buy = min(max_shares, affordable_shares)
            
            if shares_to_buy > 0:
                transaction = Transaction(
                    stock=stock,
                    date=date,
                    price=price,
                    shares=shares_to_buy,
                    transaction_type=TransactionType.BUY
                )
                transactions.append(transaction)
                available_cash -= transaction.get_total_value()
        
        return transactions 