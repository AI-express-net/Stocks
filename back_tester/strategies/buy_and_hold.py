"""
Buy and Hold strategy implementation.
"""

from typing import List, Tuple
from back_tester.strategy import Strategy
from back_tester.models.transaction import Transaction, TransactionType


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
    
    def __init__(self, target_stocks: List[str] = None, max_position_size: float = 0.1, allow_existing_positions: bool = False):
        """
        Initialize the buy and hold strategy.
        
        Args:
            target_stocks: List of stocks to buy and hold (if None, uses all available)
            max_position_size: Maximum percentage of portfolio per stock (0.1 = 10%)
            allow_existing_positions: If True, allows buying more of stocks already in portfolio
        """
        self.target_stocks = target_stocks
        self.max_position_size = max_position_size
        self.allow_existing_positions = allow_existing_positions
    
    def get_strategy_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            Strategy name as a string
        """
        if self.target_stocks:
            stocks_str = '_'.join(self.target_stocks)
            return f"BuyAndHold_{stocks_str}"
        else:
            return "BuyAndHold_All"
    
    def get_strategy_description(self) -> str:
        """
        Get a description of the strategy.
        
        Returns:
            Strategy description as a string
        """
        target_desc = f"targeting {', '.join(self.target_stocks)}" if self.target_stocks else "targeting all available stocks"
        return f"Buy and hold strategy {target_desc} with max position size {self.max_position_size*100:.0f}%"
    
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
        portfolio_stocks = set()
        for item in portfolio_items:
            if hasattr(item, 'name'):
                portfolio_stocks.add(item.name)
            elif isinstance(item, str):
                portfolio_stocks.add(item)
            else:
                # Try to get the stock name from the item
                portfolio_stocks.add(str(item))
        
        # Determine which stocks to consider
        stocks_to_consider = self.target_stocks if self.target_stocks else [stock for stock, _ in stock_values]
        
        for stock, price in stock_values:
            if stock not in stocks_to_consider:
                continue
                
            # Skip if already in portfolio (unless allow_existing_positions is True)
            if stock in portfolio_stocks and not self.allow_existing_positions:
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