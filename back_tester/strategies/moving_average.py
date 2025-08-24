"""
Moving Average strategy implementation.
"""

from typing import List, Tuple, Dict, Optional
from strategy import Strategy
from models.transaction import Transaction, TransactionType


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy.
    
    This strategy uses moving averages to generate buy/sell signals.
    It's a simplified implementation for demonstration purposes.
    
    ===== BUY/SELL CRITERIA =====
    
    BUY DECISIONS:
    - Stock is not currently in the portfolio
    - Short-term moving average (default 10 days) > Long-term moving average (default 30 days)
    - Current price > Short-term moving average
    - Sufficient cash available to purchase at least 1 share
    - Position size would not exceed max_position_size (default 20% of portfolio)
    
    SELL DECISIONS:
    - Stock is currently in the portfolio with shares > 0
    - Short-term moving average (default 10 days) < Long-term moving average (default 30 days)
    - Current price < Short-term moving average
    - Sells entire position (all shares) when sell signal is triggered
    
    HOLDING CRITERIA:
    - Stock is held as long as moving average conditions remain favorable
    - No partial position adjustments - either full position or no position
    - Positions are monitored daily for crossover signals
    
    ===== STRATEGY LOGIC =====
    This is a trend-following strategy that uses price momentum to identify entry and exit points.
    The strategy assumes that when short-term trends align with long-term trends and current
    price confirms the trend, it's a good time to buy. Conversely, when trends diverge and
    price falls below the short-term average, it's time to sell.
    """
    
    def __init__(self, short_period: int = 10, long_period: int = 30, 
                 max_position_size: float = 0.2):
        """
        Initialize the moving average strategy.
        
        Args:
            short_period: Short-term moving average period
            long_period: Long-term moving average period
            max_position_size: Maximum percentage of portfolio per stock
        """
        self.short_period = short_period
        self.long_period = long_period
        self.max_position_size = max_position_size
        self.price_history: Dict[str, List[Tuple[str, float]]] = {}
    
    def generate_transactions(self, portfolio_items: List, stock_values: List[Tuple[str, float]], 
                            date: str, available_cash: float) -> List[Transaction]:
        """
        Generate transactions based on moving average signals.
        
        Args:
            portfolio_items: Current portfolio positions
            stock_values: Available stock prices
            date: Current date
            available_cash: Available cash for trading
            
        Returns:
            List of buy/sell transactions
        """
        # Validate inputs
        self.validate_inputs(portfolio_items, stock_values, date, available_cash)
        
        # Update price history
        self._update_price_history(stock_values, date)
        
        # Get current portfolio value
        portfolio_value = self.get_portfolio_value(portfolio_items, stock_values)
        total_portfolio_value = portfolio_value + available_cash
        
        # Create portfolio lookup
        portfolio_dict = {item.name: item for item in portfolio_items}
        
        # Generate transactions using list comprehension
        transactions = [
            self._create_buy_transaction(stock, price, date, total_portfolio_value, available_cash)
            for stock, price in stock_values
            if self._should_buy(stock, price, portfolio_dict)
        ]
        
        # Filter out None transactions and update available cash
        buy_transactions = [t for t in transactions if t is not None]
        for transaction in buy_transactions:
            available_cash -= transaction.get_total_value()
        
        # Generate sell transactions
        sell_transactions = [
            self._create_sell_transaction(stock, price, date, portfolio_dict[stock])
            for stock, price in stock_values
            if self._should_sell(stock, price, portfolio_dict)
        ]
        
        # Filter out None transactions
        sell_transactions = [t for t in sell_transactions if t is not None]
        
        return buy_transactions + sell_transactions
    
    def _should_buy(self, stock: str, price: float, portfolio_dict: Dict) -> bool:
        """Determine if we should buy a stock."""
        if stock in portfolio_dict:
            return False
        
        short_ma = self._calculate_moving_average(stock, self.short_period)
        long_ma = self._calculate_moving_average(stock, self.long_period)
        
        if short_ma is None or long_ma is None:
            return False
        
        signal = self._generate_signal(short_ma, long_ma, price)
        return signal == "BUY"
    
    def _should_sell(self, stock: str, price: float, portfolio_dict: Dict) -> bool:
        """Determine if we should sell a stock."""
        if stock not in portfolio_dict:
            return False
        
        portfolio_item = portfolio_dict[stock]
        if portfolio_item.shares <= 0:
            return False
        
        short_ma = self._calculate_moving_average(stock, self.short_period)
        long_ma = self._calculate_moving_average(stock, self.long_period)
        
        if short_ma is None or long_ma is None:
            return False
        
        signal = self._generate_signal(short_ma, long_ma, price)
        return signal == "SELL"
    
    def _create_buy_transaction(self, stock: str, price: float, date: str, 
                              total_portfolio_value: float, available_cash: float) -> Optional[Transaction]:
        """Create a buy transaction if conditions are met."""
        max_position_value = total_portfolio_value * self.max_position_size
        max_shares = int(max_position_value / price)
        affordable_shares = int(available_cash / price)
        shares_to_buy = min(max_shares, affordable_shares)
        
        if shares_to_buy > 0:
            return Transaction(
                stock=stock,
                date=date,
                price=price,
                shares=shares_to_buy,
                transaction_type=TransactionType.BUY
            )
        return None
    
    def _create_sell_transaction(self, stock: str, price: float, date: str, 
                               portfolio_item) -> Optional[Transaction]:
        """Create a sell transaction if conditions are met."""
        if portfolio_item.shares > 0:
            return Transaction(
                stock=stock,
                date=date,
                price=price,
                shares=portfolio_item.shares,
                transaction_type=TransactionType.SELL
            )
        return None
    
    def _update_price_history(self, stock_values: List[Tuple[str, float]], date: str):
        """Update price history with current prices."""
        for stock, price in stock_values:
            if stock not in self.price_history:
                self.price_history[stock] = []
            self.price_history[stock].append((date, price))
    
    def _calculate_moving_average(self, stock: str, period: int) -> Optional[float]:
        """Calculate moving average for a stock."""
        if stock not in self.price_history:
            return None
        
        prices = [price for _, price in self.price_history[stock]]
        if len(prices) < period:
            return None
        
        # Use the last 'period' prices
        recent_prices = prices[-period:]
        return sum(recent_prices) / len(recent_prices)
    
    def _generate_signal(self, short_ma: float, long_ma: float, current_price: float) -> str:
        """
        Generate buy/sell signal based on moving averages.
        
        Args:
            short_ma: Short-term moving average
            long_ma: Long-term moving average
            current_price: Current stock price
            
        Returns:
            "BUY", "SELL", or "HOLD"
        """
        # Simple crossover strategy
        if short_ma > long_ma and current_price > short_ma:
            return "BUY"
        elif short_ma < long_ma and current_price < short_ma:
            return "SELL"
        else:
            return "HOLD" 