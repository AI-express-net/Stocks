"""
Moving Average strategy implementation.
"""

from typing import List, Tuple, Dict
from strategy import Strategy
from models.transaction import Transaction, TransactionType


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy.
    
    This strategy uses moving averages to generate buy/sell signals.
    It's a simplified implementation for demonstration purposes.
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
        self.price_history = {}  # Store historical prices for MA calculation
    
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
        
        transactions = []
        
        # Update price history
        self._update_price_history(stock_values, date)
        
        # Get current portfolio value
        portfolio_value = self.get_portfolio_value(portfolio_items, stock_values)
        total_portfolio_value = portfolio_value + available_cash
        
        # Create portfolio lookup
        portfolio_dict = {item.name: item for item in portfolio_items}
        
        for stock, current_price in stock_values:
            # Calculate moving averages
            short_ma = self._calculate_moving_average(stock, self.short_period)
            long_ma = self._calculate_moving_average(stock, self.long_period)
            
            if short_ma is None or long_ma is None:
                continue  # Not enough data for MA calculation
            
            # Generate signals
            signal = self._generate_signal(short_ma, long_ma, current_price)
            
            if signal == "BUY" and stock not in portfolio_dict:
                # Buy signal for stock not in portfolio
                max_position_value = total_portfolio_value * self.max_position_size
                max_shares = int(max_position_value / current_price)
                affordable_shares = int(available_cash / current_price)
                shares_to_buy = min(max_shares, affordable_shares)
                
                if shares_to_buy > 0:
                    transaction = Transaction(
                        stock=stock,
                        date=date,
                        price=current_price,
                        shares=shares_to_buy,
                        transaction_type=TransactionType.BUY
                    )
                    transactions.append(transaction)
                    available_cash -= transaction.get_total_value()
            
            elif signal == "SELL" and stock in portfolio_dict:
                # Sell signal for stock in portfolio
                portfolio_item = portfolio_dict[stock]
                if portfolio_item.shares > 0:
                    transaction = Transaction(
                        stock=stock,
                        date=date,
                        price=current_price,
                        shares=portfolio_item.shares,
                        transaction_type=TransactionType.SELL
                    )
                    transactions.append(transaction)
        
        return transactions
    
    def _update_price_history(self, stock_values: List[Tuple[str, float]], date: str):
        """Update price history with current prices."""
        for stock, price in stock_values:
            if stock not in self.price_history:
                self.price_history[stock] = []
            self.price_history[stock].append((date, price))
    
    def _calculate_moving_average(self, stock: str, period: int) -> float:
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