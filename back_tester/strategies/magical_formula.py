"""
Magical Formula Strategy Implementation

Based on Joel Greenblatt's "The Little Book That Beats the Market"
Strategy focuses on high earnings yield and high return on capital.
"""

import logging
from typing import List, Tuple, Dict, Optional
from datetime import date, timedelta
import random

from strategy import Strategy
from models.transaction import Transaction, TransactionType

logger = logging.getLogger(__name__)


class MagicalFormulaStrategy(Strategy):
    """
    Implements the Magical Formula investing strategy.
    
    The strategy:
    1. Ranks stocks by earnings yield (EBIT/Enterprise Value)
    2. Ranks stocks by return on capital (EBIT/(Net Fixed Assets + Working Capital))
    3. Combines rankings and invests in top 20-30 companies
    4. Rebalances annually
    """
    
    def __init__(self, 
                 max_position_size: float = 0.05,  # 5% max per position
                 portfolio_size: int = 25,         # Number of stocks to hold
                 min_market_cap: float = 50e6,    # $50M minimum market cap
                 rebalance_frequency_days: int = 365,  # Annual rebalancing
                 exclude_utilities: bool = True,
                 exclude_financials: bool = True,
                 exclude_adrs: bool = True):
        """
        Initialize the Magical Formula strategy.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            portfolio_size: Number of stocks to hold in portfolio
            min_market_cap: Minimum market capitalization in dollars
            rebalance_frequency_days: Days between rebalancing
            exclude_utilities: Whether to exclude utility stocks
            exclude_financials: Whether to exclude financial stocks
            exclude_adrs: Whether to exclude ADRs (foreign companies)
        """
        super().__init__()
        self.max_position_size = max_position_size
        self.portfolio_size = portfolio_size
        self.min_market_cap = min_market_cap
        self.rebalance_frequency_days = rebalance_frequency_days
        self.exclude_utilities = exclude_utilities
        self.exclude_financials = exclude_financials
        self.exclude_adrs = exclude_adrs
        
        # Track last rebalancing date
        self.last_rebalance_date = None
        self.current_positions = {}  # Track current holdings
        
        logger.info(f"Magical Formula Strategy initialized with portfolio size: {portfolio_size}")
    
    def generate_transactions(self, portfolio_items: List, stock_values: List[Tuple[str, float]],
                            date, available_cash: float) -> List[Transaction]:
        """
        Generate transactions based on the Magical Formula strategy.
        
        Args:
            portfolio_items: Current portfolio holdings
            stock_values: List of (stock, price) tuples
            date: Current date (can be string or datetime.date)
            available_cash: Available cash for investment
            
        Returns:
            List of transactions to execute
        """
        transactions = []
        # Handle date parameter - can be string or datetime.date
        if isinstance(date, str):
            from datetime import date as date_type
            current_date = date_type.fromisoformat(date)
        else:
            current_date = date
        
        # Check if it's time to rebalance
        should_rebalance = self._should_rebalance(current_date)
        
        if should_rebalance:
            logger.info(f"Rebalancing portfolio on {date}")
            # Sell all current positions
            sell_transactions = self._generate_sell_transactions(portfolio_items, stock_values, date)
            transactions.extend(sell_transactions)
            
            # Buy new positions based on rankings
            buy_transactions = self._generate_buy_transactions(stock_values, date, available_cash)
            transactions.extend(buy_transactions)
            
            self.last_rebalance_date = current_date
        else:
            # Normal monthly accumulation (2-3 positions per month)
            if len(portfolio_items) < self.portfolio_size:
                buy_transactions = self._generate_buy_transactions(stock_values, date, available_cash)
                transactions.extend(buy_transactions)
        
        return transactions
    
    def _should_rebalance(self, current_date: date) -> bool:
        """Check if it's time to rebalance the portfolio."""
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance >= self.rebalance_frequency_days
    
    def _generate_sell_transactions(self, portfolio_items: List, stock_values: List[Tuple[str, float]], 
                                  date: str) -> List[Transaction]:
        """Generate sell transactions for all current positions."""
        transactions = []
        
        for item in portfolio_items:
            if item.shares > 0:
                # Find current price
                current_price = self._get_stock_price(item.stock, stock_values)
                if current_price:
                    transaction = Transaction(
                        stock=item.stock,
                        date=date,
                        price=current_price,
                        shares=item.shares,
                        transaction_type=TransactionType.SELL
                    )
                    transactions.append(transaction)
                    logger.info(f"Selling {item.shares} shares of {item.stock} at ${current_price:.2f}")
        
        return transactions
    
    def _generate_buy_transactions(self, stock_values: List[Tuple[str, float]], 
                                 date: str, available_cash: float) -> List[Transaction]:
        """Generate buy transactions based on Magical Formula rankings."""
        transactions = []
        
        # Get ranked stocks
        ranked_stocks = self._get_ranked_stocks(stock_values)
        
        # Calculate position size
        position_size = min(available_cash / self.portfolio_size, 
                          available_cash * self.max_position_size)
        
        # Buy top ranked stocks
        for stock, price in ranked_stocks[:self.portfolio_size]:
            if price > 0 and position_size >= price:
                shares_to_buy = int(position_size / price)
                if shares_to_buy > 0:
                    transaction = Transaction(
                        stock=stock,
                        date=date,
                        price=price,
                        shares=shares_to_buy,
                        transaction_type=TransactionType.BUY
                    )
                    transactions.append(transaction)
                    logger.info(f"Buying {shares_to_buy} shares of {stock} at ${price:.2f}")
        
        return transactions
    
    def _get_ranked_stocks(self, stock_values: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Rank stocks according to the Magical Formula criteria.
        
        In a real implementation, this would:
        1. Calculate earnings yield (EBIT/Enterprise Value)
        2. Calculate return on capital (EBIT/(Net Fixed Assets + Working Capital))
        3. Rank by both metrics
        4. Filter by market cap, exclude utilities/financials/ADRs
        
        For now, we'll use a simplified ranking based on price volatility
        and simulated fundamental metrics.
        """
        if not stock_values:
            return []
        
        # Simulate ranking based on price and volatility
        # In a real implementation, this would use actual fundamental data
        ranked_stocks = []
        
        for stock, price in stock_values:
            # Skip if price is invalid
            if price <= 0:
                continue
            
            # Simulate earnings yield and return on capital
            # In reality, these would come from fundamental data
            earnings_yield = self._simulate_earnings_yield(stock, price)
            return_on_capital = self._simulate_return_on_capital(stock, price)
            
            # Combined ranking score (higher is better)
            ranking_score = earnings_yield * 0.6 + return_on_capital * 0.4
            
            ranked_stocks.append((stock, price, ranking_score))
        
        # Sort by ranking score (highest first)
        ranked_stocks.sort(key=lambda x: x[2], reverse=True)
        
        # Return just stock and price
        return [(stock, price) for stock, price, _ in ranked_stocks]
    
    def _simulate_earnings_yield(self, stock: str, price: float) -> float:
        """
        Simulate earnings yield calculation.
        
        In reality, this would be: EBIT / Enterprise Value
        For simulation, we'll use a random but consistent value.
        """
        # Use stock name hash for consistent "random" values
        seed = hash(stock) % 1000
        random.seed(seed)
        
        # Simulate earnings yield between 5% and 25%
        base_yield = random.uniform(0.05, 0.25)
        
        # Adjust based on price (lower price = higher yield)
        price_factor = max(0.5, 1.0 - (price / 1000.0))
        
        return base_yield * price_factor
    
    def _simulate_return_on_capital(self, stock: str, price: float) -> float:
        """
        Simulate return on capital calculation.
        
        In reality, this would be: EBIT / (Net Fixed Assets + Working Capital)
        For simulation, we'll use a random but consistent value.
        """
        # Use stock name hash for consistent "random" values
        seed = hash(stock) % 1000
        random.seed(seed)
        
        # Simulate return on capital between 10% and 40%
        base_roc = random.uniform(0.10, 0.40)
        
        # Adjust based on price (lower price = higher ROC)
        price_factor = max(0.6, 1.0 - (price / 800.0))
        
        return base_roc * price_factor
    
    def _get_stock_price(self, stock: str, stock_values: List[Tuple[str, float]]) -> Optional[float]:
        """Get the current price for a stock."""
        for s, price in stock_values:
            if s == stock:
                return price
        return None
    
    def validate_inputs(self, portfolio_items: List, stock_values: List[Tuple[str, float]], 
                       date, available_cash: float):
        """Validate strategy inputs."""
        if available_cash < 0:
            raise ValueError("Available cash cannot be negative")
        
        if not stock_values:
            raise ValueError("No stock values provided")
        
        # Validate date format
        try:
            if isinstance(date, str):
                from datetime import date as date_type
                date_type.fromisoformat(date)
            elif not isinstance(date, date_type):
                raise ValueError("Date must be string or datetime.date object")
        except ValueError:
            raise ValueError("Invalid date format")
    
    def __str__(self):
        return f"MagicalFormulaStrategy(portfolio_size={self.portfolio_size}, max_position_size={self.max_position_size})"
    
    def __repr__(self):
        return self.__str__() 