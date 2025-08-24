"""
Enhanced Portfolio Management

This module provides an enhanced portfolio management system with performance
tracking, risk metrics, and advanced portfolio management features.
"""

import json
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
# Python 3.6 compatibility - no dataclasses
from enum import Enum

from back_tester.models.portfolio_item import PortfolioItem
from back_tester.models.transaction import Transaction, TransactionType

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics that can be calculated."""
    TOTAL_VALUE = "total_value"
    TOTAL_RETURN = "total_return"
    DAILY_RETURN = "daily_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"


class PerformanceSnapshot:
    """Snapshot of portfolio performance at a specific date."""
    
    def __init__(self, date: date, total_value: float, cash_balance: float, 
                 stock_value: float, total_return: float, daily_return: float,
                 num_positions: int, largest_position: Optional[str], 
                 largest_position_value: float):
        self.date = date
        self.total_value = total_value
        self.cash_balance = cash_balance
        self.stock_value = stock_value
        self.total_return = total_return
        self.daily_return = daily_return
        self.num_positions = num_positions
        self.largest_position = largest_position
        self.largest_position_value = largest_position_value
    
    def __repr__(self):
        return f"PerformanceSnapshot(date={self.date}, total_value={self.total_value})"


class RiskMetrics:
    """Risk metrics for the portfolio."""
    
    def __init__(self, volatility: float, sharpe_ratio: float, max_drawdown: float,
                 var_95: float, beta: float, alpha: float):
        self.volatility = volatility
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.var_95 = var_95  # Value at Risk (95% confidence)
        self.beta = beta
        self.alpha = alpha
    
    def __repr__(self):
        return f"RiskMetrics(volatility={self.volatility:.4f}, sharpe_ratio={self.sharpe_ratio:.4f})"


class EnhancedPortfolio:
    """
    Enhanced portfolio management with performance tracking and risk metrics.
    
    This class extends the basic portfolio functionality with:
    - Performance tracking over time
    - Risk metrics calculation
    - Portfolio rebalancing capabilities
    - Advanced position sizing
    - Performance reporting
    """
    
    def __init__(self, initial_cash: float = 0.0, portfolio_file: str = "portfolio.json"):
        """
        Initialize the enhanced portfolio.
        
        Args:
            initial_cash: Initial cash balance
            portfolio_file: File to save portfolio data
        """
        self.cash_balance = initial_cash
        self.portfolio_items: Dict[str, PortfolioItem] = {}
        self.portfolio_file = portfolio_file
        self.performance_history: List[PerformanceSnapshot] = []
        self.transaction_history: List[Transaction] = []
        
        # Load existing portfolio if available
        self.load_from_file()
    
    def add_cash(self, amount: float) -> bool:
        """
        Add cash to the portfolio.
        
        Args:
            amount: Amount to add (can be negative)
            
        Returns:
            True if successful, False if would result in negative cash
        """
        if self.cash_balance + amount < 0:
            logger.warning(f"Cannot add {amount}: would result in negative cash balance")
            return False
            
        self.cash_balance += amount
        logger.info(f"Added ${amount:.2f} to portfolio. New balance: ${self.cash_balance:.2f}")
        return True
    
    def get_cash_balance(self) -> float:
        """Get current cash balance."""
        return self.cash_balance
    
    def get_portfolio_items(self) -> Dict[str, PortfolioItem]:
        """Get all portfolio items."""
        return self.portfolio_items.copy()
    
    def get_portfolio_item(self, stock_name: str) -> Optional[PortfolioItem]:
        """Get a specific portfolio item."""
        return self.portfolio_items.get(stock_name)
    
    def add_portfolio_item(self, item: PortfolioItem) -> None:
        """Add or update a portfolio item."""
        self.portfolio_items[item.name] = item
        logger.debug(f"Added/updated portfolio item: {item.name}")
    
    def remove_portfolio_item(self, stock_name: str) -> bool:
        """Remove a portfolio item."""
        if stock_name in self.portfolio_items:
            del self.portfolio_items[stock_name]
            logger.debug(f"Removed portfolio item: {stock_name}")
            return True
        return False
    
    def execute_transaction(self, transaction: Transaction) -> bool:
        """
        Execute a transaction and update portfolio.
        
        Args:
            transaction: Transaction to execute
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if transaction.transaction_type == TransactionType.BUY:
                return self._execute_buy_transaction(transaction)
            elif transaction.transaction_type == TransactionType.SELL:
                return self._execute_sell_transaction(transaction)
            else:
                logger.error(f"Unknown transaction type: {transaction.transaction_type}")
                return False
        except Exception as e:
            logger.error(f"Error executing transaction: {str(e)}")
            return False
    
    def _execute_buy_transaction(self, transaction: Transaction) -> bool:
        """Execute a buy transaction."""
        total_cost = transaction.shares * transaction.price
        
        if total_cost > self.cash_balance:
            # logger.warning(f"Insufficient cash for buy: need ${total_cost:.2f}, have ${self.cash_balance:.2f}")
            return False
        
        # Update cash balance
        self.cash_balance -= total_cost
        
        # Update or create portfolio item
        if transaction.stock in self.portfolio_items:
            # Update existing position
            item = self.portfolio_items[transaction.stock]
            item.update_position(transaction.shares, transaction.price, transaction.date)
        else:
            # Create new position
            item = PortfolioItem(
                name=transaction.stock,
                shares=transaction.shares,
                average_price=transaction.price,
                current_value=total_cost,
                date_added=transaction.date,
                last_modified=transaction.date
            )
            self.portfolio_items[transaction.stock] = item
        
        # Add to transaction history
        self.transaction_history.append(transaction)
        
        logger.info(f"Executed buy: {transaction.shares} shares of {transaction.stock} at ${transaction.price:.2f}")
        return True
    
    def _execute_sell_transaction(self, transaction: Transaction) -> bool:
        """Execute a sell transaction."""
        if transaction.stock not in self.portfolio_items:
            logger.warning(f"Cannot sell {transaction.stock}: not in portfolio")
            return False
        
        item = self.portfolio_items[transaction.stock]
        
        if item.shares < transaction.shares:
            logger.warning(f"Insufficient shares for sell: need {transaction.shares}, have {item.shares}")
            return False
        
        # Calculate proceeds
        proceeds = transaction.shares * transaction.price
        
        # Update cash balance
        self.cash_balance += proceeds
        
        # Update portfolio item
        remaining_shares = item.shares - transaction.shares
        if remaining_shares <= 0:
            # Remove the position entirely
            del self.portfolio_items[transaction.stock]
        else:
            # Update the position
            item.shares = remaining_shares
            item.last_modified = transaction.date
        
        # Add to transaction history
        self.transaction_history.append(transaction)
        
        logger.info(f"Executed sell: {transaction.shares} shares of {transaction.stock} at ${transaction.price:.2f}")
        return True
    
    def update_portfolio_values(self, stock_values: List[Tuple[str, float]]) -> None:
        """
        Update portfolio with current stock values.
        
        Args:
            stock_values: List of (stock_symbol, current_price) tuples
        """
        stock_value_dict = dict(stock_values)
        
        for stock_name, item in self.portfolio_items.items():
            if stock_name in stock_value_dict:
                current_price = stock_value_dict[stock_name]
                item.update_current_value(current_price)
        
        logger.debug(f"Updated portfolio values for {len(stock_values)} stocks")
    
    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + stocks)."""
        stock_value = sum(item.current_value for item in self.portfolio_items.values())
        return self.cash_balance + stock_value
    
    def get_stock_value(self) -> float:
        """Calculate total stock value."""
        return sum(item.current_value for item in self.portfolio_items.values())
    
    def take_performance_snapshot(self, current_date: date) -> PerformanceSnapshot:
        """
        Take a snapshot of current portfolio performance.
        
        Args:
            current_date: Date for the snapshot
            
        Returns:
            PerformanceSnapshot object
        """
        total_value = self.get_total_value()
        stock_value = self.get_stock_value()
        
        # Calculate daily return
        daily_return = 0.0
        if self.performance_history:
            prev_snapshot = self.performance_history[-1]
            if prev_snapshot.total_value > 0:
                daily_return = (total_value - prev_snapshot.total_value) / prev_snapshot.total_value
        
        # Find largest position
        largest_position = None
        largest_value = 0.0
        for item in self.portfolio_items.values():
            if item.current_value > largest_value:
                largest_value = item.current_value
                largest_position = item.name
        
        # Calculate total return (assuming initial cash was the starting point)
        total_return = 0.0
        if self.performance_history:
            initial_value = self.performance_history[0].total_value
            if initial_value > 0:
                total_return = (total_value - initial_value) / initial_value
        
        snapshot = PerformanceSnapshot(
            date=current_date,
            total_value=total_value,
            cash_balance=self.cash_balance,
            stock_value=stock_value,
            total_return=total_return,
            daily_return=daily_return,
            num_positions=len(self.portfolio_items),
            largest_position=largest_position,
            largest_position_value=largest_value
        )
        
        self.performance_history.append(snapshot)
        return snapshot
    
    def calculate_risk_metrics(self, risk_free_rate: float = 0.02) -> RiskMetrics:
        """
        Calculate risk metrics for the portfolio.
        
        Args:
            risk_free_rate: Risk-free rate (default 2%)
            
        Returns:
            RiskMetrics object
        """
        if len(self.performance_history) < 2:
            return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(self.performance_history)):
            prev_value = self.performance_history[i-1].total_value
            curr_value = self.performance_history[i].total_value
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        if not daily_returns:
            return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate metrics
        volatility = self._calculate_volatility(daily_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns, risk_free_rate)
        max_drawdown = self._calculate_max_drawdown()
        var_95 = self._calculate_var(daily_returns, 0.95)
        beta = self._calculate_beta(daily_returns)  # Simplified beta calculation
        alpha = self._calculate_alpha(daily_returns, risk_free_rate, beta)
        
        return RiskMetrics(
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            beta=beta,
            alpha=alpha
        )
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility."""
        if not returns:
            return 0.0
        
        import statistics
        mean_return = statistics.mean(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        daily_volatility = variance ** 0.5
        return daily_volatility * (252 ** 0.5)  # Annualized
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if not returns:
            return 0.0
        
        import statistics
        mean_return = statistics.mean(returns)
        volatility = self._calculate_volatility(returns)
        
        if volatility == 0:
            return 0.0
        
        # Annualized return and risk-free rate
        annual_return = mean_return * 252
        annual_risk_free = risk_free_rate
        
        return (annual_return - annual_risk_free) / volatility
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.performance_history:
            return 0.0
        
        peak = self.performance_history[0].total_value
        max_drawdown = 0.0
        
        for snapshot in self.performance_history:
            if snapshot.total_value > peak:
                peak = snapshot.total_value
            else:
                drawdown = (peak - snapshot.total_value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_var(self, returns: List[float], confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if not returns:
            return 0.0
        
        import statistics
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]
    
    def _calculate_beta(self, returns: List[float]) -> float:
        """Calculate beta (simplified - assumes market correlation)."""
        # This is a simplified beta calculation
        # In a real implementation, you would compare against a market index
        return 1.0  # Default to market beta
    
    def _calculate_alpha(self, returns: List[float], risk_free_rate: float, beta: float) -> float:
        """Calculate alpha."""
        if not returns:
            return 0.0
        
        import statistics
        portfolio_return = statistics.mean(returns) * 252
        market_return = risk_free_rate + beta * 0.08  # Assume 8% market return
        alpha = portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
        return alpha
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of portfolio performance."""
        if not self.performance_history:
            return {}
        
        latest = self.performance_history[-1]
        risk_metrics = self.calculate_risk_metrics()
        
        return {
            "total_value": latest.total_value,
            "cash_balance": latest.cash_balance,
            "stock_value": latest.stock_value,
            "total_return": latest.total_return,
            "num_positions": latest.num_positions,
            "volatility": risk_metrics.volatility,
            "sharpe_ratio": risk_metrics.sharpe_ratio,
            "max_drawdown": risk_metrics.max_drawdown,
            "var_95": risk_metrics.var_95
        }
    
    def save_to_file(self) -> None:
        """Save portfolio to file."""
        try:
            # Ensure results directory exists
            results_dir = os.path.dirname(self.portfolio_file)
            if results_dir and not os.path.exists(results_dir):
                os.makedirs(results_dir)
                logger.info(f"Created results directory: {results_dir}")
            
            data = {
                "cash_balance": self.cash_balance,
                "portfolio_items": {name: item.to_dict() for name, item in self.portfolio_items.items()},
                "performance_history": [{
                    "date": snapshot.date.isoformat(),
                    "total_value": snapshot.total_value,
                    "cash_balance": snapshot.cash_balance,
                    "stock_value": snapshot.stock_value,
                    "total_return": snapshot.total_return,
                    "daily_return": snapshot.daily_return,
                    "num_positions": snapshot.num_positions,
                    "largest_position": snapshot.largest_position,
                    "largest_position_value": snapshot.largest_position_value
                } for snapshot in self.performance_history],
                "last_updated": date.today().isoformat()
            }
            
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Portfolio saved to {self.portfolio_file}")
        except Exception as e:
            logger.error(f"Error saving portfolio: {str(e)}")
    
    def load_from_file(self) -> None:
        """Load portfolio from file."""
        try:
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)
            
            self.cash_balance = data.get("cash_balance", 0.0)
            
            # Load portfolio items
            self.portfolio_items.clear()
            for name, item_data in data.get("portfolio_items", {}).items():
                item = PortfolioItem.from_dict(item_data)
                self.portfolio_items[name] = item
            
            # Load performance history
            self.performance_history.clear()
            for snapshot_data in data.get("performance_history", []):
                # Convert date string back to date object
                snapshot_data["date"] = datetime.strptime(snapshot_data["date"], "%Y-%m-%d").date()
                snapshot = PerformanceSnapshot(**snapshot_data)
                self.performance_history.append(snapshot)
            
            logger.info(f"Portfolio loaded from {self.portfolio_file}")
        except FileNotFoundError:
            logger.info(f"No existing portfolio file found. Starting with fresh portfolio.")
        except Exception as e:
            logger.error(f"Error loading portfolio: {str(e)}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get a summary of current portfolio holdings."""
        summary = {
            "cash_balance": self.cash_balance,
            "total_value": self.get_total_value(),
            "num_positions": len(self.portfolio_items),
            "positions": []
        }
        
        for name, item in self.portfolio_items.items():
            position = {
                "stock": name,
                "shares": item.shares,
                "average_price": item.average_price,
                "current_value": item.current_value,
                "gain_loss": item.current_value - (item.shares * item.average_price),
                "gain_loss_pct": ((item.current_value / (item.shares * item.average_price)) - 1) * 100 if item.shares * item.average_price > 0 else 0
            }
            summary["positions"].append(position)
        
        # Sort positions by current value (largest first)
        summary["positions"].sort(key=lambda x: x["current_value"], reverse=True)
        
        return summary 