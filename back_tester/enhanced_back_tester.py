"""
Enhanced Back Tester Implementation

This module provides an enhanced back tester that integrates with real stock data
and includes advanced portfolio management features.
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import List, Dict, Any

from back_tester.config import BackTesterConfig
from back_tester.enhanced_portfolio import EnhancedPortfolio, PerformanceSnapshot
from back_tester.real_valuator import RealValuator
from back_tester.strategy import Strategy
from back_tester.models.transaction import Transaction, TransactionType
from back_tester.performance_tracker import PerformanceTracker

from back_tester.performance_graph import PerformanceGraph
from back_tester.strategies.buy_and_hold import BuyAndHoldStrategy

from stocks.fmp_stock import Stock



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBackTester:
    """
    Enhanced back tester with real data integration and advanced features.
    
    This back tester provides:
    - Real stock data integration
    - Performance tracking and analysis
    - Risk metrics calculation
    - Enhanced error handling
    - Detailed reporting capabilities
    """
    
    def __init__(self, config: BackTesterConfig):
        """
        Initialize the enhanced back tester.
        
        Args:
            config: Back tester configuration
        """
        self.config = config
        self.portfolio = EnhancedPortfolio(
            initial_cash=config.start_cash,
            portfolio_file=config.portfolio_file
        )
        self.valuator = RealValuator()
        self.strategy = None  # Will be set by set_strategy
        # Convert string dates to datetime objects
        from datetime import datetime
        self.current_date = datetime.strptime(config.start_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(config.end_date, '%Y-%m-%d').date()
        self.test_frequency_days = config.test_frequency_days
        self.add_amount = config.add_amount
        self.add_amount_frequency_days = config.add_amount_frequency_days
        self.last_cash_addition_date = self.current_date
        
        # Performance tracking
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.transaction_log: List[Transaction] = []
        
        # Benchmark tracking
        self.performance_tracker = PerformanceTracker()
        self.benchmark_portfolio = EnhancedPortfolio(
            initial_cash=config.start_cash,
            portfolio_file=config.portfolio_file.replace('.json', '_benchmark.json')
        )  # Use regular portfolio for benchmark
        # Create benchmark strategy using BuyAndHoldStrategy with SPY as target stock
        self.benchmark_strategy = BuyAndHoldStrategy(
            target_stocks=['SPY'], 
            max_position_size=1.0,
            allow_existing_positions=True  # Allow buying more SPY if already in portfolio
        )
        self.performance_graph = PerformanceGraph()
        
        # Dividend handling - use FMP API for live dividend data
        # Convert string dates to date objects for dividend handling
        self.test_start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d').date()
        self.test_end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d').date()
        
        # Stock cache for dividend data
        self.stock_cache: Dict[str, Stock] = {}
        
        # Load stock list - both main strategy and benchmark use the same stock list
        self.stock_list = self._load_stock_list()
        
        logger.info(f"Enhanced back tester initialized with {len(self.stock_list)} stocks")
    
    def _clear_previous_results(self) -> None:
        """Clear previous result files at the start of each run."""
        import os
        import glob
        
        try:
            # Clear results directory
            results_dir = "results"
            if os.path.exists(results_dir):
                # Remove all JSON files in results directory
                json_files = glob.glob(os.path.join(results_dir, "*.json"))
                for json_file in json_files:
                    try:
                        os.remove(json_file)
                        logger.debug(f"Removed previous result file: {json_file}")
                    except Exception as e:
                        logger.warning(f"Could not remove {json_file}: {e}")
                
                # Remove all PNG files in results directory
                png_files = glob.glob(os.path.join(results_dir, "*.png"))
                for png_file in png_files:
                    try:
                        os.remove(png_file)
                        logger.debug(f"Removed previous result file: {png_file}")
                    except Exception as e:
                        logger.warning(f"Could not remove {png_file}: {e}")
                
                logger.info(f"Cleared {len(json_files)} JSON files and {len(png_files)} PNG files from previous run")
            else:
                logger.info("Results directory does not exist, creating it")
                os.makedirs(results_dir, exist_ok=True)
                
        except Exception as e:
            logger.warning(f"Error clearing previous results: {e}")
    
    def set_strategy(self, strategy: Strategy) -> None:
        """
        Set the trading strategy to use.
        
        Args:
            strategy: Trading strategy instance
        """
        self.strategy = strategy
        logger.info(f"Strategy set: {strategy.__class__.__name__}")
    
    def run(self) -> Dict:
        """
        Run the enhanced back test.
        
        Returns:
            Dictionary containing back test results and performance metrics
        """
        if self.strategy is None:
            raise ValueError("Strategy must be set before running back test")
        
        # Clear previous results at the start of each run
        self._clear_previous_results()
        
        # Reset portfolio objects to clear any old data
        self.portfolio.reset_portfolio(self.config.start_cash)
        self.benchmark_portfolio.reset_portfolio(self.config.start_cash)
        
        # Clear transaction log to start fresh
        self.transaction_log.clear()
        
        # Dividend data will be loaded on-demand when checking for dividends
        logger.debug("Dividend data will be loaded on-demand using Stock class")
        
        logger.info(f"Starting enhanced back test from {self.current_date} to {self.end_date}")
        
        # Take initial snapshot
        initial_snapshot = self.portfolio.take_performance_snapshot(self.current_date)
        self.performance_snapshots.append(initial_snapshot)
        
        iteration_count = 0
        successful_transactions = 0
        failed_transactions = 0
        
        while self.current_date <= self.end_date:
            try:
                iteration_count += 1
                logger.debug(f"Iteration {iteration_count}: Processing date {self.current_date}")
                
                # Add cash if configured and frequency period has elapsed
                if self.add_amount > 0:
                    days_since_last_addition = (self.current_date - self.last_cash_addition_date).days
                    if days_since_last_addition >= self.add_amount_frequency_days:
                        self.portfolio.add_cash(self.add_amount)
                        self.benchmark_portfolio.add_cash(self.add_amount)  # Mirror cash addition
                        self.last_cash_addition_date = self.current_date
                        logger.info(f"Added ${self.add_amount} to portfolio and benchmark on {self.current_date}")
                        
                        # Log periodic cash addition as transaction
                        cash_transaction = self._create_cash_transaction(
                            self.add_amount, self.current_date, "Periodic cash addition"
                        )
                        self.transaction_log.append(cash_transaction)
                
                # Check for dividends and add to portfolio
                dividends_paid = self._check_dividends(self.current_date, self.portfolio)
                if dividends_paid:
                    total_dividends_today = sum(dividends_paid.values())
                    self.portfolio.add_cash(total_dividends_today)
                    logger.debug(f"Added ${total_dividends_today:.2f} in dividends to portfolio on {self.current_date}")
                    
                    # Log each dividend payment as a separate transaction
                    for stock, amount in dividends_paid.items():
                        dividend_transaction = self._create_dividend_transaction(
                            stock, amount, self.current_date
                        )
                        self.transaction_log.append(dividend_transaction)
                
                # Get current stock values
                stock_values = self.valuator.calculate_values(self.stock_list, self.current_date)
                
                if not stock_values:
                    logger.warning(f"No stock values available for {self.current_date}")
                    self._advance_date()
                    continue
                
                # Debug: Log stock values
                logger.debug(f"Stock values for {self.current_date}: {stock_values}")
                
                # Store stock values for benchmark access
                self.stock_values = stock_values
                
                # Update portfolio with current values
                self.portfolio.update_portfolio_values(stock_values)
                
                # Debug: Log portfolio values after update
                logger.debug(f"Portfolio after update - Cash: ${self.portfolio.get_cash_balance():.2f}, Stock Value: ${self.portfolio.get_stock_value():.2f}, Total: ${self.portfolio.get_total_value():.2f}")
                
                # Generate transactions from strategy
                portfolio_items = list(self.portfolio.get_portfolio_items().values())
                transactions = self.strategy.generate_transactions(
                    portfolio_items, stock_values, self.current_date.strftime('%Y-%m-%d'), self.portfolio.get_cash_balance()
                )
                
                # Execute transactions (sell first, then buy)
                sell_transactions = [t for t in transactions if t.transaction_type == TransactionType.SELL]
                buy_transactions = [t for t in transactions if t.transaction_type == TransactionType.BUY]
                
                # Execute sell transactions first
                for transaction in sell_transactions:
                    if self.portfolio.execute_transaction(transaction):
                        successful_transactions += 1
                        self.transaction_log.append(transaction)
                    else:
                        failed_transactions += 1
                        logger.debug(f"Failed to execute sell transaction: {transaction}")
                
                # Execute buy transactions
                for transaction in buy_transactions:
                    if self.portfolio.execute_transaction(transaction):
                        successful_transactions += 1
                        self.transaction_log.append(transaction)
                        
                        # Dividend data will be loaded on-demand when checking for dividends
                        logger.debug(f"Newly acquired stock: {transaction.stock} - dividend data will be loaded on-demand")
                    else:
                        failed_transactions += 1
                        logger.debug(f"Failed to execute buy transaction: {transaction}")
                
                # Take performance snapshot
                snapshot = self.portfolio.take_performance_snapshot(self.current_date)
                self.performance_snapshots.append(snapshot)
                
                # Update benchmark portfolio using SP500 buy-and-hold strategy
                self._update_benchmark_portfolio(self.current_date)
                
                # Record performance for comparison
                main_value = self.portfolio.get_total_value()
                benchmark_value = self.benchmark_portfolio.get_total_value()  # Use same method as main portfolio
                
                # Debug: Log both portfolio values
                logger.debug(f"Main portfolio: ${main_value:.2f}, Benchmark portfolio: ${benchmark_value:.2f}")
                
                self.performance_tracker.record_performance(self.current_date, main_value, benchmark_value)
                
                # Print period reporting - show both main and benchmark values every day
                main_return = main_value - self.config.start_cash
                benchmark_return = benchmark_value - self.config.start_cash
                print(f"Date: {self.current_date} - Main: ${main_value:.2f} (${main_return:+.2f}) | Benchmark: ${benchmark_value:.2f} (${benchmark_return:+.2f})")
                
                # Save portfolio periodically
                if iteration_count % 10 == 0:
                    self.portfolio.save_to_file()
                    self._save_transactions()
                
                # Advance to next date
                self._advance_date()
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration_count}: {str(e)}")
                self._advance_date()
                continue
        
        # Final save
        self.portfolio.save_to_file()
        self._save_transactions()
        
        # Generate performance graphs
        self._generate_performance_graphs()
        
        # Generate results
        results = self._generate_results(successful_transactions, failed_transactions)
        
        # Count dividend transactions for logging
        dividend_transactions = len([t for t in self.transaction_log if t.transaction_type == TransactionType.DIVIDEND])
        total_dividends = sum(t.price * t.shares for t in self.transaction_log if t.transaction_type == TransactionType.DIVIDEND)
        
        logger.info(f"Back test completed. {successful_transactions} successful, {failed_transactions} failed transactions, {dividend_transactions} dividend transactions (${total_dividends:.2f} total)")
        return results
    
    def _advance_date(self) -> None:
        """Advance the current date by the test frequency."""
        self.current_date += timedelta(days=self.test_frequency_days)
    
    def _load_stock_list(self) -> List[str]:
        """Load the stock list from the configured file."""
        try:
            stock_list_file = self.config.stock_list_file
            logger.info(f"Load stocks from {stock_list_file}")
            with open(stock_list_file, 'r') as f:
                data = json.load(f)
            
            # Handle different possible formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "stocks" in data:
                return data["stocks"]
            else:
                logger.warning(f"Unexpected stock list format in {stock_list_file}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading stock list: {str(e)}")
            return []
    

    
    def _save_transactions(self) -> None:
        """Save transaction log to file."""
        try:
            transactions_file = self.config.transactions_file
            
            # Save all current transactions (overwrite the file)
            all_transactions = [t.to_dict() for t in self.transaction_log]
            
            # Save all transactions
            with open(transactions_file, 'w') as f:
                json.dump(all_transactions, f, indent=2, default=str)
            
            logger.debug(f"Saved {len(all_transactions)} transactions")
            
        except Exception as e:
            logger.error(f"Error saving transactions: {str(e)}")
    
    def _generate_results(self, successful_transactions: int, failed_transactions: int) -> Dict:
        """
        Generate comprehensive back test results.
        
        Args:
            successful_transactions: Number of successful transactions
            failed_transactions: Number of failed transactions
            
        Returns:
            Dictionary containing all results and metrics
        """
        if not self.performance_snapshots:
            return {"error": "No performance data available"}
        
        # Get performance summary
        self.portfolio.get_performance_summary()
        risk_metrics = self.portfolio.calculate_risk_metrics()
        portfolio_summary = self.portfolio.get_portfolio_summary()
        
        # Calculate additional metrics
        initial_value = self.performance_snapshots[0].total_value
        final_value = self.performance_snapshots[-1].total_value
        total_return_pct = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
        
        # Calculate trading metrics
        total_transactions = successful_transactions + failed_transactions
        success_rate = (successful_transactions / total_transactions * 100) if total_transactions > 0 else 0
        
        # Calculate dividend metrics
        dividend_transactions = [t for t in self.transaction_log if t.transaction_type == TransactionType.DIVIDEND]
        total_dividends_received = sum(t.price * t.shares for t in dividend_transactions)
        dividend_count = len(dividend_transactions)
        
        # Group dividends by stock
        dividends_by_stock = {}
        for t in dividend_transactions:
            stock = t.stock
            amount = t.price * t.shares
            if stock not in dividends_by_stock:
                dividends_by_stock[stock] = 0
            dividends_by_stock[stock] += amount
        
        # Calculate date range
        start_date = self.performance_snapshots[0].date
        end_date = self.performance_snapshots[-1].date
        days_tested = (end_date - start_date).days
        
        results = {
            "back_test_info": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days_tested": days_tested,
                "test_frequency_days": self.test_frequency_days,
                "initial_cash": self.config.start_cash,
                "add_amount": self.add_amount,
                "num_stocks": len(self.stock_list)
            },
            "performance": {
                "initial_value": initial_value,
                "final_value": final_value,
                "total_return": final_value - initial_value,
                "total_return_pct": total_return_pct,
                "cash_balance": self.portfolio.get_cash_balance(),
                "stock_value": self.portfolio.get_stock_value(),
                "num_positions": len(self.portfolio.get_portfolio_items())
            },
            "risk_metrics": {
                "volatility": risk_metrics.volatility,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "max_drawdown": risk_metrics.max_drawdown,
                "var_95": risk_metrics.var_95,
                "beta": risk_metrics.beta,
                "alpha": risk_metrics.alpha
            },
            "trading_metrics": {
                "total_transactions": total_transactions,
                "successful_transactions": successful_transactions,
                "failed_transactions": failed_transactions,
                "success_rate": success_rate
            },
            "dividend_metrics": {
                "total_dividends_received": total_dividends_received,
                "dividend_transactions_count": dividend_count,
                "dividends_by_stock": dividends_by_stock
            },
            "portfolio_summary": portfolio_summary,
            "performance_history": [
                {
                    "date": snapshot.date.isoformat(),
                    "total_value": snapshot.total_value,
                    "cash_balance": snapshot.cash_balance,
                    "stock_value": snapshot.stock_value,
                    "total_return": snapshot.total_return,
                    "daily_return": snapshot.daily_return,
                    "num_positions": snapshot.num_positions
                }
                for snapshot in self.performance_snapshots
            ]
        }
        
        return results
    
    def get_performance_chart_data(self) -> Dict:
        """
        Get data formatted for performance charts.
        
        Returns:
            Dictionary with chart-ready data
        """
        if not self.performance_snapshots:
            return {}
        
        dates = [snapshot.date.isoformat() for snapshot in self.performance_snapshots]
        values = [snapshot.total_value for snapshot in self.performance_snapshots]
        returns = [snapshot.total_return for snapshot in self.performance_snapshots]
        
        return {
            "dates": dates,
            "values": values,
            "returns": returns,
            "cash_balance": [snapshot.cash_balance for snapshot in self.performance_snapshots],
            "stock_value": [snapshot.stock_value for snapshot in self.performance_snapshots]
        }
    
    def export_results(self, filename: str) -> None:
        """
        Export back test results to a JSON file.
        
        Args:
            filename: Output filename
        """
        try:
            # Ensure results directory exists
            results_dir = os.path.dirname(filename)
            if results_dir and not os.path.exists(results_dir):
                os.makedirs(results_dir)
                logger.info(f"Created results directory: {results_dir}")
            
            results = self._generate_results(0, 0)  # We'll get actual counts from transaction log
            
            # Count actual transactions
            successful = len([t for t in self.transaction_log if t.transaction_type.value in ["BUY", "SELL"]])
            cash_transactions = len([t for t in self.transaction_log if t.transaction_type.value == "CASH"])
            dividend_transactions = len([t for t in self.transaction_log if t.transaction_type.value == "DIVIDEND"])
            failed = 0  # Failed transactions are logged but not stored in transaction_log
            
            results["trading_metrics"]["successful_transactions"] = successful
            results["trading_metrics"]["cash_transactions"] = cash_transactions
            results["trading_metrics"]["dividend_transactions"] = dividend_transactions
            results["trading_metrics"]["failed_transactions"] = failed
            results["trading_metrics"]["total_transactions"] = successful + cash_transactions + dividend_transactions + failed
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
    
    def get_portfolio_status(self) -> Dict:
        """
        Get current portfolio status.
        
        Returns:
            Dictionary with current portfolio information
        """
        # Handle both string and date objects for current_date
        if hasattr(self.current_date, 'isoformat'):
            current_date_str = self.current_date.isoformat()
        else:
            current_date_str = str(self.current_date)
            
        return {
            "current_date": current_date_str,
            "cash_balance": self.portfolio.get_cash_balance(),
            "total_value": self.portfolio.get_total_value(),
            "num_positions": len(self.portfolio.get_portfolio_items()),
            "portfolio_summary": self.portfolio.get_portfolio_summary()
        }
    
    def reset(self) -> None:
        """Reset the back tester to initial state."""
        self.current_date = datetime.strptime(self.config.start_date, '%Y-%m-%d').date()
        self.portfolio = EnhancedPortfolio(
            initial_cash=self.config.start_cash,
            portfolio_file=self.config.portfolio_file
        )
        self.last_cash_addition_date = self.current_date
        self.strategy = None  # Clear the strategy
        self.performance_snapshots.clear()
        self.transaction_log.clear()
        self.valuator.clear_cache()
        
        logger.info("Back tester reset to initial state")
    
    def _update_benchmark_portfolio(self, date: date) -> None:
        """Update benchmark portfolio using SP500 buy-and-hold strategy."""
        # Check for dividends and add to benchmark portfolio (same logic as main portfolio)
        dividends_paid = self._check_dividends(date, self.benchmark_portfolio)
        if dividends_paid:
            total_dividends_today = sum(dividends_paid.values())
            self.benchmark_portfolio.add_cash(total_dividends_today)
            logger.debug(f"Added ${total_dividends_today:.2f} in dividends to benchmark portfolio on {date}")
            
            # Log each dividend payment as a separate transaction in benchmark portfolio
            for stock, amount in dividends_paid.items():
                dividend_transaction = self._create_dividend_transaction(
                    stock, amount, date
                )
                self.benchmark_portfolio.transaction_history.append(dividend_transaction)
        
        # Get current portfolio items and stock values for benchmark
        benchmark_portfolio_items = list(self.benchmark_portfolio.get_portfolio_items().values())
        available_cash = self.benchmark_portfolio.get_cash_balance()
        
        # Generate benchmark transactions using SP500 buy-and-hold strategy
        # Pass the same stock values list as the main strategy
        benchmark_transactions = self.benchmark_strategy.generate_transactions(
            benchmark_portfolio_items, 
            self.stock_values,  # Use same stock values list as main strategy
            date.strftime('%Y-%m-%d'), 
            available_cash
        )
        
        # Execute benchmark transactions (sell first, then buy)
        sell_transactions = [t for t in benchmark_transactions if t.transaction_type == TransactionType.SELL]
        buy_transactions = [t for t in benchmark_transactions if t.transaction_type == TransactionType.BUY]
        
        # Execute sell transactions first
        for transaction in sell_transactions:
            try:
                if self.benchmark_portfolio.execute_transaction(transaction):
                    logger.debug(f"Executed benchmark sell: {transaction.shares} shares of {transaction.stock} at ${transaction.price}")
                else:
                    logger.debug(f"Failed to execute benchmark sell transaction: {transaction}")
            except ValueError as e:
                logger.debug(f"Failed to execute benchmark sell transaction: {e}")
        
        # Execute buy transactions
        for transaction in buy_transactions:
            try:
                if self.benchmark_portfolio.execute_transaction(transaction):
                    logger.debug(f"Executed benchmark buy: {transaction.shares} shares of {transaction.stock} at ${transaction.price}")
                else:
                    logger.debug(f"Failed to execute benchmark buy transaction: {transaction}")
            except ValueError as e:
                logger.debug(f"Failed to execute benchmark buy transaction: {e}")
        
        # Update benchmark portfolio with current stock values (this was missing!)
        self.benchmark_portfolio.update_portfolio_values(self.stock_values)
        
        # Take performance snapshot for benchmark
        benchmark_snapshot = self.benchmark_portfolio.take_performance_snapshot(date)
        self.benchmark_portfolio.performance_history.append(benchmark_snapshot)
    
    def _generate_performance_graphs(self) -> None:
        """Generate performance comparison graphs."""
        try:
            # Ensure results directory exists
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                logger.info(f"Created results directory: {results_dir}")
            
            # Get performance data
            performance_data = self.performance_tracker.get_performance_data()
            
            if not performance_data['dates']:
                logger.warning("No performance data available for graphing")
                return
            
            # Create graphs
            strategy_name = self.strategy.get_strategy_name() if hasattr(self, 'strategy') and self.strategy else self.config.strategy
            output_prefix = f"results/{strategy_name}_performance"
            self.performance_graph.create_all_graphs(performance_data, output_prefix)
            
            # Save performance data
            performance_file = f"results/{strategy_name}_performance_data.json"
            self.performance_tracker.save_performance_data(performance_file)
            
            # Save benchmark files
            f"results/{strategy_name}_benchmark_portfolio.json"
            benchmark_results_file = f"results/{strategy_name}_benchmark_results.json"
            benchmark_transactions_file = f"results/{strategy_name}_benchmark_transactions.json"
            
            # Save benchmark portfolio to its configured file
            self.benchmark_portfolio.save_to_file()
            
            # Save benchmark results
            benchmark_results = {
                'benchmark_info': {
                    'strategy': 'SP500_Buy_And_Hold',
                    'start_date': self.config.start_date,
                    'end_date': self.config.end_date,
                    'initial_cash': self.config.start_cash,
                    'add_amount': self.config.add_amount,
                    'add_frequency_days': self.config.add_amount_frequency_days
                },
                'performance': self.benchmark_portfolio.get_performance_summary(),
                'transactions': [t.to_dict() for t in self.benchmark_portfolio.transaction_history]
            }
            
            with open(benchmark_results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            
            # Save benchmark transactions
            with open(benchmark_transactions_file, 'w') as f:
                json.dump([t.to_dict() for t in self.benchmark_portfolio.transaction_history], f, indent=2, default=str)
            
            logger.info("Performance graphs and data generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating performance graphs: {str(e)}")
    
    def get_benchmark_comparison(self) -> Dict[str, Any]:
        """Get benchmark comparison summary."""
        performance_summary = self.performance_tracker.get_performance_summary()
        benchmark_summary = self.benchmark_portfolio.get_performance_summary()
        
        return {
            'main_strategy': performance_summary,
            'benchmark': benchmark_summary,
            'comparison': {
                'outperformed_benchmark': performance_summary.get('outperformed_benchmark', False),
                'excess_return': performance_summary.get('excess_return', 0.0),
                'excess_return_percentage': performance_summary.get('excess_return_percentage', 0.0)
            }
        }
    
    def _create_cash_transaction(self, amount: float, date: date, description: str) -> Transaction:
        """
        Create a cash transaction for logging.
        
        Args:
            amount: Cash amount to add
            date: Transaction date
            description: Description of the cash transaction
            
        Returns:
            Transaction object representing the cash transaction
        """
        return Transaction(
            stock="CASH",
            date=date.strftime('%Y-%m-%d'),
            price=amount,
            shares=1,
            transaction_type=TransactionType.CASH,
            description=description
        )
    
    def _create_dividend_transaction(self, stock: str, amount: float, date: date) -> Transaction:
        """
        Create a dividend transaction for logging.
        
        Args:
            stock: Stock symbol that paid the dividend
            amount: Dividend amount received
            date: Transaction date
            
        Returns:
            Transaction object representing the dividend transaction
        """
        return Transaction(
            stock=stock,
            date=date.strftime('%Y-%m-%d'),
            price=amount,
            shares=1,
            transaction_type=TransactionType.DIVIDEND,
            description=f"Dividend payment from {stock}"
        )
    
    def _check_dividends(self, current_date: date, portfolio) -> Dict[str, float]:
        """
        Check for dividends on the current date using the Stock class data.
        
        Args:
            current_date: Date to check for dividends
            portfolio: Portfolio object with holdings
            
        Returns:
            Dictionary mapping stock symbols to dividend amounts
        """
        dividends_paid = {}
        portfolio_items = portfolio.get_portfolio_items()
        
        for stock_symbol, portfolio_item in portfolio_items.items():
            try:
                # Get or create Stock instance
                if stock_symbol not in self.stock_cache:
                    self.stock_cache[stock_symbol] = Stock(stock_symbol)
                stock = self.stock_cache[stock_symbol]
                
                # Fetch dividend data using the Stock class (this will use MongoDB caching)
                stock.fetch_stock_data('historical_dividends')
                
                # Get the dividend data from the stock entity
                dividend_data = stock.historical_dividends.get_data()
                
                if not dividend_data or not isinstance(dividend_data, dict):
                    continue
                    
                # Extract historical dividend records from the response
                historical_dividends = dividend_data.get('historical', [])
                if not historical_dividends or not isinstance(historical_dividends, list):
                    continue
                    
                # Check for dividends on the current date
                for dividend in historical_dividends:
                    try:
                        dividend_date = datetime.strptime(dividend.get('date', ''), '%Y-%m-%d').date()
                        
                        if dividend_date == current_date:
                            shares = portfolio_item.shares
                            dividend_amount = shares * float(dividend.get('dividend', 0))
                            
                            if dividend_amount > 0:
                                dividends_paid[stock_symbol] = dividend_amount
                                logger.debug(f"Dividend for {stock_symbol}: {shares} shares × ${dividend.get('dividend', 0)} = ${dividend_amount:.2f}")
                                
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Error processing dividend for {stock_symbol}: {e}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Error checking dividends for {stock_symbol}: {e}")
                continue
        
        return dividends_paid 