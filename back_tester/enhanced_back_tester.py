"""
Enhanced Back Tester Implementation

This module provides an enhanced back tester that integrates with real stock data
and includes advanced portfolio management features.
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any

from back_tester.config import BackTesterConfig
from back_tester.enhanced_portfolio import EnhancedPortfolio, PerformanceSnapshot
from back_tester.real_valuator import RealValuator
from back_tester.strategy import Strategy
from back_tester.models.transaction import Transaction, TransactionType
from back_tester.performance_tracker import PerformanceTracker
from back_tester.models.benchmark_portfolio import BenchmarkPortfolio
from back_tester.performance_graph import PerformanceGraph
from back_tester.strategies.benchmark_strategy import BenchmarkStrategy
from back_tester.dividend_handler import DividendHandler

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
        self.benchmark_portfolio = BenchmarkPortfolio(
            instrument=config.benchmark_instrument
        )
        self.benchmark_portfolio.initialize_with_cash(config.start_cash)  # Start with same cash
        self.performance_graph = PerformanceGraph()
        
        # Dividend handling
        self.dividend_handler = DividendHandler(config.dividend_file)
        
        # Load stock list
        self.stock_list = self._load_stock_list()
        
        logger.info(f"Enhanced back tester initialized with {len(self.stock_list)} stocks")
    
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
                dividend_amount = self.dividend_handler.check_dividends(self.current_date, self.portfolio)
                if dividend_amount > 0:
                    self.dividend_handler.add_dividend_cash(self.portfolio, dividend_amount)
                    logger.info(f"Added ${dividend_amount:.2f} in dividends to portfolio on {self.current_date}")
                    
                    # Log dividend payment as transaction
                    dividend_transaction = self._create_cash_transaction(
                        dividend_amount, self.current_date, "Dividend payment"
                    )
                    self.transaction_log.append(dividend_transaction)
                
                # Get current stock values
                stock_values = self.valuator.calculate_values(self.stock_list, self.current_date)
                
                if not stock_values:
                    logger.warning(f"No stock values available for {self.current_date}")
                    self._advance_date()
                    continue
                
                # Update portfolio with current values
                self.portfolio.update_portfolio_values(stock_values)
                
                # Generate transactions from strategy
                portfolio_items = list(self.portfolio.get_portfolio_items().values())
                transactions = self.strategy.generate_transactions(
                    portfolio_items, stock_values, self.current_date, self.portfolio.get_cash_balance()
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
                    else:
                        failed_transactions += 1
                        logger.debug(f"Failed to execute buy transaction: {transaction}")
                
                # Take performance snapshot
                snapshot = self.portfolio.take_performance_snapshot(self.current_date)
                self.performance_snapshots.append(snapshot)
                
                # Update benchmark portfolio - mirror transactions
                self._update_benchmark_portfolio(transactions, self.current_date)
                
                # Record performance for comparison
                main_value = self.portfolio.get_total_value()
                benchmark_value = self.benchmark_portfolio.get_current_value()
                self.performance_tracker.record_performance(self.current_date, main_value, benchmark_value)
                
                # Print period reporting if there were transactions
                if transactions:
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
        
        logger.info(f"Back test completed. {successful_transactions} successful, {failed_transactions} failed transactions")
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
            
            # Load existing transactions
            existing_transactions = []
            try:
                with open(transactions_file, 'r') as f:
                    existing_transactions = json.load(f)
            except FileNotFoundError:
                pass
            
            # Add new transactions
            new_transactions = [t.to_dict() for t in self.transaction_log]
            all_transactions = existing_transactions + new_transactions
            
            # Save all transactions
            with open(transactions_file, 'w') as f:
                json.dump(all_transactions, f, indent=2, default=str)
            
            logger.debug(f"Saved {len(new_transactions)} new transactions")
            
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
        performance_summary = self.portfolio.get_performance_summary()
        risk_metrics = self.portfolio.calculate_risk_metrics()
        portfolio_summary = self.portfolio.get_portfolio_summary()
        
        # Calculate additional metrics
        initial_value = self.performance_snapshots[0].total_value
        final_value = self.performance_snapshots[-1].total_value
        total_return_pct = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
        
        # Calculate trading metrics
        total_transactions = successful_transactions + failed_transactions
        success_rate = (successful_transactions / total_transactions * 100) if total_transactions > 0 else 0
        
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
            results = self._generate_results(0, 0)  # We'll get actual counts from transaction log
            
            # Count actual transactions
            successful = len([t for t in self.transaction_log if t.transaction_type.value in ["BUY", "SELL"]])
            cash_transactions = len([t for t in self.transaction_log if t.transaction_type.value == "CASH"])
            failed = 0  # Failed transactions are logged but not stored in transaction_log
            
            results["trading_metrics"]["successful_transactions"] = successful
            results["trading_metrics"]["cash_transactions"] = cash_transactions
            results["trading_metrics"]["failed_transactions"] = failed
            results["trading_metrics"]["total_transactions"] = successful + cash_transactions + failed
            
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
    
    def _update_benchmark_portfolio(self, transactions: List[Transaction], date: datetime) -> None:
        """Update benchmark portfolio by mirroring main portfolio transactions."""
        # Create benchmark strategy to generate mirror transactions
        benchmark_strategy = BenchmarkStrategy(self.config.benchmark_instrument)
        
        # Generate mirror transactions for the same dollar amounts
        benchmark_transactions = benchmark_strategy.mirror_transactions(transactions, date)
        
        # Execute benchmark transactions
        for transaction in benchmark_transactions:
            try:
                self.benchmark_portfolio.execute_transaction(transaction)
            except ValueError as e:
                logger.debug(f"Failed to execute benchmark transaction: {e}")
        
        # Update SP500 price (simplified - in real implementation would fetch actual SP500 data)
        # For now, use a simple growth model
        sp500_growth_rate = 0.0001  # 0.01% daily growth (simplified)
        current_sp500_price = self.benchmark_portfolio.sp500_price
        new_sp500_price = current_sp500_price * (1 + sp500_growth_rate)
        self.benchmark_portfolio.update_sp500_price(new_sp500_price)
        
        # Record performance snapshot
        self.benchmark_portfolio.record_performance(date)
    
    def _generate_performance_graphs(self) -> None:
        """Generate performance comparison graphs."""
        try:
            # Get performance data
            performance_data = self.performance_tracker.get_performance_data()
            
            if not performance_data['dates']:
                logger.warning("No performance data available for graphing")
                return
            
            # Create graphs
            output_prefix = f"{self.config.strategy}_performance"
            self.performance_graph.create_all_graphs(performance_data, output_prefix)
            
            # Save performance data
            performance_file = f"{self.config.strategy}_performance_data.json"
            self.performance_tracker.save_performance_data(performance_file)
            
            # Save benchmark files
            benchmark_portfolio_file = f"{self.config.strategy}_benchmark_portfolio.json"
            benchmark_results_file = f"{self.config.strategy}_benchmark_results.json"
            benchmark_transactions_file = f"{self.config.strategy}_benchmark_transactions.json"
            
            self.benchmark_portfolio.save_to_file(benchmark_portfolio_file)
            
            # Save benchmark results
            benchmark_results = {
                'benchmark_info': {
                    'instrument': self.config.benchmark_instrument,
                    'start_date': self.config.start_date,
                    'end_date': self.config.end_date,
                    'initial_cash': self.config.start_cash,
                    'add_amount': self.config.add_amount,
                    'add_frequency_days': self.config.add_amount_frequency_days
                },
                'performance': self.benchmark_portfolio.get_performance_summary(),
                'transactions': [t.to_dict() for t in self.benchmark_portfolio.transactions]
            }
            
            with open(benchmark_results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            
            # Save benchmark transactions
            with open(benchmark_transactions_file, 'w') as f:
                json.dump([t.to_dict() for t in self.benchmark_portfolio.transactions], f, indent=2, default=str)
            
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