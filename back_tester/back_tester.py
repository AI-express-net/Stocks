"""
Main back tester engine.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from pathlib import Path
from config import BackTesterConfig
from portfolio import Portfolio
from valuator import Valuator
from strategy import Strategy
from models.transaction import Transaction
from utils import save_json, load_json, append_json


class BackTester:
    """Main back tester engine."""
    
    def __init__(self, config: BackTesterConfig, valuator: Valuator, strategy: Strategy):
        """
        Initialize back tester.
        
        Args:
            config: Configuration settings
            valuator: Stock valuator implementation
            strategy: Trading strategy implementation
        """
        self.config = config
        self.valuator = valuator
        self.strategy = strategy
        self.portfolio: Optional[Portfolio] = None
        self.transactions: List[Transaction] = []
        self._is_running = False
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")
    
    def __enter__(self):
        """Context manager entry."""
        self._cleanup_files()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._is_running:
            self._save_final_state()
    
    def _cleanup_files(self):
        """Clean up output files before starting."""
        files_to_clean = [
            Path(self.config.portfolio_file),
            Path(self.config.transactions_file)
        ]
        
        for file_path in files_to_clean:
            if file_path.exists():
                file_path.unlink()
    
    def _save_final_state(self):
        """Save final portfolio state."""
        if self.portfolio:
            self.portfolio.save_to_file(self.config.portfolio_file)
    
    def run(self):
        """Run the back tester."""
        self._is_running = True
        print("Starting back tester...")
        
        # Load portfolio
        self.portfolio = Portfolio.load_from_file(self.config.portfolio_file)
        print(f"Loaded portfolio: {self.portfolio}")
        
        # Load existing transactions
        self._load_transactions()
        
        # Get stock list
        stock_list = self._load_stock_list()
        print(f"Loaded {len(stock_list)} stocks")
        
        # Initialize date
        current_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        frequency_days = self.config.test_frequency_days
        
        print(f"Running from {current_date.date()} to {end_date.date()}")
        print(f"Test frequency: {frequency_days} day(s)")
        
        # Main loop
        iteration = 0
        while current_date <= end_date:
            iteration += 1
            date_str = current_date.strftime('%Y-%m-%d')
            
            print(f"\nIteration {iteration}: {date_str}")
            print(f"Portfolio value: ${self.portfolio.get_total_value():.2f}")
            
            try:
                # Add cash amount
                add_amount = self.config.add_amount
                if add_amount > 0:
                    self.portfolio.add_cash(add_amount)
                    print(f"Added ${add_amount:.2f} to portfolio")
                
                # Get stock values for current date
                stock_values = self.valuator.calculate_values(stock_list, date_str)
                if not stock_values:
                    print(f"No stock values available for {date_str}, skipping...")
                    current_date += timedelta(days=frequency_days)
                    continue
                
                # Generate transactions from strategy
                transactions = self.strategy.generate_transactions(
                    self.portfolio.portfolio_items,
                    stock_values,
                    date_str,
                    self.portfolio.cash
                )
                
                if transactions:
                    print(f"Generated {len(transactions)} transactions")
                    
                    # Execute transactions (sell first, then buy)
                    sell_transactions = [t for t in transactions if t.transaction_type.value == 'sell']
                    buy_transactions = [t for t in transactions if t.transaction_type.value == 'buy']
                    
                    # Execute sell transactions first
                    for transaction in sell_transactions:
                        self._execute_transaction(transaction)
                    
                    # Execute buy transactions
                    for transaction in buy_transactions:
                        self._execute_transaction(transaction)
                
                # Save portfolio state
                self.portfolio.save_to_file(self.config.portfolio_file)
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                break
            
            # Update date
            current_date += timedelta(days=frequency_days)
        
        print(f"\nBack testing completed after {iteration} iterations")
        print(f"Final portfolio value: ${self.portfolio.get_total_value():.2f}")
        self._is_running = False
    
    def _load_transactions(self):
        """Load existing transactions from file."""
        try:
            transactions_file = Path(self.config.transactions_file)
            if transactions_file.exists():
                data = load_json(transactions_file, default=[])
                self.transactions = [Transaction.from_dict(t) for t in data]
                print(f"Loaded {len(self.transactions)} existing transactions")
        except Exception as e:
            print(f"Warning: Could not load transactions: {e}")
            self.transactions = []
    
    def _save_transaction(self, transaction: Transaction):
        """Save transaction to file."""
        try:
            append_json(transaction.to_dict(), Path(self.config.transactions_file))
            self.transactions.append(transaction)
        except Exception as e:
            print(f"Warning: Could not save transaction: {e}")
    
    def _execute_transaction(self, transaction: Transaction):
        """Execute a transaction."""
        try:
            self.portfolio.execute_transaction(transaction)
            self._save_transaction(transaction)
            print(f"Executed: {transaction}")
        except Exception as e:
            print(f"Failed to execute transaction: {e}")
    
    def _load_stock_list(self) -> List[str]:
        """Load stock list from file."""
        try:
            stock_file = Path(self.config.stock_list_file)
            if stock_file.exists():
                data = load_json(stock_file, default=[])
                if isinstance(data, list):
                    return data
                else:
                    print("Warning: Stock list file is not a list, using empty list")
                    return []
            else:
                print(f"Warning: Stock list file not found: {stock_file}")
                return []
        except Exception as e:
            print(f"Warning: Could not load stock list: {e}")
            return []
    
    def get_results(self) -> dict:
        """Get back testing results."""
        if not self.portfolio:
            return {}
        
        return {
            'portfolio': self.portfolio.to_dict(),
            'transactions': [t.to_dict() for t in self.transactions],
            'total_transactions': len(self.transactions),
            'final_value': self.portfolio.get_total_value()
        } 