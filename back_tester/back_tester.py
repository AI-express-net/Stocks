"""
Main back tester engine.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Tuple
from config import BackTesterConfig
from portfolio import Portfolio
from valuator import Valuator
from strategy import Strategy
from models.transaction import Transaction


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
        self.portfolio = None
        self.transactions = []
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")
    
    def run(self):
        """Run the back tester."""
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
                else:
                    print("No transactions generated")
                
                        # Save portfolio
        self.portfolio.save_to_file(self.config.portfolio_file)
                
            except Exception as e:
                print(f"Error on {date_str}: {e}")
                # Continue to next iteration
            
            # Move to next date
            current_date += timedelta(days=frequency_days)
        
        print(f"\nBack testing completed after {iteration} iterations")
        print(f"Final portfolio value: ${self.portfolio.get_total_value():.2f}")
    
    def _load_transactions(self):
        """Load existing transactions from file."""
        transactions_file = self.config.transactions_file
        if os.path.exists(transactions_file):
            try:
                with open(transactions_file, 'r') as f:
                    transactions_data = json.load(f)
                self.transactions = [Transaction.from_dict(t) for t in transactions_data]
                print(f"Loaded {len(self.transactions)} existing transactions")
            except Exception as e:
                print(f"Warning: Could not load transactions: {e}")
                self.transactions = []
        else:
            self.transactions = []
    
    def _save_transaction(self, transaction: Transaction):
        """Save transaction to file."""
        self.transactions.append(transaction)
        
        transactions_file = self.config.transactions_file
        try:
            os.makedirs(os.path.dirname(transactions_file), exist_ok=True)
            with open(transactions_file, 'w') as f:
                json.dump([t.to_dict() for t in self.transactions], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save transaction: {e}")
    
    def _execute_transaction(self, transaction: Transaction):
        """Execute a single transaction."""
        try:
            self.portfolio.execute_transaction(transaction)
            self._save_transaction(transaction)
            print(f"Executed: {transaction}")
        except ValueError as e:
            print(f"Transaction failed: {e}")
        except Exception as e:
            print(f"Unexpected error executing transaction: {e}")
    
    def _load_stock_list(self) -> List[str]:
        """Load stock list from file."""
        stock_list_file = self.config.stock_list_file
        try:
            with open(stock_list_file, 'r') as f:
                data = json.load(f)
            return data.get('stocks', [])
        except Exception as e:
            print(f"Warning: Could not load stock list: {e}")
            return []
    
    def get_results(self) -> dict:
        """Get back testing results."""
        return {
            'final_portfolio_value': self.portfolio.get_total_value(),
            'total_transactions': len(self.transactions),
            'final_cash': self.portfolio.cash,
            'final_positions': len(self.portfolio.portfolio_items),
            'portfolio_summary': self.portfolio.to_dict()
        } 