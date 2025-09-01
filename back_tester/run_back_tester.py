#!/usr/bin/env python3
"""
Command-line interface for running the stock back tester.
"""

import argparse
import os
import sys
from datetime import date

from back_tester.config import BackTesterConfig
from back_tester.enhanced_back_tester import EnhancedBackTester
from back_tester.strategies.moving_average import MovingAverageStrategy
from back_tester.strategies.buy_and_hold import BuyAndHoldStrategy

from back_tester.strategies.magical_formula import MagicalFormulaStrategy


def _cleanup_saved_files(strategy_name):
    """Delete existing saved files for the given strategy."""
    import os
    
    files_to_delete = [
        f"results/{strategy_name}_portfolio.json",
        f"results/{strategy_name}_transactions.json", 
        f"results/{strategy_name}_results.json",
        f"results/{strategy_name}_performance_comparison.png",
        f"results/{strategy_name}_performance_returns.png",
        f"results/{strategy_name}_performance_drawdown.png",
        f"results/{strategy_name}_performance_data.json",
        f"results/{strategy_name}_benchmark_portfolio.json",
        f"results/{strategy_name}_benchmark_results.json",
        f"results/{strategy_name}_benchmark_transactions.json"
    ]
    
    # Also clean up files with dynamic strategy names for specific strategies
    if strategy_name.lower() == 'buy_and_hold':
        files_to_delete.extend([
            f"results/BuyAndHold_SPY_portfolio.json",
            f"results/BuyAndHold_SPY_transactions.json", 
            f"results/BuyAndHold_SPY_results.json",
            f"results/BuyAndHold_SPY_performance_comparison.png",
            f"results/BuyAndHold_SPY_performance_returns.png",
            f"results/BuyAndHold_SPY_performance_drawdown.png",
            f"results/BuyAndHold_SPY_performance_data.json",
            f"results/BuyAndHold_SPY_benchmark_portfolio.json",
            f"results/BuyAndHold_SPY_benchmark_results.json",
            f"results/BuyAndHold_SPY_benchmark_transactions.json"
        ])
    
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")


def create_config_from_args(args):
    """Create configuration from command line arguments."""
    config_dict = {
        'start_cash': args.start_cash,
        'add_amount': args.add_amount,
        'add_amount_frequency_days': args.add_frequency,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'test_frequency_days': args.frequency,
        'portfolio_file': f"results/{args.strategy}_portfolio.json",
        'transactions_file': f"results/{args.strategy}_transactions.json"
    }
    
    # Add stock list file if provided
    if hasattr(args, 'stock_list_file') and args.stock_list_file:
        config_dict['stock_list_file'] = args.stock_list_file
    
    # Add benchmark instrument if provided
    if hasattr(args, 'benchmark') and args.benchmark:
        config_dict['benchmark_instrument'] = args.benchmark
    
    return BackTesterConfig(**config_dict)


def create_strategy(strategy_name, **kwargs):
    """Create strategy based on name."""
    if strategy_name.lower() == 'moving_average':
        return MovingAverageStrategy(
            short_period=kwargs.get('short_period', 10),
            long_period=kwargs.get('long_period', 30),
            max_position_size=kwargs.get('max_position_size', 0.2)
        )
    elif strategy_name.lower() == 'buy_and_hold':
        return BuyAndHoldStrategy(
            max_position_size=kwargs.get('max_position_size', 0.2)
        )
    elif strategy_name.lower() == 'sp500_buy_and_hold':
        return BuyAndHoldStrategy(
            target_stocks=['SPY'],
            max_position_size=kwargs.get('max_position_size', 1.0),
            allow_existing_positions=True
        )
    elif strategy_name.lower() == 'magical_formula':
        return MagicalFormulaStrategy(
            portfolio_size=kwargs.get('portfolio_size', 25),
            max_position_size=kwargs.get('max_position_size', 0.05),
            rebalance_frequency_days=kwargs.get('rebalance_frequency_days', 365)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_back_tester(args):
    """Run the back tester with specified parameters."""
    
    print("=== Stock Back Tester ===\n")
    
    # Delete existing saved files as per instructions
    _cleanup_saved_files(args.strategy)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Create strategy
    strategy_kwargs = {
        'max_position_size': args.max_position_size
    }
    
    if args.strategy.lower() == 'moving_average':
        strategy_kwargs.update({
            'short_period': args.short_period,
            'long_period': args.long_period
        })
    elif args.strategy.lower() == 'magical_formula':
        strategy_kwargs.update({
            'portfolio_size': args.portfolio_size,
            'rebalance_frequency_days': args.rebalance_frequency_days
        })
    
    strategy = create_strategy(args.strategy, **strategy_kwargs)
    
    # Create back tester
    back_tester = EnhancedBackTester(config)
    back_tester.set_strategy(strategy)
    
    print(f"Configuration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Start Cash: ${config.start_cash:,.2f}")
    print(f"  Add Amount: ${config.add_amount:,.2f}")
    print(f"  Add Frequency: {config.add_amount_frequency_days} days")
    print(f"  Date Range: {config.start_date} to {config.end_date}")
    print(f"  Test Frequency: {config.test_frequency_days} days")
    print(f"  Stock List File: {config.stock_list_file}")
    print(f"  Benchmark: {config.benchmark_instrument}")
    
    if args.strategy.lower() == 'moving_average':
        print(f"  Moving Average: {args.short_period}/{args.long_period} days")
    elif args.strategy.lower() == 'magical_formula':
        print(f"  Portfolio Size: {args.portfolio_size} stocks")
        print(f"  Rebalance Frequency: {args.rebalance_frequency_days} days")
    print(f"  Max Position Size: {args.max_position_size * 100}%")
    print()
    
    print("Running back test...")
    results = back_tester.run()
    
    # Display results
    print("\n=== Results ===")
    if 'error' in results:
        print(f"Error: {results['error']}")
        return results
    
    print(f"Final Portfolio Value: ${results['performance']['final_value']:,.2f}")
    print(f"Total Return: ${results['performance']['total_return']:,.2f} ({results['performance']['total_return_pct']:.2f}%)")
    print(f"Sharpe Ratio: {results['risk_metrics']['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['risk_metrics']['max_drawdown']:.2%}")
    print(f"Volatility: {results['risk_metrics']['volatility']:.2%}")
    print(f"Total Transactions: {results['trading_metrics']['total_transactions']}")
    print(f"Successful Transactions: {results['trading_metrics']['successful_transactions']}")
    print(f"Success Rate: {results['trading_metrics']['success_rate']:.1f}%")
    
    # Display dividend information
    if 'dividend_metrics' in results:
        print(f"Dividend Transactions: {results['dividend_metrics']['dividend_transactions_count']}")
        print(f"Total Dividends Received: ${results['dividend_metrics']['total_dividends_received']:.2f}")
        
        # Show top dividend-paying stocks
        dividends_by_stock = results['dividend_metrics']['dividends_by_stock']
        if dividends_by_stock:
            print("Top Dividend-Paying Stocks:")
            sorted_dividends = sorted(dividends_by_stock.items(), key=lambda x: x[1], reverse=True)
            for stock, amount in sorted_dividends[:5]:  # Show top 5
                print(f"  {stock}: ${amount:.2f}")
    
    # Export results
    strategy_name = back_tester.strategy.get_strategy_name() if hasattr(back_tester, 'strategy') and back_tester.strategy else args.strategy
    results_file = f"results/{strategy_name}_results.json"
    back_tester.export_results(results_file)
    results_dir = os.path.abspath(os.path.dirname(results_file))
    print(f"\nResults exported to: {results_file}")
    print(f"All result files saved in directory: {results_dir}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Stock Back Tester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Moving Average strategy
  python run_back_tester.py --strategy moving_average --start-cash 10000 --add-amount 1000

  # Run Buy and Hold strategy
  python run_back_tester.py --strategy buy_and_hold --start-cash 5000 --add-amount 500

  # Run Magical Formula strategy
  python run_back_tester.py --strategy magical_formula --start-cash 10000 --add-amount 1000

  # Custom Moving Average parameters
  python run_back_tester.py --strategy moving_average --short-period 5 --long-period 20 --max-position-size 0.15

  # Custom Magical Formula parameters
  python run_back_tester.py --strategy magical_formula --portfolio-size 30 --rebalance-frequency-days 180

  # Use custom stock list file
  python run_back_tester.py --strategy moving_average --stock-list-file my_stocks.json --start-cash 10000
        """
    )

    testConfig = BackTesterConfig()

    # Strategy options
    parser.add_argument('--strategy', choices=['moving_average', 'buy_and_hold', 'magical_formula'], 
                       default='moving_average', help='Trading strategy to use')
    
    # Configuration options
    parser.add_argument('--start-cash', type=float, default=testConfig.start_cash,
                       help=f'Starting cash amount (default: {testConfig.start_cash})')
    parser.add_argument('--add-amount', type=float, default=testConfig.add_amount,
                       help=f'Amount to add each frequency period (default: {testConfig.add_amount})')
    parser.add_argument('--add-frequency', type=int, default=testConfig.add_amount_frequency_days,
                       help=f'Days between cash additions (default: {testConfig.add_amount_frequency_days})')
    parser.add_argument('--start-date', default='2023-01-01',
                       help='Start date (YYYY-MM-DD, default: 2023-01-01)')
    parser.add_argument('--end-date', default='2023-03-31',
                       help='End date (YYYY-MM-DD, default: 2023-03-31)')
    parser.add_argument('--frequency', type=int, default=1,
                       help='Test frequency in days (default: 1)')
    
    # Strategy-specific options
    parser.add_argument('--short-period', type=int, default=10,
                       help='Short period for moving average (default: 10)')
    parser.add_argument('--long-period', type=int, default=30,
                       help='Long period for moving average (default: 30)')
    parser.add_argument('--max-position-size', type=float, default=0.2,
                       help='Maximum position size as fraction (default: 0.2)')
    parser.add_argument('--portfolio-size', type=int, default=25,
                       help='Number of stocks to hold in portfolio (default: 25)')
    parser.add_argument('--rebalance-frequency-days', type=int, default=365,
                       help='Days between portfolio rebalancing (default: 365)')

    # Stock list file option
    parser.add_argument('--stock-list-file', type=str, default=testConfig.stock_list_file,
                       help=f'Path to JSON file containing list of stocks (default: uses {testConfig.stock_list_file})')
    
    # Trading options
    parser.add_argument('--valuator', type=str, default=testConfig.valuator,
                       help=f'Stock valuator to use  (default: uses {testConfig.valuator})')
    parser.add_argument('--benchmark', type=str, default=testConfig.benchmark_instrument,
                       help=f'Benchmark instrument (default: {testConfig.benchmark_instrument})')

    args = parser.parse_args()
    
    try:
        run_back_tester(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 