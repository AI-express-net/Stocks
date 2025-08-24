#!/usr/bin/env python3
"""
Example script to run the back tester with Moving Average strategy.
"""

from datetime import date, timedelta

from back_tester.config import BackTesterConfig
from back_tester.enhanced_back_tester import EnhancedBackTester
from back_tester.strategies.moving_average import MovingAverageStrategy
from back_tester.mock_valuator import MockValuator


def run_moving_average_example():
    """Run the back tester with Moving Average strategy."""
    
    print("=== Stock Back Tester - Moving Average Strategy Example ===\n")
    
    # Create configuration
    config = BackTesterConfig({
        'start_cash': 10000.0,  # Start with $10,000
        'add_amount': 1000.0,   # Add $1,000 each iteration
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'test_frequency_days': 7,  # Weekly testing
        'portfolio_file': 'back_tester/results/moving_average_portfolio.json',
        'transactions_file': 'back_tester/results/moving_average_transactions.json'
    })
    
    print(f"Configuration:")
    print(f"  Start Cash: ${config.start_cash:,.2f}")
    print(f"  Add Amount: ${config.add_amount:,.2f}")
    print(f"  Date Range: {config.start_date} to {config.end_date}")
    print(f"  Test Frequency: {config.test_frequency_days} days")
    print()
    
    # Create components
    valuator = MockValuator()  # Use mock valuator for testing
    strategy = MovingAverageStrategy(
        short_period=10,    # 10-day moving average
        long_period=30,     # 30-day moving average
        max_position_size=0.2  # Max 20% of portfolio in any single stock
    )
    
    print(f"Strategy: Moving Average ({strategy.short_period}/{strategy.long_period})")
    print(f"Max Position Size: {strategy.max_position_size * 100}%")
    print()
    
    # Create and run back tester
    back_tester = EnhancedBackTester(config)
    back_tester.set_strategy(strategy)
    
    print("Running back test...")
    results = back_tester.run()
    
    # Display results
    print("\n=== Results ===")
    print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['risk_metrics']['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['risk_metrics']['max_drawdown']:.2%}")
    print(f"Volatility: {results['risk_metrics']['volatility']:.2%}")
    print(f"Total Transactions: {results['trading_metrics']['total_transactions']}")
    print(f"Successful Transactions: {results['trading_metrics']['successful_transactions']}")
    
    # Export results
    back_tester.export_results('back_tester/results/moving_average_results.json')
    print(f"\nResults exported to: back_tester/results/moving_average_results.json")
    
    return results


def run_with_real_data():
    """Run with real API data (requires API key)."""
    
    print("=== Stock Back Tester - Moving Average Strategy with Real Data ===\n")
    
    # Create configuration
    config = BackTesterConfig({
        'start_cash': 10000.0,
        'add_amount': 500.0,
        'start_date': '2024-01-01',
        'end_date': '2024-06-30',
        'test_frequency_days': 7,
        'portfolio_file': 'back_tester/results/real_moving_average_portfolio.json',
        'transactions_file': 'back_tester/results/real_moving_average_transactions.json'
    })
    
    # Import real valuator (requires API key setup)
    try:
        from real_valuator import RealValuator
        valuator = RealValuator()
        print("Using RealValuator with live API data")
    except ImportError:
        print("RealValuator not available, using MockValuator")
        valuator = MockValuator()
    
    strategy = MovingAverageStrategy(
        short_period=5,     # 5-day moving average
        long_period=20,     # 20-day moving average
        max_position_size=0.15  # Max 15% of portfolio in any single stock
    )
    
    back_tester = EnhancedBackTester(config)
    back_tester.set_strategy(strategy)
    
    print("Running back test with real data...")
    results = back_tester.run()
    
    print(f"\nFinal Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Moving Average Back Tester')
    parser.add_argument('--real-data', action='store_true', 
                       help='Use real API data (requires API key setup)')
    
    args = parser.parse_args()
    
    if args.real_data:
        run_with_real_data()
    else:
        run_moving_average_example() 