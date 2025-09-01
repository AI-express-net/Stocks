"""
Test to verify SP500 benchmark restructuring.
This test runs the back tester with both main strategy and benchmark strategy
as SP500 buy_and_hold, which should produce identical results.
"""


from back_tester.config import BackTesterConfig
from back_tester.enhanced_back_tester import EnhancedBackTester
from back_tester.strategies.buy_and_hold import BuyAndHoldStrategy


def test_sp500_benchmark_identical():
    """Test that SP500 buy_and_hold strategy produces identical results for main and benchmark."""

    # Clean up any existing test files
    import os
    test_files = [
        'results/test_sp500_portfolio.json',
        'results/test_sp500_transactions.json',
        'results/test_sp500_portfolio_benchmark.json'
    ]
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass  # Ignore errors if file doesn't exist

    # Create configuration for 3-month test period
    # Use SP500_index.json for both main strategy and benchmark - they should be identical
    config = BackTesterConfig(
        start_cash=10000.0,
        add_amount=0.0,
        add_amount_frequency_days=30,
        start_date='2023-01-01',
        end_date='2023-03-31',  # 3-month period (Jan-Mar 2023)
        test_frequency_days=1,
        stock_list_file='tests/SP500_index.json',  # Both strategies use same stock list
        portfolio_file='results/test_sp500_portfolio.json',
        transactions_file='results/test_sp500_transactions.json',
        strategy='sp500_buy_and_hold',
        benchmark_instrument='SPY'
    )

    # Create SP500 buy_and_hold strategy for main strategy
    main_strategy = BuyAndHoldStrategy(target_stocks=['SPY'], max_position_size=1.0, allow_existing_positions=True)

    # Create back tester
    back_tester = EnhancedBackTester(config)
    back_tester.set_strategy(main_strategy)

    # Run back test
    results = back_tester.run()

    # Check for errors
    assert 'error' not in results, f"Back test failed with error: {results.get('error')}"

    # Get benchmark comparison
    benchmark_comparison = back_tester.get_benchmark_comparison()
    
    # Extract values
    main_value = results['performance']['final_value']
    main_return = results['performance']['total_return']
    benchmark_value = benchmark_comparison['benchmark']['total_value']
    benchmark_return = benchmark_comparison['benchmark']['total_return']

    # Check if results are identical
    value_difference = abs(main_value - benchmark_value)
    return_difference = abs(main_return - benchmark_return)

    # Tolerance for floating point differences
    tolerance = 50.0  # $50.00 tolerance for small timing differences

    assert value_difference <= tolerance, f"Value difference ${value_difference:.2f} exceeds tolerance ${tolerance:.2f}"
    assert return_difference <= tolerance, f"Return difference ${return_difference:.2f} exceeds tolerance ${tolerance:.2f}"
    
    # Additional assertions to verify the results make sense
    assert main_value > 10000.0, f"Main strategy final value ${main_value:.2f} should be greater than initial cash $10000.00"
    assert benchmark_value > 10000.0, f"Benchmark final value ${benchmark_value:.2f} should be greater than initial cash $10000.00"
    assert main_return > 0, f"Main strategy return ${main_return:.2f} should be positive"
    assert benchmark_return > 0, f"Benchmark return ${benchmark_return:.2f} should be positive"
