"""
Integration tests using pytest framework.
"""


from back_tester.config import BackTesterConfig
from back_tester.portfolio import Portfolio
from back_tester.tests.mock_valuator import MockValuator as ExampleValuator
from back_tester.strategies.buy_and_hold import BuyAndHoldStrategy
from back_tester.models.portfolio_item import PortfolioItem


class TestIntegration:
    """Test basic integration."""
    
    def test_component_creation(self):
        """Test that all components can be created."""
        config = BackTesterConfig(
            start_cash=10000.0,
            add_amount=0.0,
            start_date='2025-01-01',
            end_date='2025-01-02',
            test_frequency_days=1
        )
        
        valuator = ExampleValuator()
        strategy = BuyAndHoldStrategy(target_stocks=["AAPL", "GOOGL"])
        
        assert config is not None
        assert valuator is not None
        assert strategy is not None
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        portfolio = Portfolio(cash=10000.0)
        item = PortfolioItem("AAPL", 100, 150.0, 15500.0)
        portfolio.add_portfolio_item(item)
        
        stock_values = [("AAPL", 160.0)]
        total_value = portfolio.get_total_value(stock_values)
        
        assert total_value == 26000.0  # 10000 cash + (100 * 160)
