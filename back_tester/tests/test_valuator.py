"""
Valuator tests using pytest framework.
"""


from back_tester.tests.mock_valuator import MockValuator as ExampleValuator


class TestValuator:
    """Test valuator interface."""
    
    def test_valuator_creation(self):
        """Test valuator creation."""
        valuator = ExampleValuator()
        assert valuator is not None
    
    def test_calculate_values(self):
        """Test stock value calculation."""
        from datetime import date
        valuator = ExampleValuator()
        values = valuator.calculate_values(["AAPL", "GOOGL"], date(2025, 1, 1))
        
        assert len(values) == 2
        assert all(isinstance(v, tuple) and len(v) == 2 for v in values)
        assert all(isinstance(v[0], str) and isinstance(v[1], float) for v in values)
    
    def test_date_validation(self):
        """Test date validation."""
        from datetime import date
        valuator = ExampleValuator()
        assert valuator.validate_date(date(2025, 1, 1)) == True
        assert valuator.validate_date(None) == False
