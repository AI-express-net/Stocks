import pytest
from back_tester.models.transaction import Transaction, TransactionType

def test_cash_transaction_type():
    """Test that CASH transaction type exists and works."""
    assert TransactionType.CASH.value == "cash"
    
    cash_tx = Transaction(
        stock="CASH",
        date="2023-01-01", 
        price=1000.0,
        shares=1,
        transaction_type=TransactionType.CASH,
        description="Test cash transaction"
    )
    
    assert cash_tx.stock == "CASH"
    assert cash_tx.price == 1000.0
    assert cash_tx.shares == 1
    assert cash_tx.transaction_type == TransactionType.CASH
    assert cash_tx.description == "Test cash transaction"

def test_cash_transaction_validation():
    """Test cash transaction validation rules."""
    # Valid cash transaction
    cash_tx = Transaction(
        stock="CASH",
        date="2023-01-01",
        price=1000.0, 
        shares=1,
        transaction_type=TransactionType.CASH,
        description="Valid cash transaction"
    )
    assert cash_tx is not None
    
    # Invalid cash transaction - shares must be 1
    with pytest.raises(ValueError):
        Transaction(
            stock="CASH",
            date="2023-01-01",
            price=1000.0,
            shares=2,  # Should be 1 for CASH transactions
            transaction_type=TransactionType.CASH,
            description="Invalid cash transaction"
        )


def test_dividend_transaction_creation():
    """Test that dividend transactions can be created correctly."""
    
    # Create dividend transaction directly
    dividend_tx = Transaction(
        stock="AAPL",
        date="2023-02-10",
        price=23.0,  # $23.00 dividend payment
        shares=1,
        transaction_type=TransactionType.DIVIDEND,
        description="Dividend payment from AAPL"
    )
    
    assert dividend_tx.transaction_type == TransactionType.DIVIDEND
    assert dividend_tx.description == "Dividend payment from AAPL"
    assert dividend_tx.price == 23.0
    assert dividend_tx.stock == "AAPL"
    assert dividend_tx.date == "2023-02-10"

def test_periodic_cash_addition_transaction():
    """Test that periodic cash additions can be logged as cash transactions."""
    from back_tester.config import BackTesterConfig
    from back_tester.enhanced_back_tester import EnhancedBackTester
    from back_tester.models.transaction import TransactionType
    from datetime import date
    
    config = BackTesterConfig()
    config.add_amount = 1000.0
    
    tester = EnhancedBackTester(config)
    
    # Create cash transaction for periodic addition
    cash_tx = tester._create_cash_transaction(
        1000.0, date(2023, 1, 1), "Periodic cash addition"
    )
    
    assert cash_tx.transaction_type == TransactionType.CASH
    assert cash_tx.description == "Periodic cash addition"
    assert cash_tx.price == 1000.0

