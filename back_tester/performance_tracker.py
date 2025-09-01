from typing import Dict, Any
from datetime import datetime


class PerformanceTracker:
    """Track performance over time for both main and benchmark portfolios."""
    
    def __init__(self):
        self.main_performance = []
        self.benchmark_performance = []
        self.dates = []
        self.performance_snapshots = []
    
    def record_performance(self, date: datetime, main_value: float, benchmark_value: float):
        """Record performance for a given date."""
        date_str = date.isoformat() if isinstance(date, datetime) else str(date)
        self.dates.append(date_str)
        self.main_performance.append(main_value)
        self.benchmark_performance.append(benchmark_value)
        
        # Create performance snapshot
        snapshot = {
            'date': date_str,
            'main_value': main_value,
            'benchmark_value': benchmark_value,
            'main_return': main_value - self.main_performance[0] if self.main_performance else 0.0,
            'benchmark_return': benchmark_value - self.benchmark_performance[0] if self.benchmark_performance else 0.0,
            'excess_return': (main_value - self.main_performance[0]) - (benchmark_value - self.benchmark_performance[0]) if self.main_performance and self.benchmark_performance else 0.0
        }
        self.performance_snapshots.append(snapshot)
    
    def get_performance_data(self) -> Dict[str, Any]:
        """Get performance data for graphing."""
        return {
            'dates': self.dates,
            'main_performance': self.main_performance,
            'benchmark_performance': self.benchmark_performance,
            'snapshots': self.performance_snapshots
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.main_performance or not self.benchmark_performance:
            return {}
        
        initial_main = self.main_performance[0]
        final_main = self.main_performance[-1]
        initial_benchmark = self.benchmark_performance[0]
        final_benchmark = self.benchmark_performance[-1]
        
        main_return = final_main - initial_main
        benchmark_return = final_benchmark - initial_benchmark
        excess_return = main_return - benchmark_return
        
        main_return_pct = (main_return / initial_main * 100) if initial_main > 0 else 0.0
        benchmark_return_pct = (benchmark_return / initial_benchmark * 100) if initial_benchmark > 0 else 0.0
        excess_return_pct = main_return_pct - benchmark_return_pct
        
        return {
            'main_total_return': main_return,
            'main_return_percentage': main_return_pct,
            'benchmark_total_return': benchmark_return,
            'benchmark_return_percentage': benchmark_return_pct,
            'excess_return': excess_return,
            'excess_return_percentage': excess_return_pct,
            'outperformed_benchmark': excess_return > 0,
            'total_days': len(self.dates)
        }
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for main portfolio."""
        if len(self.main_performance) < 2:
            return 0.0
        
        import numpy as np
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(self.main_performance)):
            if self.main_performance[i-1] > 0:
                daily_return = (self.main_performance[i] / self.main_performance[i-1]) - 1
                returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        # Calculate Sharpe ratio
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily data)
        annualized_return = avg_return * 252
        annualized_volatility = std_return * np.sqrt(252)
        
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        return sharpe_ratio
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown for main portfolio."""
        if not self.main_performance:
            return 0.0
        
        peak = self.main_performance[0]
        max_drawdown = 0.0
        
        for value in self.main_performance:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100  # Return as percentage
    
    def save_performance_data(self, file_path: str):
        """Save performance data to file."""
        import json
        import os
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        data = {
            'performance_data': self.get_performance_data(),
            'performance_summary': self.get_performance_summary(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
