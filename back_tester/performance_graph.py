import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PerformanceGraph:
    """Generate performance comparison graphs."""
    
    def __init__(self):
        self.figure_size = (12, 8)
        self.dpi = 300
    
    def create_performance_comparison(self, performance_data: Dict[str, Any], 
                                    output_file: str = 'performance_comparison.png'):
        """Create performance comparison graph."""
        dates = performance_data['dates']
        main_perf = performance_data['main_performance']
        benchmark_perf = performance_data['benchmark_performance']
        
        # Convert date strings to datetime objects for proper plotting
        date_objects = [datetime.fromisoformat(date) for date in dates]
        
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot main strategy performance
        plt.plot(date_objects, main_perf, label='Back Tester Strategy', 
                linewidth=2, color='blue', alpha=0.8)
        
        # Plot benchmark performance
        plt.plot(date_objects, benchmark_perf, label='SP500 Benchmark', 
                linewidth=2, color='red', alpha=0.8)
        
        # Customize the plot
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.title('Back Tester vs Benchmark Performance Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add value formatting to y-axis
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        logger.info(f"Saving performance comparison graph to: {output_file}")
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison graph saved to: {output_file}")
    
    def create_returns_comparison(self, performance_data: Dict[str, Any], 
                                output_file: str = 'returns_comparison.png'):
        """Create returns comparison graph."""
        dates = performance_data['dates']
        main_perf = performance_data['main_performance']
        benchmark_perf = performance_data['benchmark_performance']
        
        # Calculate cumulative returns
        if main_perf and benchmark_perf:
            initial_main = main_perf[0]
            initial_benchmark = benchmark_perf[0]
            
            main_returns = [(value / initial_main - 1) * 100 for value in main_perf]
            benchmark_returns = [(value / initial_benchmark - 1) * 100 for value in benchmark_perf]
            
            # Convert date strings to datetime objects
            date_objects = [datetime.fromisoformat(date) for date in dates]
            
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot returns
            plt.plot(date_objects, main_returns, label='Back Tester Strategy', 
                    linewidth=2, color='blue', alpha=0.8)
            plt.plot(date_objects, benchmark_returns, label='SP500 Benchmark', 
                    linewidth=2, color='red', alpha=0.8)
            
            # Add zero line
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Customize the plot
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return (%)', fontsize=12)
            plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
            
            # Add percentage formatting to y-axis
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
            
            plt.tight_layout()
            logger.info(f"Saving returns comparison graph to: {output_file}")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"Returns comparison graph saved to: {output_file}")
    
    def create_drawdown_chart(self, performance_data: Dict[str, Any], 
                            output_file: str = 'drawdown_chart.png'):
        """Create drawdown chart for main strategy."""
        dates = performance_data['dates']
        main_perf = performance_data['main_performance']
        
        if not main_perf:
            return
        
        # Calculate drawdown
        peak = main_perf[0]
        drawdowns = []
        
        for value in main_perf:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(drawdown)
        
        # Convert date strings to datetime objects
        date_objects = [datetime.fromisoformat(date) for date in dates]
        
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot drawdown
        plt.fill_between(date_objects, drawdowns, 0, alpha=0.3, color='red')
        plt.plot(date_objects, drawdowns, color='red', linewidth=1)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Customize the plot
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add percentage formatting to y-axis
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        plt.tight_layout()
        logger.info(f"Saving drawdown chart to: {output_file}")
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Drawdown chart saved to: {output_file}")
    
    def create_all_graphs(self, performance_data: Dict[str, Any], 
                         output_prefix: str = 'performance'):
        """Create all performance graphs."""
        self.create_performance_comparison(performance_data, f'{output_prefix}_comparison.png')
        self.create_returns_comparison(performance_data, f'{output_prefix}_returns.png')
        self.create_drawdown_chart(performance_data, f'{output_prefix}_drawdown.png')
        
        print(f"All performance graphs created with prefix: {output_prefix}")
