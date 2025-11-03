"""
Backtesting Module - Professional backtesting engine with comprehensive analytics

Features:
- Realistic order simulation (Market, Limit, Stop-Loss orders)
- Commission and slippage modeling
- Walk-forward analysis for robustness testing
- Monte Carlo simulation for probability analysis
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, etc.)
- Professional reporting with visualizations
- HTML/JSON export capabilities

Example:
    from backtesting import BacktestEngine, MetricsCalculator, ReportGenerator
    from strategies import RSIMACDStrategy
    from risk import RiskManagement
    
    # Setup
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    strategy = RSIMACDStrategy()
    risk_mgmt = RiskManagement(initial_capital=100000)
    
    # Run backtest
    results = engine.run(data, strategy, "BTC/USDT", risk_mgmt)
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(results)
    
    # Generate reports
    reporter = ReportGenerator()
    summary = reporter.generate_summary_report(results, metrics)
    reporter.plot_equity_curve(results['equity_curve'])
"""

from backtesting.backtest_engine import (
    BacktestEngine,
    BacktestOrder,
    BacktestPosition,
    BacktestTrade,
    OrderType,
    OrderStatus
)
from backtesting.metrics import (
    MetricsCalculator,
    PerformanceMetrics
)
from backtesting.reports import ReportGenerator

__all__ = [
    # Main classes
    'BacktestEngine',
    'MetricsCalculator',
    'ReportGenerator',
    
    # Data classes
    'BacktestOrder',
    'BacktestPosition',
    'BacktestTrade',
    'PerformanceMetrics',
    
    # Enums
    'OrderType',
    'OrderStatus'
]

