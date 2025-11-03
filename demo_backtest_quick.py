"""
Quick Backtest Demo - Simple test of Phase 6 functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime

from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import MetricsCalculator  
from backtesting.reports import ReportGenerator
from strategies.rsi_macd import RSIMACDStrategy
from risk import RiskManagement
from core.logger import get_logger


def main():
    """Quick demo of backtest functionality"""
    logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("PHASE 6 BACKTEST QUICK DEMO")
    logger.info("=" * 80)
    
    # Generate simple sample data
    logger.info("\n1. Generating sample data...")
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1D')
    np.random.seed(42)
    prices = 45000 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 100),
        'symbol': 'BTC/USDT'
    })
    logger.info(f"   Data: {len(data)} bars from {dates[0].date()} to {dates[-1].date()}")
    logger.info(f"   Price range: ${prices.min():.2f} to ${prices.max():.2f}")
    
    # Initialize strategy
    logger.info("\n2. Initializing RSI+MACD Strategy...")
    strategy = RSIMACDStrategy(config={
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    })
    
    # Initialize risk management
    logger.info("\n3. Initializing Risk Management...")
    risk_mgmt = RiskManagement(config={
        'initial_capital': 100000,
        'risk_management': {
            'max_position_size': 0.1,
            'max_daily_loss': 0.05
        }
    })
    
    # Initialize backtest engine
    logger.info("\n4. Initializing Backtest Engine...")
    engine = BacktestEngine(config={
        'initial_capital': 100000,
        'backtesting': {
            'commission_rate': 0.1,
            'slippage_model': 'percentage',
            'max_slippage_pct': 0.05
        }
    })
    
    # Run backtest
    logger.info("\n5. Running backtest...")
    results = engine.run(
        data=data,
        strategy=strategy,
        symbol="BTC/USDT",
        risk_mgmt=risk_mgmt
    )
    
    # Calculate metrics
    logger.info("\n6. Calculating performance metrics...")
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        initial_capital=results['initial_capital']
    )
    
    # Generate report
    logger.info("\n7. Generating report...")
    reporter = ReportGenerator()
    summary = reporter.generate_summary_report(results, metrics)
    print(summary)
    
    # Export results
    logger.info("\n8. Exporting results...")
    reporter.export_to_json(results, metrics, "backtest_quick_results.json")
    logger.info("   JSON: backtest_quick_results.json")
    
    # Generate HTML report
    trades_df = reporter.generate_trade_list(results['trades'])
    reporter.generate_html_report(results, metrics, trades_df, "backtest_quick_report.html")
    logger.info("   HTML: backtest_quick_report.html")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ PHASE 6 QUICK DEMO COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nKey Results:")
    logger.info(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    logger.info(f"  Final Equity: ${results['final_equity']:,.2f}")
    logger.info(f"  Total Return: {metrics.total_return:+.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
    logger.info(f"  Total Trades: {metrics.total_trades}")
    logger.info(f"  Win Rate: {metrics.win_rate:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = get_logger()
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        raise
