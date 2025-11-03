"""
Backtest Demo - Test complete backtesting workflow
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import MetricsCalculator
from backtesting.reports import ReportGenerator
from strategies.rsi_macd import RSIMACDStrategy
from strategies.bollinger import BollingerBandsStrategy
from risk import RiskManagement
from core.logger import get_logger


def generate_sample_data(symbol: str = "BTC/USDT", days: int = 365) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='1D')
    
    # Generate realistic price movements
    np.random.seed(42)
    base_price = 45000
    returns = np.random.normal(0.001, 0.02, days)
    prices = base_price * (1 + returns).cumprod()
    
    # Generate OHLCV
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, days)),
        'low': prices * (1 + np.random.uniform(-0.02, 0, days)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, days)
    })
    
    data['symbol'] = symbol
    return data


def demo_single_strategy():
    """Demo 1: Backtest single strategy"""
    logger = get_logger()
    logger.info("=" * 80)
    logger.info("DEMO 1: Single Strategy Backtest")
    logger.info("=" * 80)
    
    # Generate sample data
    logger.info("Generating sample data...")
    data = generate_sample_data("BTC/USDT", days=365)
    
    # Initialize components
    logger.info("Initializing strategy and risk management...")
    strategy = RSIMACDStrategy(config={
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    })
    
    risk_mgmt = RiskManagement(config={
        'initial_capital': 100000,
        'risk_management': {
            'max_position_size': 0.1,
            'max_daily_loss': 0.05,
            'max_open_positions': 5
        }
    })
    
    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(config={
        'initial_capital': 100000,
        'backtesting': {
            'commission_rate': 0.1,
            'slippage_model': 'percentage',
            'max_slippage_pct': 0.05
        }
    })
    
    results = engine.run(
        data=data,
        strategy=strategy,
        symbol="BTC/USDT",
        risk_management=risk_mgmt
    )
    
    # Calculate metrics
    logger.info("Calculating performance metrics...")
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(results)
    
    # Generate reports
    logger.info("Generating reports...")
    reporter = ReportGenerator()
    
    # Text report
    summary = reporter.generate_summary_report(results, metrics)
    print(summary)
    
    # Trade list
    trades_df = reporter.generate_trade_list(results['trades'])
    logger.info(f"\nTrade List (first 10 trades):")
    print(trades_df.head(10).to_string())
    
    # Export JSON
    json_path = "backtest_results.json"
    reporter.export_to_json(results, metrics, json_path)
    logger.info(f"Results exported to {json_path}")
    
    # Generate HTML report
    html_path = "backtest_report.html"
    reporter.generate_html_report(results, metrics, trades_df, html_path)
    logger.info(f"HTML report generated: {html_path}")
    
    logger.info("✓ Demo 1 completed successfully")
    return results, metrics


def demo_strategy_comparison():
    """Demo 2: Compare multiple strategies"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 2: Strategy Comparison")
    logger.info("=" * 80)
    
    # Generate sample data
    data = generate_sample_data("BTC/USDT", days=365)
    
    # Define strategies
    strategies = {
        "RSI+MACD": RSIMACDStrategy(config={
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }),
        "Bollinger Bands": BollingerBandsStrategy(config={
            'period': 20,
            'std_dev': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        })
    }
    
    # Run backtests for each strategy
    results_comparison = {}
    metrics_comparison = {}
    
    risk_mgmt = RiskManagement(config={
        'initial_capital': 100000,
        'risk_management': {
            'max_position_size': 0.1,
            'max_daily_loss': 0.05,
            'max_open_positions': 5
        }
    })
    
    engine = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
    calculator = MetricsCalculator()
    
    for name, strategy in strategies.items():
        logger.info(f"\nBacktesting {name}...")
        
        results = engine.run(
            data=data,
            strategy=strategy,
            symbol="BTC/USDT",
            risk_management=risk_mgmt
        )
        
        metrics = calculator.calculate_all_metrics(results)
        
        results_comparison[name] = results
        metrics_comparison[name] = metrics
        
        logger.info(f"{name} Results:")
        logger.info(f"  Total Return: {metrics.total_return:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
        logger.info(f"  Win Rate: {metrics.win_rate:.1f}%")
        logger.info(f"  Total Trades: {metrics.total_trades}")
    
    # Comparison table
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 80)
    
    comparison_df = pd.DataFrame({
        'Strategy': list(metrics_comparison.keys()),
        'Total Return (%)': [m.total_return for m in metrics_comparison.values()],
        'Sharpe Ratio': [m.sharpe_ratio for m in metrics_comparison.values()],
        'Max DD (%)': [m.max_drawdown for m in metrics_comparison.values()],
        'Win Rate (%)': [m.win_rate for m in metrics_comparison.values()],
        'Profit Factor': [m.profit_factor for m in metrics_comparison.values()],
        'Total Trades': [m.total_trades for m in metrics_comparison.values()]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Find best strategy
    best_strategy = max(metrics_comparison.items(), key=lambda x: x[1].sharpe_ratio)
    logger.info(f"\n🏆 Best Strategy (by Sharpe): {best_strategy[0]}")
    
    logger.info("✓ Demo 2 completed successfully")
    return results_comparison, metrics_comparison


def demo_walk_forward():
    """Demo 3: Walk-forward analysis"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 3: Walk-Forward Analysis")
    logger.info("=" * 80)
    
    # Generate sample data
    logger.info("Generating sample data...")
    data = generate_sample_data("BTC/USDT", days=500)  # Need more data
    
    # Initialize components
    strategy = RSIMACDStrategy(config={})
    risk_mgmt = RiskManagement(config={'initial_capital': 100000})
    
    # Run walk-forward analysis
    logger.info("Running walk-forward analysis...")
    logger.info("  Training Period: 252 days")
    logger.info("  Testing Period: 63 days")
    
    engine = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
    
    wf_results = engine.walk_forward_analysis(
        data=data,
        strategy=strategy,
        symbol="BTC/USDT",
        risk_management=risk_mgmt,
        train_period=252,
        test_period=63
    )
    
    # Analyze results
    logger.info(f"\nWalk-Forward Results:")
    logger.info(f"  Total Windows: {len(wf_results)}")
    
    returns = [r['total_return'] for r in wf_results]
    sharpes = [r['sharpe_ratio'] for r in wf_results]
    
    logger.info(f"\nPerformance Across Windows:")
    logger.info(f"  Average Return: {np.mean(returns):.2f}%")
    logger.info(f"  Std Dev of Returns: {np.std(returns):.2f}%")
    logger.info(f"  Average Sharpe: {np.mean(sharpes):.2f}")
    logger.info(f"  Min Return: {min(returns):.2f}%")
    logger.info(f"  Max Return: {max(returns):.2f}%")
    
    # Window-by-window breakdown
    logger.info("\nWindow Breakdown:")
    for i, result in enumerate(wf_results, 1):
        logger.info(f"  Window {i}: Return={result['total_return']:.2f}%, "
                   f"Sharpe={result['sharpe_ratio']:.2f}, "
                   f"Trades={result['total_trades']}")
    
    logger.info("✓ Demo 3 completed successfully")
    return wf_results


def demo_monte_carlo():
    """Demo 4: Monte Carlo simulation"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 4: Monte Carlo Simulation")
    logger.info("=" * 80)
    
    # Generate sample data and run backtest
    data = generate_sample_data("BTC/USDT", days=365)
    strategy = RSIMACDStrategy(config={})
    risk_mgmt = RiskManagement(config={'initial_capital': 100000})
    
    engine = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
    
    logger.info("Running initial backtest...")
    results = engine.run(
        data=data,
        strategy=strategy,
        symbol="BTC/USDT",
        risk_management=risk_mgmt
    )
    
    # Run Monte Carlo simulation
    logger.info("Running Monte Carlo simulation (1000 iterations)...")
    mc_results = engine.monte_carlo_simulation(
        trades=results['trades'],
        initial_capital=100000,
        num_simulations=1000
    )
    
    # Display results
    logger.info("\nMonte Carlo Results:")
    logger.info(f"  Mean Final Equity: ${mc_results['mean_final_equity']:,.2f}")
    logger.info(f"  Median Final Equity: ${mc_results['median_final_equity']:,.2f}")
    logger.info(f"  Std Dev: ${mc_results['std_final_equity']:,.2f}")
    
    logger.info(f"\nPercentiles:")
    logger.info(f"  5th: ${mc_results['percentile_5']:,.2f}")
    logger.info(f"  25th: ${mc_results['percentile_25']:,.2f}")
    logger.info(f"  50th (Median): ${mc_results['percentile_50']:,.2f}")
    logger.info(f"  75th: ${mc_results['percentile_75']:,.2f}")
    logger.info(f"  95th: ${mc_results['percentile_95']:,.2f}")
    
    logger.info(f"\nProbabilities:")
    logger.info(f"  Profit: {mc_results['probability_profit']:.1f}%")
    logger.info(f"  Loss: {mc_results['probability_loss']:.1f}%")
    
    logger.info(f"\nRisk Metrics:")
    logger.info(f"  Max Simulated Equity: ${mc_results['max_final_equity']:,.2f}")
    logger.info(f"  Min Simulated Equity: ${mc_results['min_final_equity']:,.2f}")
    logger.info(f"  Range: ${mc_results['max_final_equity'] - mc_results['min_final_equity']:,.2f}")
    
    logger.info("✓ Demo 4 completed successfully")
    return mc_results


def demo_visualization():
    """Demo 5: Generate all visualizations"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 5: Visualization Generation")
    logger.info("=" * 80)
    
    # Run backtest
    data = generate_sample_data("BTC/USDT", days=365)
    strategy = RSIMACDStrategy(config={})
    risk_mgmt = RiskManagement(config={'initial_capital': 100000})
    
    engine = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
    results = engine.run(data, strategy, "BTC/USDT", risk_mgmt)
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(results)
    
    reporter = ReportGenerator()
    
    # Generate visualizations
    logger.info("Generating equity curve...")
    reporter.plot_equity_curve(results['equity_curve'])
    
    logger.info("Generating drawdown chart...")
    reporter.plot_drawdown(results['equity_curve'])
    
    logger.info("Generating trade distribution...")
    reporter.plot_trade_distribution(results['trades'])
    
    logger.info("Generating returns heatmap...")
    reporter.plot_returns_heatmap(results['equity_curve'])
    
    logger.info("✓ Demo 5 completed successfully")


def main():
    """Run all demos"""
    logger = get_logger()
    
    try:
        logger.info("Starting Backtesting Demos...")
        logger.info("=" * 80)
        
        # Demo 1: Single strategy
        results1, metrics1 = demo_single_strategy()
        
        # Demo 2: Strategy comparison
        results2, metrics2 = demo_strategy_comparison()
        
        # Demo 3: Walk-forward analysis
        wf_results = demo_walk_forward()
        
        # Demo 4: Monte Carlo
        mc_results = demo_monte_carlo()
        
        # Demo 5: Visualization (uncomment to show plots)
        # demo_visualization()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL DEMOS COMPLETED SUCCESSFULLY! ✓")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
