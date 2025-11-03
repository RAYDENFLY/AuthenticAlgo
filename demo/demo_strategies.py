"""
Trading Strategies Demo
Demonstrates all 3 trading strategies with simple backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.logger import logger
from strategies import create_strategy, STRATEGY_REGISTRY


def create_sample_data(periods=500):
    """Create realistic sample price data for testing"""
    logger.info(f"Creating {periods} periods of sample data...")
    
    # Generate dates
    dates = pd.date_range(datetime.now() - timedelta(hours=periods), periods=periods, freq='1H')
    
    # Create trending data with noise
    np.random.seed(42)
    trend = np.cumsum(np.random.randn(periods) * 0.01) + 100
    noise = np.random.randn(periods) * 0.5
    
    # Generate OHLCV data
    close = trend + noise
    high = close + np.abs(np.random.randn(periods) * 0.3)
    low = close - np.abs(np.random.randn(periods) * 0.3)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.randint(1000, 10000, periods)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Sample data created: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
    return df


def simple_backtest(strategy, data, initial_capital=10000):
    """Simple backtest simulation"""
    logger.info(f"\n🔄 Backtesting {strategy.name}...")
    logger.info("=" * 70)
    
    capital = initial_capital
    position_size = 0
    trades = []
    
    # Calculate indicators once
    data_with_indicators = strategy.calculate_indicators(data.copy())
    
    for i in range(len(data_with_indicators)):
        if i < 50:  # Wait for enough data
            continue
        
        current_data = data_with_indicators.iloc[:i+1]
        current_price = current_data['close'].iloc[-1]
        current_time = current_data.index[-1]
        
        # Check risk management first
        if strategy.position:
            risk_signal = strategy.check_risk_management(current_price, current_time)
            if risk_signal:
                # Execute risk-based exit
                if position_size > 0:
                    capital = position_size * current_price
                    position_size = 0
                
                strategy.update_position('EXIT', current_price, current_time, risk_signal['reason'])
                trades.append(('RISK_EXIT', current_price, current_time, risk_signal['reason']))
        
        # Check for entry/exit signals
        if strategy.position is None:
            # Look for entry
            entry_signal = strategy.should_enter(current_data)
            
            if entry_signal['signal'] in ['BUY', 'SELL'] and entry_signal['confidence'] > 0.6:
                if entry_signal['signal'] == 'BUY':
                    position_size = capital / current_price
                    capital = 0
                    
                    strategy.update_position(
                        'BUY',
                        current_price,
                        current_time,
                        entry_signal['metadata']['reason']
                    )
                    trades.append(('ENTER', current_price, current_time, entry_signal['signal']))
        
        else:
            # Look for exit
            exit_signal = strategy.should_exit(current_data)
            if exit_signal['signal'] == 'EXIT':
                if position_size > 0:
                    capital = position_size * current_price
                    position_size = 0
                
                strategy.update_position(
                    'EXIT',
                    current_price,
                    current_time,
                    exit_signal['reason']
                )
                trades.append(('EXIT', current_price, current_time, exit_signal['reason']))
    
    # Close any remaining position
    if position_size > 0:
        capital = position_size * data['close'].iloc[-1]
    
    # Calculate results
    total_return = (capital - initial_capital) / initial_capital * 100
    total_trades = len([t for t in trades if t[0] == 'ENTER'])
    
    logger.info(f"\n📊 Results:")
    logger.info(f"  Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"  Final Capital: ${capital:,.2f}")
    logger.info(f"  Total Return: {total_return:.2f}%")
    logger.info(f"  Total Trades: {total_trades}")
    
    # Strategy performance metrics
    metrics = strategy.get_performance_metrics()
    if metrics:
        logger.info(f"\n📈 Performance Metrics:")
        logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
        logger.info(f"  Average Profit: {metrics['avg_profit']:.2f}%")
        logger.info(f"  Average Win: {metrics['avg_win']:.2f}%")
        logger.info(f"  Average Loss: {metrics['avg_loss']:.2f}%")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Total Profit: {metrics['total_profit']:.2f}%")
    
    return {
        'final_capital': capital,
        'total_return': total_return,
        'total_trades': total_trades,
        'metrics': metrics
    }


def demo_all_strategies():
    """Demo all three strategies"""
    logger.info("🚀 Trading Strategies Demo")
    logger.info("=" * 70)
    
    # Create sample data
    data = create_sample_data(periods=500)
    
    # Define strategies to test
    strategies_config = {
        'RSI_MACD_Strategy': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0,
            'use_trailing_stop': True,
            'trailing_stop_pct': 1.5
        },
        'BollingerBands_Strategy': {
            'bb_period': 20,
            'bb_std': 2.0,
            'require_volume_spike': True,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 3.0
        },
        'ML_Strategy': {
            'use_fallback': True,
            'confidence_threshold': 0.7,
            'fallback_config': {
                'rsi_period': 14,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0
            }
        }
    }
    
    results = {}
    
    # Test each strategy
    for strategy_name, config in strategies_config.items():
        logger.info(f"\n\n{'='*70}")
        logger.info(f"📈 Testing: {strategy_name}")
        logger.info(f"{'='*70}")
        
        # Create strategy
        strategy = create_strategy(strategy_name, config)
        
        # Run backtest
        result = simple_backtest(strategy, data.copy())
        results[strategy_name] = result
    
    # Compare results
    logger.info(f"\n\n{'='*70}")
    logger.info("🏆 STRATEGY COMPARISON")
    logger.info(f"{'='*70}")
    
    for strategy_name, result in results.items():
        logger.info(f"\n{strategy_name}:")
        logger.info(f"  Return: {result['total_return']:.2f}%")
        logger.info(f"  Trades: {result['total_trades']}")
        if result['metrics']:
            logger.info(f"  Win Rate: {result['metrics']['win_rate']:.1%}")
            logger.info(f"  Profit Factor: {result['metrics']['profit_factor']:.2f}")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['total_return'])
    logger.info(f"\n🥇 Best Strategy: {best_strategy[0]} ({best_strategy[1]['total_return']:.2f}% return)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✨ Phase 4 Trading Strategies: COMPLETE! ✨")
    logger.info("=" * 70)
    
    logger.info("\n📝 What we built:")
    logger.info("  ✅ BaseStrategy - Abstract base with risk management")
    logger.info("  ✅ RSI+MACD Strategy - Momentum-based entries")
    logger.info("  ✅ Bollinger Bands Strategy - Mean reversion")
    logger.info("  ✅ ML Strategy - With fallback to traditional")
    logger.info("  ✅ Position management - Entry/exit tracking")
    logger.info("  ✅ Risk management - Stop-loss, take-profit, trailing stops")
    logger.info("  ✅ Performance metrics - Win rate, profit factor, etc.")
    
    logger.info("\n🚀 Ready for Phase 5: Risk Management Module!")


if __name__ == "__main__":
    demo_all_strategies()
