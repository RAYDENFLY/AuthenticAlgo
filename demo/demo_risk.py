"""
Risk Management Demo
Comprehensive demonstration of all risk management features
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from risk import RiskManagement
from core.logger import get_logger

# Get logger instance
logger = get_logger()


def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    base_price = 100
    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }
    
    current_price = base_price
    for _ in range(periods):
        change = np.random.randn() * 2
        open_price = current_price
        close_price = current_price + change
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.5)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.5)
        volume = np.random.randint(1000, 10000)
        
        data['open'].append(open_price)
        data['high'].append(high_price)
        data['low'].append(low_price)
        data['close'].append(close_price)
        data['volume'].append(volume)
        
        current_price = close_price
    
    return pd.DataFrame(data)


def demo_risk_manager():
    """Demo RiskManager functionality"""
    logger.info("=" * 60)
    logger.info("DEMO 1: Risk Manager - Trade Validation & Position Sizing")
    logger.info("=" * 60)
    
    config = {
        'initial_capital': 50000,
        'risk_management': {
            'max_position_size_pct': 5.0,
            'max_daily_loss_pct': 2.0,
            'max_drawdown_pct': 15.0,
            'risk_per_trade_pct': 1.0,
            'max_portfolio_exposure_pct': 20.0,
            'correlation_threshold': 0.6,
            'circuit_breakers': {
                'volatility_threshold': 5.0,
                'max_consecutive_losses': 2
            }
        },
        'stop_loss': {},
        'portfolio': {}
    }
    
    risk_mgmt = RiskManagement(config)
    
    # Test 1: Validate a normal trade
    print("\n📊 Test 1: Normal Trade Validation")
    result = risk_mgmt.validate_trade(
        symbol='BTC/USDT',
        quantity=0.5,
        price=45000.0,
        order_type='BUY',
        current_positions={}
    )
    print(f"  Approved: {result['approved']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Adjusted Quantity: {result['adjusted_quantity']:.4f}")
    
    # Test 2: Position size too large
    print("\n📊 Test 2: Oversized Position Validation")
    result = risk_mgmt.validate_trade(
        symbol='BTC/USDT',
        quantity=2.0,  # Too large
        price=45000.0,
        order_type='BUY',
        current_positions={}
    )
    print(f"  Approved: {result['approved']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Adjusted Quantity: {result['adjusted_quantity']:.4f}")
    
    # Test 3: Calculate optimal position size
    print("\n📊 Test 3: Position Size Calculation")
    position_size = risk_mgmt.calculate_position_size(
        symbol='BTC/USDT',
        price=45000.0,
        stop_loss_price=44000.0,
        account_balance=50000
    )
    print(f"  Symbol: BTC/USDT")
    print(f"  Price: $45,000")
    print(f"  Stop Loss: $44,000 (2.22% risk)")
    print(f"  Optimal Position: {position_size:.4f} BTC (${position_size * 45000:.2f})")
    
    # Test 4: Simulate consecutive losses (trigger circuit breaker)
    print("\n📊 Test 4: Circuit Breaker Test")
    for i in range(3):
        risk_mgmt.risk_manager.update_trade_result(
            symbol='BTC/USDT',
            quantity=0.5,
            entry_price=45000.0,
            exit_price=44000.0,
            pnl=-500.0,
            timestamp=datetime.now()
        )
        print(f"  Loss #{i+1}: -$500, Consecutive losses: {risk_mgmt.risk_manager.consecutive_losses}")
    
    print(f"\n  Circuit Breaker Active: {risk_mgmt.risk_manager.circuit_breaker_active}")
    print(f"  Reason: {risk_mgmt.risk_manager.circuit_breaker_reason}")
    
    # Test 5: Try to trade with circuit breaker active
    print("\n📊 Test 5: Trading with Circuit Breaker Active")
    result = risk_mgmt.validate_trade(
        symbol='ETH/USDT',
        quantity=10.0,
        price=2500.0,
        order_type='BUY',
        current_positions={}
    )
    print(f"  Approved: {result['approved']}")
    print(f"  Reason: {result['reason']}")
    
    # Test 6: Risk Report
    print("\n📊 Test 6: Risk Report")
    report = risk_mgmt.risk_manager.generate_risk_report()
    print(f"  Daily PnL: ${report['daily_pnl']:.2f}")
    print(f"  Daily Loss Limit: ${report['daily_loss_limit']:.2f}")
    print(f"  Total Trades: {report['total_trades']}")
    print(f"  Winning Trades: {report['winning_trades']}")
    print(f"  Losing Trades: {report['losing_trades']}")
    print(f"  Overall Risk Score: {report['risk_metrics']['overall_risk_score']:.2f}")
    print(f"  Recommendations:")
    for rec in report['recommendations']:
        print(f"    - {rec}")


def demo_stop_loss_manager():
    """Demo StopLossManager functionality"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Stop-Loss Manager - Multiple Stop Strategies")
    logger.info("=" * 60)
    
    config = {
        'initial_capital': 50000,
        'risk_management': {},
        'stop_loss': {
            'default_type': 'atr_based',
            'fixed_stop_percentage': 2.0,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'trailing_activation_pct': 1.0,
            'trailing_percentage': 1.0
        },
        'portfolio': {}
    }
    
    risk_mgmt = RiskManagement(config)
    sample_data = create_sample_data(100)
    
    entry_price = 45000.0
    current_volatility = 0.02  # 2% volatility
    
    print("\n📊 Stop-Loss Recommendations for BTC/USDT Long Position")
    print(f"  Entry Price: ${entry_price:,.2f}")
    print(f"  Current Volatility: {current_volatility * 100:.1f}%")
    
    recommendations = risk_mgmt.stop_loss_manager.get_stop_loss_recommendation(
        symbol='BTC/USDT',
        entry_price=entry_price,
        position_type='long',
        data=sample_data,
        volatility=current_volatility
    )
    
    print("\n  Stop-Loss Methods:")
    for method, rec in recommendations.items():
        if 'error' not in rec:
            print(f"\n  {method.upper().replace('_', ' ')}:")
            print(f"    Stop Price: ${rec['stop_price']:,.2f}")
            print(f"    Distance: {rec['distance_pct']:.2f}%")
            print(f"    Risk/Reward: {rec['risk_reward_ratio']:.2f}:1")
    
    # Test trailing stop updates
    print("\n📊 Trailing Stop Updates Simulation")
    entry_price = 45000.0
    risk_mgmt.stop_loss_manager.set_active_stop(
        symbol='BTC/USDT',
        stop_price=44550.0,  # Initial stop 1% below
        stop_type=risk_mgmt.stop_loss_manager.default_type,
        entry_price=entry_price,
        position_type='long'
    )
    
    prices = [45000, 45500, 46000, 46500, 46200, 45800]
    print(f"  Initial Entry: ${entry_price:,.2f}")
    print(f"  Initial Stop: ${44550:,.2f}\n")
    
    for i, price in enumerate(prices, 1):
        new_stop = risk_mgmt.stop_loss_manager.update_stop_loss(
            symbol='BTC/USDT',
            current_price=price,
            position_type='long',
            data=sample_data
        )
        
        triggered, reason = risk_mgmt.stop_loss_manager.check_stop_loss('BTC/USDT', price)
        
        print(f"  Update #{i}: Price ${price:,.2f}", end="")
        if new_stop:
            print(f" → Stop moved to ${new_stop:,.2f} ✓")
        elif triggered:
            print(f" → {reason} ⚠️")
        else:
            print(" → No update")


def demo_portfolio_manager():
    """Demo PortfolioManager functionality"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Portfolio Manager - Tracking & Metrics")
    logger.info("=" * 60)
    
    config = {
        'initial_capital': 100000,
        'risk_management': {},
        'stop_loss': {},
        'portfolio': {
            'max_positions': 5,
            'rebalancing_threshold': 3.0,
            'target_allocations': {
                'BTC/USDT': 10.0,
                'ETH/USDT': 8.0,
                'ADA/USDT': 5.0
            }
        }
    }
    
    risk_mgmt = RiskManagement(config)
    
    print(f"\n📊 Initial Portfolio State")
    print(f"  Starting Capital: ${config['initial_capital']:,.2f}")
    print(f"  Cash Balance: ${risk_mgmt.portfolio_manager.cash_balance:,.2f}")
    print(f"  Total Value: ${risk_mgmt.portfolio_manager.get_total_value():,.2f}")
    
    # Execute some trades
    print(f"\n📊 Executing Sample Trades")
    
    trades = [
        ('BTC/USDT', 1.0, 45000.0, 'BUY'),
        ('ETH/USDT', 20.0, 2500.0, 'BUY'),
        ('ADA/USDT', 5000.0, 0.50, 'BUY'),
    ]
    
    for symbol, qty, price, action in trades:
        risk_mgmt.update_portfolio(symbol, qty, price, action, datetime.now())
        print(f"  {action} {qty} {symbol} @ ${price:,.2f}")
    
    # Update market prices (simulate price movements)
    print(f"\n📊 Market Price Updates")
    price_updates = {
        'BTC/USDT': 46500.0,  # +3.33%
        'ETH/USDT': 2600.0,   # +4.00%
        'ADA/USDT': 0.52      # +4.00%
    }
    
    risk_mgmt.portfolio_manager.update_market_prices(price_updates, datetime.now())
    
    for symbol, price in price_updates.items():
        pos = risk_mgmt.portfolio_manager.positions[symbol]
        print(f"  {symbol}: ${price:,.2f} ({pos.unrealized_pnl_pct:+.2f}%)")
    
    # Get portfolio metrics
    print(f"\n📊 Portfolio Metrics")
    metrics = risk_mgmt.portfolio_manager.get_portfolio_metrics()
    print(f"  Total Value: ${metrics.total_value:,.2f}")
    print(f"  Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:+.2f}%)")
    print(f"  Unrealized P&L: ${metrics.unrealized_pnl:,.2f}")
    print(f"  Realized P&L: ${metrics.realized_pnl:,.2f}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
    print(f"  Volatility: {metrics.volatility:.2f}%")
    
    # Position allocations
    print(f"\n📊 Position Allocations")
    allocations = risk_mgmt.portfolio_manager.get_position_allocations()
    for symbol, alloc in allocations.items():
        target = config['portfolio']['target_allocations'].get(symbol, 0)
        print(f"  {symbol}: {alloc:.2f}% (Target: {target:.2f}%)")
    
    # Rebalancing recommendations
    print(f"\n📊 Rebalancing Analysis")
    rebalancing = risk_mgmt.portfolio_manager.check_rebalancing_needed()
    if rebalancing:
        print("  Recommendations:")
        for symbol, info in rebalancing.items():
            print(f"    {symbol}: {info['action']} (Current: {info['current_allocation']:.2f}%, "
                  f"Target: {info['target_allocation']:.2f}%, Deviation: {info['deviation']:.2f}%)")
    else:
        print("  ✓ No rebalancing needed")


def demo_comprehensive_report():
    """Demo comprehensive risk management report"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Comprehensive Risk Management Report")
    logger.info("=" * 60)
    
    config = {
        'initial_capital': 100000,
        'risk_management': {
            'max_position_size_pct': 10.0,
            'max_daily_loss_pct': 3.0,
            'risk_per_trade_pct': 1.5,
            'circuit_breakers': {
                'max_consecutive_losses': 3
            }
        },
        'stop_loss': {
            'default_type': 'atr_based',
            'atr_multiplier': 2.5
        },
        'portfolio': {
            'max_positions': 10,
            'target_allocations': {
                'BTC/USDT': 15.0,
                'ETH/USDT': 10.0
            }
        }
    }
    
    risk_mgmt = RiskManagement(config)
    
    # Simulate some trading activity
    risk_mgmt.update_portfolio('BTC/USDT', 1.5, 45000.0, 'BUY', datetime.now())
    risk_mgmt.update_portfolio('ETH/USDT', 30.0, 2500.0, 'BUY', datetime.now())
    
    # Update prices
    risk_mgmt.portfolio_manager.update_market_prices({
        'BTC/USDT': 47000.0,
        'ETH/USDT': 2650.0
    }, datetime.now())
    
    # Close a position
    risk_mgmt.update_portfolio('ETH/USDT', 30.0, 2650.0, 'SELL', datetime.now())
    
    # Get comprehensive report
    report = risk_mgmt.get_comprehensive_report()
    
    print(f"\n{'=' * 60}")
    print("COMPREHENSIVE RISK MANAGEMENT REPORT")
    print(f"{'=' * 60}")
    
    print(f"\n📈 PORTFOLIO SUMMARY")
    ps = report['portfolio']['portfolio_summary']
    print(f"  Total Value: ${ps['total_value']:,.2f}")
    print(f"  Cash Balance: ${ps['cash_balance']:,.2f}")
    print(f"  Positions Value: ${ps['positions_value']:,.2f}")
    print(f"  Total P&L: ${ps['total_pnl']:,.2f} ({ps['total_pnl_pct']:+.2f}%)")
    print(f"  Open Positions: {ps['total_positions']}")
    
    print(f"\n📊 PERFORMANCE METRICS")
    pm = report['portfolio']['performance_metrics']
    print(f"  Sharpe Ratio: {pm['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {pm['max_drawdown']:.2f}%")
    print(f"  Volatility: {pm['volatility']:.2f}%")
    print(f"  Win Rate: {pm['win_rate']:.2f}%")
    
    print(f"\n⚠️ RISK MANAGEMENT")
    rm = report['risk_management']
    print(f"  Daily P&L: ${rm['daily_pnl']:,.2f}")
    print(f"  Remaining Daily Loss: ${rm['remaining_daily_loss']:,.2f}")
    print(f"  Circuit Breaker: {'🔴 ACTIVE' if rm['circuit_breaker_active'] else '🟢 Inactive'}")
    print(f"  Overall Risk Score: {rm['risk_metrics']['overall_risk_score']:.2f}")
    
    print(f"\n💡 RECOMMENDATIONS")
    for rec in rm['recommendations']:
        print(f"  {rec}")
    
    print(f"\n🎯 CURRENT POSITIONS")
    for symbol, pos in report['portfolio']['current_positions'].items():
        print(f"  {symbol}:")
        print(f"    Quantity: {pos['quantity']:.4f}")
        print(f"    Entry: ${pos['entry_price']:,.2f} → Current: ${pos['current_price']:,.2f}")
        print(f"    P&L: ${pos['unrealized_pnl']:,.2f} ({pos['unrealized_pnl_pct']:+.2f}%)")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print(" " * 20 + "RISK MANAGEMENT MODULE DEMO")
    print("=" * 80)
    
    try:
        demo_risk_manager()
        demo_stop_loss_manager()
        demo_portfolio_manager()
        demo_comprehensive_report()
        
        print("\n" + "=" * 80)
        print("✅ All Risk Management Demos Completed Successfully!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
