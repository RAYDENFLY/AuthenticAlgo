"""
Demo: Monitoring Module

This script demonstrates all monitoring features:
1. Telegram bot notifications
2. Discord webhook alerts
3. Dashboard integration
4. Complete monitoring workflow
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import random

from monitoring import MonitoringModule
from core.logger import get_logger


# Sample data generators
def generate_trade_opened_data() -> Dict[str, Any]:
    """Generate sample trade opened data"""
    return {
        'symbol': random.choice(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']),
        'side': random.choice(['BUY', 'SELL']),
        'entry_price': random.uniform(40000, 50000),
        'quantity': random.uniform(0.1, 1.0),
        'strategy': random.choice(['RSI_MACD', 'Bollinger', 'ML Strategy']),
        'stop_loss': random.uniform(38000, 39000),
        'take_profit': random.uniform(52000, 55000)
    }


def generate_trade_closed_data() -> Dict[str, Any]:
    """Generate sample trade closed data"""
    entry_price = random.uniform(40000, 50000)
    exit_price = entry_price * random.uniform(0.95, 1.05)
    quantity = random.uniform(0.1, 1.0)
    pnl = (exit_price - entry_price) * quantity
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    
    return {
        'symbol': random.choice(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']),
        'side': random.choice(['BUY', 'SELL']),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'quantity': quantity,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'reason': random.choice(['Take Profit', 'Stop Loss', 'Manual']),
        'duration': f"{random.randint(1, 24)} hours"
    }


def generate_error_data() -> Dict[str, Any]:
    """Generate sample error data"""
    return {
        'type': random.choice(['ConnectionError', 'ValidationError', 'OrderError']),
        'message': 'This is a sample error message for testing purposes',
        'module': random.choice(['data.collector', 'execution.exchange', 'strategies.rsi_macd']),
        'severity': random.choice(['WARNING', 'ERROR', 'CRITICAL']),
        'traceback': 'File "main.py", line 123...'
    }


def generate_daily_summary_data() -> Dict[str, Any]:
    """Generate sample daily summary data"""
    total_trades = random.randint(5, 20)
    winning_trades = random.randint(0, total_trades)
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': random.uniform(-500, 1500),
        'total_pnl_pct': random.uniform(-5, 15),
        'best_trade': random.uniform(100, 500),
        'worst_trade': random.uniform(-300, -50),
        'sharpe_ratio': random.uniform(0.5, 2.5)
    }


def generate_portfolio_data() -> Dict[str, Any]:
    """Generate sample portfolio data"""
    total_value = 10000 + random.uniform(-1000, 3000)
    cash = total_value * random.uniform(0.3, 0.7)
    positions_value = total_value - cash
    initial_capital = 10000
    total_pnl = total_value - initial_capital
    total_pnl_pct = (total_pnl / initial_capital) * 100
    
    return {
        'total_value': total_value,
        'cash': cash,
        'positions_value': positions_value,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'active_positions': random.randint(0, 5)
    }


def generate_performance_data() -> Dict[str, Any]:
    """Generate sample performance data"""
    return {
        'total_return': random.uniform(-10, 30),
        'annual_return': random.uniform(-5, 20),
        'sharpe_ratio': random.uniform(0.5, 2.5),
        'sortino_ratio': random.uniform(0.6, 3.0),
        'calmar_ratio': random.uniform(0.3, 2.0),
        'win_rate': random.uniform(40, 70),
        'profit_factor': random.uniform(0.8, 2.5),
        'total_trades': random.randint(50, 200),
        'max_drawdown': random.uniform(-20, -5),
        'avg_drawdown': random.uniform(-10, -2),
        'current_drawdown': random.uniform(-15, 0),
        'volatility': random.uniform(10, 30),
        'var_95': random.uniform(-5, -2),
        'expected_shortfall': random.uniform(-8, -3),
        'beta': random.uniform(0.5, 1.5),
        'alpha': random.uniform(-2, 5)
    }


async def demo_telegram_notifications(monitoring: MonitoringModule):
    """Demo 1: Telegram Bot Notifications"""
    print("\n" + "="*80)
    print("DEMO 1: Telegram Bot Notifications")
    print("="*80)
    
    print("\n1. Testing trade opened notification...")
    trade_opened = generate_trade_opened_data()
    await monitoring.notify_trade_opened(trade_opened)
    print(f"   ✓ Trade opened notification sent for {trade_opened['symbol']}")
    await asyncio.sleep(2)
    
    print("\n2. Testing trade closed notification...")
    trade_closed = generate_trade_closed_data()
    await monitoring.notify_trade_closed(trade_closed)
    print(f"   ✓ Trade closed notification sent (P&L: ${trade_closed['pnl']:.2f})")
    await asyncio.sleep(2)
    
    print("\n3. Testing error notification...")
    error_data = generate_error_data()
    await monitoring.notify_error(error_data)
    print(f"   ✓ Error notification sent ({error_data['severity']})")
    await asyncio.sleep(2)
    
    print("\n4. Testing daily summary...")
    summary_data = generate_daily_summary_data()
    await monitoring.send_daily_summary(summary_data)
    print(f"   ✓ Daily summary sent ({summary_data['total_trades']} trades)")
    
    print("\n✓ Demo 1 completed successfully")


async def demo_discord_alerts(monitoring: MonitoringModule):
    """Demo 2: Discord Webhook Alerts"""
    print("\n" + "="*80)
    print("DEMO 2: Discord Webhook Alerts")
    print("="*80)
    
    print("\n1. Testing Discord trade opened...")
    trade_opened = generate_trade_opened_data()
    await monitoring.notify_trade_opened(trade_opened)
    print(f"   ✓ Discord alert sent for trade opened")
    await asyncio.sleep(2)
    
    print("\n2. Testing Discord trade closed...")
    trade_closed = generate_trade_closed_data()
    await monitoring.notify_trade_closed(trade_closed)
    print(f"   ✓ Discord alert sent for trade closed")
    await asyncio.sleep(2)
    
    print("\n3. Testing Discord portfolio update...")
    portfolio_data = generate_portfolio_data()
    await monitoring.send_portfolio_update(portfolio_data)
    print(f"   ✓ Portfolio update sent to Discord")
    await asyncio.sleep(2)
    
    print("\n4. Testing Discord performance metrics...")
    performance_data = generate_performance_data()
    await monitoring.send_performance_metrics(performance_data)
    print(f"   ✓ Performance metrics sent to Discord")
    
    print("\n✓ Demo 2 completed successfully")


async def demo_complete_workflow(monitoring: MonitoringModule):
    """Demo 3: Complete Monitoring Workflow"""
    print("\n" + "="*80)
    print("DEMO 3: Complete Monitoring Workflow")
    print("="*80)
    
    print("\n1. Simulating trading day...")
    
    # Morning: Portfolio check
    print("\n   Morning - Portfolio Update:")
    portfolio = generate_portfolio_data()
    await monitoring.send_portfolio_update(portfolio)
    print(f"   ✓ Portfolio: ${portfolio['total_value']:,.2f}")
    await asyncio.sleep(2)
    
    # Day: Multiple trades
    print("\n   Day - Trading Activity:")
    for i in range(3):
        # Open trade
        trade_opened = generate_trade_opened_data()
        await monitoring.notify_trade_opened(trade_opened)
        print(f"   ✓ Trade {i+1} opened: {trade_opened['symbol']} {trade_opened['side']}")
        await asyncio.sleep(1)
        
        # Close trade
        trade_closed = generate_trade_closed_data()
        await monitoring.notify_trade_closed(trade_closed)
        print(f"   ✓ Trade {i+1} closed: P&L ${trade_closed['pnl']:.2f}")
        await asyncio.sleep(1)
    
    # Evening: Daily summary
    print("\n   Evening - Daily Summary:")
    summary = generate_daily_summary_data()
    await monitoring.send_daily_summary(summary)
    print(f"   ✓ Summary: {summary['total_trades']} trades, Win Rate: {summary['win_rate']:.1f}%")
    
    # Performance metrics
    print("\n   Performance Metrics:")
    performance = generate_performance_data()
    await monitoring.send_performance_metrics(performance)
    print(f"   ✓ Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   ✓ Total Return: {performance['total_return']:.2f}%")
    
    print("\n✓ Demo 3 completed successfully")


async def demo_error_handling(monitoring: MonitoringModule):
    """Demo 4: Error Handling and Monitoring"""
    print("\n" + "="*80)
    print("DEMO 4: Error Handling and Monitoring")
    print("="*80)
    
    print("\n1. Testing different error severities...")
    
    severities = ['WARNING', 'ERROR', 'CRITICAL']
    for severity in severities:
        error_data = {
            'type': f'{severity}Exception',
            'message': f'This is a {severity} level error for testing',
            'module': 'demo.module',
            'severity': severity,
            'traceback': 'Sample traceback...'
        }
        await monitoring.notify_error(error_data)
        print(f"   ✓ {severity} notification sent")
        await asyncio.sleep(2)
    
    print("\n2. Testing service status...")
    status = monitoring.get_service_status()
    print("   Service Status:")
    for service, running in status.items():
        emoji = "🟢" if running else "🔴"
        print(f"   {emoji} {service}: {'Running' if running else 'Stopped'}")
    
    print("\n3. Getting statistics...")
    stats = monitoring.get_stats()
    print("   Statistics:")
    for service, service_stats in stats.items():
        if service_stats:
            print(f"   {service}:")
            for key, value in service_stats.items():
                print(f"     - {key}: {value}")
    
    print("\n✓ Demo 4 completed successfully")


async def main():
    """Main demo function"""
    logger = get_logger()
    
    print("\n" + "="*80)
    print("MONITORING MODULE DEMO")
    print("="*80)
    print("\nThis demo will test all monitoring features.")
    print("Make sure to configure your .env file with:")
    print("  - TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    print("  - DISCORD_WEBHOOK_URL")
    print("\nNote: Some demos may not work if services are not configured.")
    print("="*80)
    
    # Configuration
    config = {
        'monitoring': {
            'telegram': {
                'enabled': False,  # Set to True if configured
                'bot_token': 'your_token_here',
                'chat_id': 'your_chat_id_here',
                'events': ['trade_opened', 'trade_closed', 'error', 'daily_summary']
            },
            'discord': {
                'enabled': False,  # Set to True if configured
                'webhook_url': 'your_webhook_url_here',
                'events': ['trade_opened', 'trade_closed', 'error', 'daily_summary'],
                'username': 'Trading Bot',
                'avatar_url': ''
            },
            'dashboard': {
                'refresh_interval': 5
            }
        }
    }
    
    # Initialize monitoring
    print("\nInitializing MonitoringModule...")
    monitoring = MonitoringModule(config)
    
    # Start services
    print("Starting monitoring services...")
    await monitoring.start()
    
    try:
        # Run demos
        await demo_telegram_notifications(monitoring)
        await asyncio.sleep(2)
        
        await demo_discord_alerts(monitoring)
        await asyncio.sleep(2)
        
        await demo_complete_workflow(monitoring)
        await asyncio.sleep(2)
        
        await demo_error_handling(monitoring)
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nTo use the dashboard:")
        print("  streamlit run monitoring/dashboard.py")
        print("\nTo enable Telegram/Discord:")
        print("  1. Configure tokens in .env file")
        print("  2. Set enabled=True in config")
        print("  3. Run demos again")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Demo error: {e}")
    finally:
        # Stop services
        print("\nStopping monitoring services...")
        await monitoring.stop()
        print("✓ Monitoring services stopped")


if __name__ == "__main__":
    asyncio.run(main())
