"""
Quick ML Trading Bot Test (10 minutes)
Test BTCUSDT 1h XGBoost Model - Best Validated
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import datetime
from core.logger import get_logger

logger = get_logger()


async def quick_test():
    """Quick 10-minute test"""
    
    logger.info("="*80)
    logger.info("🚀 QUICK ML TRADING BOT TEST (10 MINUTES)")
    logger.info("="*80)
    logger.info("")
    logger.info("📊 Configuration:")
    logger.info("   Model:        BTCUSDT 1h XGBoost (Best Validated)")
    logger.info("   Capital:      $10.00")
    logger.info("   Leverage:     10x")
    logger.info("   Test Duration: 10 minutes")
    logger.info("   Accuracy:     96.05%")
    logger.info("   Win Rate:     100%")
    logger.info("")
    logger.info("="*80)
    
    # Import components
    from demo.demo_ml_production import ProductionMLBot
    
    # Create bot
    logger.info("\n🔧 Initializing bot...")
    bot = ProductionMLBot(capital=10.0)
    
    # Initialize
    await bot.initialize()
    
    # Print config
    bot.print_configuration()
    
    logger.info("\n\n" + "="*80)
    logger.info("✅ STARTING 10-MINUTE TEST RUN")
    logger.info("="*80)
    logger.info("Bot will:")
    logger.info("  1. Fetch live BTCUSDT data")
    logger.info("  2. Generate ML predictions")
    logger.info("  3. Simulate trades (paper mode)")
    logger.info("  4. Report results after 10 minutes")
    logger.info("")
    logger.info("Press Ctrl+C to stop early")
    logger.info("="*80 + "\n")
    
    # Run for 10 minutes (simulate by running 10 cycles with 1 min each)
    try:
        # Simulate 10 cycles (1 per minute)
        from data.collector import DataCollector
        from execution.exchange import Exchange
        
        exchange = Exchange('binance', '', '', testnet=True)
        collector = DataCollector(exchange)
        
        for cycle in range(1, 11):
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 Cycle {cycle}/10 ({cycle} minutes elapsed)")
            logger.info(f"{'='*80}")
            
            # Fetch data
            logger.info(f"📊 Fetching BTCUSDT 1h data...")
            data = await collector.fetch_ohlcv('BTCUSDT', '1h', limit=200)
            
            if data is not None and len(data) >= 100:
                logger.info(f"✅ Got {len(data)} candles")
                logger.info(f"   Latest Price: ${data['close'].iloc[-1]:.2f}")
                logger.info(f"   24h Change: {((data['close'].iloc[-1] / data['close'].iloc[-24]) - 1) * 100:+.2f}%")
                
                # Generate signal
                logger.info(f"\n🤖 Generating ML prediction...")
                signal = bot.strategy.generate_signal(data)
                
                if signal:
                    logger.info(f"✅ Signal Generated!")
                    logger.info(f"   Action: {signal.action.upper()}")
                    logger.info(f"   Confidence: {signal.confidence:.1%}")
                    logger.info(f"   Entry: ${signal.price:.2f}")
                    logger.info(f"   Stop Loss: ${signal.stop_loss:.2f}")
                    logger.info(f"   Take Profit: ${signal.take_profit:.2f}")
                    logger.info(f"   Risk/Reward: 1:{(signal.take_profit - signal.price) / (signal.price - signal.stop_loss):.2f}")
                    
                    # Simulate trade
                    position_size = bot.capital * 0.95  # 95% position
                    trade = await bot.simulate_trade(signal, position_size)
                    bot.trades.append(trade)
                    
                    logger.info(f"\n💰 Trade Executed (Simulated):")
                    logger.info(f"   Result: {'WIN ✅' if trade['is_winner'] else 'LOSS ❌'}")
                    logger.info(f"   PnL: ${trade['pnl']:.2f}")
                    logger.info(f"   Return: {trade['return']:.2%}")
                    logger.info(f"   New Balance: ${bot.capital + sum(t['pnl'] for t in bot.trades):.2f}")
                else:
                    logger.info(f"⏸️ No signal (confidence below {bot.model_config['confidence_threshold']:.0%} threshold)")
            else:
                logger.warning("⚠️ Insufficient data")
            
            # Progress
            if bot.trades:
                wins = sum(1 for t in bot.trades if t['is_winner'])
                logger.info(f"\n📊 Progress: {len(bot.trades)} trades, {wins} wins ({wins/len(bot.trades)*100:.0f}% win rate)")
            
            # Wait 1 minute (or less for faster testing)
            if cycle < 10:
                logger.info(f"\n⏳ Waiting 60 seconds...")
                await asyncio.sleep(60)
        
        # Final report
        logger.info("\n\n" + "="*80)
        logger.info("🏁 TEST COMPLETED!")
        logger.info("="*80)
        bot.generate_final_report()
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Test interrupted by user")
        if bot.trades:
            bot.generate_final_report()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(quick_test())
