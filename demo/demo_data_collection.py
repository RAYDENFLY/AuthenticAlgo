"""
Demo: AsterDEX Data Collection
Demonstrates how to collect and manage historical market data
"""

import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from data.asterdex_collector import AsterDEXDataCollector
from core.logger import setup_logger
from loguru import logger

# Setup
setup_logger()
load_dotenv()


async def demo_collect_single_symbol():
    """Demo 1: Collect data for single symbol"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Collect Single Symbol Data")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    try:
        # Collect 7 days of 1h data for BTC/USDT
        symbol = 'BTCUSDT'
        interval = '1h'
        days = 7
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"\n📊 Collecting {symbol} {interval} data")
        logger.info(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Expected candles: ~{days * 24}")
        
        df = await collector.collect_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            save_to_file=True
        )
        
        if not df.empty:
            logger.info(f"\n✅ Successfully collected {len(df)} candles")
            logger.info(f"\n📈 Data Preview (first 5):")
            logger.info(f"\n{df.head().to_string()}")
            logger.info(f"\n📈 Data Preview (last 5):")
            logger.info(f"\n{df.tail().to_string()}")
            
            # Validate data quality
            validation = await collector.validate_data_quality(df)
            logger.info(f"\n🔍 Data Quality:")
            logger.info(f"   Total rows: {validation['total_rows']}")
            logger.info(f"   Missing values: {sum(validation['missing_values'].values())}")
            logger.info(f"   Duplicate rows: {validation['duplicate_rows']}")
            logger.info(f"   Valid: {'✅ YES' if validation['valid'] else '❌ NO'}")
            
            if validation['issues']:
                logger.warning(f"   Issues: {', '.join(validation['issues'])}")
        else:
            logger.error("❌ No data collected")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        await collector.close()


async def demo_collect_multiple_timeframes():
    """Demo 2: Collect multiple timeframes for same symbol"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: Collect Multiple Timeframes")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    try:
        symbol = 'BTCUSDT'
        intervals = ['15m', '1h', '4h']
        days = 30  # 1 month
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"\n📊 Collecting {symbol} data for {days} days")
        logger.info(f"   Timeframes: {', '.join(intervals)}")
        
        results = {}
        
        for interval in intervals:
            logger.info(f"\n🔄 Processing {interval}...")
            
            df = await collector.collect_klines(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                save_to_file=True
            )
            
            results[interval] = df
            
            if not df.empty:
                logger.info(f"   ✅ {interval}: {len(df)} candles")
                logger.info(f"   Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
                logger.info(f"   Avg volume: {df['volume'].mean():,.2f}")
            
            await asyncio.sleep(1)  # Rate limit
        
        # Summary
        logger.info(f"\n📊 Collection Summary:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Period: {days} days")
        for interval, df in results.items():
            logger.info(f"   {interval:>5}: {len(df):>5} candles")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        await collector.close()


async def demo_collect_multiple_symbols():
    """Demo 3: Collect data for multiple symbols"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: Collect Multiple Symbols")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    try:
        # Popular trading pairs
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
        interval = '1h'
        days = 14  # 2 weeks
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"\n📊 Collecting data for {len(symbols)} symbols")
        logger.info(f"   Symbols: {', '.join(symbols)}")
        logger.info(f"   Interval: {interval}")
        logger.info(f"   Period: {days} days")
        
        results = await collector.collect_multiple_symbols(
            symbols=symbols,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        # Summary
        logger.info(f"\n📊 Collection Summary:")
        for symbol, df in results.items():
            if not df.empty:
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
                emoji = "🟢" if price_change > 0 else "🔴"
                logger.info(f"   {emoji} {symbol:>10}: {len(df):>4} candles, "
                          f"${df['close'].iloc[-1]:>8,.2f} ({price_change:+.2f}%)")
            else:
                logger.warning(f"   ⚠️  {symbol:>10}: NO DATA")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        await collector.close()


async def demo_market_summary():
    """Demo 4: Get market summary"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Market Summary")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    try:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
        
        logger.info(f"\n📊 Getting market summary for {len(symbols)} symbols\n")
        
        for symbol in symbols:
            try:
                summary = await collector.get_market_summary(symbol)
                
                # Format output
                price = summary['price']
                change_pct = summary['price_change_percent']
                volume = summary['volume_24h']
                high = summary['high_24h']
                low = summary['low_24h']
                
                emoji = "🟢" if change_pct > 0 else "🔴"
                
                logger.info(f"{emoji} {symbol:>10}")
                logger.info(f"   Price:        ${price:>12,.2f}")
                logger.info(f"   24h Change:   {change_pct:>12.2f}%")
                logger.info(f"   24h High:     ${high:>12,.2f}")
                logger.info(f"   24h Low:      ${low:>12,.2f}")
                logger.info(f"   24h Volume:   {volume:>12,.2f}")
                logger.info("")
                
                await asyncio.sleep(0.5)  # Rate limit
                
            except Exception as e:
                logger.error(f"   ❌ Error for {symbol}: {e}")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    finally:
        await collector.close()


async def demo_load_saved_data():
    """Demo 5: Load previously saved data"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 5: Load Saved Data")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    try:
        # Try to load previously saved data
        symbol = 'BTCUSDT'
        interval = '1h'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"\n📂 Trying to load saved data for {symbol} {interval}")
        
        df = collector.load_saved_data(symbol, interval, start_date, end_date)
        
        if df is not None:
            logger.info(f"✅ Loaded {len(df)} rows from file")
            logger.info(f"\n📈 Data Preview:")
            logger.info(f"\n{df.head().to_string()}")
        else:
            logger.warning("⚠️ No saved data found")
            logger.info("\n💡 Run Demo 1 first to collect and save data")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    finally:
        await collector.close()


async def demo_large_dataset():
    """Demo 6: Collect large dataset (3 months)"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 6: Collect Large Dataset (3 months)")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    try:
        symbol = 'BTCUSDT'
        interval = '1h'
        days = 90  # 3 months
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"\n📊 Collecting LARGE dataset")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Interval: {interval}")
        logger.info(f"   Period: {days} days (~{days * 24} candles)")
        logger.info(f"\n⏳ This may take a few minutes...")
        
        start_time = datetime.now()
        
        df = await collector.collect_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            save_to_file=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if not df.empty:
            logger.info(f"\n✅ Collection completed in {duration:.1f} seconds")
            logger.info(f"\n📊 Dataset Statistics:")
            logger.info(f"   Total candles: {len(df)}")
            logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"   Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
            logger.info(f"   Avg volume: {df['volume'].mean():,.2f}")
            logger.info(f"   Total volume: {df['volume'].sum():,.2f}")
            
            # Validate
            validation = await collector.validate_data_quality(df)
            if validation['valid']:
                logger.info(f"\n✅ Data quality: EXCELLENT")
            else:
                logger.warning(f"\n⚠️ Data quality issues detected")
                for issue in validation['issues']:
                    logger.warning(f"   - {issue}")
            
            logger.info(f"\n💡 Data ready for:")
            logger.info(f"   - Backtesting strategies")
            logger.info(f"   - ML model training")
            logger.info(f"   - Technical analysis")
            logger.info(f"   - Pattern recognition")
        else:
            logger.error("❌ No data collected")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        await collector.close()


async def main():
    """Run all demos"""
    logger.info("="*60)
    logger.info("AsterDEX Data Collection Demo")
    logger.info("="*60)
    
    # Demo 1: Single symbol
    await demo_collect_single_symbol()
    await asyncio.sleep(2)
    
    # Demo 2: Multiple timeframes
    await demo_collect_multiple_timeframes()
    await asyncio.sleep(2)
    
    # Demo 3: Multiple symbols
    await demo_collect_multiple_symbols()
    await asyncio.sleep(2)
    
    # Demo 4: Market summary
    await demo_market_summary()
    await asyncio.sleep(2)
    
    # Demo 5: Load saved data
    await demo_load_saved_data()
    await asyncio.sleep(2)
    
    # Demo 6: Large dataset (optional - takes time)
    logger.info("\n💡 Running Demo 6 (Large Dataset - 3 months)...")
    await demo_large_dataset()
    
    logger.info("\n" + "="*60)
    logger.info("✅ All demos completed!")
    logger.info("="*60)
    logger.info("\n📁 Check 'data/historical/' folder for saved CSV files")
    logger.info("\n💡 Next steps:")
    logger.info("   1. Collect more data: python -m data.asterdex_collector ETHUSDT 1h 90")
    logger.info("   2. Use data for backtesting: python demo_backtesting.py")
    logger.info("   3. Train ML models: python demo_ml.py")
    logger.info("   4. Run live strategies: python main.py")


if __name__ == "__main__":
    asyncio.run(main())
