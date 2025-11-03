"""
Script: Collect Large Historical Dataset
Collects 3-6 months of data for multiple symbols and timeframes
Optimized for backtesting and ML training
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from datetime import datetime, timedelta
from data.asterdex_collector import AsterDEXDataCollector
from core.logger import setup_logger
from loguru import logger

# Setup
setup_logger()
load_dotenv()


async def collect_all_data():
    """Collect comprehensive dataset for trading"""
    logger.info("="*80)
    logger.info("🚀 LARGE DATASET COLLECTION - AsterDEX Futures")
    logger.info("="*80)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    # Configuration
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    intervals = ['15m', '1h', '4h', '1d']
    days = 90  # 3 months
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"\n📊 Collection Configuration:")
    logger.info(f"   Symbols:     {', '.join(symbols)} ({len(symbols)} pairs)")
    logger.info(f"   Timeframes:  {', '.join(intervals)} ({len(intervals)} timeframes)")
    logger.info(f"   Period:      {days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    logger.info(f"   Total tasks: {len(symbols) * len(intervals)} collections")
    logger.info(f"\n⏳ Estimated time: 5-10 minutes (depending on network)")
    logger.info(f"🔄 Rate limit friendly: 0.2s delay between requests\n")
    
    start_time = datetime.now()
    total_candles = 0
    successful = 0
    failed = 0
    
    try:
        for symbol_idx, symbol in enumerate(symbols, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"📈 Processing {symbol} ({symbol_idx}/{len(symbols)})")
            logger.info(f"{'='*80}")
            
            for interval_idx, interval in enumerate(intervals, 1):
                try:
                    logger.info(f"\n🔄 [{symbol_idx}.{interval_idx}] {symbol} {interval}...")
                    
                    df = await collector.collect_klines(
                        symbol=symbol,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date,
                        save_to_file=True
                    )
                    
                    if not df.empty:
                        # Validate data quality
                        validation = await collector.validate_data_quality(df)
                        
                        candles = len(df)
                        total_candles += candles
                        successful += 1
                        
                        status = "✅ GOOD" if validation['valid'] else "⚠️ ISSUES"
                        logger.info(f"   ✅ Success: {candles:>5} candles | Quality: {status}")
                        
                        if not validation['valid']:
                            logger.warning(f"   ⚠️ Issues: {', '.join(validation['issues'])}")
                    else:
                        failed += 1
                        logger.error(f"   ❌ Failed: No data received")
                    
                    # Rate limit delay
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    failed += 1
                    logger.error(f"   ❌ Error: {e}")
                    continue
            
            # Longer delay between symbols
            if symbol_idx < len(symbols):
                logger.info(f"\n💤 Cooling down 3 seconds before next symbol...")
                await asyncio.sleep(3)
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ COLLECTION COMPLETED!")
        logger.info(f"{'='*80}")
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Duration:         {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"   Total candles:    {total_candles:,}")
        logger.info(f"   Successful:       {successful}/{len(symbols) * len(intervals)}")
        logger.info(f"   Failed:           {failed}/{len(symbols) * len(intervals)}")
        logger.info(f"   Success rate:     {(successful/(len(symbols) * len(intervals))*100):.1f}%")
        logger.info(f"   Avg per task:     {duration/(len(symbols) * len(intervals)):.1f}s")
        
        logger.info(f"\n📁 Data saved to: data/historical/")
        logger.info(f"\n💡 Next steps:")
        logger.info(f"   1. View collected files: ls data/historical/")
        logger.info(f"   2. Run backtests: python demo/demo_backtesting.py")
        logger.info(f"   3. Train ML models: python demo/demo_ml.py")
        logger.info(f"   4. Paper trade: python main.py --paper-trading")
        
    except Exception as e:
        logger.error(f"\n❌ Collection failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        await collector.close()
        logger.info(f"\n{'='*80}")


async def collect_specific(symbol: str = 'BTCUSDT', interval: str = '1h', days: int = 90):
    """Collect data for specific symbol and timeframe"""
    logger.info("="*80)
    logger.info(f"📊 Collecting {symbol} {interval} - {days} days")
    logger.info("="*80)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"\n📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"⏳ Please wait...\n")
        
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
            logger.info(f"\n✅ Success!")
            logger.info(f"   Candles:      {len(df):,}")
            logger.info(f"   Duration:     {duration:.1f}s")
            logger.info(f"   Date range:   {df.index.min()} to {df.index.max()}")
            logger.info(f"   Price range:  ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
            
            # Validate
            validation = await collector.validate_data_quality(df)
            if validation['valid']:
                logger.info(f"   Quality:      ✅ EXCELLENT")
            else:
                logger.warning(f"   Quality:      ⚠️ Has issues")
                for issue in validation['issues']:
                    logger.warning(f"      - {issue}")
        else:
            logger.error("❌ No data collected")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        await collector.close()


async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        # Specific collection: python collect_large_dataset.py ETHUSDT 1h 90
        symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT'
        interval = sys.argv[2] if len(sys.argv) > 2 else '1h'
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 90
        
        await collect_specific(symbol, interval, days)
    else:
        # Collect all
        await collect_all_data()


if __name__ == "__main__":
    asyncio.run(main())
