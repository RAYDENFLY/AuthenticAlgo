"""
Simple demo of Data module functionality
Shows basic usage of DataCollector and DataStorage
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import DataStorage
from core import setup_logger, get_logger


def demo_storage():
    """Demonstrate DataStorage functionality"""
    logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("🎯 Data Storage Demo")
    logger.info("=" * 60)
    
    # Create sample OHLCV data
    logger.info("\n📊 Creating sample data...")
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': [50000 + i * 10 for i in range(100)],
        'high': [50100 + i * 10 for i in range(100)],
        'low': [49900 + i * 10 for i in range(100)],
        'close': [50050 + i * 10 for i in range(100)],
        'volume': [1000 + i * 5 for i in range(100)],
    })
    sample_data.set_index('timestamp', inplace=True)
    
    logger.info(f"Created {len(sample_data)} sample candles")
    logger.info(f"\nSample data (first 5 rows):")
    logger.info(sample_data.head())
    
    # Initialize storage
    with DataStorage() as storage:
        
        # Save data
        logger.info("\n💾 Saving data to database...")
        symbol = 'BTC/USDT'
        timeframe = '1h'
        rows_saved = storage.save_ohlcv(symbol, timeframe, sample_data, replace=True)
        logger.info(f"✅ Saved {rows_saved} rows")
        
        # Load data
        logger.info("\n📂 Loading data from database...")
        df_loaded = storage.load_ohlcv(symbol, timeframe, limit=10)
        logger.info(f"✅ Loaded {len(df_loaded)} rows")
        logger.info(f"\nLoaded data:")
        logger.info(df_loaded)
        
        # Save a trade
        logger.info("\n💰 Saving sample trade...")
        trade_data = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'type': 'market',
            'entry_price': 51000.0,
            'quantity': 0.1,
            'entry_time': datetime.now(),
            'status': 'open',
            'strategy': 'demo_strategy',
            'notes': 'This is a demo trade'
        }
        trade_id = storage.save_trade(trade_data)
        logger.info(f"✅ Trade saved with ID: {trade_id}")
        
        # Get trades
        logger.info("\n📋 Retrieving trades...")
        trades_df = storage.get_trades(symbol='BTC/USDT')
        logger.info(f"✅ Found {len(trades_df)} trade(s)")
        if not trades_df.empty:
            logger.info(f"\nTrade details:")
            logger.info(trades_df[['symbol', 'side', 'entry_price', 'quantity', 'status']])
        
        # Get statistics
        logger.info("\n📊 Database statistics:")
        stats = storage.get_statistics()
        logger.info(f"  📈 OHLCV records: {stats['ohlcv_records']}")
        logger.info(f"  💼 Total trades: {stats['total_trades']}")
        logger.info(f"  🪙 Symbols: {stats['symbols']}")
        logger.info(f"  💾 Database: {stats['database_path']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Demo completed successfully!")
    logger.info("=" * 60)


def demo_collector_info():
    """Show DataCollector capabilities"""
    logger = get_logger()
    
    logger.info("\n" + "=" * 60)
    logger.info("📡 DataCollector Capabilities")
    logger.info("=" * 60)
    
    logger.info("\nDataCollector can:")
    logger.info("  ✅ Fetch OHLCV (candlestick) data")
    logger.info("  ✅ Fetch ticker (current prices)")
    logger.info("  ✅ Fetch order book (bids/asks)")
    logger.info("  ✅ Fetch recent trades")
    logger.info("  ✅ Fetch funding rates (futures)")
    logger.info("  ✅ Handle pagination for large date ranges")
    logger.info("  ✅ Automatic retry on network errors")
    logger.info("  ✅ Rate limiting protection")
    
    logger.info("\nExample usage:")
    logger.info("""
    from data import DataCollector
    
    # Initialize collector
    collector = DataCollector(exchange_name='binance', testnet=True)
    
    # Fetch recent data
    df = collector.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
    
    # Fetch date range
    df_range = collector.fetch_ohlcv_range(
        'BTC/USDT', '1h',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )
    
    # Get current price
    ticker = collector.fetch_ticker('BTC/USDT')
    print(f"BTC Price: ${ticker['last']}")
    """)


def main():
    """Main demo function"""
    # Setup logger
    setup_logger(log_level='INFO')
    logger = get_logger()
    
    logger.info("🚀 Bot Trading V2 - Data Module Demo")
    logger.info("=" * 60)
    
    try:
        # Demo storage (works offline)
        demo_storage()
        
        # Show collector capabilities
        demo_collector_info()
        
        logger.info("\n" + "=" * 60)
        logger.info("✨ Phase 1 Data Module: COMPLETE! ✨")
        logger.info("=" * 60)
        logger.info("\n📝 What we built:")
        logger.info("  ✅ DataCollector - Fetch data from exchanges")
        logger.info("  ✅ DataStorage - SQLite database integration")
        logger.info("  ✅ OHLCV data management")
        logger.info("  ✅ Trade history tracking")
        logger.info("  ✅ Robust error handling")
        logger.info("  ✅ Clean, documented code")
        
        logger.info("\n🚀 Ready for Phase 2: Technical Indicators!")
        
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
