"""
Test script for Data module
Tests DataCollector and DataStorage functionality
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import DataCollector, DataStorage
from core import setup_logger, get_logger


def test_data_collector():
    """Test DataCollector functionality"""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("Testing DataCollector")
    logger.info("=" * 60)
    
    try:
        # Initialize collector
        with DataCollector(exchange_name='binance', testnet=True) as collector:
            
            # Test 1: Get available symbols
            logger.info("\n📊 Test 1: Get available symbols")
            symbols = collector.get_available_symbols()
            logger.info(f"Found {len(symbols)} symbols")
            logger.info(f"First 5 symbols: {symbols[:5]}")
            
            # Test 2: Fetch ticker
            logger.info("\n📊 Test 2: Fetch ticker")
            symbol = 'BTC/USDT'
            ticker = collector.fetch_ticker(symbol)
            logger.info(f"Ticker for {symbol}:")
            logger.info(f"  Last Price: ${ticker['last']:,.2f}")
            logger.info(f"  Bid: ${ticker['bid']:,.2f}")
            logger.info(f"  Ask: ${ticker['ask']:,.2f}")
            logger.info(f"  24h Volume: {ticker['baseVolume']:,.2f}")
            
            # Test 3: Fetch OHLCV
            logger.info("\n📊 Test 3: Fetch OHLCV (1 hour data)")
            df = collector.fetch_ohlcv(
                symbol=symbol,
                timeframe='1h',
                limit=100
            )
            logger.info(f"Fetched {len(df)} candles")
            logger.info(f"\nFirst 5 rows:")
            logger.info(df.head())
            logger.info(f"\nLast 5 rows:")
            logger.info(df.tail())
            
            # Test 4: Fetch OHLCV range
            logger.info("\n📊 Test 4: Fetch OHLCV range (7 days)")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            df_range = collector.fetch_ohlcv_range(
                symbol=symbol,
                timeframe='1h',
                start_date=start_date,
                end_date=end_date
            )
            logger.info(f"Fetched {len(df_range)} candles for date range")
            logger.info(f"Date range: {df_range.index[0]} to {df_range.index[-1]}")
            
            # Test 5: Fetch order book
            logger.info("\n📊 Test 5: Fetch order book")
            order_book = collector.fetch_order_book(symbol, limit=5)
            logger.info(f"Order book for {symbol}:")
            logger.info(f"  Top 3 bids: {order_book['bids'][:3]}")
            logger.info(f"  Top 3 asks: {order_book['asks'][:3]}")
            
            logger.info("\n✅ DataCollector tests completed successfully!")
            return df  # Return for storage testing
            
    except Exception as e:
        logger.error(f"❌ DataCollector test failed: {e}")
        raise


def test_data_storage(df):
    """Test DataStorage functionality"""
    logger = get_logger()
    logger.info("\n" + "=" * 60)
    logger.info("Testing DataStorage")
    logger.info("=" * 60)
    
    try:
        # Initialize storage
        with DataStorage() as storage:
            
            # Test 1: Save OHLCV data
            logger.info("\n💾 Test 1: Save OHLCV data")
            symbol = 'BTC/USDT'
            timeframe = '1h'
            rows_saved = storage.save_ohlcv(symbol, timeframe, df)
            logger.info(f"Saved {rows_saved} rows to database")
            
            # Test 2: Load OHLCV data
            logger.info("\n💾 Test 2: Load OHLCV data")
            df_loaded = storage.load_ohlcv(symbol, timeframe, limit=10)
            logger.info(f"Loaded {len(df_loaded)} rows from database")
            logger.info(f"\nLoaded data:")
            logger.info(df_loaded)
            
            # Test 3: Load with date filter
            logger.info("\n💾 Test 3: Load with date filter")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3)
            df_filtered = storage.load_ohlcv(
                symbol, timeframe,
                start_date=start_date,
                end_date=end_date
            )
            logger.info(f"Loaded {len(df_filtered)} rows with date filter")
            
            # Test 4: Save trade
            logger.info("\n💾 Test 4: Save trade record")
            trade_data = {
                'symbol': 'BTC/USDT',
                'side': 'long',
                'type': 'market',
                'entry_price': 50000.0,
                'quantity': 0.1,
                'entry_time': datetime.now(),
                'status': 'open',
                'strategy': 'test_strategy',
                'notes': 'Test trade'
            }
            trade_id = storage.save_trade(trade_data)
            logger.info(f"Saved trade with ID: {trade_id}")
            
            # Test 5: Get trades
            logger.info("\n💾 Test 5: Get trade history")
            trades_df = storage.get_trades(limit=10)
            logger.info(f"Retrieved {len(trades_df)} trades")
            logger.info(f"\nTrades:")
            logger.info(trades_df)
            
            # Test 6: Get statistics
            logger.info("\n💾 Test 6: Get database statistics")
            stats = storage.get_statistics()
            logger.info(f"Database Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            logger.info("\n✅ DataStorage tests completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ DataStorage test failed: {e}")
        raise


def main():
    """Main test function"""
    # Setup logger
    setup_logger(log_level='INFO')
    logger = get_logger()
    
    logger.info("🚀 Starting Data Module Tests")
    logger.info("=" * 60)
    
    try:
        # Test DataCollector
        df = test_data_collector()
        
        # Test DataStorage
        test_data_storage(df)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 All tests passed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\n❌ Tests failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
