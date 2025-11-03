"""
AsterDEX Data Collector
Collects historical and real-time market data from AsterDEX Futures
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from execution.asterdex import AsterDEXFutures
from core.logger import setup_logger
from loguru import logger
from core.exceptions import DataError

# Setup logger
setup_logger()


class AsterDEXDataCollector:
    """
    Collects market data from AsterDEX Futures exchange
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AsterDEX data collector
        
        Args:
            config: Configuration dictionary with API credentials
        """
        self.exchange = AsterDEXFutures(config)
        self.data_dir = Path("data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ AsterDEX Data Collector initialized")
    
    async def collect_klines(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        save_to_file: bool = True
    ) -> pd.DataFrame:
        """
        Collect historical kline/candlestick data
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start datetime
            end_date: End datetime (default: now)
            save_to_file: Save to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"📊 Collecting klines: {symbol} {interval} from {start_date} to {end_date}")
        
        all_klines = []
        current_time = start_date
        
        # Calculate interval in milliseconds
        interval_map = {
            '1m': 60000,
            '3m': 180000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '2h': 7200000,
            '4h': 14400000,
            '6h': 21600000,
            '12h': 43200000,
            '1d': 86400000,
            '1w': 604800000
        }
        
        if interval not in interval_map:
            raise DataError(f"Invalid interval: {interval}")
        
        interval_ms = interval_map[interval]
        max_limit = 1500  # AsterDEX max limit per request
        
        try:
            while current_time < end_date:
                start_time_ms = int(current_time.timestamp() * 1000)
                end_time_ms = int(end_date.timestamp() * 1000)
                
                # Fetch batch
                klines = await self.exchange.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=max_limit,
                    start_time=start_time_ms,
                    end_time=end_time_ms
                )
                
                if not klines:
                    logger.warning(f"⚠️ No more data available after {current_time}")
                    break
                
                all_klines.extend(klines)
                
                # Move to next batch
                last_kline_time = klines[-1][0]  # Open time of last kline
                current_time = datetime.fromtimestamp(last_kline_time / 1000) + timedelta(milliseconds=interval_ms)
                
                logger.info(f"   Fetched {len(klines)} klines, total: {len(all_klines)}")
                
                # Avoid rate limits
                await asyncio.sleep(0.2)
            
            # Convert to DataFrame
            df = self._klines_to_dataframe(all_klines)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Filter to exact range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"✅ Collected {len(df)} klines for {symbol} ({interval})")
            
            # Save to file
            if save_to_file and not df.empty:
                filename = self._generate_filename(symbol, interval, start_date, end_date)
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"💾 Saved to: {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error collecting klines: {e}")
            raise DataError(f"Failed to collect klines: {e}")
    
    def _klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        Convert klines list to DataFrame
        
        Args:
            klines: List of klines [timestamp, open, high, low, close, volume, ...]
            
        Returns:
            DataFrame with OHLCV data
        """
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def _generate_filename(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """
        Generate filename for saved data
        
        Args:
            symbol: Trading symbol
            interval: Kline interval
            start_date: Start date
            end_date: End date
            
        Returns:
            Filename string
        """
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        symbol_clean = symbol.replace('/', '_')
        
        return f"asterdex_{symbol_clean}_{interval}_{start_str}_to_{end_str}.csv"
    
    async def collect_multiple_symbols(
        self,
        symbols: List[str],
        interval: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple symbols
        
        Args:
            symbols: List of trading symbols
            interval: Kline interval
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"\n📈 Processing {symbol}...")
                df = await self.collect_klines(symbol, interval, start_date, end_date)
                results[symbol] = df
                
                # Small delay between symbols
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Failed to collect {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    async def collect_recent_trades(
        self,
        symbol: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Collect recent trades
        
        Args:
            symbol: Trading symbol
            limit: Number of trades
            
        Returns:
            DataFrame with trade data
        """
        logger.info(f"📊 Collecting recent {limit} trades for {symbol}")
        
        try:
            trades = await self.exchange.get_recent_trades(symbol, limit=limit)
            
            if not trades:
                logger.warning(f"⚠️ No trades found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Convert timestamp
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Collected {len(df)} trades for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error collecting trades: {e}")
            raise DataError(f"Failed to collect trades: {e}")
    
    async def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get market summary for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with market summary
        """
        logger.info(f"📊 Getting market summary for {symbol}")
        
        try:
            # Get 24h ticker
            ticker = await self.exchange.get_ticker_24h(symbol)
            
            # Get current price
            price_data = await self.exchange.get_ticker_price(symbol)
            
            summary = {
                'symbol': symbol,
                'price': float(price_data.get('price', 0)),
                'price_change': float(ticker.get('priceChange', 0)),
                'price_change_percent': float(ticker.get('priceChangePercent', 0)),
                'high_24h': float(ticker.get('highPrice', 0)),
                'low_24h': float(ticker.get('lowPrice', 0)),
                'volume_24h': float(ticker.get('volume', 0)),
                'quote_volume_24h': float(ticker.get('quoteVolume', 0)),
                'trades_24h': int(ticker.get('count', 0)),
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ {symbol}: ${summary['price']:,.2f} ({summary['price_change_percent']:+.2f}%)")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting market summary: {e}")
            raise DataError(f"Failed to get market summary: {e}")
    
    def load_saved_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load previously saved data from file
        
        Args:
            symbol: Trading symbol
            interval: Kline interval
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame or None if not found
        """
        filename = self._generate_filename(symbol, interval, start_date, end_date)
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"⚠️ Data file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            logger.info(f"✅ Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    async def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        if df.empty:
            return {'valid': False, 'reason': 'Empty DataFrame'}
        
        validation = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.index.duplicated().sum(),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max()
            },
            'price_stats': {
                'min': df['close'].min(),
                'max': df['close'].max(),
                'mean': df['close'].mean(),
                'std': df['close'].std()
            }
        }
        
        # Check for issues
        issues = []
        
        # Check missing values
        total_missing = sum(validation['missing_values'].values())
        if total_missing > 0:
            issues.append(f"{total_missing} missing values")
        
        # Check duplicates
        if validation['duplicate_rows'] > 0:
            issues.append(f"{validation['duplicate_rows']} duplicate rows")
        
        # Check for zero or negative prices
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            issues.append("Zero or negative prices found")
        
        # Check for unrealistic price changes (>50% in one candle)
        price_changes = df['close'].pct_change().abs()
        extreme_changes = (price_changes > 0.5).sum()
        if extreme_changes > 0:
            issues.append(f"{extreme_changes} extreme price changes (>50%)")
        
        validation['issues'] = issues
        validation['valid'] = len(issues) == 0
        
        if validation['valid']:
            logger.info("✅ Data quality: GOOD")
        else:
            logger.warning(f"⚠️ Data quality issues: {', '.join(issues)}")
        
        return validation
    
    async def close(self):
        """Close exchange connection"""
        await self.exchange.close()
        logger.info("✅ AsterDEX Data Collector closed")


async def quick_collect(
    symbol: str = 'BTCUSDT',
    interval: str = '1h',
    days: int = 90
) -> pd.DataFrame:
    """
    Quick function to collect recent data
    
    Args:
        symbol: Trading symbol
        interval: Kline interval
        days: Number of days to collect
        
    Returns:
        DataFrame with data
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    collector = AsterDEXDataCollector(config)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = await collector.collect_klines(symbol, interval, start_date, end_date)
        
        # Validate
        validation = await collector.validate_data_quality(df)
        logger.info(f"📊 Validation: {validation}")
        
        return df
        
    finally:
        await collector.close()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = 'BTCUSDT'
    
    if len(sys.argv) > 2:
        interval = sys.argv[2]
    else:
        interval = '1h'
    
    if len(sys.argv) > 3:
        days = int(sys.argv[3])
    else:
        days = 90
    
    logger.info(f"🚀 Collecting {symbol} {interval} for {days} days")
    df = asyncio.run(quick_collect(symbol, interval, days))
    
    if not df.empty:
        logger.info(f"\n📊 Data Preview:")
        logger.info(f"\n{df.head()}")
        logger.info(f"\n{df.tail()}")
        logger.info(f"\n📈 Statistics:")
        logger.info(f"\n{df.describe()}")
