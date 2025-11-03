"""
Data Collector for fetching market data from exchanges
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import ccxt
from ccxt.base.errors import (
    NetworkError,
    ExchangeError as CCXTExchangeError,
    RateLimitExceeded
)

from core import get_logger, get_config
from core.exceptions import ExchangeError, DataError
from core.utils import retry_on_exception, timestamp_to_datetime


class DataCollector:
    """Collects historical and real-time data from exchanges"""
    
    def __init__(self, exchange_name: str = 'binance', testnet: bool = True):
        """
        Initialize data collector
        
        Args:
            exchange_name: Name of the exchange (binance, asterdex, etc.)
            testnet: Use testnet if True
        """
        self.logger = get_logger()
        self.config = get_config()
        self.exchange_name = exchange_name.lower()
        self.testnet = testnet
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        self.logger.info(f"DataCollector initialized for {exchange_name} (testnet={testnet})")
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Initialize CCXT exchange instance
        
        Returns:
            CCXT exchange instance
            
        Raises:
            ExchangeError: If initialization fails
        """
        try:
            # Get API credentials from config
            api_key = self.config.get_env(f'{self.exchange_name.upper()}_API_KEY', '')
            api_secret = self.config.get_env(f'{self.exchange_name.upper()}_API_SECRET', '')
            
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange_params = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures by default
                }
            }
            
            # Set testnet if enabled
            if self.testnet and hasattr(exchange_class, 'set_sandbox_mode'):
                exchange_params['options']['sandboxMode'] = True
            
            exchange = exchange_class(exchange_params)
            
            # Test connection
            exchange.load_markets()
            
            self.logger.info(f"Successfully connected to {self.exchange_name}")
            return exchange
            
        except Exception as e:
            error_msg = f"Failed to initialize {self.exchange_name}: {e}"
            self.logger.error(error_msg)
            raise ExchangeError(error_msg)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            since: Start datetime (if None, fetches recent data)
            limit: Number of candles to fetch (max 1000)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            DataError: If data fetching fails
        """
        try:
            self.logger.debug(f"Fetching OHLCV for {symbol} ({timeframe})")
            
            # Convert datetime to timestamp
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)
            
            # Fetch data with retry
            def fetch():
                return self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since_ts,
                    limit=limit
                )
            
            ohlcv = retry_on_exception(fetch, max_retries=3, delay=1.0)
            
            if not ohlcv:
                raise DataError(f"No data returned for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df
            
        except (NetworkError, RateLimitExceeded) as e:
            error_msg = f"Network error fetching OHLCV for {symbol}: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
        except CCXTExchangeError as e:
            error_msg = f"Exchange error fetching OHLCV for {symbol}: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching OHLCV for {symbol}: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a date range (handles pagination)
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with all data in the range
        """
        self.logger.info(f"Fetching OHLCV range for {symbol}: {start_date} to {end_date}")
        
        all_data = []
        current_date = start_date
        
        # Calculate timeframe in seconds
        timeframe_map = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200,
            '1d': 86400, '1w': 604800
        }
        
        if timeframe not in timeframe_map:
            raise DataError(f"Invalid timeframe: {timeframe}")
        
        tf_seconds = timeframe_map[timeframe]
        max_candles = 1000  # Most exchanges limit to 1000 candles per request
        
        while current_date < end_date:
            # Fetch batch
            df = self.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_date,
                limit=max_candles
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # Move to next batch
            last_timestamp = df.index[-1]
            current_date = last_timestamp + timedelta(seconds=tf_seconds)
            
            # Sleep to respect rate limits
            time.sleep(0.5)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep='first')]  # Remove duplicates
        result = result.sort_index()
        
        # Filter to exact range
        result = result[(result.index >= start_date) & (result.index <= end_date)]
        
        self.logger.info(f"Fetched total {len(result)} candles for {symbol}")
        return result
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker (latest price info)
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with ticker information
        """
        try:
            self.logger.debug(f"Fetching ticker for {symbol}")
            
            def fetch():
                return self.exchange.fetch_ticker(symbol)
            
            ticker = retry_on_exception(fetch, max_retries=3, delay=0.5)
            
            self.logger.debug(f"Ticker for {symbol}: {ticker['last']}")
            return ticker
            
        except Exception as e:
            error_msg = f"Error fetching ticker for {symbol}: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch order book (bids and asks)
        
        Args:
            symbol: Trading pair symbol
            limit: Depth limit (number of orders)
            
        Returns:
            Dictionary with bids and asks
        """
        try:
            self.logger.debug(f"Fetching order book for {symbol}")
            
            def fetch():
                return self.exchange.fetch_order_book(symbol, limit=limit)
            
            order_book = retry_on_exception(fetch, max_retries=3, delay=0.5)
            return order_book
            
        except Exception as e:
            error_msg = f"Error fetching order book for {symbol}: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def fetch_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch recent trades
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch
            
        Returns:
            List of trade dictionaries
        """
        try:
            self.logger.debug(f"Fetching recent trades for {symbol}")
            
            def fetch():
                return self.exchange.fetch_trades(symbol, limit=limit)
            
            trades = retry_on_exception(fetch, max_retries=3, delay=0.5)
            return trades
            
        except Exception as e:
            error_msg = f"Error fetching trades for {symbol}: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Fetch current funding rate (for futures)
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current funding rate or None if not available
        """
        try:
            if not hasattr(self.exchange, 'fetch_funding_rate'):
                self.logger.warning(f"Funding rate not supported by {self.exchange_name}")
                return None
            
            self.logger.debug(f"Fetching funding rate for {symbol}")
            
            def fetch():
                return self.exchange.fetch_funding_rate(symbol)
            
            funding = retry_on_exception(fetch, max_retries=3, delay=0.5)
            
            rate = funding.get('fundingRate', None)
            self.logger.debug(f"Funding rate for {symbol}: {rate}")
            return rate
            
        except Exception as e:
            self.logger.warning(f"Error fetching funding rate for {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols
        
        Returns:
            List of symbol strings
        """
        try:
            if not self.exchange.markets:
                self.exchange.load_markets()
            
            symbols = list(self.exchange.markets.keys())
            self.logger.info(f"Found {len(symbols)} available symbols")
            return symbols
            
        except Exception as e:
            error_msg = f"Error getting available symbols: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with symbol information
        """
        try:
            if not self.exchange.markets:
                self.exchange.load_markets()
            
            if symbol not in self.exchange.markets:
                raise DataError(f"Symbol {symbol} not found")
            
            return self.exchange.markets[symbol]
            
        except Exception as e:
            error_msg = f"Error getting symbol info for {symbol}: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def close(self) -> None:
        """Close exchange connection"""
        if hasattr(self.exchange, 'close') and callable(self.exchange.close):
            self.exchange.close()
        self.logger.info(f"Closed connection to {self.exchange_name}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __repr__(self) -> str:
        return f"DataCollector(exchange={self.exchange_name}, testnet={self.testnet})"
