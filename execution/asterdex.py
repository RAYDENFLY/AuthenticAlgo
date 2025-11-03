"""
AsterDEX Futures Exchange Integration
Supports Binance-compatible Futures API
"""

import hmac
import hashlib
import time
import urllib.parse
from typing import Dict, List, Optional, Any
from decimal import Decimal
import aiohttp
import asyncio
from datetime import datetime

from core.logger import setup_logger
from core.exceptions import ExchangeError, AuthenticationError, RateLimitError

logger = setup_logger(__name__)


class AsterDEXFutures:
    """
    AsterDEX Futures Exchange Adapter
    Compatible with Binance Futures API format
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AsterDEX Futures client
        
        Args:
            config: Configuration dictionary containing API credentials and settings
        """
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.base_url = config.get('base_url', 'https://fapi.asterdex.com')
        self.ws_url = config.get('ws_url', 'wss://fstream.asterdex.com')
        
        # Futures settings
        self.leverage = config.get('leverage', 10)
        self.margin_type = config.get('margin_type', 'ISOLATED')
        self.position_mode = config.get('position_mode', 'ONE_WAY')
        
        # Rate limit settings
        self.rate_limits = config.get('rate_limits', {
            'requests_per_minute': 2400,
            'orders_per_minute': 1200,
            'weight_per_minute': 2400
        })
        
        # API settings
        self.recv_window = config.get('recv_window', 5000)
        self.timeout = config.get('timeout', 30)
        
        # Internal tracking
        self.request_count = 0
        self.order_count = 0
        self.weight_used = 0
        self.last_reset = time.time()
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"✅ AsterDEX Futures initialized - Leverage: {self.leverage}x, Margin: {self.margin_type}")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for authenticated requests
        
        Args:
            params: Request parameters
            
        Returns:
            Signature string
        """
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _prepare_params(self, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Dict[str, Any]:
        """
        Prepare request parameters with timestamp and signature
        
        Args:
            params: Request parameters
            signed: Whether request needs signature
            
        Returns:
            Prepared parameters
        """
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = self.recv_window
            params['signature'] = self._generate_signature(params)
        
        return params
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers with API key
        
        Returns:
            Headers dictionary
        """
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset counters every minute
        if current_time - self.last_reset >= 60:
            self.request_count = 0
            self.order_count = 0
            self.weight_used = 0
            self.last_reset = current_time
        
        # Check limits
        if self.request_count >= self.rate_limits['requests_per_minute']:
            wait_time = 60 - (current_time - self.last_reset)
            logger.warning(f"⚠️ Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.last_reset = time.time()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        weight: int = 1
    ) -> Dict[str, Any]:
        """
        Make HTTP request to AsterDEX API
        
        Args:
            method: HTTP method (GET, POST, DELETE, PUT)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether request needs signature
            weight: Request weight for rate limiting
            
        Returns:
            API response data
        """
        await self._check_rate_limit()
        
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{endpoint}"
        params = self._prepare_params(params, signed)
        headers = self._get_headers()
        
        try:
            async with self.session.request(
                method,
                url,
                params=params if method == 'GET' else None,
                json=params if method != 'GET' else None,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                # Track rate limits from headers
                self.weight_used = int(response.headers.get('X-MBX-USED-WEIGHT-1M', self.weight_used))
                self.request_count += 1
                
                data = await response.json()
                
                if response.status != 200:
                    error_code = data.get('code', 'UNKNOWN')
                    error_msg = data.get('msg', 'Unknown error')
                    
                    if error_code == -1001:
                        raise RateLimitError(f"Disconnected: {error_msg}")
                    elif error_code == -1010:
                        raise ExchangeError(f"Error message received: {error_msg}")
                    elif error_code == -2010:
                        raise ExchangeError(f"New order rejected: {error_msg}")
                    elif error_code in [-1021, -1022]:
                        raise AuthenticationError(f"Timestamp/Signature error: {error_msg}")
                    else:
                        raise ExchangeError(f"API Error {error_code}: {error_msg}")
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"❌ Connection error: {e}")
            raise ExchangeError(f"Connection failed: {e}")
        except asyncio.TimeoutError:
            logger.error(f"❌ Request timeout")
            raise ExchangeError("Request timeout")
    
    # ====================== PUBLIC ENDPOINTS ======================
    
    async def ping(self) -> bool:
        """
        Test connectivity to the API
        
        Returns:
            True if connected
        """
        try:
            await self._request('GET', '/fapi/v1/ping')
            logger.info("✅ AsterDEX: Connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ AsterDEX: Connection failed - {e}")
            return False
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information
        
        Returns:
            Exchange information
        """
        data = await self._request('GET', '/fapi/v1/exchangeInfo', weight=10)
        logger.info(f"✅ Exchange info: {len(data.get('symbols', []))} symbols")
        return data
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book (market depth)
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Depth limit (default 100, max 1000)
            
        Returns:
            Order book data
        """
        params = {'symbol': symbol, 'limit': limit}
        data = await self._request('GET', '/fapi/v1/depth', params, weight=limit//10 + 1)
        return data
    
    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get recent trades
        
        Args:
            symbol: Trading symbol
            limit: Number of trades (default 500, max 1000)
            
        Returns:
            List of recent trades
        """
        params = {'symbol': symbol, 'limit': limit}
        data = await self._request('GET', '/fapi/v1/trades', params, weight=1)
        return data
    
    async def get_ticker_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get latest price for symbol(s)
        
        Args:
            symbol: Trading symbol (optional, returns all if None)
            
        Returns:
            Price ticker data
        """
        params = {'symbol': symbol} if symbol else {}
        data = await self._request('GET', '/fapi/v1/ticker/price', params, weight=1)
        return data
    
    async def get_ticker_24h(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get 24-hour ticker statistics
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            24-hour ticker data
        """
        params = {'symbol': symbol} if symbol else {}
        weight = 1 if symbol else 40
        data = await self._request('GET', '/fapi/v1/ticker/24hr', params, weight=weight)
        return data
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        Get candlestick/kline data
        
        Args:
            symbol: Trading symbol
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of klines (default 500, max 1500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of klines
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        data = await self._request('GET', '/fapi/v1/klines', params, weight=1)
        logger.info(f"✅ Fetched {len(data)} klines for {symbol} ({interval})")
        return data
    
    # ====================== AUTHENTICATED ENDPOINTS ======================
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information
        
        Returns:
            Account information including balances and positions
        """
        data = await self._request('GET', '/fapi/v2/account', signed=True, weight=5)
        logger.info(f"✅ Account balance: {data.get('totalWalletBalance', 'N/A')} USDT")
        return data
    
    async def get_balance(self) -> List[Dict[str, Any]]:
        """
        Get account balances
        
        Returns:
            List of asset balances
        """
        data = await self._request('GET', '/fapi/v2/balance', signed=True, weight=5)
        return data
    
    async def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get current position information
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            Position information
        """
        params = {'symbol': symbol} if symbol else {}
        data = await self._request('GET', '/fapi/v2/positionRisk', params, signed=True, weight=5)
        return data
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Change initial leverage
        
        Args:
            symbol: Trading symbol
            leverage: Target leverage (1-125)
            
        Returns:
            Leverage change result
        """
        params = {'symbol': symbol, 'leverage': leverage}
        data = await self._request('POST', '/fapi/v1/leverage', params, signed=True, weight=1)
        logger.info(f"✅ {symbol} leverage set to {leverage}x")
        return data
    
    async def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """
        Change margin type (ISOLATED or CROSS)
        
        Args:
            symbol: Trading symbol
            margin_type: 'ISOLATED' or 'CROSS'
            
        Returns:
            Margin type change result
        """
        params = {'symbol': symbol, 'marginType': margin_type}
        data = await self._request('POST', '/fapi/v1/marginType', params, signed=True, weight=1)
        logger.info(f"✅ {symbol} margin type set to {margin_type}")
        return data
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: str = 'GTC',
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        close_position: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new order
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            order_type: Order type (LIMIT, MARKET, STOP, etc.)
            quantity: Order quantity
            price: Order price (for LIMIT orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            stop_price: Stop price (for STOP orders)
            reduce_only: Reduce only flag
            close_position: Close position flag
            
        Returns:
            Order creation result
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type
        }
        
        if quantity:
            params['quantity'] = quantity
        if price:
            params['price'] = price
        if time_in_force and order_type == 'LIMIT':
            params['timeInForce'] = time_in_force
        if stop_price:
            params['stopPrice'] = stop_price
        if reduce_only:
            params['reduceOnly'] = 'true'
        if close_position:
            params['closePosition'] = 'true'
        
        # Add any additional parameters
        params.update(kwargs)
        
        self.order_count += 1
        data = await self._request('POST', '/fapi/v1/order', params, signed=True, weight=1)
        logger.info(f"✅ Order created: {symbol} {side} {quantity} @ {price or 'MARKET'}")
        return data
    
    async def cancel_order(self, symbol: str, order_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Cancel an active order
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            Order cancellation result
        """
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        
        data = await self._request('DELETE', '/fapi/v1/order', params, signed=True, weight=1)
        logger.info(f"✅ Order cancelled: {symbol} #{order_id}")
        return data
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Cancel all open orders for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Cancellation result
        """
        params = {'symbol': symbol}
        data = await self._request('DELETE', '/fapi/v1/allOpenOrders', params, signed=True, weight=1)
        logger.info(f"✅ All orders cancelled: {symbol}")
        return data
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of open orders
        """
        params = {'symbol': symbol} if symbol else {}
        weight = 1 if symbol else 40
        data = await self._request('GET', '/fapi/v1/openOrders', params, signed=True, weight=weight)
        return data
    
    async def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Query order status
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            
        Returns:
            Order information
        """
        params = {'symbol': symbol, 'orderId': order_id}
        data = await self._request('GET', '/fapi/v1/order', params, signed=True, weight=2)
        return data
    
    async def get_trades(
        self,
        symbol: str,
        limit: int = 500,
        from_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get account trade list
        
        Args:
            symbol: Trading symbol
            limit: Number of trades (default 500, max 1000)
            from_id: Trade ID to fetch from
            
        Returns:
            List of trades
        """
        params = {'symbol': symbol, 'limit': limit}
        if from_id:
            params['fromId'] = from_id
        
        data = await self._request('GET', '/fapi/v1/userTrades', params, signed=True, weight=5)
        return data
    
    # ====================== UTILITY METHODS ======================
    
    async def initialize_futures_settings(self, symbol: str):
        """
        Initialize futures settings for a symbol
        
        Args:
            symbol: Trading symbol
        """
        try:
            # Set leverage
            await self.set_leverage(symbol, self.leverage)
            
            # Set margin type
            try:
                await self.change_margin_type(symbol, self.margin_type)
            except ExchangeError as e:
                # Margin type might already be set
                logger.warning(f"⚠️ Could not change margin type: {e}")
            
            logger.info(f"✅ {symbol} initialized: {self.leverage}x {self.margin_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {symbol}: {e}")
            raise
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status
        
        Returns:
            Rate limit information
        """
        return {
            'request_count': self.request_count,
            'order_count': self.order_count,
            'weight_used': self.weight_used,
            'requests_limit': self.rate_limits['requests_per_minute'],
            'orders_limit': self.rate_limits['orders_per_minute'],
            'weight_limit': self.rate_limits['weight_per_minute']
        }
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("✅ AsterDEX session closed")
