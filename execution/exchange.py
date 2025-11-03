"""
Exchange Execution Module
Handles all exchange interactions for order placement and account management.
Priority: AsterDEX with Binance fallback endpoints
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import hmac
import hashlib
import asyncio
from datetime import datetime
import ccxt

from core.logger import logger
from core.exceptions import ExchangeError, ConfigurationError
from core.config import get_config


class OrderType(Enum):
    """Order types supported by the exchange"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    """Position side for futures trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


@dataclass
class Balance:
    """Account balance information"""
    asset: str
    free: float
    locked: float
    total: float
    
    @property
    def available(self) -> float:
        """Available balance for trading"""
        return self.free


@dataclass
class Position:
    """Position information for futures trading"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: int
    liquidation_price: Optional[float] = None
    
    @property
    def pnl_percent(self) -> float:
        """PnL as percentage of position value"""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.entry_price * abs(self.size))) * 100


@dataclass
class Order:
    """Order information"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    filled_quantity: float
    status: str
    timestamp: int
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status.upper() in ["FILLED", "COMPLETELY_FILLED"]
    
    @property
    def is_open(self) -> bool:
        """Check if order is still open"""
        return self.status.upper() in ["NEW", "PARTIALLY_FILLED", "PENDING_NEW"]


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int = 1200, time_window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
        
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        # If at limit, wait until oldest call expires
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]) + 0.1
            logger.warning(f"Rate limit reached, waiting {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            self.calls.pop(0)
        
        # Record this call
        self.calls.append(now)


class BaseExchange(ABC):
    """
    Abstract base class for exchange implementations
    All exchanges must implement these methods
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize exchange connection
        
        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            testnet: Whether to use testnet (default: True for safety)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.connected = False
        self.rate_limiter = RateLimiter()
        
        logger.info(f"Initializing {self.__class__.__name__} (testnet={testnet})")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to exchange and verify credentials
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Balance]:
        """
        Fetch account balance
        
        Returns:
            Dictionary of asset -> Balance
        """
        pass
    
    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Create a new order
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: Buy or Sell
            order_type: Market, Limit, etc.
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            **kwargs: Additional exchange-specific parameters
            
        Returns:
            Order object
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair
            
        Returns:
            True if cancelled successfully
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of open orders
        """
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all open positions (for futures)
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of positions
        """
        pass
    
    async def close(self):
        """Close exchange connection and cleanup"""
        self.connected = False
        logger.info(f"{self.__class__.__name__} connection closed")


class AsterDEXExchange(BaseExchange):
    """
    AsterDEX Exchange Implementation
    Primary exchange with fallback to Binance endpoints
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        
        # AsterDEX endpoint (priority)
        self.asterdex_url = 'https://fapi.asterdex.com/fapi/v1'
        
        # Fallback Binance endpoints if AsterDEX is down
        self.binance_endpoints = [
            'https://api1.binance.com/api/v3',
            'https://api2.binance.com/api/v3',
            'https://api3.binance.com/api/v3',
            'https://data-api.binance.vision/api/v3'
        ]
        
        # Initialize CCXT exchange
        self.exchange: Optional[ccxt.binance] = None
        self.current_endpoint = self.asterdex_url
        
    async def connect(self) -> bool:
        """
        Connect to AsterDEX with fallback to Binance
        
        Returns:
            True if connected successfully
        """
        try:
            # Try AsterDEX first
            logger.info("🎯 Attempting connection to AsterDEX...")
            
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Futures trading
                    'adjustForTimeDifference': True,
                }
            })
            
            # Set AsterDEX URL
            if self.testnet:
                self.exchange.urls['api'] = 'https://testnet.binancefuture.com'
                logger.info("📍 Using Binance Testnet (AsterDEX testnet not available)")
            else:
                self.exchange.urls['api'] = self.asterdex_url
                self.current_endpoint = self.asterdex_url
            
            # Test connection
            await self.rate_limiter.acquire()
            balance = await self._safe_api_call(self.exchange.fetch_balance)
            
            if balance:
                self.connected = True
                logger.info(f"✅ Connected to {self.current_endpoint}")
                logger.info(f"📊 Account has {len(balance.get('total', {}))} assets")
                return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to AsterDEX: {e}")
            
            # Try fallback Binance endpoints
            if not self.testnet:
                logger.info("🔄 Trying fallback Binance endpoints...")
                for endpoint in self.binance_endpoints:
                    try:
                        self.exchange.urls['api'] = endpoint
                        self.current_endpoint = endpoint
                        
                        await self.rate_limiter.acquire()
                        balance = await self._safe_api_call(self.exchange.fetch_balance)
                        
                        if balance:
                            self.connected = True
                            logger.info(f"✅ Connected to fallback: {endpoint}")
                            return True
                            
                    except Exception as fallback_error:
                        logger.warning(f"Fallback {endpoint} failed: {fallback_error}")
                        continue
            
            raise ExchangeError(f"Failed to connect to any endpoint: {e}")
    
    async def _safe_api_call(self, func, *args, **kwargs):
        """
        Safely call API function with error handling
        
        Args:
            func: API function to call
            *args, **kwargs: Function arguments
            
        Returns:
            API response or None on error
        """
        try:
            await self.rate_limiter.acquire()
            
            # Handle async functions
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
                
        except ccxt.RateLimitExceeded as e:
            logger.error(f"Rate limit exceeded: {e}")
            await asyncio.sleep(60)  # Wait 1 minute
            return await self._safe_api_call(func, *args, **kwargs)  # Retry
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            return None
            
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            raise ExchangeError(f"Exchange error: {e}")
    
    async def fetch_balance(self) -> Dict[str, Balance]:
        """
        Fetch account balance from AsterDEX
        
        Returns:
            Dictionary of asset -> Balance
        """
        if not self.connected:
            raise ExchangeError("Not connected to exchange. Call connect() first.")
        
        try:
            balance_data = await self._safe_api_call(self.exchange.fetch_balance)
            
            if not balance_data:
                raise ExchangeError("Failed to fetch balance")
            
            balances = {}
            for asset, amounts in balance_data.get('total', {}).items():
                if amounts > 0:  # Only include non-zero balances
                    balances[asset] = Balance(
                        asset=asset,
                        free=balance_data.get('free', {}).get(asset, 0.0),
                        locked=balance_data.get('used', {}).get(asset, 0.0),
                        total=amounts
                    )
            
            logger.info(f"📊 Fetched balance: {len(balances)} assets")
            return balances
            
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise ExchangeError(f"Failed to fetch balance: {e}")
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Create order on AsterDEX
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: Buy or Sell
            order_type: Market, Limit, etc.
            quantity: Order quantity
            price: Limit price
            stop_price: Stop price
            **kwargs: Additional parameters
            
        Returns:
            Order object
        """
        if not self.connected:
            raise ExchangeError("Not connected to exchange")
        
        try:
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'type': order_type.value.lower(),
                'side': side.value.lower(),
                'amount': quantity,
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                if price is None:
                    raise ValueError(f"{order_type.value} requires price parameter")
                order_params['price'] = price
            
            # Add stop price for stop orders
            if order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT]:
                if stop_price is None:
                    raise ValueError(f"{order_type.value} requires stop_price parameter")
                order_params['stopPrice'] = stop_price
            
            # Merge additional parameters
            order_params.update(kwargs)
            
            logger.info(f"📝 Creating {side.value} {order_type.value} order: {quantity} {symbol}")
            
            # Create order
            result = await self._safe_api_call(
                self.exchange.create_order,
                **order_params
            )
            
            if not result:
                raise ExchangeError("Failed to create order")
            
            # Parse order response
            order = Order(
                order_id=str(result.get('id', result.get('orderId', ''))),
                symbol=result['symbol'],
                side=OrderSide[result['side'].upper()],
                order_type=OrderType[result['type'].upper()],
                price=float(result.get('price', 0)),
                quantity=float(result.get('amount', quantity)),
                filled_quantity=float(result.get('filled', 0)),
                status=result.get('status', 'UNKNOWN'),
                timestamp=result.get('timestamp', int(time.time() * 1000))
            )
            
            logger.info(f"✅ Order created: ID={order.order_id}, Status={order.status}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise ExchangeError(f"Failed to create order: {e}")
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel order on AsterDEX
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair
            
        Returns:
            True if cancelled successfully
        """
        if not self.connected:
            raise ExchangeError("Not connected to exchange")
        
        try:
            logger.info(f"❌ Cancelling order {order_id} for {symbol}")
            
            result = await self._safe_api_call(
                self.exchange.cancel_order,
                order_id,
                symbol
            )
            
            if result:
                logger.info(f"✅ Order {order_id} cancelled successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise ExchangeError(f"Failed to cancel order: {e}")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders from AsterDEX
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of open orders
        """
        if not self.connected:
            raise ExchangeError("Not connected to exchange")
        
        try:
            if symbol:
                orders_data = await self._safe_api_call(
                    self.exchange.fetch_open_orders,
                    symbol
                )
            else:
                orders_data = await self._safe_api_call(
                    self.exchange.fetch_open_orders
                )
            
            if not orders_data:
                return []
            
            orders = []
            for order_data in orders_data:
                order = Order(
                    order_id=str(order_data.get('id', order_data.get('orderId', ''))),
                    symbol=order_data['symbol'],
                    side=OrderSide[order_data['side'].upper()],
                    order_type=OrderType[order_data['type'].upper()],
                    price=float(order_data.get('price', 0)),
                    quantity=float(order_data.get('amount', 0)),
                    filled_quantity=float(order_data.get('filled', 0)),
                    status=order_data.get('status', 'UNKNOWN'),
                    timestamp=order_data.get('timestamp', 0)
                )
                orders.append(order)
            
            logger.info(f"📋 Fetched {len(orders)} open orders")
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get open positions from AsterDEX (futures)
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of positions
        """
        if not self.connected:
            raise ExchangeError("Not connected to exchange")
        
        try:
            positions_data = await self._safe_api_call(
                self.exchange.fetch_positions,
                [symbol] if symbol else None
            )
            
            if not positions_data:
                return []
            
            positions = []
            for pos_data in positions_data:
                # Skip empty positions
                contracts = float(pos_data.get('contracts', 0))
                if contracts == 0:
                    continue
                
                position = Position(
                    symbol=pos_data['symbol'],
                    side=PositionSide.LONG if float(pos_data.get('contracts', 0)) > 0 else PositionSide.SHORT,
                    size=abs(float(pos_data.get('contracts', 0))),
                    entry_price=float(pos_data.get('entryPrice', 0)),
                    current_price=float(pos_data.get('markPrice', 0)),
                    unrealized_pnl=float(pos_data.get('unrealizedPnl', 0)),
                    realized_pnl=float(pos_data.get('realizedPnl', 0)),
                    leverage=int(pos_data.get('leverage', 1)),
                    liquidation_price=float(pos_data.get('liquidationPrice', 0)) if pos_data.get('liquidationPrice') else None
                )
                positions.append(position)
            
            logger.info(f"📊 Fetched {len(positions)} open positions")
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []


# Factory function for easy exchange creation
def create_exchange(exchange_name: str = "asterdex", testnet: bool = True) -> BaseExchange:
    """
    Create exchange instance from configuration
    
    Args:
        exchange_name: Name of exchange ('asterdex', 'binance')
        testnet: Use testnet if available
        
    Returns:
        BaseExchange instance
    """
    config = get_config()
    
    # Get API credentials from config
    if exchange_name.lower() == "asterdex":
        api_key = config.get('exchanges', {}).get('asterdex', {}).get('api_key')
        api_secret = config.get('exchanges', {}).get('asterdex', {}).get('api_secret')
        
        if not api_key or not api_secret:
            raise ConfigurationError("AsterDEX API credentials not found in config")
        
        return AsterDEXExchange(api_key, api_secret, testnet)
    
    else:
        raise ValueError(f"Unknown exchange: {exchange_name}")


if __name__ == "__main__":
    # Quick test
    async def test():
        exchange = create_exchange("asterdex", testnet=True)
        await exchange.connect()
        balance = await exchange.fetch_balance()
        print(f"Balance: {balance}")
        await exchange.close()
    
    asyncio.run(test())
