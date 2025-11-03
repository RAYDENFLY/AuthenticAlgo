"""
Order Management Module
Handles order placement, tracking, and lifecycle management
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from collections import defaultdict

from core.logger import logger
from core.exceptions import ExchangeError, ValidationError
from execution.exchange import (
    BaseExchange, Order, OrderType, OrderSide,
    create_exchange
)


class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class ManagedOrder:
    """
    Enhanced order with tracking and metadata
    """
    order: Order
    strategy_name: str
    entry_order: bool = True
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update(self):
        """Update timestamp"""
        self.updated_at = datetime.now()


class OrderManager:
    """
    Manages order lifecycle, tracking, and validation
    """
    
    def __init__(
        self,
        exchange: BaseExchange,
        enable_validation: bool = True,
        max_slippage_percent: float = 1.0
    ):
        """
        Initialize OrderManager
        
        Args:
            exchange: Exchange instance to use
            enable_validation: Enable order validation
            max_slippage_percent: Maximum allowed slippage percentage
        """
        self.exchange = exchange
        self.enable_validation = enable_validation
        self.max_slippage_percent = max_slippage_percent
        
        # Order tracking
        self.active_orders: Dict[str, ManagedOrder] = {}
        self.order_history: List[ManagedOrder] = []
        
        # Callbacks for order events
        self.order_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info(f"OrderManager initialized (validation={enable_validation})")
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register callback for order events
        
        Args:
            event: Event name ('filled', 'cancelled', 'rejected', etc.)
            callback: Callback function
        """
        self.order_callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")
    
    async def _trigger_callbacks(self, event: str, order: ManagedOrder):
        """Trigger all callbacks for an event"""
        for callback in self.order_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def _validate_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None
    ):
        """
        Validate order parameters before submission
        
        Args:
            symbol: Trading pair
            side: Order side
            quantity: Order quantity
            price: Order price (for limit orders)
            
        Raises:
            ValidationError: If order validation fails
        """
        if not self.enable_validation:
            return
        
        # Basic validation
        if quantity <= 0:
            raise ValidationError(f"Invalid quantity: {quantity}")
        
        if price is not None and price <= 0:
            raise ValidationError(f"Invalid price: {price}")
        
        # Symbol format check
        if '/' not in symbol:
            raise ValidationError(f"Invalid symbol format: {symbol}")
        
        logger.debug(f"✅ Order validation passed for {symbol}")
    
    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        strategy_name: str = "manual",
        **kwargs
    ) -> ManagedOrder:
        """
        Place a market order
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: Buy or Sell
            quantity: Order quantity
            strategy_name: Name of strategy placing order
            **kwargs: Additional parameters
            
        Returns:
            ManagedOrder object
        """
        try:
            # Validate order
            self._validate_order(symbol, side, quantity)
            
            logger.info(f"📝 Placing MARKET {side.value} order: {quantity} {symbol}")
            
            # Create order on exchange
            order = await self.exchange.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                **kwargs
            )
            
            # Create managed order
            managed_order = ManagedOrder(
                order=order,
                strategy_name=strategy_name,
                entry_order=True,
                notes=f"Market {side.value}"
            )
            
            # Track order
            self.active_orders[order.order_id] = managed_order
            
            logger.info(f"✅ Market order placed: ID={order.order_id}")
            
            # Trigger callbacks
            await self._trigger_callbacks('order_placed', managed_order)
            
            return managed_order
            
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            raise ExchangeError(f"Failed to place market order: {e}")
    
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        strategy_name: str = "manual",
        time_in_force: str = "GTC",
        **kwargs
    ) -> ManagedOrder:
        """
        Place a limit order
        
        Args:
            symbol: Trading pair
            side: Buy or Sell
            quantity: Order quantity
            price: Limit price
            strategy_name: Strategy name
            time_in_force: GTC, IOC, FOK
            **kwargs: Additional parameters
            
        Returns:
            ManagedOrder object
        """
        try:
            # Validate order
            self._validate_order(symbol, side, quantity, price)
            
            logger.info(f"📝 Placing LIMIT {side.value} order: {quantity} {symbol} @ {price}")
            
            # Create order on exchange
            order = await self.exchange.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                timeInForce=time_in_force,
                **kwargs
            )
            
            # Create managed order
            managed_order = ManagedOrder(
                order=order,
                strategy_name=strategy_name,
                entry_order=True,
                notes=f"Limit {side.value} @ {price}"
            )
            
            # Track order
            self.active_orders[order.order_id] = managed_order
            
            logger.info(f"✅ Limit order placed: ID={order.order_id}")
            
            # Trigger callbacks
            await self._trigger_callbacks('order_placed', managed_order)
            
            return managed_order
            
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise ExchangeError(f"Failed to place limit order: {e}")
    
    async def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        strategy_name: str = "manual",
        parent_order_id: Optional[str] = None,
        **kwargs
    ) -> ManagedOrder:
        """
        Place a stop-loss order
        
        Args:
            symbol: Trading pair
            side: Buy or Sell (opposite of position)
            quantity: Order quantity
            stop_price: Stop trigger price
            limit_price: Limit price (if None, uses STOP_LOSS_MARKET)
            strategy_name: Strategy name
            parent_order_id: Parent position order ID
            **kwargs: Additional parameters
            
        Returns:
            ManagedOrder object
        """
        try:
            # Validate order
            self._validate_order(symbol, side, quantity)
            
            # Determine order type
            order_type = OrderType.STOP_LOSS_LIMIT if limit_price else OrderType.STOP_LOSS
            
            logger.info(f"🛡️ Placing STOP LOSS {side.value} order: {quantity} {symbol} @ stop={stop_price}")
            
            # Create order on exchange
            order = await self.exchange.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=limit_price,
                stop_price=stop_price,
                **kwargs
            )
            
            # Create managed order
            managed_order = ManagedOrder(
                order=order,
                strategy_name=strategy_name,
                entry_order=False,
                parent_order_id=parent_order_id,
                notes=f"Stop Loss @ {stop_price}"
            )
            
            # Track order
            self.active_orders[order.order_id] = managed_order
            
            # Link to parent order if exists
            if parent_order_id and parent_order_id in self.active_orders:
                self.active_orders[parent_order_id].stop_loss_order_id = order.order_id
            
            logger.info(f"✅ Stop loss order placed: ID={order.order_id}")
            
            # Trigger callbacks
            await self._trigger_callbacks('stop_loss_placed', managed_order)
            
            return managed_order
            
        except Exception as e:
            logger.error(f"Failed to place stop loss order: {e}")
            raise ExchangeError(f"Failed to place stop loss order: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            managed_order = self.active_orders[order_id]
            order = managed_order.order
            
            logger.info(f"❌ Cancelling order {order_id} for {order.symbol}")
            
            # Cancel on exchange
            success = await self.exchange.cancel_order(order_id, order.symbol)
            
            if success:
                # Move to history
                self.order_history.append(managed_order)
                del self.active_orders[order_id]
                
                logger.info(f"✅ Order {order_id} cancelled")
                
                # Trigger callbacks
                await self._trigger_callbacks('order_cancelled', managed_order)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> bool:
        """
        Modify an existing order (cancel and replace)
        
        Args:
            order_id: Order ID to modify
            new_quantity: New quantity (if None, keep original)
            new_price: New price (if None, keep original)
            
        Returns:
            True if modified successfully
        """
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found")
                return False
            
            managed_order = self.active_orders[order_id]
            order = managed_order.order
            
            # Use original values if not specified
            quantity = new_quantity or order.quantity
            price = new_price or order.price
            
            logger.info(f"🔄 Modifying order {order_id}: qty={quantity}, price={price}")
            
            # Cancel original order
            cancel_success = await self.cancel_order(order_id)
            
            if not cancel_success:
                logger.error(f"Failed to cancel order {order_id} for modification")
                return False
            
            # Place new order with updated parameters
            if order.order_type == OrderType.LIMIT:
                new_order = await self.place_limit_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=quantity,
                    price=price,
                    strategy_name=managed_order.strategy_name
                )
            else:
                logger.warning(f"Cannot modify {order.order_type.value} orders")
                return False
            
            logger.info(f"✅ Order modified: Old ID={order_id}, New ID={new_order.order.order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to modify order: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all active orders
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        
        # Get orders to cancel
        orders_to_cancel = [
            order_id for order_id, managed_order in self.active_orders.items()
            if symbol is None or managed_order.order.symbol == symbol
        ]
        
        logger.info(f"❌ Cancelling {len(orders_to_cancel)} orders...")
        
        # Cancel each order
        for order_id in orders_to_cancel:
            success = await self.cancel_order(order_id)
            if success:
                cancelled_count += 1
        
        logger.info(f"✅ Cancelled {cancelled_count}/{len(orders_to_cancel)} orders")
        
        return cancelled_count
    
    async def update_order_status(self, order_id: str) -> bool:
        """
        Update order status from exchange
        
        Args:
            order_id: Order ID to update
            
        Returns:
            True if status updated successfully
        """
        try:
            if order_id not in self.active_orders:
                return False
            
            managed_order = self.active_orders[order_id]
            
            # Fetch latest order info from exchange
            open_orders = await self.exchange.get_open_orders(managed_order.order.symbol)
            
            # Find our order
            updated_order = None
            for order in open_orders:
                if order.order_id == order_id:
                    updated_order = order
                    break
            
            if updated_order:
                # Update managed order
                managed_order.order = updated_order
                managed_order.update()
                
                # Check if filled
                if updated_order.is_filled:
                    logger.info(f"✅ Order {order_id} FILLED")
                    
                    # Move to history
                    self.order_history.append(managed_order)
                    del self.active_orders[order_id]
                    
                    # Trigger callbacks
                    await self._trigger_callbacks('order_filled', managed_order)
                
                return True
            else:
                # Order not in open orders (might be filled/cancelled)
                logger.warning(f"Order {order_id} not found in open orders")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update order status: {e}")
            return False
    
    async def update_all_orders(self):
        """Update status of all active orders"""
        order_ids = list(self.active_orders.keys())
        
        logger.debug(f"Updating {len(order_ids)} active orders...")
        
        for order_id in order_ids:
            await self.update_order_status(order_id)
    
    def get_active_orders(self, strategy_name: Optional[str] = None) -> List[ManagedOrder]:
        """
        Get active orders
        
        Args:
            strategy_name: Filter by strategy (optional)
            
        Returns:
            List of active orders
        """
        if strategy_name:
            return [
                order for order in self.active_orders.values()
                if order.strategy_name == strategy_name
            ]
        return list(self.active_orders.values())
    
    def get_order_history(
        self,
        strategy_name: Optional[str] = None,
        limit: int = 100
    ) -> List[ManagedOrder]:
        """
        Get order history
        
        Args:
            strategy_name: Filter by strategy (optional)
            limit: Maximum number of orders to return
            
        Returns:
            List of historical orders
        """
        history = self.order_history
        
        if strategy_name:
            history = [
                order for order in history
                if order.strategy_name == strategy_name
            ]
        
        # Return most recent orders
        return history[-limit:]
    
    def get_statistics(self) -> Dict:
        """
        Get order statistics
        
        Returns:
            Dictionary with order statistics
        """
        total_orders = len(self.active_orders) + len(self.order_history)
        
        filled_orders = len([
            order for order in self.order_history
            if order.order.is_filled
        ])
        
        cancelled_orders = len([
            order for order in self.order_history
            if not order.order.is_filled
        ])
        
        return {
            'total_orders': total_orders,
            'active_orders': len(self.active_orders),
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0
        }


if __name__ == "__main__":
    # Quick test
    async def test():
        exchange = create_exchange("asterdex", testnet=True)
        await exchange.connect()
        
        order_manager = OrderManager(exchange)
        
        # Print statistics
        stats = order_manager.get_statistics()
        print(f"Order Statistics: {stats}")
        
        await exchange.close()
    
    asyncio.run(test())
