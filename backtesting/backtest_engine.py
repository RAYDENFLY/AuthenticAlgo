"""
Backtest Engine Module
Professional backtesting engine with realistic order simulation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from core.logger import get_logger
from strategies.base_strategy import BaseStrategy
from risk import RiskManagement


class OrderType(Enum):
    """Order types for backtesting"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BacktestOrder:
    """Order object for backtesting"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestPosition:
    """Position object for backtesting"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BacktestTrade:
    """Completed trade record"""
    trade_id: int
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    duration: timedelta
    exit_reason: str


class BacktestEngine:
    """
    Professional backtesting engine with realistic order simulation
    
    Features:
    - Historical data processing
    - Realistic order fills with slippage
    - Commission calculation
    - Multiple position management
    - Strategy integration
    - Risk management integration
    - Walk-forward analysis
    - Monte Carlo simulation
    - Performance metrics calculation
    
    Attributes:
        initial_capital: Starting capital
        commission_rate: Commission as percentage (0.1 = 0.1%)
        slippage_model: 'fixed', 'percentage', or 'volume_based'
        max_slippage_pct: Maximum slippage percentage
    """
    
    def __init__(self, config: dict):
        """
        Initialize backtest engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger()
        
        # Backtest parameters
        self.backtest_config = config.get('backtesting', {})
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission_rate = self.backtest_config.get('commission_rate', 0.1)  # 0.1%
        self.slippage_model = self.backtest_config.get('slippage_model', 'percentage')
        self.max_slippage_pct = self.backtest_config.get('max_slippage_pct', 0.05)  # 0.05%
        
        # State tracking
        self.cash = self.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.pending_orders: List[BacktestOrder] = []
        self.filled_orders: List[BacktestOrder] = []
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Counters
        self.order_counter = 0
        self.trade_counter = 0
        
        # Current state
        self.current_time: Optional[datetime] = None
        self.current_data: Optional[pd.DataFrame] = None
        
        self.logger.info(f"Backtest Engine initialized: Capital=${self.initial_capital:,.2f}, "
                        f"Commission={self.commission_rate}%, Slippage={self.slippage_model}")
    
    def run(self, data: pd.DataFrame, strategy: BaseStrategy, 
            symbol: str = 'BTC/USDT', risk_mgmt: Optional[RiskManagement] = None) -> dict:
        """
        Run backtest on historical data
        
        Args:
            data: Historical OHLCV data
            strategy: Trading strategy instance
            symbol: Trading symbol
            risk_mgmt: Optional risk management instance
            
        Returns:
            Dict with backtest results
        """
        self.logger.info(f"Starting backtest for {symbol} with {strategy.__class__.__name__}")
        self.logger.info(f"Data period: {data.index[0]} to {data.index[-1]} ({len(data)} bars)")
        
        # Calculate indicators
        self.logger.info("Calculating technical indicators...")
        data = strategy.calculate_indicators(data)
        
        # Reset state
        self._reset_state()
        
        # Store symbol
        self.symbol = symbol
        self.strategy = strategy
        self.risk_mgmt = risk_mgmt
        
        # Main backtest loop
        for idx in range(len(data)):
            current_bar = data.iloc[idx:idx+1]
            self.current_time = current_bar.index[0]
            self.current_data = data.iloc[:idx+1]  # All data up to current bar
            
            # Update positions with current prices
            self._update_positions(current_bar)
            
            # Process pending orders
            self._process_pending_orders(current_bar)
            
            # Check strategy signals
            self._check_strategy_signals(current_bar)
            
            # Record equity
            equity = self._calculate_equity()
            self.equity_curve.append((self.current_time, equity))
            
            # Log progress every 100 bars
            if (idx + 1) % 100 == 0:
                self.logger.debug(f"Progress: {idx + 1}/{len(data)} bars, "
                                 f"Equity: ${equity:,.2f}, "
                                 f"Trades: {len(self.trades)}")
        
        # Close all remaining positions
        self._close_all_positions(data.iloc[-1:])
        
        # Generate results
        results = self._generate_results()
        
        self.logger.info(f"Backtest complete: {len(self.trades)} trades, "
                        f"Final equity: ${results['final_equity']:,.2f}")
        
        return results
    
    def _reset_state(self):
        """Reset backtest state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.pending_orders = []
        self.filled_orders = []
        self.trades = []
        self.equity_curve = [(datetime.now(), self.initial_capital)]
        self.order_counter = 0
        self.trade_counter = 0
    
    def _update_positions(self, current_bar: pd.DataFrame):
        """Update position prices and unrealized P&L"""
        current_price = current_bar['close'].iloc[0]
        
        for symbol, position in self.positions.items():
            position.current_price = current_price
            
            if position.quantity > 0:  # Long position
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # Short position
                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)
            
            # Check stop-loss and take-profit
            self._check_position_exits(position, current_bar)
    
    def _check_position_exits(self, position: BacktestPosition, current_bar: pd.DataFrame):
        """Check if position should be closed due to stop-loss or take-profit"""
        current_price = current_bar['close'].iloc[0]
        high = current_bar['high'].iloc[0]
        low = current_bar['low'].iloc[0]
        
        if position.quantity > 0:  # Long position
            # Check stop-loss
            if position.stop_loss and low <= position.stop_loss:
                self._close_position(position, position.stop_loss, "Stop-loss hit")
            # Check take-profit
            elif position.take_profit and high >= position.take_profit:
                self._close_position(position, position.take_profit, "Take-profit hit")
        
        else:  # Short position
            # Check stop-loss
            if position.stop_loss and high >= position.stop_loss:
                self._close_position(position, position.stop_loss, "Stop-loss hit")
            # Check take-profit
            elif position.take_profit and low <= position.take_profit:
                self._close_position(position, position.take_profit, "Take-profit hit")
    
    def _process_pending_orders(self, current_bar: pd.DataFrame):
        """Process pending orders and fill if conditions met"""
        high = current_bar['high'].iloc[0]
        low = current_bar['low'].iloc[0]
        close = current_bar['close'].iloc[0]
        
        orders_to_remove = []
        
        for order in self.pending_orders:
            filled = False
            fill_price = order.price
            
            if order.order_type == OrderType.MARKET:
                # Market orders fill at close with slippage
                fill_price = self._apply_slippage(close, order.side)
                filled = True
            
            elif order.order_type == OrderType.LIMIT:
                # Limit buy fills if low <= limit price
                if order.side == 'BUY' and low <= order.price:
                    fill_price = order.price
                    filled = True
                # Limit sell fills if high >= limit price
                elif order.side == 'SELL' and high >= order.price:
                    fill_price = order.price
                    filled = True
            
            elif order.order_type == OrderType.STOP_LOSS:
                # Stop-loss buy triggers if high >= stop price
                if order.side == 'BUY' and high >= order.price:
                    fill_price = self._apply_slippage(order.price, order.side)
                    filled = True
                # Stop-loss sell triggers if low <= stop price
                elif order.side == 'SELL' and low <= order.price:
                    fill_price = self._apply_slippage(order.price, order.side)
                    filled = True
            
            if filled:
                self._fill_order(order, fill_price)
                orders_to_remove.append(order)
        
        # Remove filled orders from pending
        for order in orders_to_remove:
            self.pending_orders.remove(order)
    
    def _check_strategy_signals(self, current_bar: pd.DataFrame):
        """Check strategy for entry/exit signals"""
        if len(self.current_data) < 50:  # Need minimum data for indicators
            return
        
        symbol = self.symbol
        
        # Check for entry signal if no position
        if symbol not in self.positions:
            entry_signal = self.strategy.should_enter(self.current_data)
            
            if entry_signal['signal'] != 'HOLD':
                # Calculate position size
                position_size = self._calculate_position_size(
                    entry_signal, 
                    current_bar['close'].iloc[0]
                )
                
                if position_size > 0:
                    # Create entry order
                    order = self._create_order(
                        symbol=symbol,
                        side=entry_signal['signal'],
                        quantity=position_size,
                        price=current_bar['close'].iloc[0],
                        order_type=OrderType.MARKET
                    )
                    self.pending_orders.append(order)
        
        # Check for exit signal if have position
        else:
            position = self.positions[symbol]
            exit_signal = self.strategy.should_exit(self.current_data, position.quantity > 0)
            
            if exit_signal['signal'] in ['SELL', 'BUY']:
                self._close_position(
                    position, 
                    current_bar['close'].iloc[0], 
                    exit_signal.get('reason', 'Strategy signal')
                )
    
    def _calculate_position_size(self, signal: dict, price: float) -> float:
        """Calculate position size based on risk management"""
        if self.risk_mgmt:
            # Use risk management for position sizing
            position_size = self.risk_mgmt.calculate_position_size(
                symbol=self.symbol,
                price=price,
                stop_loss_price=signal.get('stop_loss'),
                account_balance=self.cash
            )
        else:
            # Default: 10% of capital
            position_value = self.cash * 0.10
            position_size = position_value / price
        
        return position_size
    
    def _create_order(self, symbol: str, side: str, quantity: float, 
                     price: float, order_type: OrderType) -> BacktestOrder:
        """Create new order"""
        self.order_counter += 1
        
        order = BacktestOrder(
            order_id=f"ORD_{self.order_counter}",
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=self.current_time
        )
        
        return order
    
    def _fill_order(self, order: BacktestOrder, fill_price: float):
        """Fill order and update position"""
        # Calculate commission
        commission = (fill_price * order.quantity) * (self.commission_rate / 100)
        
        # Calculate slippage
        slippage = abs(fill_price - order.price) * order.quantity
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.slippage = slippage
        
        self.filled_orders.append(order)
        
        # Update position
        if order.side == 'BUY':
            self._open_position(order, fill_price)
        else:
            self._close_position_by_order(order, fill_price)
        
        self.logger.debug(f"Order filled: {order.side} {order.quantity:.4f} {order.symbol} "
                         f"@ {fill_price:.2f} (Commission: ${commission:.2f})")
    
    def _open_position(self, order: BacktestOrder, fill_price: float):
        """Open new position"""
        cost = fill_price * order.quantity + order.commission
        
        if cost > self.cash:
            self.logger.warning(f"Insufficient cash: ${self.cash:.2f} < ${cost:.2f}")
            return
        
        self.cash -= cost
        
        # Get stop-loss and take-profit from strategy
        stop_loss = None
        take_profit = None
        if hasattr(self.strategy, 'config'):
            sl_pct = self.strategy.config.get('stop_loss_pct', 2.0)
            tp_pct = self.strategy.config.get('take_profit_pct', 4.0)
            
            stop_loss = fill_price * (1 - sl_pct / 100)
            take_profit = fill_price * (1 + tp_pct / 100)
        
        position = BacktestPosition(
            symbol=order.symbol,
            quantity=order.quantity,
            entry_price=fill_price,
            entry_time=self.current_time,
            current_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[order.symbol] = position
        
        self.logger.info(f"Position opened: {order.quantity:.4f} {order.symbol} @ {fill_price:.2f}")
    
    def _close_position(self, position: BacktestPosition, exit_price: float, reason: str):
        """Close existing position"""
        # Calculate P&L
        if position.quantity > 0:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * abs(position.quantity)
        
        # Calculate commission
        commission = (exit_price * abs(position.quantity)) * (self.commission_rate / 100)
        pnl -= commission
        
        # Update cash
        self.cash += (exit_price * abs(position.quantity)) - commission
        
        # Record trade
        self.trade_counter += 1
        trade = BacktestTrade(
            trade_id=self.trade_counter,
            symbol=position.symbol,
            side='LONG' if position.quantity > 0 else 'SHORT',
            quantity=abs(position.quantity),
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            pnl=pnl,
            pnl_pct=(pnl / (position.entry_price * abs(position.quantity))) * 100,
            commission=commission,
            slippage=0.0,
            duration=self.current_time - position.entry_time,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        
        # Remove position
        del self.positions[position.symbol]
        
        self.logger.info(f"Position closed: {position.symbol} P&L=${pnl:.2f} ({trade.pnl_pct:+.2f}%) - {reason}")
    
    def _close_position_by_order(self, order: BacktestOrder, fill_price: float):
        """Close position via order"""
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            self._close_position(position, fill_price, "Manual close")
    
    def _close_all_positions(self, final_bar: pd.DataFrame):
        """Close all remaining positions at end of backtest"""
        close_price = final_bar['close'].iloc[0]
        
        for symbol, position in list(self.positions.items()):
            self._close_position(position, close_price, "Backtest end")
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price"""
        if self.slippage_model == 'fixed':
            slippage_amount = price * (self.max_slippage_pct / 100)
        elif self.slippage_model == 'percentage':
            slippage_amount = price * (np.random.uniform(0, self.max_slippage_pct) / 100)
        else:  # volume_based (simplified)
            slippage_amount = price * (np.random.uniform(0, self.max_slippage_pct / 2) / 100)
        
        if side == 'BUY':
            return price + slippage_amount
        else:
            return price - slippage_amount
    
    def _calculate_equity(self) -> float:
        """Calculate current total equity"""
        positions_value = sum(
            pos.current_price * abs(pos.quantity) 
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def _generate_results(self) -> dict:
        """Generate backtest results"""
        final_equity = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': ((final_equity - self.initial_capital) / self.initial_capital) * 100,
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t.pnl > 0]),
            'losing_trades': len([t for t in self.trades if t.pnl < 0]),
            'win_rate': (len([t for t in self.trades if t.pnl > 0]) / len(self.trades) * 100) if self.trades else 0,
            'total_pnl': sum(t.pnl for t in self.trades),
            'total_commission': sum(t.commission for t in self.trades),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'orders': self.filled_orders
        }
    
    def walk_forward_analysis(self, data: pd.DataFrame, strategy: BaseStrategy,
                            train_period: int = 252, test_period: int = 63) -> List[dict]:
        """
        Perform walk-forward analysis
        
        Args:
            data: Historical data
            strategy: Trading strategy
            train_period: Training period in bars
            test_period: Testing period in bars
            
        Returns:
            List of results for each window
        """
        self.logger.info(f"Starting walk-forward analysis: Train={train_period}, Test={test_period}")
        
        results = []
        total_windows = (len(data) - train_period) // test_period
        
        for i in range(total_windows):
            start_idx = i * test_period
            train_end_idx = start_idx + train_period
            test_end_idx = train_end_idx + test_period
            
            if test_end_idx > len(data):
                break
            
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:test_end_idx]
            
            # Run backtest on test period
            window_results = self.run(test_data, strategy)
            window_results['window'] = i + 1
            window_results['train_period'] = (train_data.index[0], train_data.index[-1])
            window_results['test_period'] = (test_data.index[0], test_data.index[-1])
            
            results.append(window_results)
            
            self.logger.info(f"Window {i+1}/{total_windows}: Return={window_results['total_return']:.2f}%")
        
        return results
    
    def monte_carlo_simulation(self, trades: List[BacktestTrade], 
                              num_simulations: int = 1000) -> dict:
        """
        Run Monte Carlo simulation on trade results
        
        Args:
            trades: Historical trades
            num_simulations: Number of simulations
            
        Returns:
            Dict with simulation statistics
        """
        if not trades:
            return {}
        
        self.logger.info(f"Running Monte Carlo simulation: {num_simulations} iterations")
        
        # Extract trade returns
        trade_returns = [t.pnl / (t.entry_price * t.quantity) for t in trades]
        
        simulation_results = []
        
        for _ in range(num_simulations):
            # Randomly resample trades with replacement
            resampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate cumulative return
            cumulative_return = np.prod([1 + r for r in resampled_returns]) - 1
            final_equity = self.initial_capital * (1 + cumulative_return)
            
            simulation_results.append(final_equity)
        
        simulation_results = np.array(simulation_results)
        
        return {
            'mean_final_equity': np.mean(simulation_results),
            'median_final_equity': np.median(simulation_results),
            'std_final_equity': np.std(simulation_results),
            'min_final_equity': np.min(simulation_results),
            'max_final_equity': np.max(simulation_results),
            'percentile_5': np.percentile(simulation_results, 5),
            'percentile_95': np.percentile(simulation_results, 95),
            'probability_profit': (np.sum(simulation_results > self.initial_capital) / num_simulations) * 100
        }
