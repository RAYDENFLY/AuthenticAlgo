"""
Portfolio Manager Module
Comprehensive portfolio tracking and management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.logger import get_logger


@dataclass
class Position:
    """Portfolio position data"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    position_value: float
    entry_time: datetime
    last_updated: datetime


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation_matrix: pd.DataFrame


class PortfolioManager:
    """
    Comprehensive portfolio management system
    
    Features:
    - Real-time position tracking
    - Performance metrics calculation (Sharpe, Sortino, Drawdown)
    - Portfolio rebalancing logic
    - Correlation analysis
    - Risk-adjusted returns
    - Detailed reporting
    
    Attributes:
        initial_capital: Starting capital amount
        cash_balance: Current cash balance
        positions: Dict of current open positions
        max_positions: Maximum number of concurrent positions
        rebalancing_threshold: % deviation before rebalancing
        target_allocations: Target allocation percentages
    """
    
    def __init__(self, config: dict):
        """
        Initialize portfolio manager
        
        Args:
            config: Configuration dictionary with portfolio parameters
        """
        self.config = config
        self.logger = get_logger()
        
        # Portfolio parameters
        self.portfolio_config = config.get('portfolio', {})
        self.max_positions = self.portfolio_config.get('max_positions', 10)
        self.rebalancing_threshold = self.portfolio_config.get('rebalancing_threshold', 5.0)
        self.target_allocations = self.portfolio_config.get('target_allocations', {})
        
        # Portfolio state
        self.initial_capital = config.get('initial_capital', 10000)
        self.cash_balance = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        self.daily_pnl_history = []
        
        # Performance tracking
        self.portfolio_values = [self.initial_capital]
        self.portfolio_timestamps = [datetime.now()]
        
        self.logger.info(f"Portfolio Manager initialized with ${self.initial_capital:,.2f} capital, "
                        f"max positions: {self.max_positions}")
    
    def update_position(self, symbol: str, quantity: float, price: float,
                       action: str, timestamp: datetime):
        """
        Update portfolio with new trade
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            action: Trade action ('BUY', 'SELL', 'SHORT', 'COVER')
            timestamp: Trade timestamp
        """
        cost = quantity * price
        
        if action in ['BUY', 'SHORT']:
            # Opening position
            if symbol in self.positions:
                # Average existing position
                existing = self.positions[symbol]
                total_quantity = existing.quantity + (quantity if action == 'BUY' else -quantity)
                total_cost = (existing.quantity * existing.entry_price) + cost
                
                if abs(total_quantity) > 1e-8:
                    new_entry_price = total_cost / abs(total_quantity)
                else:
                    new_entry_price = 0
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_quantity,
                    entry_price=new_entry_price,
                    current_price=price,
                    unrealized_pnl=(price - new_entry_price) * total_quantity if total_quantity > 0 else (new_entry_price - price) * abs(total_quantity),
                    unrealized_pnl_pct=((price - new_entry_price) / new_entry_price * 100) if new_entry_price > 0 and total_quantity > 0 else ((new_entry_price - price) / new_entry_price * 100) if new_entry_price > 0 else 0,
                    position_value=abs(total_quantity) * price,
                    entry_time=min(existing.entry_time, timestamp),
                    last_updated=timestamp
                )
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity if action == 'BUY' else -quantity,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    position_value=cost,
                    entry_time=timestamp,
                    last_updated=timestamp
                )
            
            self.cash_balance -= cost if action == 'BUY' else -cost
            self.logger.info(f"Opened {action} position: {quantity} {symbol} @ {price:.4f}")
        
        elif action in ['SELL', 'COVER']:
            # Closing position
            if symbol not in self.positions:
                self.logger.error(f"Cannot close non-existent position: {symbol}")
                return
            
            existing = self.positions[symbol]
            
            # Calculate realized PnL
            if action == 'SELL' and existing.quantity > 0:
                realized_pnl = (price - existing.entry_price) * quantity
            elif action == 'COVER' and existing.quantity < 0:
                realized_pnl = (existing.entry_price - price) * abs(quantity)
            else:
                self.logger.error(f"Invalid close action: {action} for position: {existing.quantity}")
                return
            
            # Update position
            new_quantity = existing.quantity - quantity if action == 'SELL' else existing.quantity + quantity
            
            if abs(new_quantity) < 1e-8:  # Position fully closed
                del self.positions[symbol]
                self.logger.info(f"Closed position: {symbol}, realized PnL: ${realized_pnl:.2f}")
            else:
                self.positions[symbol].quantity = new_quantity
                self.positions[symbol].position_value = abs(new_quantity) * price
                self.positions[symbol].last_updated = timestamp
                self.logger.info(f"Partially closed position: {symbol}, realized PnL: ${realized_pnl:.2f}")
            
            self.cash_balance += cost
            self._record_trade(symbol, quantity, price, action, realized_pnl, timestamp)
        
        self._update_portfolio_value(timestamp)
    
    def update_market_prices(self, price_updates: Dict[str, float], timestamp: datetime):
        """
        Update current prices for all positions
        
        Args:
            price_updates: Dict of symbol -> current price
            timestamp: Update timestamp
        """
        for symbol, price in price_updates.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.position_value = abs(position.quantity) * price
                
                # Calculate unrealized PnL
                if position.quantity > 0:  # Long position
                    position.unrealized_pnl = (price - position.entry_price) * position.quantity
                    position.unrealized_pnl_pct = ((price - position.entry_price) / position.entry_price) * 100
                else:  # Short position
                    position.unrealized_pnl = (position.entry_price - price) * abs(position.quantity)
                    position.unrealized_pnl_pct = ((position.entry_price - price) / position.entry_price) * 100
                
                position.last_updated = timestamp
        
        self._update_portfolio_value(timestamp)
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics
        
        Returns:
            PortfolioMetrics dataclass with current metrics
        """
        total_value = self.get_total_value()
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized_pnl = sum(trade['realized_pnl'] for trade in self.trade_history)
        
        # Calculate daily PnL
        daily_pnl = self._calculate_daily_pnl()
        
        # Calculate advanced metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        volatility = self._calculate_volatility()
        beta = self._calculate_beta()
        correlation_matrix = self._calculate_correlation_matrix()
        
        return PortfolioMetrics(
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            daily_pnl=daily_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=beta,
            correlation_matrix=correlation_matrix
        )
    
    def get_position_allocations(self) -> Dict[str, float]:
        """
        Get current position allocations as percentages
        
        Returns:
            Dict of symbol -> allocation percentage
        """
        total_value = self.get_total_value()
        allocations = {}
        
        for symbol, position in self.positions.items():
            allocation = (position.position_value / total_value) * 100 if total_value > 0 else 0
            allocations[symbol] = allocation
        
        return allocations
    
    def check_rebalancing_needed(self) -> Dict[str, dict]:
        """
        Check if portfolio rebalancing is needed
        
        Returns:
            Dict with rebalancing recommendations
        """
        current_allocations = self.get_position_allocations()
        rebalancing_needed = {}
        
        for symbol, current_alloc in current_allocations.items():
            target_alloc = self.target_allocations.get(symbol, 0.0)
            
            if abs(current_alloc - target_alloc) > self.rebalancing_threshold:
                rebalancing_needed[symbol] = {
                    'current_allocation': current_alloc,
                    'target_allocation': target_alloc,
                    'deviation': current_alloc - target_alloc,
                    'action': 'REDUCE' if current_alloc > target_alloc else 'INCREASE'
                }
        
        # Check for new positions to add
        for symbol, target_alloc in self.target_allocations.items():
            if symbol not in current_allocations and target_alloc > 0:
                rebalancing_needed[symbol] = {
                    'current_allocation': 0.0,
                    'target_allocation': target_alloc,
                    'deviation': -target_alloc,
                    'action': 'ADD'
                }
        
        return rebalancing_needed
    
    def generate_rebalancing_orders(self) -> List[dict]:
        """
        Generate rebalancing orders based on target allocations
        
        Returns:
            List of order dicts with symbol, action, value, reason
        """
        rebalancing_needed = self.check_rebalancing_needed()
        orders = []
        
        total_value = self.get_total_value()
        
        for symbol, rebalance_info in rebalancing_needed.items():
            target_value = total_value * (rebalance_info['target_allocation'] / 100)
            
            if rebalance_info['action'] == 'ADD':
                # New position
                if symbol in self.positions:
                    current_value = self.positions[symbol].position_value
                else:
                    current_value = 0
                
                order_value = target_value - current_value
                
                if order_value > 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'value': order_value,
                        'reason': f'Rebalance to {rebalance_info["target_allocation"]:.1f}%'
                    })
            
            elif rebalance_info['action'] == 'INCREASE':
                # Increase existing position
                current_value = self.positions[symbol].position_value
                order_value = target_value - current_value
                
                if order_value > 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'value': order_value,
                        'reason': f'Rebalance to {rebalance_info["target_allocation"]:.1f}%'
                    })
            
            elif rebalance_info['action'] == 'REDUCE':
                # Reduce existing position
                current_value = self.positions[symbol].position_value
                order_value = current_value - target_value
                
                if order_value > 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'value': order_value,
                        'reason': f'Rebalance to {rebalance_info["target_allocation"]:.1f}%'
                    })
        
        return orders
    
    def get_total_value(self) -> float:
        """
        Get total portfolio value (cash + positions)
        
        Returns:
            Total portfolio value
        """
        positions_value = sum(pos.position_value for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    def get_correlation_analysis(self) -> dict:
        """
        Get correlation analysis for current portfolio
        
        Returns:
            Dict with correlation information
        """
        positions = list(self.positions.keys())
        correlation_analysis = {}
        
        if len(positions) < 2:
            return {'message': 'Insufficient positions for correlation analysis'}
        
        # Simplified correlation based on asset classes
        asset_classes = self._classify_assets(positions)
        
        for i, sym1 in enumerate(positions):
            for sym2 in positions[i+1:]:
                class1 = asset_classes.get(sym1, 'unknown')
                class2 = asset_classes.get(sym2, 'unknown')
                
                # Simplified correlation estimation
                if class1 == class2:
                    correlation = 0.8  # High correlation within same class
                elif class1 in ['crypto', 'tech'] and class2 in ['crypto', 'tech']:
                    correlation = 0.6  # Medium correlation
                else:
                    correlation = 0.3  # Low correlation
                
                correlation_analysis[f"{sym1}-{sym2}"] = {
                    'correlation': correlation,
                    'risk_level': 'HIGH' if correlation > 0.7 else 'MEDIUM' if correlation > 0.4 else 'LOW'
                }
        
        return correlation_analysis
    
    def _classify_assets(self, symbols: List[str]) -> Dict[str, str]:
        """
        Classify assets into broad categories
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dict of symbol -> asset class
        """
        asset_classes = {}
        
        crypto_keywords = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'SOL', 'AVAX', 'MATIC', 'USDT']
        tech_keywords = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        
        for symbol in symbols:
            if any(keyword in symbol for keyword in crypto_keywords):
                asset_classes[symbol] = 'crypto'
            elif any(keyword in symbol for keyword in tech_keywords):
                asset_classes[symbol] = 'tech'
            else:
                asset_classes[symbol] = 'other'
        
        return asset_classes
    
    def _record_trade(self, symbol: str, quantity: float, price: float,
                     action: str, realized_pnl: float, timestamp: datetime):
        """
        Record trade in history
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            action: Trade action
            realized_pnl: Realized profit/loss
            timestamp: Trade timestamp
        """
        trade = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'action': action,
            'realized_pnl': realized_pnl,
            'timestamp': timestamp,
            'portfolio_value': self.get_total_value()
        }
        self.trade_history.append(trade)
    
    def _update_portfolio_value(self, timestamp: datetime):
        """
        Update portfolio value history
        
        Args:
            timestamp: Current timestamp
        """
        current_value = self.get_total_value()
        self.portfolio_values.append(current_value)
        self.portfolio_timestamps.append(timestamp)
    
    def _calculate_daily_pnl(self) -> float:
        """
        Calculate today's PnL
        
        Returns:
            Daily PnL amount
        """
        if not self.trade_history:
            return 0.0
        
        today = datetime.now().date()
        today_trades = [t for t in self.trade_history if t['timestamp'].date() == today]
        
        return sum(trade['realized_pnl'] for trade in today_trades)
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (annualized)
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(self.portfolio_values) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        excess_returns = [r - (risk_free_rate / 252) for r in returns]  # Daily risk-free rate
        avg_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        if std_excess_return == 0:
            return 0.0
        
        return (avg_excess_return / std_excess_return) * np.sqrt(252)  # Annualize
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown
        
        Returns:
            Max drawdown as percentage
        """
        if len(self.portfolio_values) < 2:
            return 0.0
        
        peak = self.portfolio_values[0]
        max_dd = 0.0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100  # Return as percentage
    
    def _calculate_volatility(self) -> float:
        """
        Calculate portfolio volatility (annualized)
        
        Returns:
            Volatility as percentage
        """
        if len(self.portfolio_values) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        return np.std(returns) * np.sqrt(252) * 100  # Annualized percentage
    
    def _calculate_beta(self) -> float:
        """
        Calculate portfolio beta (simplified)
        
        Returns:
            Beta value
        """
        # In production, this would compare portfolio returns to benchmark returns
        # For now, return a placeholder
        return 1.0
    
    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix (placeholder)
        
        Returns:
            Correlation matrix DataFrame
        """
        # In production, this would use historical price data
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def generate_portfolio_report(self) -> dict:
        """
        Generate comprehensive portfolio report
        
        Returns:
            Dict with complete portfolio information
        """
        metrics = self.get_portfolio_metrics()
        allocations = self.get_position_allocations()
        rebalancing_info = self.check_rebalancing_needed()
        correlation_analysis = self.get_correlation_analysis()
        
        # Win rate calculation
        winning_trades = len([t for t in self.trade_history if t['realized_pnl'] > 0])
        losing_trades = len([t for t in self.trade_history if t['realized_pnl'] < 0])
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'total_value': metrics.total_value,
                'cash_balance': self.cash_balance,
                'positions_value': metrics.total_value - self.cash_balance,
                'total_positions': len(self.positions),
                'unrealized_pnl': metrics.unrealized_pnl,
                'realized_pnl': metrics.realized_pnl,
                'total_pnl': metrics.total_pnl,
                'total_pnl_pct': metrics.total_pnl_pct
            },
            'performance_metrics': {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'beta': metrics.beta,
                'daily_pnl': metrics.daily_pnl,
                'win_rate': win_rate
            },
            'position_allocations': allocations,
            'rebalancing_recommendations': rebalancing_info,
            'correlation_analysis': correlation_analysis,
            'recent_trades': self.trade_history[-10:] if self.trade_history else [],
            'current_positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'position_value': pos.position_value,
                    'duration': str(datetime.now() - pos.entry_time)
                }
                for symbol, pos in self.positions.items()
            }
        }
