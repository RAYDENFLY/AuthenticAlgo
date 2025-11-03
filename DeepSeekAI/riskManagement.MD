🛡️ RISK MANAGEMENT MODULE
1. risk/init.py
python
"""
Risk Management Module
Comprehensive risk management for trading operations
"""

from .risk_manager import RiskManager
from .stop_loss import StopLossManager
from .portfolio import PortfolioManager

class RiskManagement:
    """Main interface for all risk management components"""
    
    def __init__(self, config: dict):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.stop_loss_manager = StopLossManager(config)
        self.portfolio_manager = PortfolioManager(config)
        
    def validate_trade(self, symbol: str, quantity: float, price: float, 
                      order_type: str, current_positions: dict) -> dict:
        """Comprehensive trade validation"""
        return self.risk_manager.validate_trade(symbol, quantity, price, order_type, current_positions)
    
    def calculate_position_size(self, symbol: str, price: float, 
                              stop_loss_price: float, account_balance: float) -> float:
        """Calculate optimal position size"""
        return self.risk_manager.calculate_position_size(symbol, price, stop_loss_price, account_balance)
    
    def update_portfolio(self, symbol: str, quantity: float, price: float, 
                        action: str, timestamp: str):
        """Update portfolio with new trade"""
        return self.portfolio_manager.update_position(symbol, quantity, price, action, timestamp)

__all__ = ['RiskManagement', 'RiskManager', 'StopLossManager', 'PortfolioManager']
2. risk/risk_manager.py
python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Risk metrics for a single trade"""
    position_size: float
    risk_per_trade: float
    max_drawdown: float
    daily_loss: float
    correlation_risk: float
    overall_risk_score: float

class RiskManager:
    """
    Comprehensive risk management system
    Handles position sizing, drawdown limits, daily loss limits, and correlation checks
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("risk.risk_manager")
        
        # Risk parameters
        self.risk_params = config.get('risk_management', {})
        self.max_position_size_pct = self.risk_params.get('max_position_size_pct', 10.0)
        self.max_daily_loss_pct = self.risk_params.get('max_daily_loss_pct', 5.0)
        self.max_drawdown_pct = self.risk_params.get('max_drawdown_pct', 15.0)
        self.risk_per_trade_pct = self.risk_params.get('risk_per_trade_pct', 2.0)
        self.max_portfolio_exposure_pct = self.risk_params.get('max_portfolio_exposure_pct', 25.0)
        self.correlation_threshold = self.risk_params.get('correlation_threshold', 0.7)
        
        # Circuit breaker settings
        self.circuit_breakers = self.risk_params.get('circuit_breakers', {})
        self.volatility_threshold = self.circuit_breakers.get('volatility_threshold', 5.0)
        self.max_consecutive_losses = self.circuit_breakers.get('max_consecutive_losses', 3)
        
        # Tracking
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.0
        self.consecutive_losses = 0
        self.trade_history = []
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""
        
        # Initialize daily limits
        self._reset_daily_limits()
        
        self.logger.info("Risk Manager initialized")
    
    def _reset_daily_limits(self):
        """Reset daily limits at the start of each day"""
        account_balance = self.config.get('initial_capital', 10000)
        self.max_daily_loss = account_balance * (self.max_daily_loss_pct / 100)
        self.daily_pnl = 0.0
        self.logger.info(f"Daily loss limit reset to: ${self.max_daily_loss:.2f}")
    
    def validate_trade(self, symbol: str, quantity: float, price: float, 
                      order_type: str, current_positions: dict) -> Dict[str, any]:
        """
        Comprehensive trade validation
        Returns: dict with 'approved', 'reason', 'adjusted_quantity'
        """
        # Check circuit breaker first
        if self.circuit_breaker_active:
            return {
                'approved': False,
                'reason': f'Circuit breaker active: {self.circuit_breaker_reason}',
                'adjusted_quantity': 0.0
            }
        
        # Calculate position value
        position_value = quantity * price
        account_balance = self.config.get('initial_capital', 10000)
        
        # 1. Position size validation
        position_size_pct = (position_value / account_balance) * 100
        if position_size_pct > self.max_position_size_pct:
            adjusted_quantity = (account_balance * (self.max_position_size_pct / 100)) / price
            return {
                'approved': True,
                'reason': f'Position size adjusted from {position_size_pct:.1f}% to {self.max_position_size_pct}%',
                'adjusted_quantity': adjusted_quantity
            }
        
        # 2. Daily loss limit check
        if self.daily_pnl <= -self.max_daily_loss:
            return {
                'approved': False,
                'reason': f'Daily loss limit reached: ${self.daily_pnl:.2f}',
                'adjusted_quantity': 0.0
            }
        
        # 3. Portfolio exposure check
        total_exposure = self._calculate_portfolio_exposure(current_positions, price, quantity)
        if total_exposure > self.max_portfolio_exposure_pct:
            return {
                'approved': False,
                'reason': f'Portfolio exposure limit exceeded: {total_exposure:.1f}%',
                'adjusted_quantity': 0.0
            }
        
        # 4. Correlation check (for multi-asset portfolios)
        if len(current_positions) > 0:
            correlation_risk = self._check_correlation_risk(symbol, current_positions)
            if correlation_risk > self.correlation_threshold:
                return {
                    'approved': False,
                    'reason': f'High correlation risk: {correlation_risk:.2f}',
                    'adjusted_quantity': 0.0
                }
        
        # 5. Volatility check
        volatility_risk = self._check_volatility_risk(symbol, price)
        if volatility_risk == RiskLevel.HIGH:
            adjusted_quantity = quantity * 0.5  # Reduce position in high volatility
            return {
                'approved': True,
                'reason': 'High volatility detected - position reduced by 50%',
                'adjusted_quantity': adjusted_quantity
            }
        elif volatility_risk == RiskLevel.CRITICAL:
            return {
                'approved': False,
                'reason': 'Critical volatility - trading suspended',
                'adjusted_quantity': 0.0
            }
        
        return {
            'approved': True,
            'reason': 'Trade validated successfully',
            'adjusted_quantity': quantity
        }
    
    def calculate_position_size(self, symbol: str, price: float, 
                              stop_loss_price: float, account_balance: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk-based methods
        """
        # Method 1: Fixed fractional position sizing
        risk_amount = account_balance * (self.risk_per_trade_pct / 100)
        
        # Calculate stop distance
        if stop_loss_price:
            stop_distance_pct = abs(price - stop_loss_price) / price * 100
            if stop_distance_pct > 0:
                # Position size based on risk amount and stop distance
                position_size_risk = risk_amount / (stop_distance_pct / 100)
            else:
                position_size_risk = account_balance * (self.max_position_size_pct / 100)
        else:
            position_size_risk = account_balance * (self.max_position_size_pct / 100)
        
        # Method 2: Volatility-adjusted position sizing
        volatility_factor = self._get_volatility_factor(symbol)
        position_size_vol = account_balance * (self.max_position_size_pct / 100) * volatility_factor
        
        # Use the more conservative approach
        position_size = min(position_size_risk, position_size_vol)
        
        # Convert to quantity
        quantity = position_size / price
        
        self.logger.debug(f"Position size calculated: {quantity:.4f} units (${position_size:.2f})")
        return quantity
    
    def update_trade_result(self, symbol: str, quantity: float, entry_price: float,
                           exit_price: float, pnl: float, timestamp: datetime):
        """
        Update risk metrics with trade results
        """
        trade_data = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'timestamp': timestamp
        }
        self.trade_history.append(trade_data)
        
        # Update daily PnL
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check for circuit breaker triggers
        self._check_circuit_breakers()
        
        self.logger.info(f"Trade updated: {symbol} PnL: ${pnl:.2f}, Daily PnL: ${self.daily_pnl:.2f}")
    
    def _calculate_portfolio_exposure(self, current_positions: dict, 
                                    new_price: float, new_quantity: float) -> float:
        """Calculate total portfolio exposure percentage"""
        account_balance = self.config.get('initial_capital', 10000)
        
        # Calculate existing positions value
        existing_exposure = 0.0
        for symbol, position in current_positions.items():
            existing_exposure += position['quantity'] * position['current_price']
        
        # Add new position
        new_exposure = new_quantity * new_price
        total_exposure = existing_exposure + new_exposure
        
        return (total_exposure / account_balance) * 100
    
    def _check_correlation_risk(self, new_symbol: str, current_positions: dict) -> float:
        """
        Check correlation risk between new symbol and existing positions
        Returns: correlation score (0-1)
        """
        # This is a simplified correlation check
        # In production, you would use historical price data
        
        # Group symbols by asset class for basic correlation estimation
        crypto_pairs = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        new_asset_class = None
        if any(pair in new_symbol for pair in crypto_pairs):
            new_asset_class = 'crypto'
        elif any(stock in new_symbol for stock in tech_stocks):
            new_asset_class = 'tech'
        
        correlated_positions = 0
        for symbol in current_positions.keys():
            if any(pair in symbol for pair in crypto_pairs) and new_asset_class == 'crypto':
                correlated_positions += 1
            elif any(stock in symbol for stock in tech_stocks) and new_asset_class == 'tech':
                correlated_positions += 1
        
        return correlated_positions / len(current_positions) if current_positions else 0.0
    
    def _check_volatility_risk(self, symbol: str, current_price: float) -> RiskLevel:
        """
        Check volatility risk for a symbol
        Returns: RiskLevel enum
        """
        # Simplified volatility check
        # In production, use ATR, historical volatility, or VIX
        
        # Mock volatility data - replace with real data source
        high_volatility_symbols = ['BTC', 'ETH', 'TSLA']
        critical_volatility_events = []  # Could be fed from news or market data
        
        if symbol in critical_volatility_events:
            return RiskLevel.CRITICAL
        elif any(vol_symbol in symbol for vol_symbol in high_volatility_symbols):
            return RiskLevel.HIGH
        else:
            return RiskLevel.LOW
    
    def _get_volatility_factor(self, symbol: str) -> float:
        """Get volatility factor for position sizing (0.1 - 1.0)"""
        risk_level = self._check_volatility_risk(symbol, 0)
        
        volatility_factors = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.1
        }
        
        return volatility_factors.get(risk_level, 0.5)
    
    def _check_circuit_breakers(self):
        """Check and activate circuit breakers if necessary"""
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = f"Consecutive losses: {self.consecutive_losses}"
            self.logger.warning(f"Circuit breaker activated: {self.circuit_breaker_reason}")
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = f"Daily loss limit exceeded: ${self.daily_pnl:.2f}"
            self.logger.warning(f"Circuit breaker activated: {self.circuit_breaker_reason}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention required)"""
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""
        self.logger.info("Circuit breaker reset")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        account_balance = self.config.get('initial_capital', 10000)
        
        # Calculate max drawdown
        equity_curve = [trade['pnl'] for trade in self.trade_history]
        running_max = 0
        max_drawdown = 0
        current_value = account_balance
        
        for pnl in equity_curve:
            current_value += pnl
            if current_value > running_max:
                running_max = current_value
            drawdown = (running_max - current_value) / running_max * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return RiskMetrics(
            position_size=self.max_position_size_pct,
            risk_per_trade=self.risk_per_trade_pct,
            max_drawdown=max_drawdown,
            daily_loss=abs(self.daily_pnl),
            correlation_risk=0.0,  # Would need portfolio data
            overall_risk_score=self._calculate_overall_risk_score(max_drawdown)
        )
    
    def _calculate_overall_risk_score(self, max_drawdown: float) -> float:
        """Calculate overall risk score (0-1, where 1 is highest risk)"""
        drawdown_score = min(1.0, max_drawdown / self.max_drawdown_pct)
        daily_loss_score = min(1.0, abs(self.daily_pnl) / self.max_daily_loss)
        consecutive_loss_score = min(1.0, self.consecutive_losses / self.max_consecutive_losses)
        
        return (drawdown_score + daily_loss_score + consecutive_loss_score) / 3
    
    def generate_risk_report(self) -> dict:
        """Generate comprehensive risk report"""
        metrics = self.get_risk_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': self.max_daily_loss,
            'remaining_daily_loss': self.max_daily_loss + self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'total_trades': len(self.trade_history),
            'winning_trades': len([t for t in self.trade_history if t['pnl'] > 0]),
            'losing_trades': len([t for t in self.trade_history if t['pnl'] < 0]),
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_reason': self.circuit_breaker_reason,
            'risk_metrics': {
                'position_size_pct': metrics.position_size,
                'risk_per_trade_pct': metrics.risk_per_trade,
                'max_drawdown_pct': metrics.max_drawdown,
                'daily_loss_pct': (abs(self.daily_pnl) / self.config.get('initial_capital', 10000)) * 100,
                'overall_risk_score': metrics.overall_risk_score
            },
            'recommendations': self._generate_risk_recommendations(metrics)
        }
    
    def _generate_risk_recommendations(self, metrics: RiskMetrics) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if metrics.overall_risk_score > 0.8:
            recommendations.append("CRITICAL: Consider stopping trading temporarily")
        elif metrics.overall_risk_score > 0.6:
            recommendations.append("HIGH: Reduce position sizes significantly")
        elif metrics.overall_risk_score > 0.4:
            recommendations.append("MEDIUM: Consider reducing position sizes")
        
        if self.consecutive_losses >= 2:
            recommendations.append(f"Warning: {self.consecutive_losses} consecutive losses")
        
        if metrics.max_drawdown > self.max_drawdown_pct * 0.8:
            recommendations.append(f"Warning: Drawdown approaching limit ({metrics.max_drawdown:.1f}%)")
        
        return recommendations
3. risk/stop_loss.py
python
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum
import logging

class StopLossType(Enum):
    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"
    MOVING_AVERAGE = "moving_average"
    SUPPORT_RESISTANCE = "support_resistance"

class StopLossManager:
    """
    Advanced stop-loss management system
    Multiple stop-loss types with dynamic adjustment
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("risk.stop_loss")
        
        # Stop-loss parameters
        self.stop_loss_config = config.get('stop_loss', {})
        self.default_type = self.stop_loss_config.get('default_type', StopLossType.FIXED_PERCENTAGE)
        self.atr_period = self.stop_loss_config.get('atr_period', 14)
        self.atr_multiplier = self.stop_loss_config.get('atr_multiplier', 2.0)
        self.fixed_stop_percentage = self.stop_loss_config.get('fixed_stop_percentage', 2.0)
        self.trailing_activation_pct = self.stop_loss_config.get('trailing_activation_pct', 1.0)
        self.trailing_percentage = self.stop_loss_config.get('trailing_percentage', 1.0)
        
        # Active stop-loss tracking
        self.active_stops = {}  # symbol -> stop_loss_data
        
        self.logger.info("Stop-Loss Manager initialized")
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, position_type: str,
                          data: pd.DataFrame, stop_type: Optional[StopLossType] = None) -> float:
        """
        Calculate stop-loss price based on selected method
        """
        if stop_type is None:
            stop_type = self.default_type
        
        if stop_type == StopLossType.FIXED_PERCENTAGE:
            return self._fixed_percentage_stop(entry_price, position_type)
        
        elif stop_type == StopLossType.ATR_BASED:
            return self._atr_based_stop(symbol, entry_price, position_type, data)
        
        elif stop_type == StopLossType.TRAILING:
            return self._trailing_stop(entry_price, position_type, entry_price)
        
        elif stop_type == StopLossType.MOVING_AVERAGE:
            return self._moving_average_stop(symbol, entry_price, position_type, data)
        
        elif stop_type == StopLossType.SUPPORT_RESISTANCE:
            return self._support_resistance_stop(symbol, entry_price, position_type, data)
        
        else:
            self.logger.warning(f"Unknown stop-loss type: {stop_type}, using fixed percentage")
            return self._fixed_percentage_stop(entry_price, position_type)
    
    def update_stop_loss(self, symbol: str, current_price: float, position_type: str,
                        data: pd.DataFrame) -> Optional[float]:
        """
        Update stop-loss for existing position (mainly for trailing stops)
        Returns new stop price if updated, None otherwise
        """
        if symbol not in self.active_stops:
            return None
        
        stop_data = self.active_stops[symbol]
        
        if stop_data['type'] == StopLossType.TRAILING:
            return self._update_trailing_stop(symbol, current_price, position_type, stop_data)
        
        elif stop_data['type'] == StopLossType.MOVING_AVERAGE:
            return self._update_moving_average_stop(symbol, current_price, position_type, data, stop_data)
        
        return None
    
    def set_active_stop(self, symbol: str, stop_price: float, stop_type: StopLossType,
                       entry_price: float, position_type: str):
        """Set active stop-loss for a position"""
        self.active_stops[symbol] = {
            'stop_price': stop_price,
            'type': stop_type,
            'entry_price': entry_price,
            'position_type': position_type,
            'highest_price': entry_price if position_type == 'long' else float('inf'),
            'lowest_price': entry_price if position_type == 'short' else 0.0
        }
        
        self.logger.info(f"Set {stop_type.value} stop for {symbol} at {stop_price}")
    
    def check_stop_loss(self, symbol: str, current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Check if stop-loss is triggered
        Returns: (triggered, reason)
        """
        if symbol not in self.active_stops:
            return False, None
        
        stop_data = self.active_stops[symbol]
        stop_price = stop_data['stop_price']
        position_type = stop_data['position_type']
        
        if position_type == 'long' and current_price <= stop_price:
            reason = f"Long stop-loss triggered at {stop_price}, current price {current_price}"
            self._remove_active_stop(symbol)
            return True, reason
        
        elif position_type == 'short' and current_price >= stop_price:
            reason = f"Short stop-loss triggered at {stop_price}, current price {current_price}"
            self._remove_active_stop(symbol)
            return True, reason
        
        return False, None
    
    def _fixed_percentage_stop(self, entry_price: float, position_type: str) -> float:
        """Fixed percentage stop-loss"""
        if position_type == 'long':
            return entry_price * (1 - self.fixed_stop_percentage / 100)
        else:  # short
            return entry_price * (1 + self.fixed_stop_percentage / 100)
    
    def _atr_based_stop(self, symbol: str, entry_price: float, 
                       position_type: str, data: pd.DataFrame) -> float:
        """ATR-based stop-loss"""
        atr = self._calculate_atr(data)
        
        if atr is None:
            self.logger.warning(f"Could not calculate ATR for {symbol}, using fixed percentage")
            return self._fixed_percentage_stop(entry_price, position_type)
        
        atr_stop_distance = atr * self.atr_multiplier
        
        if position_type == 'long':
            return entry_price - atr_stop_distance
        else:  # short
            return entry_price + atr_stop_distance
    
    def _trailing_stop(self, entry_price: float, position_type: str, 
                      current_price: float) -> float:
        """Trailing stop-loss (initial calculation)"""
        if position_type == 'long':
            stop_distance = entry_price * (self.trailing_percentage / 100)
            return entry_price - stop_distance
        else:  # short
            stop_distance = entry_price * (self.trailing_percentage / 100)
            return entry_price + stop_distance
    
    def _update_trailing_stop(self, symbol: str, current_price: float,
                            position_type: str, stop_data: dict) -> Optional[float]:
        """Update trailing stop-loss"""
        if position_type == 'long':
            # Update highest price
            if current_price > stop_data['highest_price']:
                stop_data['highest_price'] = current_price
                
                # Calculate new stop only if we've moved enough
                price_move_pct = ((current_price - stop_data['entry_price']) / stop_data['entry_price']) * 100
                if price_move_pct >= self.trailing_activation_pct:
                    new_stop = current_price * (1 - self.trailing_percentage / 100)
                    if new_stop > stop_data['stop_price']:
                        stop_data['stop_price'] = new_stop
                        self.logger.debug(f"Updated trailing stop for {symbol} to {new_stop}")
                        return new_stop
        
        else:  # short
            # Update lowest price
            if current_price < stop_data['lowest_price']:
                stop_data['lowest_price'] = current_price
                
                # Calculate new stop only if we've moved enough
                price_move_pct = ((stop_data['entry_price'] - current_price) / stop_data['entry_price']) * 100
                if price_move_pct >= self.trailing_activation_pct:
                    new_stop = current_price * (1 + self.trailing_percentage / 100)
                    if new_stop < stop_data['stop_price']:
                        stop_data['stop_price'] = new_stop
                        self.logger.debug(f"Updated trailing stop for {symbol} to {new_stop}")
                        return new_stop
        
        return None
    
    def _moving_average_stop(self, symbol: str, entry_price: float,
                           position_type: str, data: pd.DataFrame) -> float:
        """Moving average based stop-loss"""
        if len(data) < 20:
            self.logger.warning(f"Insufficient data for MA stop on {symbol}, using fixed percentage")
            return self._fixed_percentage_stop(entry_price, position_type)
        
        # Use 20-period EMA as dynamic stop
        ema_20 = data['close'].ewm(span=20).mean().iloc[-1]
        
        if position_type == 'long':
            return min(ema_20, self._fixed_percentage_stop(entry_price, position_type))
        else:  # short
            return max(ema_20, self._fixed_percentage_stop(entry_price, position_type))
    
    def _update_moving_average_stop(self, symbol: str, current_price: float,
                                  position_type: str, data: pd.DataFrame, 
                                  stop_data: dict) -> Optional[float]:
        """Update moving average stop-loss"""
        if len(data) < 20:
            return None
        
        ema_20 = data['close'].ewm(span=20).mean().iloc[-1]
        
        if position_type == 'long':
            new_stop = ema_20
            # Only move stop up, not down
            if new_stop > stop_data['stop_price']:
                stop_data['stop_price'] = new_stop
                self.logger.debug(f"Updated MA stop for {symbol} to {new_stop}")
                return new_stop
        
        else:  # short
            new_stop = ema_20
            # Only move stop down, not up
            if new_stop < stop_data['stop_price']:
                stop_data['stop_price'] = new_stop
                self.logger.debug(f"Updated MA stop for {symbol} to {new_stop}")
                return new_stop
        
        return None
    
    def _support_resistance_stop(self, symbol: str, entry_price: float,
                               position_type: str, data: pd.DataFrame) -> float:
        """Support/resistance based stop-loss"""
        if len(data) < 50:
            self.logger.warning(f"Insufficient data for S/R stop on {symbol}, using fixed percentage")
            return self._fixed_percentage_stop(entry_price, position_type)
        
        # Find recent support and resistance levels
        support_level = self._find_support_level(data)
        resistance_level = self._find_resistance_level(data)
        
        if position_type == 'long':
            # Use the more conservative stop (higher of support or fixed percentage)
            fixed_stop = self._fixed_percentage_stop(entry_price, position_type)
            return max(support_level, fixed_stop) if support_level else fixed_stop
        else:  # short
            # Use the more conservative stop (lower of resistance or fixed percentage)
            fixed_stop = self._fixed_percentage_stop(entry_price, position_type)
            return min(resistance_level, fixed_stop) if resistance_level else fixed_stop
    
    def _find_support_level(self, data: pd.DataFrame, lookback: int = 20) -> Optional[float]:
        """Find recent support level"""
        if len(data) < lookback:
            return None
        
        recent_lows = data['low'].tail(lookback)
        return recent_lows.min()
    
    def _find_resistance_level(self, data: pd.DataFrame, lookback: int = 20) -> Optional[float]:
        """Find recent resistance level"""
        if len(data) < lookback:
            return None
        
        recent_highs = data['high'].tail(lookback)
        return recent_highs.max()
    
    def _calculate_atr(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate Average True Range"""
        if len(data) < self.atr_period + 1:
            return None
        
        high = data['high']
        low = data['low']
        close_prev = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
        
        return atr
    
    def _remove_active_stop(self, symbol: str):
        """Remove active stop-loss for a symbol"""
        if symbol in self.active_stops:
            del self.active_stops[symbol]
            self.logger.info(f"Removed active stop for {symbol}")
    
    def get_active_stops(self) -> dict:
        """Get all active stop-loss orders"""
        return self.active_stops.copy()
    
    def get_stop_loss_recommendation(self, symbol: str, entry_price: float,
                                   position_type: str, data: pd.DataFrame,
                                   volatility: float) -> dict:
        """
        Get stop-loss recommendation with multiple methods
        """
        recommendations = {}
        
        # Calculate stops for all methods
        for stop_type in StopLossType:
            stop_price = self.calculate_stop_loss(symbol, entry_price, position_type, data, stop_type)
            stop_distance_pct = abs(entry_price - stop_price) / entry_price * 100
            
            recommendations[stop_type.value] = {
                'stop_price': stop_price,
                'distance_pct': stop_distance_pct,
                'risk_reward_ratio': self._calculate_risk_reward_ratio(entry_price, stop_price, position_type)
            }
        
        # Add volatility-adjusted recommendation
        vol_adjusted_stop = self._volatility_adjusted_stop(entry_price, position_type, volatility)
        recommendations['volatility_adjusted'] = {
            'stop_price': vol_adjusted_stop,
            'distance_pct': abs(entry_price - vol_adjusted_stop) / entry_price * 100,
            'risk_reward_ratio': self._calculate_risk_reward_ratio(entry_price, vol_adjusted_stop, position_type)
        }
        
        return recommendations
    
    def _volatility_adjusted_stop(self, entry_price: float, position_type: str,
                                volatility: float) -> float:
        """Volatility-adjusted stop-loss"""
        # Use volatility to adjust stop distance (higher volatility = wider stop)
        base_distance = self.fixed_stop_percentage / 100
        adjusted_distance = base_distance * (1 + volatility)
        
        if position_type == 'long':
            return entry_price * (1 - adjusted_distance)
        else:  # short
            return entry_price * (1 + adjusted_distance)
    
    def _calculate_risk_reward_ratio(self, entry_price: float, stop_price: float,
                                   position_type: str, target_price: float = None) -> float:
        """Calculate risk-reward ratio"""
        risk = abs(entry_price - stop_price)
        
        if target_price is None:
            # Assume 2:1 reward if no target specified
            if position_type == 'long':
                target_price = entry_price + (2 * risk)
            else:
                target_price = entry_price - (2 * risk)
        
        reward = abs(entry_price - target_price)
        
        return reward / risk if risk > 0 else 0.0
4. risk/portfolio.py
python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

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
    Tracks positions, calculates metrics, and manages rebalancing
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("risk.portfolio")
        
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
        
        self.logger.info(f"Portfolio Manager initialized with ${self.initial_capital:,.2f}")
    
    def update_position(self, symbol: str, quantity: float, price: float, 
                       action: str, timestamp: datetime):
        """
        Update portfolio with new trade
        Actions: 'BUY', 'SELL', 'SHORT', 'COVER'
        """
        cost = quantity * price
        
        if action in ['BUY', 'SHORT']:
            # Opening position
            if symbol in self.positions:
                # Average existing position
                existing = self.positions[symbol]
                total_quantity = existing.quantity + (quantity if action == 'BUY' else -quantity)
                total_cost = (existing.quantity * existing.entry_price) + cost
                
                if total_quantity != 0:
                    new_entry_price = total_cost / total_quantity
                else:
                    new_entry_price = 0
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_quantity,
                    entry_price=new_entry_price,
                    current_price=price,
                    unrealized_pnl=(price - new_entry_price) * total_quantity if action == 'BUY' else (new_entry_price - price) * abs(total_quantity),
                    unrealized_pnl_pct=(price - new_entry_price) / new_entry_price * 100 if action == 'BUY' else (new_entry_price - price) / new_entry_price * 100,
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
            else:
                self.positions[symbol].quantity = new_quantity
                self.positions[symbol].position_value = abs(new_quantity) * price
                self.positions[symbol].last_updated = timestamp
            
            self.cash_balance += cost
            self._record_trade(symbol, quantity, price, action, realized_pnl, timestamp)
        
        self._update_portfolio_value(timestamp)
        self.logger.info(f"Portfolio updated: {action} {quantity} {symbol} @ {price}")
    
    def update_market_prices(self, price_updates: Dict[str, float], timestamp: datetime):
        """Update current prices for all positions"""
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
        """Calculate comprehensive portfolio metrics"""
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
        """Get current position allocations as percentages"""
        total_value = self.get_total_value()
        allocations = {}
        
        for symbol, position in self.positions.items():
            allocation = (position.position_value / total_value) * 100
            allocations[symbol] = allocation
        
        return allocations
    
    def check_rebalancing_needed(self) -> Dict[str, dict]:
        """
        Check if portfolio rebalancing is needed
        Returns: dict with rebalancing recommendations
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
        """Generate rebalancing orders based on target allocations"""
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
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(pos.position_value for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    def get_correlation_analysis(self) -> dict:
        """Get correlation analysis for current portfolio"""
        # This would typically use historical price data
        # For now, return a simplified analysis
        
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
        """Classify assets into broad categories"""
        asset_classes = {}
        
        crypto_keywords = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'USDT']
        tech_keywords = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        
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
        """Record trade in history"""
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
        """Update portfolio value history"""
        current_value = self.get_total_value()
        self.portfolio_values.append(current_value)
        self.portfolio_timestamps.append(timestamp)
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's PnL"""
        if not self.trade_history:
            return 0.0
        
        today = datetime.now().date()
        today_trades = [t for t in self.trade_history if t['timestamp'].date() == today]
        
        return sum(trade['realized_pnl'] for trade in today_trades)
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)"""
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
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        peak = self.portfolio_values[0]
        max_dd = 0.0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100  # Return as percentage
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility (annualized)"""
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
        """Calculate portfolio beta (simplified)"""
        # In production, this would compare portfolio returns to benchmark returns
        # For now, return a placeholder
        return 1.0
    
    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix (placeholder)"""
        # In production, this would use historical price data
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def generate_portfolio_report(self) -> dict:
        """Generate comprehensive portfolio report"""
        metrics = self.get_portfolio_metrics()
        allocations = self.get_position_allocations()
        rebalancing_info = self.check_rebalancing_needed()
        correlation_analysis = self.get_correlation_analysis()
        
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
                'daily_pnl': metrics.daily_pnl
            },
            'position_allocations': allocations,
            'rebalancing_recommendations': rebalancing_info,
            'correlation_analysis': correlation_analysis,
            'recent_trades': self.trade_history[-10:] if self.trade_history else []
        }
5. tests/test_risk.py
python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.risk_manager import RiskManager, RiskLevel
from risk.stop_loss import StopLossManager, StopLossType
from risk.portfolio import PortfolioManager

class TestRiskManagement:
    """Test suite for risk management module"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for risk management"""
        return {
            'initial_capital': 10000,
            'risk_management': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_drawdown_pct': 15.0,
                'risk_per_trade_pct': 2.0,
                'max_portfolio_exposure_pct': 25.0,
                'correlation_threshold': 0.7,
                'circuit_breakers': {
                    'volatility_threshold': 5.0,
                    'max_consecutive_losses': 3
                }
            },
            'stop_loss': {
                'default_type': 'fixed_percentage',
                'fixed_stop_percentage': 2.0,
                'atr_period': 14,
                'atr_multiplier': 2.0,
                'trailing_activation_pct': 1.0,
                'trailing_percentage': 1.0
            },
            'portfolio': {
                'max_positions': 10,
                'rebalancing_threshold': 5.0,
                'target_allocations': {
                    'BTC/USDT': 10.0,
                    'ETH/USDT': 8.0
                }
            }
        }
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        data = {
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 0.5,
            'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 0.5,
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_risk_manager_creation(self, sample_config):
        """Test risk manager creation"""
        risk_manager = RiskManager(sample_config)
        
        assert risk_manager.max_position_size_pct == 10.0
        assert risk_manager.max_daily_loss_pct == 5.0
        assert risk_manager.risk_per_trade_pct == 2.0
    
    def test_trade_validation(self, sample_config):
        """Test trade validation"""
        risk_manager = RiskManager(sample_config)
        
        # Test valid trade
        result = risk_manager.validate_trade(
            symbol='BTC/USDT',
            quantity=1.0,
            price=100.0,
            order_type='BUY',
            current_positions={}
        )
        
        assert 'approved' in result
        assert 'reason' in result
        assert 'adjusted_quantity' in result
    
    def test_position_size_calculation(self, sample_config):
        """Test position size calculation"""
        risk_manager = RiskManager(sample_config)
        
        position_size = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            price=100.0,
            stop_loss_price=98.0,
            account_balance=10000
        )
        
        assert position_size > 0
        assert position_size <= (10000 * 0.1) / 100  # Max position size
    
    def test_stop_loss_manager_creation(self, sample_config):
        """Test stop-loss manager creation"""
        stop_manager = StopLossManager(sample_config)
        
        assert stop_manager.default_type == StopLossType.FIXED_PERCENTAGE
        assert stop_manager.fixed_stop_percentage == 2.0
    
    def test_fixed_percentage_stop(self, sample_config):
        """Test fixed percentage stop-loss"""
        stop_manager = StopLossManager(sample_config)
        
        # Long position
        stop_price_long = stop_manager.calculate_stop_loss(
            symbol='BTC/USDT',
            entry_price=100.0,
            position_type='long',
            data=pd.DataFrame(),
            stop_type=StopLossType.FIXED_PERCENTAGE
        )
        
        assert stop_price_long == 98.0  # 100 - 2%
        
        # Short position
        stop_price_short = stop_manager.calculate_stop_loss(
            symbol='BTC/USDT',
            entry_price=100.0,
            position_type='short',
            data=pd.DataFrame(),
            stop_type=StopLossType.FIXED_PERCENTAGE
        )
        
        assert stop_price_short == 102.0  # 100 + 2%
    
    def test_portfolio_manager_creation(self, sample_config):
        """Test portfolio manager creation"""
        portfolio_manager = PortfolioManager(sample_config)
        
        assert portfolio_manager.initial_capital == 10000
        assert portfolio_manager.cash_balance == 10000
        assert len(portfolio_manager.positions) == 0
    
    def test_portfolio_position_management(self, sample_config):
        """Test portfolio position management"""
        portfolio_manager = PortfolioManager(sample_config)
        
        # Test buying a position
        portfolio_manager.update_position(
            symbol='BTC/USDT',
            quantity=1.0,
            price=100.0,
            action='BUY',
            timestamp=datetime.now()
        )
        
        assert 'BTC/USDT' in portfolio_manager.positions
        assert portfolio_manager.positions['BTC/USDT'].quantity == 1.0
        assert portfolio_manager.cash_balance == 10000 - 100.0
        
        # Test selling the position
        portfolio_manager.update_position(
            symbol='BTC/USDT',
            quantity=1.0,
            price=110.0,
            action='SELL',
            timestamp=datetime.now()
        )
        
        assert 'BTC/USDT' not in portfolio_manager.positions
        assert portfolio_manager.cash_balance == 10000 + 10.0  # Initial + profit
    
    def test_portfolio_metrics(self, sample_config):
        """Test portfolio metrics calculation"""
        portfolio_manager = PortfolioManager(sample_config)
        
        # Add some positions
        portfolio_manager.update_position(
            symbol='BTC/USDT',
            quantity=1.0,
            price=100.0,
            action='BUY',
            timestamp=datetime.now()
        )
        
        # Update market prices
        portfolio_manager.update_market_prices(
            {'BTC/USDT': 110.0},
            datetime.now()
        )
        
        metrics = portfolio_manager.get_portfolio_metrics()
        
        assert metrics.total_value > 0
        assert metrics.unrealized_pnl == 10.0  # 110 - 100
    
    def test_circuit_breaker_activation(self, sample_config):
        """Test circuit breaker activation"""
        risk_manager = RiskManager(sample_config)
        
        # Simulate consecutive losses
        for i in range(3):
            risk_manager.update_trade_result(
                symbol='BTC/USDT',
                quantity=1.0,
                entry_price=100.0,
                exit_price=95.0,  # 5% loss
                pnl=-50.0,
                timestamp=datetime.now()
            )
        
        assert risk_manager.circuit_breaker_active == True
        assert "consecutive losses" in risk_manager.circuit_breaker_reason.lower()
    
    def test_risk_report_generation(self, sample_config):
        """Test risk report generation"""
        risk_manager = RiskManager(sample_config)
        
        report = risk_manager.generate_risk_report()
        
        assert 'timestamp' in report
        assert 'daily_pnl' in report
        assert 'risk_metrics' in report
        assert 'recommendations' in report
    
    def test_rebalancing_recommendations(self, sample_config):
        """Test portfolio rebalancing recommendations"""
        portfolio_manager = PortfolioManager(sample_config)
        
        # Add a position that exceeds target allocation
        portfolio_manager.update_position(
            symbol='BTC/USDT',
            quantity=2.0,
            price=1000.0,  # $2000 position, 20% of portfolio
            action='BUY',
            timestamp=datetime.now()
        )
        
        rebalancing = portfolio_manager.check_rebalancing_needed()
        
        assert 'BTC/USDT' in rebalancing
        assert rebalancing['BTC/USDT']['action'] == 'REDUCE'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
6. Example Usage
python
# examples/risk_management_example.py
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk import RiskManagement

def run_risk_management_example():
    """Example of using the risk management module"""
    
    # Configuration
    config = {
        'initial_capital': 50000,
        'risk_management': {
            'max_position_size_pct': 5.0,
            'max_daily_loss_pct': 2.0,
            'risk_per_trade_pct': 1.0,
            'max_portfolio_exposure_pct': 20.0,
            'correlation_threshold': 0.6,
            'circuit_breakers': {
                'max_consecutive_losses': 2
            }
        },
        'stop_loss': {
            'default_type': 'atr_based',
            'atr_period': 14,
            'atr_multiplier': 2.0
        },
        'portfolio': {
            'max_positions': 8,
            'rebalancing_threshold': 3.0,
            'target_allocations': {
                'BTC/USDT': 5.0,
                'ETH/USDT': 4.0,
                'ADA/USDT': 3.0
            }
        }
    }
    
    # Initialize risk management
    risk_management = RiskManagement(config)
    
    # Example 1: Validate a trade
    validation_result = risk_management.validate_trade(
        symbol='BTC/USDT',
        quantity=1.0,
        price=45000.0,
        order_type='BUY',
        current_positions={}
    )
    
    print("=== TRADE VALIDATION ===")
    print(f"Approved: {validation_result['approved']}")
    print(f"Reason: {validation_result['reason']}")
    print(f"Adjusted Quantity: {validation_result['adjusted_quantity']}")
    
    # Example 2: Calculate position size
    position_size = risk_management.risk_manager.calculate_position_size(
        symbol='BTC/USDT',
        price=45000.0,
        stop_loss_price=44000.0,
        account_balance=50000
    )
    
    print(f"\n=== POSITION SIZING ===")
    print(f"Recommended position size: {position_size:.4f} BTC")
    print(f"Position value: ${position_size * 45000:.2f}")
    
    # Example 3: Generate risk report
    risk_report = risk_management.risk_manager.generate_risk_report()
    print(f"\n=== RISK REPORT ===")
    print(f"Daily PnL: ${risk_report['daily_pnl']:.2f}")
    print(f"Consecutive Losses: {risk_report['consecutive_losses']}")
    print(f"Circuit Breaker Active: {risk_report['circuit_breaker_active']}")
    
    # Example 4: Portfolio management
    risk_management.update_portfolio(
        symbol='BTC/USDT',
        quantity=0.5,
        price=45000.0,
        action='BUY',
        timestamp=datetime.now()
    )
    
    portfolio_report = risk_management.portfolio_manager.generate_portfolio_report()
    print(f"\n=== PORTFOLIO REPORT ===")
    print(f"Total Value: ${portfolio_report['portfolio_summary']['total_value']:.2f}")
    print(f"Positions: {portfolio_report['portfolio_summary']['total_positions']}")
    
    # Example 5: Stop-loss recommendations
    import pandas as pd
    sample_data = pd.DataFrame({
        'open': [44800, 44900, 45100],
        'high': [45200, 45300, 45400],
        'low': [44700, 44800, 44900],
        'close': [45000, 45100, 45200],
        'volume': [1000, 1200, 1100]
    })
    
    stop_recommendations = risk_management.stop_loss_manager.get_stop_loss_recommendation(
        symbol='BTC/USDT',
        entry_price=45000.0,
        position_type='long',
        data=sample_data,
        volatility=2.5
    )
    
    print(f"\n=== STOP-LOSS RECOMMENDATIONS ===")
    for method, recommendation in stop_recommendations.items():
        print(f"{method}: ${recommendation['stop_price']:.2f} "
              f"(Distance: {recommendation['distance_pct']:.2f}%)")

if __name__ == "__main__":
    run_risk_management_example()
🛡️ Key Features Implemented
Risk Manager
✅ Position Size Validation - Prevents over-concentration
✅ Daily Loss Limits - Circuit breaker for bad days
✅ Max Drawdown Protection - Portfolio-level risk control
✅ Correlation Analysis - Avoids correlated positions
✅ Circuit Breakers - Automatic trading suspension
✅ Volatility Adjustments - Dynamic risk parameters

Stop-Loss Manager
✅ Multiple Stop Types - Fixed, ATR, Trailing, MA, Support/Resistance
✅ Dynamic Updates - Real-time stop adjustment
✅ Risk-Reward Analysis - Stop-loss recommendations
✅ Volatility Adjustment - Adaptive stop distances

Portfolio Manager
✅ Real-time Tracking - Live position monitoring
✅ Performance Metrics - Sharpe ratio, drawdown, volatility
✅ Rebalancing Logic - Automatic portfolio optimization
✅ Correlation Analysis - Multi-asset risk assessment
✅ Comprehensive Reporting - Detailed portfolio insights

📊 Risk Management Benefits
Capital Preservation - Strict position and loss limits

Emotional Discipline - Automated risk controls

Portfolio Optimization - Smart rebalancing and allocation

Risk-Adjusted Returns - Better Sharpe ratios

Drawdown Control - Protection during market downturns