"""
Stop-Loss Manager Module
Advanced stop-loss management with multiple strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum

from core.logger import get_logger


class StopLossType(Enum):
    """Stop-loss calculation methods"""
    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"
    MOVING_AVERAGE = "moving_average"
    SUPPORT_RESISTANCE = "support_resistance"


class StopLossManager:
    """
    Advanced stop-loss management system
    
    Features:
    - Multiple stop-loss calculation methods
    - Dynamic stop-loss adjustments (trailing)
    - ATR-based adaptive stops
    - Moving average stops
    - Support/resistance level stops
    - Volatility-adjusted stops
    - Risk-reward ratio calculations
    
    Attributes:
        default_type: Default stop-loss type to use
        atr_period: Period for ATR calculation
        atr_multiplier: ATR multiplier for stop distance
        fixed_stop_percentage: Fixed stop percentage
        trailing_activation_pct: Profit % before trailing activates
        trailing_percentage: Trailing stop distance %
    """
    
    def __init__(self, config: dict):
        """
        Initialize stop-loss manager
        
        Args:
            config: Configuration dictionary with stop-loss parameters
        """
        self.config = config
        self.logger = get_logger()
        
        # Stop-loss parameters
        self.stop_loss_config = config.get('stop_loss', {})
        default_type_str = self.stop_loss_config.get('default_type', 'fixed_percentage')
        self.default_type = StopLossType(default_type_str) if isinstance(default_type_str, str) else default_type_str
        self.atr_period = self.stop_loss_config.get('atr_period', 14)
        self.atr_multiplier = self.stop_loss_config.get('atr_multiplier', 2.0)
        self.fixed_stop_percentage = self.stop_loss_config.get('fixed_stop_percentage', 2.0)
        self.trailing_activation_pct = self.stop_loss_config.get('trailing_activation_pct', 1.0)
        self.trailing_percentage = self.stop_loss_config.get('trailing_percentage', 1.0)
        
        # Active stop-loss tracking
        self.active_stops = {}  # symbol -> stop_loss_data
        
        self.logger.info(f"Stop-Loss Manager initialized with default type: {self.default_type.value}")
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, position_type: str,
                          data: pd.DataFrame, stop_type: Optional[StopLossType] = None) -> float:
        """
        Calculate stop-loss price based on selected method
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price of position
            position_type: 'long' or 'short'
            data: Price data DataFrame with OHLCV
            stop_type: Stop-loss type (optional, uses default if None)
            
        Returns:
            Stop-loss price
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
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            position_type: 'long' or 'short'
            data: Price data DataFrame
            
        Returns:
            New stop price if updated, None otherwise
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
        """
        Set active stop-loss for a position
        
        Args:
            symbol: Trading symbol
            stop_price: Initial stop price
            stop_type: Type of stop-loss
            entry_price: Entry price
            position_type: 'long' or 'short'
        """
        self.active_stops[symbol] = {
            'stop_price': stop_price,
            'type': stop_type,
            'entry_price': entry_price,
            'position_type': position_type,
            'highest_price': entry_price if position_type == 'long' else float('inf'),
            'lowest_price': entry_price if position_type == 'short' else 0.0
        }
        
        self.logger.info(f"Set {stop_type.value} stop for {symbol} at {stop_price:.4f}")
    
    def check_stop_loss(self, symbol: str, current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Check if stop-loss is triggered
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Tuple of (triggered: bool, reason: str or None)
        """
        if symbol not in self.active_stops:
            return False, None
        
        stop_data = self.active_stops[symbol]
        stop_price = stop_data['stop_price']
        position_type = stop_data['position_type']
        
        if position_type == 'long' and current_price <= stop_price:
            reason = f"Long stop-loss triggered at {stop_price:.4f}, current price {current_price:.4f}"
            self._remove_active_stop(symbol)
            return True, reason
        
        elif position_type == 'short' and current_price >= stop_price:
            reason = f"Short stop-loss triggered at {stop_price:.4f}, current price {current_price:.4f}"
            self._remove_active_stop(symbol)
            return True, reason
        
        return False, None
    
    def _fixed_percentage_stop(self, entry_price: float, position_type: str) -> float:
        """
        Fixed percentage stop-loss
        
        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            
        Returns:
            Stop price
        """
        if position_type == 'long':
            return entry_price * (1 - self.fixed_stop_percentage / 100)
        else:  # short
            return entry_price * (1 + self.fixed_stop_percentage / 100)
    
    def _atr_based_stop(self, symbol: str, entry_price: float,
                       position_type: str, data: pd.DataFrame) -> float:
        """
        ATR-based stop-loss (adapts to volatility)
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            data: Price data DataFrame
            
        Returns:
            Stop price
        """
        atr = self._calculate_atr(data)
        
        if atr is None or atr == 0:
            self.logger.warning(f"Could not calculate ATR for {symbol}, using fixed percentage")
            return self._fixed_percentage_stop(entry_price, position_type)
        
        atr_stop_distance = atr * self.atr_multiplier
        
        if position_type == 'long':
            return entry_price - atr_stop_distance
        else:  # short
            return entry_price + atr_stop_distance
    
    def _trailing_stop(self, entry_price: float, position_type: str,
                      current_price: float) -> float:
        """
        Trailing stop-loss (initial calculation)
        
        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            current_price: Current price
            
        Returns:
            Initial stop price
        """
        if position_type == 'long':
            stop_distance = entry_price * (self.trailing_percentage / 100)
            return entry_price - stop_distance
        else:  # short
            stop_distance = entry_price * (self.trailing_percentage / 100)
            return entry_price + stop_distance
    
    def _update_trailing_stop(self, symbol: str, current_price: float,
                            position_type: str, stop_data: dict) -> Optional[float]:
        """
        Update trailing stop-loss as price moves favorably
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            position_type: 'long' or 'short'
            stop_data: Stop-loss data dict
            
        Returns:
            New stop price if updated, None otherwise
        """
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
                        self.logger.debug(f"Updated trailing stop for {symbol} to {new_stop:.4f}")
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
                        self.logger.debug(f"Updated trailing stop for {symbol} to {new_stop:.4f}")
                        return new_stop
        
        return None
    
    def _moving_average_stop(self, symbol: str, entry_price: float,
                           position_type: str, data: pd.DataFrame) -> float:
        """
        Moving average based stop-loss
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            data: Price data DataFrame
            
        Returns:
            Stop price
        """
        if len(data) < 20:
            self.logger.warning(f"Insufficient data for MA stop on {symbol}, using fixed percentage")
            return self._fixed_percentage_stop(entry_price, position_type)
        
        # Use 20-period EMA as dynamic stop
        ema_20 = data['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        
        if position_type == 'long':
            # Use the higher of EMA or fixed stop (more conservative)
            fixed_stop = self._fixed_percentage_stop(entry_price, position_type)
            return max(ema_20, fixed_stop)
        else:  # short
            # Use the lower of EMA or fixed stop (more conservative)
            fixed_stop = self._fixed_percentage_stop(entry_price, position_type)
            return min(ema_20, fixed_stop)
    
    def _update_moving_average_stop(self, symbol: str, current_price: float,
                                  position_type: str, data: pd.DataFrame,
                                  stop_data: dict) -> Optional[float]:
        """
        Update moving average stop-loss
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            position_type: 'long' or 'short'
            data: Price data DataFrame
            stop_data: Stop-loss data dict
            
        Returns:
            New stop price if updated, None otherwise
        """
        if len(data) < 20:
            return None
        
        ema_20 = data['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        
        if position_type == 'long':
            new_stop = ema_20
            # Only move stop up, not down
            if new_stop > stop_data['stop_price']:
                stop_data['stop_price'] = new_stop
                self.logger.debug(f"Updated MA stop for {symbol} to {new_stop:.4f}")
                return new_stop
        
        else:  # short
            new_stop = ema_20
            # Only move stop down, not up
            if new_stop < stop_data['stop_price']:
                stop_data['stop_price'] = new_stop
                self.logger.debug(f"Updated MA stop for {symbol} to {new_stop:.4f}")
                return new_stop
        
        return None
    
    def _support_resistance_stop(self, symbol: str, entry_price: float,
                               position_type: str, data: pd.DataFrame) -> float:
        """
        Support/resistance based stop-loss
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            data: Price data DataFrame
            
        Returns:
            Stop price
        """
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
        """
        Find recent support level
        
        Args:
            data: Price data DataFrame
            lookback: Number of periods to look back
            
        Returns:
            Support price level or None
        """
        if len(data) < lookback:
            return None
        
        recent_lows = data['low'].tail(lookback)
        return recent_lows.min()
    
    def _find_resistance_level(self, data: pd.DataFrame, lookback: int = 20) -> Optional[float]:
        """
        Find recent resistance level
        
        Args:
            data: Price data DataFrame
            lookback: Number of periods to look back
            
        Returns:
            Resistance price level or None
        """
        if len(data) < lookback:
            return None
        
        recent_highs = data['high'].tail(lookback)
        return recent_highs.max()
    
    def _calculate_atr(self, data: pd.DataFrame) -> Optional[float]:
        """
        Calculate Average True Range
        
        Args:
            data: Price data DataFrame with high, low, close
            
        Returns:
            ATR value or None
        """
        if len(data) < self.atr_period + 1:
            return None
        
        try:
            high = data['high']
            low = data['low']
            close_prev = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
            
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return None
    
    def _remove_active_stop(self, symbol: str):
        """
        Remove active stop-loss for a symbol
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.active_stops:
            del self.active_stops[symbol]
            self.logger.info(f"Removed active stop for {symbol}")
    
    def get_active_stops(self) -> dict:
        """
        Get all active stop-loss orders
        
        Returns:
            Dict of active stops
        """
        return self.active_stops.copy()
    
    def get_stop_loss_recommendation(self, symbol: str, entry_price: float,
                                   position_type: str, data: pd.DataFrame,
                                   volatility: float) -> dict:
        """
        Get stop-loss recommendation with multiple methods
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            data: Price data DataFrame
            volatility: Current volatility measure
            
        Returns:
            Dict with recommendations for each stop-loss method
        """
        recommendations = {}
        
        # Calculate stops for all methods
        for stop_type in StopLossType:
            try:
                stop_price = self.calculate_stop_loss(symbol, entry_price, position_type, data, stop_type)
                stop_distance_pct = abs(entry_price - stop_price) / entry_price * 100
                
                recommendations[stop_type.value] = {
                    'stop_price': stop_price,
                    'distance_pct': stop_distance_pct,
                    'risk_reward_ratio': self._calculate_risk_reward_ratio(entry_price, stop_price, position_type)
                }
            except Exception as e:
                self.logger.error(f"Error calculating {stop_type.value} stop: {e}")
                recommendations[stop_type.value] = {
                    'stop_price': None,
                    'distance_pct': None,
                    'risk_reward_ratio': None,
                    'error': str(e)
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
        """
        Volatility-adjusted stop-loss
        
        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            volatility: Volatility measure (0-1)
            
        Returns:
            Stop price
        """
        # Use volatility to adjust stop distance (higher volatility = wider stop)
        base_distance = self.fixed_stop_percentage / 100
        adjusted_distance = base_distance * (1 + volatility)
        
        if position_type == 'long':
            return entry_price * (1 - adjusted_distance)
        else:  # short
            return entry_price * (1 + adjusted_distance)
    
    def _calculate_risk_reward_ratio(self, entry_price: float, stop_price: float,
                                   position_type: str, target_price: float = None) -> float:
        """
        Calculate risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_price: Stop-loss price
            position_type: 'long' or 'short'
            target_price: Optional target price (assumes 2:1 if None)
            
        Returns:
            Risk-reward ratio
        """
        risk = abs(entry_price - stop_price)
        
        if target_price is None:
            # Assume 2:1 reward if no target specified
            if position_type == 'long':
                target_price = entry_price + (2 * risk)
            else:
                target_price = entry_price - (2 * risk)
        
        reward = abs(entry_price - target_price)
        
        return reward / risk if risk > 0 else 0.0
