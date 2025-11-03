"""
Risk Manager Module
Comprehensive risk management with position sizing, drawdown limits, and circuit breakers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio analysis"""
    position_size: float
    risk_per_trade: float
    max_drawdown: float
    daily_loss: float
    correlation_risk: float
    overall_risk_score: float


class RiskManager:
    """
    Comprehensive risk management system
    
    Features:
    - Position size validation and adjustment
    - Daily loss limits with circuit breakers
    - Maximum drawdown monitoring
    - Portfolio exposure limits
    - Correlation risk analysis
    - Volatility-based risk adjustments
    - Consecutive loss tracking
    
    Attributes:
        max_position_size_pct: Maximum position size as % of account
        max_daily_loss_pct: Maximum daily loss as % of account
        max_drawdown_pct: Maximum portfolio drawdown allowed
        risk_per_trade_pct: Risk per trade as % of account
        max_portfolio_exposure_pct: Maximum total exposure
        correlation_threshold: Maximum correlation between positions
    """
    
    def __init__(self, config: dict):
        """
        Initialize risk manager
        
        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config
        self.logger = get_logger()
        
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
        
        self.logger.info("Risk Manager initialized with parameters: "
                        f"max_position={self.max_position_size_pct}%, "
                        f"max_daily_loss={self.max_daily_loss_pct}%, "
                        f"risk_per_trade={self.risk_per_trade_pct}%")
    
    def _reset_daily_limits(self):
        """Reset daily limits at the start of each day"""
        account_balance = self.config.get('initial_capital', 10000)
        self.max_daily_loss = account_balance * (self.max_daily_loss_pct / 100)
        self.daily_pnl = 0.0
        self.logger.info(f"Daily loss limit reset to: ${self.max_daily_loss:.2f}")
    
    def validate_trade(self, symbol: str, quantity: float, price: float,
                      order_type: str, current_positions: dict) -> Dict[str, any]:
        """
        Comprehensive trade validation with multiple risk checks
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Current price
            order_type: Order type (BUY/SELL)
            current_positions: Dict of current open positions
            
        Returns:
            Dict with 'approved' (bool), 'reason' (str), 'adjusted_quantity' (float)
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
            self.logger.warning(f"Position size reduced from {position_size_pct:.1f}% to {self.max_position_size_pct}%")
            return {
                'approved': True,
                'reason': f'Position size adjusted from {position_size_pct:.1f}% to {self.max_position_size_pct}%',
                'adjusted_quantity': adjusted_quantity
            }
        
        # 2. Daily loss limit check
        if self.daily_pnl <= -self.max_daily_loss:
            self.logger.error(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return {
                'approved': False,
                'reason': f'Daily loss limit reached: ${self.daily_pnl:.2f}',
                'adjusted_quantity': 0.0
            }
        
        # 3. Portfolio exposure check
        total_exposure = self._calculate_portfolio_exposure(current_positions, price, quantity)
        if total_exposure > self.max_portfolio_exposure_pct:
            self.logger.warning(f"Portfolio exposure limit exceeded: {total_exposure:.1f}%")
            return {
                'approved': False,
                'reason': f'Portfolio exposure limit exceeded: {total_exposure:.1f}%',
                'adjusted_quantity': 0.0
            }
        
        # 4. Correlation check (for multi-asset portfolios)
        if len(current_positions) > 0:
            correlation_risk = self._check_correlation_risk(symbol, current_positions)
            if correlation_risk > self.correlation_threshold:
                self.logger.warning(f"High correlation risk detected: {correlation_risk:.2f}")
                return {
                    'approved': False,
                    'reason': f'High correlation risk: {correlation_risk:.2f}',
                    'adjusted_quantity': 0.0
                }
        
        # 5. Volatility check
        volatility_risk = self._check_volatility_risk(symbol, price)
        if volatility_risk == RiskLevel.HIGH:
            adjusted_quantity = quantity * 0.5  # Reduce position in high volatility
            self.logger.warning(f"High volatility - position reduced by 50%")
            return {
                'approved': True,
                'reason': 'High volatility detected - position reduced by 50%',
                'adjusted_quantity': adjusted_quantity
            }
        elif volatility_risk == RiskLevel.CRITICAL:
            self.logger.error("Critical volatility - trading suspended")
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
        
        Args:
            symbol: Trading symbol
            price: Current price
            stop_loss_price: Stop-loss price
            account_balance: Current account balance
            
        Returns:
            Position quantity
        """
        # Method 1: Fixed fractional position sizing
        risk_amount = account_balance * (self.risk_per_trade_pct / 100)
        
        # Calculate stop distance
        if stop_loss_price and stop_loss_price > 0:
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
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss amount
            timestamp: Trade timestamp
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
        
        self.logger.info(f"Trade updated: {symbol} PnL: ${pnl:.2f}, Daily PnL: ${self.daily_pnl:.2f}, "
                        f"Consecutive losses: {self.consecutive_losses}")
    
    def _calculate_portfolio_exposure(self, current_positions: dict,
                                    new_price: float, new_quantity: float) -> float:
        """
        Calculate total portfolio exposure percentage
        
        Args:
            current_positions: Dict of current positions
            new_price: New position price
            new_quantity: New position quantity
            
        Returns:
            Total exposure as percentage
        """
        account_balance = self.config.get('initial_capital', 10000)
        
        # Calculate existing positions value
        existing_exposure = 0.0
        for symbol, position in current_positions.items():
            existing_exposure += position.get('quantity', 0) * position.get('current_price', 0)
        
        # Add new position
        new_exposure = new_quantity * new_price
        total_exposure = existing_exposure + new_exposure
        
        return (total_exposure / account_balance) * 100
    
    def _check_correlation_risk(self, new_symbol: str, current_positions: dict) -> float:
        """
        Check correlation risk between new symbol and existing positions
        
        Args:
            new_symbol: Symbol to check
            current_positions: Current open positions
            
        Returns:
            Correlation score (0-1)
        """
        # Group symbols by asset class for basic correlation estimation
        crypto_pairs = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'SOL', 'AVAX', 'MATIC']
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        
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
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            
        Returns:
            RiskLevel enum
        """
        # High volatility symbols (crypto and meme stocks)
        high_volatility_symbols = ['BTC', 'ETH', 'DOGE', 'SHIB', 'TSLA', 'GME', 'AMC']
        critical_volatility_events = []  # Could be fed from news or market data
        
        if symbol in critical_volatility_events:
            return RiskLevel.CRITICAL
        elif any(vol_symbol in symbol for vol_symbol in high_volatility_symbols):
            return RiskLevel.HIGH
        else:
            return RiskLevel.LOW
    
    def _get_volatility_factor(self, symbol: str) -> float:
        """
        Get volatility factor for position sizing (0.1 - 1.0)
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Volatility adjustment factor
        """
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
            self.logger.warning(f"⚠️ Circuit breaker activated: {self.circuit_breaker_reason}")
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = f"Daily loss limit exceeded: ${self.daily_pnl:.2f}"
            self.logger.warning(f"⚠️ Circuit breaker activated: {self.circuit_breaker_reason}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention required)"""
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""
        self.consecutive_losses = 0
        self.logger.info("✅ Circuit breaker reset - trading resumed")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Get current risk metrics
        
        Returns:
            RiskMetrics dataclass with current metrics
        """
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
            drawdown = (running_max - current_value) / running_max * 100 if running_max > 0 else 0
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
        """
        Calculate overall risk score (0-1, where 1 is highest risk)
        
        Args:
            max_drawdown: Maximum drawdown percentage
            
        Returns:
            Risk score (0-1)
        """
        drawdown_score = min(1.0, max_drawdown / self.max_drawdown_pct)
        daily_loss_score = min(1.0, abs(self.daily_pnl) / self.max_daily_loss) if self.max_daily_loss > 0 else 0
        consecutive_loss_score = min(1.0, self.consecutive_losses / self.max_consecutive_losses)
        
        return (drawdown_score + daily_loss_score + consecutive_loss_score) / 3
    
    def generate_risk_report(self) -> dict:
        """
        Generate comprehensive risk report
        
        Returns:
            Dict with comprehensive risk metrics and recommendations
        """
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
        """
        Generate risk management recommendations
        
        Args:
            metrics: Current risk metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if metrics.overall_risk_score > 0.8:
            recommendations.append("🚨 CRITICAL: Consider stopping trading temporarily")
        elif metrics.overall_risk_score > 0.6:
            recommendations.append("⚠️ HIGH: Reduce position sizes significantly")
        elif metrics.overall_risk_score > 0.4:
            recommendations.append("⚡ MEDIUM: Consider reducing position sizes")
        
        if self.consecutive_losses >= 2:
            recommendations.append(f"📉 Warning: {self.consecutive_losses} consecutive losses detected")
        
        if metrics.max_drawdown > self.max_drawdown_pct * 0.8:
            recommendations.append(f"📊 Warning: Drawdown approaching limit ({metrics.max_drawdown:.1f}%)")
        
        if abs(self.daily_pnl) > self.max_daily_loss * 0.7:
            recommendations.append(f"💰 Warning: Daily loss approaching limit (${abs(self.daily_pnl):.2f})")
        
        if not recommendations:
            recommendations.append("✅ All risk metrics within acceptable ranges")
        
        return recommendations
