"""
Position Sizing Module
Calculate optimal position sizes based on various risk management strategies
"""

from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass
import math

from core.logger import logger
from core.exceptions import ValidationError


class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_PERCENTAGE = "FIXED_PERCENTAGE"
    KELLY_CRITERION = "KELLY_CRITERION"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    RISK_BASED = "RISK_BASED"
    ATR_BASED = "ATR_BASED"


@dataclass
class PositionSize:
    """Position size calculation result"""
    quantity: float
    method: SizingMethod
    risk_amount: float
    risk_percent: float
    position_value: float
    leverage: int = 1
    notes: str = ""
    
    def __str__(self) -> str:
        return (
            f"PositionSize(qty={self.quantity:.4f}, "
            f"risk={self.risk_percent:.2f}%, "
            f"value=${self.position_value:.2f}, "
            f"method={self.method.value})"
        )


class PositionSizer:
    """
    Calculate position sizes using various risk management strategies
    """
    
    def __init__(
        self,
        account_balance: float,
        max_risk_percent: float = 2.0,
        max_position_percent: float = 10.0,
        leverage: int = 1
    ):
        """
        Initialize PositionSizer
        
        Args:
            account_balance: Total account balance
            max_risk_percent: Maximum risk per trade (%)
            max_position_percent: Maximum position size as % of balance
            leverage: Trading leverage (1 = no leverage)
        """
        if account_balance <= 0:
            raise ValidationError("Account balance must be positive")
        
        if not 0 < max_risk_percent <= 100:
            raise ValidationError("Max risk percent must be between 0 and 100")
        
        if not 0 < max_position_percent <= 100:
            raise ValidationError("Max position percent must be between 0 and 100")
        
        if leverage < 1:
            raise ValidationError("Leverage must be >= 1")
        
        self.account_balance = account_balance
        self.max_risk_percent = max_risk_percent
        self.max_position_percent = max_position_percent
        self.leverage = leverage
        
        logger.info(
            f"PositionSizer initialized: "
            f"Balance=${account_balance:.2f}, "
            f"MaxRisk={max_risk_percent}%, "
            f"MaxPosition={max_position_percent}%, "
            f"Leverage={leverage}x"
        )
    
    def fixed_percentage(
        self,
        current_price: float,
        position_percent: Optional[float] = None
    ) -> PositionSize:
        """
        Fixed percentage position sizing
        
        Args:
            current_price: Current asset price
            position_percent: Position size as % of balance (uses max if None)
            
        Returns:
            PositionSize object
        """
        if current_price <= 0:
            raise ValidationError("Price must be positive")
        
        # Use max position percent if not specified
        pos_pct = position_percent or self.max_position_percent
        pos_pct = min(pos_pct, self.max_position_percent)  # Cap at max
        
        # Calculate position value
        position_value = (self.account_balance * pos_pct / 100) * self.leverage
        
        # Calculate quantity
        quantity = position_value / current_price
        
        # Risk amount (assuming full loss)
        risk_amount = position_value / self.leverage  # Without leverage
        risk_percent = (risk_amount / self.account_balance) * 100
        
        logger.debug(
            f"Fixed %: {pos_pct}% of ${self.account_balance:.2f} "
            f"= {quantity:.4f} units @ ${current_price:.2f}"
        )
        
        return PositionSize(
            quantity=quantity,
            method=SizingMethod.FIXED_PERCENTAGE,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            position_value=position_value,
            leverage=self.leverage,
            notes=f"Fixed {pos_pct}% of balance"
        )
    
    def kelly_criterion(
        self,
        current_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> PositionSize:
        """
        Kelly Criterion position sizing
        Formula: f = (p * b - q) / b
        Where: f = fraction to bet, p = win probability, q = loss probability,
               b = ratio of avg win to avg loss
        
        Args:
            current_price: Current asset price
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)
            
        Returns:
            PositionSize object
        """
        if not 0 <= win_rate <= 1:
            raise ValidationError("Win rate must be between 0 and 1")
        
        if avg_win <= 0 or avg_loss <= 0:
            raise ValidationError("Average win and loss must be positive")
        
        # Kelly formula
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly_fraction = (p * b - q) / b
        
        # Apply Kelly only if positive (otherwise no edge)
        if kelly_fraction <= 0:
            logger.warning("No positive edge detected (Kelly ≤ 0)")
            kelly_fraction = 0.01  # Minimum 1%
        
        # Use half-Kelly for safety (common practice)
        kelly_fraction = kelly_fraction / 2
        
        # Cap at max position size
        kelly_fraction = min(kelly_fraction, self.max_position_percent / 100)
        
        # Calculate position value
        position_value = (self.account_balance * kelly_fraction) * self.leverage
        
        # Calculate quantity
        quantity = position_value / current_price
        
        # Risk calculation
        risk_amount = position_value / self.leverage
        risk_percent = (risk_amount / self.account_balance) * 100
        
        logger.debug(
            f"Kelly: WinRate={win_rate:.2%}, "
            f"W/L Ratio={b:.2f}, "
            f"Kelly%={kelly_fraction*100:.2f}%"
        )
        
        return PositionSize(
            quantity=quantity,
            method=SizingMethod.KELLY_CRITERION,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            position_value=position_value,
            leverage=self.leverage,
            notes=f"Half-Kelly {kelly_fraction*100:.2f}%"
        )
    
    def volatility_based(
        self,
        current_price: float,
        volatility: float,
        target_volatility: float = 0.02
    ) -> PositionSize:
        """
        Volatility-based position sizing (Vol Targeting)
        Adjust position size inversely to volatility
        
        Args:
            current_price: Current asset price
            volatility: Current volatility (e.g., 0.03 = 3%)
            target_volatility: Target portfolio volatility (default: 2%)
            
        Returns:
            PositionSize object
        """
        if volatility <= 0:
            raise ValidationError("Volatility must be positive")
        
        if target_volatility <= 0:
            raise ValidationError("Target volatility must be positive")
        
        # Calculate position size based on volatility ratio
        # Higher volatility = smaller position
        vol_ratio = target_volatility / volatility
        
        # Position size as percentage
        position_percent = self.max_position_percent * vol_ratio
        position_percent = min(position_percent, self.max_position_percent)
        
        # Calculate position value
        position_value = (self.account_balance * position_percent / 100) * self.leverage
        
        # Calculate quantity
        quantity = position_value / current_price
        
        # Risk calculation
        risk_amount = position_value / self.leverage
        risk_percent = (risk_amount / self.account_balance) * 100
        
        logger.debug(
            f"Vol-based: Current={volatility:.2%}, "
            f"Target={target_volatility:.2%}, "
            f"Size={position_percent:.2f}%"
        )
        
        return PositionSize(
            quantity=quantity,
            method=SizingMethod.VOLATILITY_BASED,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            position_value=position_value,
            leverage=self.leverage,
            notes=f"Vol-adjusted {position_percent:.2f}%"
        )
    
    def risk_based(
        self,
        current_price: float,
        stop_loss_price: float,
        risk_percent: Optional[float] = None
    ) -> PositionSize:
        """
        Risk-based position sizing
        Size position so that if stop-loss is hit, only risk X% of account
        
        Args:
            current_price: Current asset price
            stop_loss_price: Stop-loss price level
            risk_percent: Risk percentage (uses max if None)
            
        Returns:
            PositionSize object
        """
        if current_price <= 0 or stop_loss_price <= 0:
            raise ValidationError("Prices must be positive")
        
        # Calculate price risk per unit
        price_risk = abs(current_price - stop_loss_price)
        
        if price_risk == 0:
            raise ValidationError("Stop-loss price must differ from current price")
        
        # Use max risk if not specified
        risk_pct = risk_percent or self.max_risk_percent
        risk_pct = min(risk_pct, self.max_risk_percent)  # Cap at max
        
        # Calculate risk amount
        risk_amount = self.account_balance * risk_pct / 100
        
        # Calculate quantity based on risk
        # quantity * price_risk = risk_amount
        quantity = risk_amount / price_risk
        
        # Position value
        position_value = quantity * current_price
        
        # Check if position exceeds max position size
        max_position_value = self.account_balance * self.max_position_percent / 100
        if position_value > max_position_value:
            # Scale down to max position size
            quantity = max_position_value / current_price
            position_value = max_position_value
            # Recalculate actual risk
            actual_risk = quantity * price_risk
            risk_percent_actual = (actual_risk / self.account_balance) * 100
            
            logger.warning(
                f"Position size capped: Reduced from "
                f"{risk_pct}% risk to {risk_percent_actual:.2f}%"
            )
            risk_pct = risk_percent_actual
        
        logger.debug(
            f"Risk-based: Risk={risk_pct}%, "
            f"Stop=${stop_loss_price:.2f}, "
            f"PriceRisk=${price_risk:.2f}"
        )
        
        return PositionSize(
            quantity=quantity,
            method=SizingMethod.RISK_BASED,
            risk_amount=risk_amount,
            risk_percent=risk_pct,
            position_value=position_value,
            leverage=self.leverage,
            notes=f"Risk-based {risk_pct}% risk"
        )
    
    def atr_based(
        self,
        current_price: float,
        atr: float,
        atr_multiplier: float = 2.0
    ) -> PositionSize:
        """
        ATR-based position sizing
        Uses ATR as a proxy for volatility and risk
        
        Args:
            current_price: Current asset price
            atr: Average True Range value
            atr_multiplier: ATR multiplier for stop-loss (default: 2x)
            
        Returns:
            PositionSize object
        """
        if atr <= 0:
            raise ValidationError("ATR must be positive")
        
        # Calculate stop-loss distance using ATR
        stop_distance = atr * atr_multiplier
        
        # Use risk-based sizing with ATR stop
        stop_loss_price = current_price - stop_distance
        
        position_size = self.risk_based(
            current_price=current_price,
            stop_loss_price=stop_loss_price
        )
        
        # Update method and notes
        position_size.method = SizingMethod.ATR_BASED
        position_size.notes = f"ATR-based ({atr_multiplier}x ATR stop)"
        
        logger.debug(
            f"ATR-based: ATR=${atr:.2f}, "
            f"Stop Distance=${stop_distance:.2f}"
        )
        
        return position_size
    
    def calculate(
        self,
        method: SizingMethod,
        current_price: float,
        **kwargs
    ) -> PositionSize:
        """
        Calculate position size using specified method
        
        Args:
            method: Sizing method to use
            current_price: Current asset price
            **kwargs: Method-specific parameters
            
        Returns:
            PositionSize object
        """
        if method == SizingMethod.FIXED_PERCENTAGE:
            return self.fixed_percentage(
                current_price,
                kwargs.get('position_percent')
            )
        
        elif method == SizingMethod.KELLY_CRITERION:
            return self.kelly_criterion(
                current_price,
                kwargs['win_rate'],
                kwargs['avg_win'],
                kwargs['avg_loss']
            )
        
        elif method == SizingMethod.VOLATILITY_BASED:
            return self.volatility_based(
                current_price,
                kwargs['volatility'],
                kwargs.get('target_volatility', 0.02)
            )
        
        elif method == SizingMethod.RISK_BASED:
            return self.risk_based(
                current_price,
                kwargs['stop_loss_price'],
                kwargs.get('risk_percent')
            )
        
        elif method == SizingMethod.ATR_BASED:
            return self.atr_based(
                current_price,
                kwargs['atr'],
                kwargs.get('atr_multiplier', 2.0)
            )
        
        else:
            raise ValueError(f"Unknown sizing method: {method}")
    
    def validate_position_size(self, position_size: PositionSize) -> bool:
        """
        Validate if position size is within limits
        
        Args:
            position_size: PositionSize to validate
            
        Returns:
            True if valid
        """
        # Check risk limit
        if position_size.risk_percent > self.max_risk_percent:
            logger.warning(
                f"Risk too high: {position_size.risk_percent:.2f}% "
                f"> {self.max_risk_percent}%"
            )
            return False
        
        # Check position size limit
        position_percent = (position_size.position_value / self.leverage / self.account_balance) * 100
        if position_percent > self.max_position_percent:
            logger.warning(
                f"Position too large: {position_percent:.2f}% "
                f"> {self.max_position_percent}%"
            )
            return False
        
        # Check quantity is positive
        if position_size.quantity <= 0:
            logger.warning("Position quantity must be positive")
            return False
        
        return True
    
    def update_balance(self, new_balance: float):
        """
        Update account balance
        
        Args:
            new_balance: New account balance
        """
        if new_balance <= 0:
            raise ValidationError("Balance must be positive")
        
        logger.info(
            f"Balance updated: ${self.account_balance:.2f} → ${new_balance:.2f} "
            f"({((new_balance/self.account_balance - 1) * 100):+.2f}%)"
        )
        
        self.account_balance = new_balance
    
    def get_statistics(self) -> Dict:
        """
        Get position sizer statistics
        
        Returns:
            Dictionary with sizer settings
        """
        return {
            'account_balance': self.account_balance,
            'max_risk_percent': self.max_risk_percent,
            'max_position_percent': self.max_position_percent,
            'leverage': self.leverage,
            'max_risk_amount': self.account_balance * self.max_risk_percent / 100,
            'max_position_value': self.account_balance * self.max_position_percent / 100 * self.leverage
        }


if __name__ == "__main__":
    # Quick test
    sizer = PositionSizer(
        account_balance=10000,
        max_risk_percent=2.0,
        max_position_percent=10.0,
        leverage=3
    )
    
    print("\n=== Position Sizing Tests ===\n")
    
    # Test 1: Fixed percentage
    size1 = sizer.fixed_percentage(current_price=50000)
    print(f"1. Fixed %: {size1}\n")
    
    # Test 2: Kelly Criterion
    size2 = sizer.kelly_criterion(
        current_price=50000,
        win_rate=0.55,
        avg_win=500,
        avg_loss=300
    )
    print(f"2. Kelly: {size2}\n")
    
    # Test 3: Risk-based
    size3 = sizer.risk_based(
        current_price=50000,
        stop_loss_price=48000
    )
    print(f"3. Risk-based: {size3}\n")
    
    # Test 4: ATR-based
    size4 = sizer.atr_based(
        current_price=50000,
        atr=800
    )
    print(f"4. ATR-based: {size4}\n")
    
    # Statistics
    print(f"Statistics: {sizer.get_statistics()}")
