"""
Enhanced Trading Bot with TP/SL Strategy
Implements multiple take-profit levels and stop-loss
Based on ATR and support/resistance levels
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import ta
from datetime import datetime


class TPSLStrategy:
    """Take Profit & Stop Loss Strategy Manager"""
    
    def __init__(self):
        self.atr_multiplier_sl = 1.5   # Stop loss at 1.5x ATR
        self.atr_multiplier_tp1 = 2.0  # TP1 at 2x ATR
        self.atr_multiplier_tp2 = 3.5  # TP2 at 3.5x ATR
        self.atr_multiplier_tp3 = 5.0  # TP3 at 5x ATR
        
        # TP allocation (what % of position to close at each TP)
        self.tp1_close_pct = 0.33  # Close 33% at TP1
        self.tp2_close_pct = 0.33  # Close 33% at TP2
        self.tp3_close_pct = 0.34  # Close 34% at TP3 (remaining)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for volatility-based TP/SL"""
        atr = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=period
        )
        return float(atr.iloc[-1])
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate recent support and resistance levels"""
        # Support: recent lows
        support = df['low'].rolling(20).min().iloc[-1]
        
        # Resistance: recent highs
        resistance = df['high'].rolling(20).max().iloc[-1]
        
        return float(support), float(resistance)
    
    def calculate_tp_sl_levels(
        self, 
        entry_price: float, 
        direction: str,
        df: pd.DataFrame,
        confidence: float
    ) -> Dict:
        """
        Calculate TP/SL levels based on ATR and confidence
        
        Returns dict with TP1, TP2, TP3, SL levels
        """
        atr = self.calculate_atr(df)
        support, resistance = self.calculate_support_resistance(df)
        current_price = float(df['close'].iloc[-1])
        
        # Adjust multipliers based on confidence
        # Higher confidence = wider TPs (more profit potential)
        confidence_boost = 1 + (confidence - 0.5)  # 0.5-1.5x boost
        
        if direction == 'long':
            # Stop Loss: below entry
            sl_distance = atr * self.atr_multiplier_sl
            sl = max(entry_price - sl_distance, support * 0.99)  # Don't go below support
            
            # Take Profits: above entry
            tp1 = entry_price + (atr * self.atr_multiplier_tp1 * confidence_boost)
            tp2 = entry_price + (atr * self.atr_multiplier_tp2 * confidence_boost)
            tp3 = entry_price + (atr * self.atr_multiplier_tp3 * confidence_boost)
            
            # Don't set TP3 beyond resistance (unrealistic)
            if tp3 > resistance * 1.1:
                tp3 = resistance * 1.05
            
        else:  # short
            # Stop Loss: above entry
            sl_distance = atr * self.atr_multiplier_sl
            sl = min(entry_price + sl_distance, resistance * 1.01)  # Don't go above resistance
            
            # Take Profits: below entry
            tp1 = entry_price - (atr * self.atr_multiplier_tp1 * confidence_boost)
            tp2 = entry_price - (atr * self.atr_multiplier_tp2 * confidence_boost)
            tp3 = entry_price - (atr * self.atr_multiplier_tp3 * confidence_boost)
            
            # Don't set TP3 beyond support (unrealistic)
            if tp3 < support * 0.9:
                tp3 = support * 0.95
        
        # Calculate risk:reward ratio
        risk = abs(entry_price - sl)
        reward = abs(entry_price - tp3)
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'entry': entry_price,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'atr': atr,
            'support': support,
            'resistance': resistance,
            'risk_reward': risk_reward,
            'direction': direction
        }
    
    def check_tp_sl_hit(
        self, 
        current_price: float, 
        levels: Dict,
        position_remaining: float = 1.0
    ) -> Dict:
        """
        Check if current price hit any TP or SL
        
        Returns:
            hit_type: 'tp1', 'tp2', 'tp3', 'sl', or None
            close_pct: % of position to close
            reason: explanation
        """
        direction = levels['direction']
        
        if direction == 'long':
            # Check SL hit
            if current_price <= levels['sl']:
                return {
                    'hit_type': 'sl',
                    'close_pct': position_remaining,
                    'reason': f"Stop Loss hit at ${current_price:.4f}",
                    'pnl_pct': (current_price - levels['entry']) / levels['entry']
                }
            
            # Check TP3 hit (highest target)
            if current_price >= levels['tp3'] and position_remaining > 0:
                return {
                    'hit_type': 'tp3',
                    'close_pct': position_remaining,
                    'reason': f"🎯 TP3 HIT! ${current_price:.4f}",
                    'pnl_pct': (current_price - levels['entry']) / levels['entry']
                }
            
            # Check TP2 hit
            if current_price >= levels['tp2'] and position_remaining > 0.34:
                return {
                    'hit_type': 'tp2',
                    'close_pct': self.tp2_close_pct * position_remaining,
                    'reason': f"🎯 TP2 HIT! ${current_price:.4f}",
                    'pnl_pct': (current_price - levels['entry']) / levels['entry']
                }
            
            # Check TP1 hit
            if current_price >= levels['tp1'] and position_remaining > 0.67:
                return {
                    'hit_type': 'tp1',
                    'close_pct': self.tp1_close_pct,
                    'reason': f"🎯 TP1 HIT! ${current_price:.4f}",
                    'pnl_pct': (current_price - levels['entry']) / levels['entry']
                }
        
        else:  # short
            # Check SL hit
            if current_price >= levels['sl']:
                return {
                    'hit_type': 'sl',
                    'close_pct': position_remaining,
                    'reason': f"Stop Loss hit at ${current_price:.4f}",
                    'pnl_pct': (levels['entry'] - current_price) / levels['entry']
                }
            
            # Check TP3 hit
            if current_price <= levels['tp3'] and position_remaining > 0:
                return {
                    'hit_type': 'tp3',
                    'close_pct': position_remaining,
                    'reason': f"🎯 TP3 HIT! ${current_price:.4f}",
                    'pnl_pct': (levels['entry'] - current_price) / levels['entry']
                }
            
            # Check TP2 hit
            if current_price <= levels['tp2'] and position_remaining > 0.34:
                return {
                    'hit_type': 'tp2',
                    'close_pct': self.tp2_close_pct * position_remaining,
                    'reason': f"🎯 TP2 HIT! ${current_price:.4f}",
                    'pnl_pct': (levels['entry'] - current_price) / levels['entry']
                }
            
            # Check TP1 hit
            if current_price <= levels['tp1'] and position_remaining > 0.67:
                return {
                    'hit_type': 'tp1',
                    'close_pct': self.tp1_close_pct,
                    'reason': f"🎯 TP1 HIT! ${current_price:.4f}",
                    'pnl_pct': (levels['entry'] - current_price) / levels['entry']
                }
        
        return None
    
    def format_tp_sl_message(self, symbol: str, levels: Dict, score: int = 80) -> str:
        """
        Format TP/SL message like your bot's output
        
        Example:
        VIRTUAL/USDT - STRONG BUY
        Score: 84/100 | ML: BULLISH (50%) | Quality: HIGH ✨
        💰 Entry Price: $1.4470
        📊 Action: STRONG BUY
        🎯 TP1: $1.5300 | TP2: $1.6200 | TP3: $1.7000
        🛡️ SL: $1.3600
        """
        direction_label = "STRONG BUY" if levels['direction'] == 'long' else "STRONG SELL"
        ml_label = "BULLISH" if levels['direction'] == 'long' else "BEARISH"
        
        # Determine quality based on risk:reward
        if levels['risk_reward'] >= 3.0:
            quality = "HIGH ✨"
        elif levels['risk_reward'] >= 2.0:
            quality = "GOOD ✅"
        else:
            quality = "MEDIUM ⚠️"
        
        msg = f"""
{symbol} - {direction_label}
Score: {score}/100 | ML: {ml_label} | Quality: {quality}

💰 Entry Price
${levels['entry']:.4f}

📊 Action
{direction_label}

🎯 Take Profit & Stop Loss
🎯 TP1: ${levels['tp1']:.4f}
🎯 TP2: ${levels['tp2']:.4f}
🎯 TP3: ${levels['tp3']:.4f}
🛡️ SL: ${levels['sl']:.4f}

📊 Risk Management
• Risk:Reward = 1:{levels['risk_reward']:.2f}
• ATR = ${levels['atr']:.4f}
• Support = ${levels['support']:.4f}
• Resistance = ${levels['resistance']:.4f}
"""
        return msg
    
    def calculate_optimal_leverage(self, levels: Dict, max_risk_pct: float = 0.02) -> int:
        """
        Calculate optimal leverage based on SL distance
        
        Args:
            levels: TP/SL levels dict
            max_risk_pct: Maximum % of capital to risk (default 2%)
        
        Returns:
            Recommended leverage (5-125x)
        """
        # Distance from entry to SL (as %)
        sl_distance_pct = abs(levels['entry'] - levels['sl']) / levels['entry']
        
        # Leverage needed to risk only max_risk_pct
        # If SL is 1% away, need 2x leverage to risk 2% capital
        optimal_leverage = max_risk_pct / sl_distance_pct
        
        # Clamp to reasonable range
        leverage = int(max(5, min(125, optimal_leverage)))
        
        return leverage
