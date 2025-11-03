"""
Strategy 3: Hybrid (Technical Analysis + ML)
Combines TA signals with ML confidence
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import ta
from trader_technical import TechnicalAnalysisTrader
from trader_ml import MLTrader
from base_trader import BaseTrader


class HybridTrader(BaseTrader):
    """Hybrid strategy combining TA and ML"""
    
    def __init__(self, capital: float = 10.0):
        super().__init__("Hybrid_TA_ML", capital)
        
        # Initialize sub-strategies (without running them)
        self.ta_trader = TechnicalAnalysisTrader(capital)
        self.ml_trader = MLTrader(capital)
        
        # Weights
        self.ta_weight = 0.4
        self.ml_weight = 0.6
    
    def score_symbol(self, df: pd.DataFrame) -> float:
        """Score symbol using both TA and ML"""
        try:
            ta_score = self.ta_trader.score_symbol(df)
            ml_score = self.ml_trader.score_symbol(df)
            
            # Weighted combination
            combined_score = (ta_score * self.ta_weight + ml_score * self.ml_weight)
            
            return combined_score
            
        except Exception as e:
            return 0.0
    
    def screen_symbols(self, data_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Screen symbols using hybrid approach"""
        best_symbol = None
        best_score = 0.0
        
        for symbol, df in data_dict.items():
            score = self.score_symbol(df)
            if score > best_score:
                best_score = score
                best_symbol = symbol
        
        return best_symbol if best_score > 0.5 else None
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Generate hybrid signal"""
        try:
            # Get signals from both strategies
            ta_signal = self.ta_trader.generate_signal(df, symbol)
            ml_signal = self.ml_trader.generate_signal(df, symbol)
            
            # Combine signals
            ta_dir = ta_signal['direction']
            ml_dir = ml_signal['direction']
            
            ta_conf = ta_signal.get('confidence', 0.5)
            ml_conf = ml_signal.get('confidence', 0.5)
            
            # Decision logic
            if ta_dir == ml_dir and ta_dir != 'hold':
                # Both agree
                direction = ta_dir
                confidence = (ta_conf * self.ta_weight + ml_conf * self.ml_weight) * 1.2  # Boost for agreement
                confidence = min(confidence, 0.99)
            elif ta_dir == 'hold' or ml_dir == 'hold':
                # One says hold
                if ta_dir != 'hold':
                    direction = ta_dir
                    confidence = ta_conf * 0.7
                elif ml_dir != 'hold':
                    direction = ml_dir
                    confidence = ml_conf * 0.7
                else:
                    direction = 'hold'
                    confidence = 0.5
            else:
                # They disagree
                if ml_conf > ta_conf:
                    direction = ml_dir
                    confidence = ml_conf * 0.8
                else:
                    direction = ta_dir
                    confidence = ta_conf * 0.8
            
            latest_price = df['close'].iloc[-1] if not df.empty else 0
            
            return {
                'direction': direction,
                'confidence': confidence,
                'price': latest_price,
                'ta_signal': ta_dir,
                'ml_signal': ml_dir,
                'ta_confidence': ta_conf,
                'ml_confidence': ml_conf
            }
            
        except Exception as e:
            return {'direction': 'hold', 'confidence': 0.0, 'price': 0}
    
    def calculate_leverage(self, signal: Dict) -> int:
        """Dynamic leverage for hybrid"""
        confidence = signal.get('confidence', 0.5)
        ta_conf = signal.get('ta_confidence', 0.5)
        ml_conf = signal.get('ml_confidence', 0.5)
        
        # Moderate: 10x-60x
        # Base leverage from confidence
        base_leverage = int(10 + (confidence - 0.5) * 100)
        
        # Bonus if both signals agree
        ta_dir = signal.get('ta_signal', 'hold')
        ml_dir = signal.get('ml_signal', 'hold')
        
        if ta_dir == ml_dir and ta_dir != 'hold':
            # Agreement bonus
            base_leverage = int(base_leverage * 1.3)
        
        return max(10, min(60, base_leverage))
