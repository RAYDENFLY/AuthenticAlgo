"""
Trading Strategies Module
Modular and configurable trading strategies
"""

from strategies.base_strategy import BaseStrategy
from strategies.rsi_macd import RSIMACDStrategy
from strategies.bollinger import BollingerBandsStrategy
from strategies.ml_strategy import MLStrategy

STRATEGY_REGISTRY = {
    'RSI_MACD_Strategy': RSIMACDStrategy,
    'BollingerBands_Strategy': BollingerBandsStrategy,
    'ML_Strategy': MLStrategy
}

def create_strategy(strategy_name: str, config: dict) -> BaseStrategy:
    """Factory function to create strategy instances"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy {strategy_name} not found. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    return STRATEGY_REGISTRY[strategy_name](config)

__all__ = [
    'BaseStrategy',
    'RSIMACDStrategy',
    'BollingerBandsStrategy',
    'MLStrategy',
    'create_strategy',
    'STRATEGY_REGISTRY'
]
