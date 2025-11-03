"""
Core module for Bot Trading V2
Contains utilities, configuration, and base classes
"""

from .config import Config, get_config, reload_config
from .logger import setup_logger, get_logger
from .exceptions import (
    BotTradingException,
    ConfigurationError,
    ExchangeError,
    StrategyError,
    RiskManagementError,
)

__version__ = "2.0.0"
__all__ = [
    "Config",
    "get_config",
    "reload_config",
    "setup_logger",
    "get_logger",
    "BotTradingException",
    "ConfigurationError",
    "ExchangeError",
    "StrategyError",
    "RiskManagementError",
]
