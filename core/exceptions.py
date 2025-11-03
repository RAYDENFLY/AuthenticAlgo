"""
Custom exceptions for Bot Trading V2
"""


class BotTradingException(Exception):
    """Base exception for all bot trading errors"""
    pass


class ConfigurationError(BotTradingException):
    """Raised when there's a configuration error"""
    pass


class ExchangeError(BotTradingException):
    """Raised when there's an exchange-related error"""
    pass


class StrategyError(BotTradingException):
    """Raised when there's a strategy-related error"""
    pass


class RiskManagementError(BotTradingException):
    """Raised when risk management rules are violated"""
    pass


class DataError(BotTradingException):
    """Raised when there's a data-related error"""
    pass


class OrderError(BotTradingException):
    """Raised when there's an order-related error"""
    pass


class InsufficientBalanceError(OrderError):
    """Raised when there's insufficient balance"""
    pass


class InvalidOrderError(OrderError):
    """Raised when order parameters are invalid"""
    pass


class ValidationError(BotTradingException):
    """Raised when validation fails"""
    pass
