"""
Risk Management Module
Comprehensive risk management for trading operations
"""

from .risk_manager import RiskManager, RiskLevel, RiskMetrics
from .stop_loss import StopLossManager, StopLossType
from .portfolio import PortfolioManager, Position, PortfolioMetrics


class RiskManagement:
    """
    Main interface for all risk management components
    
    Integrates:
    - RiskManager: Position sizing, limits, circuit breakers
    - StopLossManager: Multiple stop-loss strategies
    - PortfolioManager: Portfolio tracking and metrics
    
    Example:
        config = {
            'initial_capital': 10000,
            'risk_management': {...},
            'stop_loss': {...},
            'portfolio': {...}
        }
        
        risk_mgmt = RiskManagement(config)
        
        # Validate trade
        validation = risk_mgmt.validate_trade('BTC/USDT', 1.0, 45000, 'BUY', {})
        
        # Calculate position size
        size = risk_mgmt.calculate_position_size('BTC/USDT', 45000, 44000, 10000)
        
        # Update portfolio
        risk_mgmt.update_portfolio('BTC/USDT', 1.0, 45000, 'BUY', datetime.now())
    """
    
    def __init__(self, config: dict):
        """
        Initialize all risk management components
        
        Args:
            config: Configuration dictionary with all risk parameters
        """
        self.config = config
        self.risk_manager = RiskManager(config)
        self.stop_loss_manager = StopLossManager(config)
        self.portfolio_manager = PortfolioManager(config)
    
    def validate_trade(self, symbol: str, quantity: float, price: float,
                      order_type: str, current_positions: dict) -> dict:
        """
        Comprehensive trade validation
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Current price
            order_type: Order type (BUY/SELL)
            current_positions: Dict of current positions
            
        Returns:
            Dict with validation result
        """
        return self.risk_manager.validate_trade(symbol, quantity, price, order_type, current_positions)
    
    def calculate_position_size(self, symbol: str, price: float,
                              stop_loss_price: float, account_balance: float) -> float:
        """
        Calculate optimal position size
        
        Args:
            symbol: Trading symbol
            price: Current price
            stop_loss_price: Stop-loss price
            account_balance: Current account balance
            
        Returns:
            Position quantity
        """
        return self.risk_manager.calculate_position_size(symbol, price, stop_loss_price, account_balance)
    
    def update_portfolio(self, symbol: str, quantity: float, price: float,
                        action: str, timestamp):
        """
        Update portfolio with new trade
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            action: Trade action ('BUY', 'SELL', 'SHORT', 'COVER')
            timestamp: Trade timestamp
        """
        return self.portfolio_manager.update_position(symbol, quantity, price, action, timestamp)
    
    def get_comprehensive_report(self) -> dict:
        """
        Get comprehensive risk and portfolio report
        
        Returns:
            Dict with all risk management information
        """
        risk_report = self.risk_manager.generate_risk_report()
        portfolio_report = self.portfolio_manager.generate_portfolio_report()
        
        return {
            'risk_management': risk_report,
            'portfolio': portfolio_report,
            'active_stops': self.stop_loss_manager.get_active_stops()
        }


__all__ = [
    'RiskManagement',
    'RiskManager',
    'StopLossManager',
    'PortfolioManager',
    'RiskLevel',
    'StopLossType',
    'RiskMetrics',
    'Position',
    'PortfolioMetrics'
]
