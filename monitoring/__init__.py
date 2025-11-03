"""
Monitoring Module for Trading Bot

This module provides comprehensive monitoring capabilities including:
- Telegram bot for interactive notifications
- Discord bot for rich embed alerts
- Streamlit dashboard for real-time visualization

All monitoring services can be configured and managed through
the unified MonitoringModule interface.
"""

import asyncio
from typing import Dict, List, Optional, Any

from core.logger import get_logger
from core.exceptions import BotTradingException

# Import monitoring components
from monitoring.telegram_bot import TelegramBot
from monitoring.discord_bot import DiscordBot
from monitoring.dashboard import TradingDashboard


class MonitoringModule:
    """
    Unified interface for all monitoring services
    
    Features:
    - Telegram notifications
    - Discord alerts
    - Web dashboard
    - Unified notification API
    - Service health monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize monitoring module
        
        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger()
        self.config = config
        
        # Initialize components
        self.telegram = TelegramBot(config)
        self.discord = DiscordBot(config)
        self.dashboard = TradingDashboard(config)
        
        # Service status
        self.services_running = {
            'telegram': False,
            'discord': False,
            'dashboard': False
        }
        
        self.logger.info("MonitoringModule initialized")
    
    async def start(self):
        """Start all enabled monitoring services"""
        self.logger.info("Starting monitoring services...")
        
        # Start Telegram bot
        if self.telegram.enabled:
            try:
                await self.telegram.initialize()
                await self.telegram.start()
                self.services_running['telegram'] = True
                self.logger.info("Telegram bot started")
            except Exception as e:
                self.logger.error(f"Failed to start Telegram bot: {e}")
        
        # Discord is webhook-based, no startup needed
        if self.discord.enabled:
            self.services_running['discord'] = True
            self.logger.info("Discord bot ready")
        
        self.logger.info("Monitoring services started")
    
    async def stop(self):
        """Stop all monitoring services"""
        self.logger.info("Stopping monitoring services...")
        
        # Stop Telegram bot
        if self.telegram.enabled and self.services_running['telegram']:
            try:
                await self.telegram.stop()
                self.services_running['telegram'] = False
                self.logger.info("Telegram bot stopped")
            except Exception as e:
                self.logger.error(f"Error stopping Telegram bot: {e}")
        
        self.logger.info("Monitoring services stopped")
    
    # Notification methods
    
    async def notify_trade_opened(self, trade_data: Dict[str, Any]):
        """
        Notify all services about a trade opening
        
        Args:
            trade_data: Trade information
        """
        tasks = []
        
        if self.telegram.enabled and self.services_running['telegram']:
            tasks.append(self.telegram.notify_trade_opened(trade_data))
        
        if self.discord.enabled and self.services_running['discord']:
            tasks.append(
                asyncio.to_thread(self.discord.notify_trade_opened, trade_data)
            )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def notify_trade_closed(self, trade_data: Dict[str, Any]):
        """
        Notify all services about a trade closing
        
        Args:
            trade_data: Trade information
        """
        tasks = []
        
        if self.telegram.enabled and self.services_running['telegram']:
            tasks.append(self.telegram.notify_trade_closed(trade_data))
        
        if self.discord.enabled and self.services_running['discord']:
            tasks.append(
                asyncio.to_thread(self.discord.notify_trade_closed, trade_data)
            )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def notify_error(self, error_data: Dict[str, Any]):
        """
        Notify all services about an error
        
        Args:
            error_data: Error information
        """
        tasks = []
        
        if self.telegram.enabled and self.services_running['telegram']:
            tasks.append(self.telegram.notify_error(error_data))
        
        if self.discord.enabled and self.services_running['discord']:
            tasks.append(
                asyncio.to_thread(self.discord.notify_error, error_data)
            )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]):
        """
        Send daily summary to all services
        
        Args:
            summary_data: Summary statistics
        """
        tasks = []
        
        if self.telegram.enabled and self.services_running['telegram']:
            tasks.append(self.telegram.send_daily_summary(summary_data))
        
        if self.discord.enabled and self.services_running['discord']:
            tasks.append(
                asyncio.to_thread(self.discord.send_daily_summary, summary_data)
            )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """
        Send portfolio update to Discord
        
        Args:
            portfolio_data: Portfolio information
        """
        if self.discord.enabled and self.services_running['discord']:
            await asyncio.to_thread(
                self.discord.send_portfolio_update,
                portfolio_data
            )
    
    async def send_performance_metrics(self, metrics_data: Dict[str, Any]):
        """
        Send performance metrics to Discord
        
        Args:
            metrics_data: Performance metrics
        """
        if self.discord.enabled and self.services_running['discord']:
            await asyncio.to_thread(
                self.discord.send_performance_metrics,
                metrics_data
            )
    
    # Callback setters for dashboard
    
    def set_portfolio_callback(self, callback):
        """Set callback for portfolio data"""
        self.telegram.set_portfolio_callback(callback)
        self.dashboard.set_portfolio_callback(callback)
    
    def set_performance_callback(self, callback):
        """Set callback for performance data"""
        self.telegram.set_performance_callback(callback)
        self.dashboard.set_performance_callback(callback)
    
    def set_positions_callback(self, callback):
        """Set callback for positions data"""
        self.telegram.set_positions_callback(callback)
        self.dashboard.set_positions_callback(callback)
    
    def set_trades_callback(self, callback):
        """Set callback for trades data"""
        self.dashboard.set_trades_callback(callback)
    
    def set_equity_curve_callback(self, callback):
        """Set callback for equity curve data"""
        self.dashboard.set_equity_curve_callback(callback)
    
    # Service health
    
    def get_service_status(self) -> Dict[str, bool]:
        """
        Get status of all monitoring services
        
        Returns:
            Dictionary with service statuses
        """
        return {
            'telegram': self.services_running['telegram'],
            'discord': self.services_running['discord'],
            'dashboard': self.services_running['dashboard']
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all services
        
        Returns:
            Dictionary with service statistics
        """
        return {
            'telegram': self.telegram.stats if self.telegram.enabled else {},
            'discord': self.discord.get_stats() if self.discord.enabled else {}
        }


__all__ = [
    'MonitoringModule',
    'TelegramBot',
    'DiscordBot',
    'TradingDashboard'
]
