"""
Discord Bot Module for Trading Bot Monitoring and Alerts

This module provides real-time notifications via Discord webhooks
with rich embeds for trading activities.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from core.logger import get_logger
from core.exceptions import BotTradingException

try:
    from discord_webhook import DiscordWebhook, DiscordEmbed
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False


class DiscordBot:
    """
    Discord webhook bot for trading notifications
    
    Features:
    - Rich embed messages
    - Trade notifications (entry/exit)
    - Error alerts
    - Daily performance summary
    - Color-coded messages
    - Custom formatting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Discord bot
        
        Args:
            config: Configuration dictionary with discord settings
        """
        self.logger = get_logger()
        self.config = config.get('monitoring', {}).get('discord', {})
        self.webhook_url = self.config.get('webhook_url', '')
        self.enabled = self.config.get('enabled', False)
        self.events = self.config.get('events', [])
        self.username = self.config.get('username', 'Trading Bot')
        self.avatar_url = self.config.get('avatar_url', '')
        
        # Stats
        self.stats = {
            'messages_sent': 0,
            'errors': 0
        }
        
        if not DISCORD_AVAILABLE:
            self.logger.warning("discord-webhook not installed. Discord features disabled.")
            self.enabled = False
        
        if self.enabled and not self.webhook_url:
            self.logger.error("Discord webhook URL not configured")
            self.enabled = False
    
    def send_embed(
        self,
        title: str,
        description: str = "",
        color: str = "03b2f8",
        fields: List[Dict[str, Any]] = None,
        footer: str = None,
        thumbnail: str = None
    ) -> bool:
        """
        Send an embed message to Discord
        
        Args:
            title: Embed title
            description: Embed description
            color: Hex color code (without #)
            fields: List of field dictionaries with 'name', 'value', 'inline'
            footer: Footer text
            thumbnail: Thumbnail URL
            
        Returns:
            True if message sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            webhook = DiscordWebhook(
                url=self.webhook_url,
                username=self.username,
                avatar_url=self.avatar_url
            )
            
            embed = DiscordEmbed(
                title=title,
                description=description,
                color=color
            )
            
            # Add fields
            if fields:
                for field in fields:
                    embed.add_embed_field(
                        name=field.get('name', ''),
                        value=field.get('value', ''),
                        inline=field.get('inline', True)
                    )
            
            # Add footer
            if footer:
                embed.set_footer(text=footer)
            else:
                embed.set_footer(text=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Add thumbnail
            if thumbnail:
                embed.set_thumbnail(url=thumbnail)
            
            # Set timestamp
            embed.set_timestamp()
            
            webhook.add_embed(embed)
            response = webhook.execute()
            
            self.stats['messages_sent'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Discord message: {e}")
            self.stats['errors'] += 1
            return False
    
    def notify_trade_opened(self, trade_data: Dict[str, Any]) -> bool:
        """
        Send notification when a trade is opened
        
        Args:
            trade_data: Trade information
            
        Returns:
            True if notification sent successfully
        """
        if not self.enabled or 'trade_opened' not in self.events:
            return False
        
        try:
            symbol = trade_data.get('symbol', 'N/A')
            side = trade_data.get('side', 'N/A')
            entry_price = trade_data.get('entry_price', 0)
            quantity = trade_data.get('quantity', 0)
            strategy = trade_data.get('strategy', 'N/A')
            stop_loss = trade_data.get('stop_loss', 'N/A')
            take_profit = trade_data.get('take_profit', 'N/A')
            
            # Color based on side
            color = "2ecc71" if side.upper() == "BUY" else "e74c3c"
            
            fields = [
                {
                    'name': '📊 Symbol',
                    'value': f'**{symbol}**',
                    'inline': True
                },
                {
                    'name': '📈 Side',
                    'value': f'**{side}**',
                    'inline': True
                },
                {
                    'name': '💰 Entry Price',
                    'value': f'${entry_price:,.2f}',
                    'inline': True
                },
                {
                    'name': '📦 Quantity',
                    'value': f'{quantity:.4f}',
                    'inline': True
                },
                {
                    'name': '💵 Total Value',
                    'value': f'${entry_price * quantity:,.2f}',
                    'inline': True
                },
                {
                    'name': '🎯 Strategy',
                    'value': strategy,
                    'inline': True
                },
                {
                    'name': '🛡️ Stop Loss',
                    'value': str(stop_loss),
                    'inline': True
                },
                {
                    'name': '🎯 Take Profit',
                    'value': str(take_profit),
                    'inline': True
                }
            ]
            
            return self.send_embed(
                title="🟢 Trade Opened" if side.upper() == "BUY" else "🔴 Trade Opened",
                description=f"New {side} position opened for {symbol}",
                color=color,
                fields=fields
            )
            
        except Exception as e:
            self.logger.error(f"Error sending trade opened notification: {e}")
            return False
    
    def notify_trade_closed(self, trade_data: Dict[str, Any]) -> bool:
        """
        Send notification when a trade is closed
        
        Args:
            trade_data: Trade information
            
        Returns:
            True if notification sent successfully
        """
        if not self.enabled or 'trade_closed' not in self.events:
            return False
        
        try:
            symbol = trade_data.get('symbol', 'N/A')
            side = trade_data.get('side', 'N/A')
            entry_price = trade_data.get('entry_price', 0)
            exit_price = trade_data.get('exit_price', 0)
            quantity = trade_data.get('quantity', 0)
            pnl = trade_data.get('pnl', 0)
            pnl_pct = trade_data.get('pnl_pct', 0)
            reason = trade_data.get('reason', 'Manual')
            duration = trade_data.get('duration', 'N/A')
            
            # Color based on profit/loss
            if pnl > 0:
                color = "2ecc71"  # Green
                emoji = "✅"
            elif pnl < 0:
                color = "e74c3c"  # Red
                emoji = "❌"
            else:
                color = "95a5a6"  # Gray
                emoji = "➖"
            
            fields = [
                {
                    'name': '📊 Symbol',
                    'value': f'**{symbol}**',
                    'inline': True
                },
                {
                    'name': '📈 Side',
                    'value': f'**{side}**',
                    'inline': True
                },
                {
                    'name': '💰 Entry Price',
                    'value': f'${entry_price:,.2f}',
                    'inline': True
                },
                {
                    'name': '💰 Exit Price',
                    'value': f'${exit_price:,.2f}',
                    'inline': True
                },
                {
                    'name': '📦 Quantity',
                    'value': f'{quantity:.4f}',
                    'inline': True
                },
                {
                    'name': f'{"💵" if pnl >= 0 else "💸"} P&L',
                    'value': f'**${pnl:,.2f}** ({pnl_pct:+.2f}%)',
                    'inline': True
                },
                {
                    'name': '📝 Reason',
                    'value': reason,
                    'inline': True
                },
                {
                    'name': '⏱️ Duration',
                    'value': str(duration),
                    'inline': True
                }
            ]
            
            return self.send_embed(
                title=f"{emoji} Trade Closed",
                description=f"{side} position closed for {symbol}",
                color=color,
                fields=fields
            )
            
        except Exception as e:
            self.logger.error(f"Error sending trade closed notification: {e}")
            return False
    
    def notify_error(self, error_data: Dict[str, Any]) -> bool:
        """
        Send error notification
        
        Args:
            error_data: Error information
            
        Returns:
            True if notification sent successfully
        """
        if not self.enabled or 'error' not in self.events:
            return False
        
        try:
            error_type = error_data.get('type', 'Unknown')
            error_msg = error_data.get('message', 'No details')
            module = error_data.get('module', 'Unknown')
            severity = error_data.get('severity', 'ERROR')
            traceback = error_data.get('traceback', '')
            
            # Color based on severity
            if severity == 'CRITICAL':
                color = "992d22"  # Dark red
                emoji = "🚨"
            elif severity == 'ERROR':
                color = "e74c3c"  # Red
                emoji = "⚠️"
            else:
                color = "f39c12"  # Orange
                emoji = "ℹ️"
            
            fields = [
                {
                    'name': '🔧 Module',
                    'value': f'`{module}`',
                    'inline': True
                },
                {
                    'name': '🏷️ Type',
                    'value': f'`{error_type}`',
                    'inline': True
                },
                {
                    'name': '📝 Message',
                    'value': f'```{error_msg[:1000]}```',
                    'inline': False
                }
            ]
            
            # Add traceback if available (truncated)
            if traceback:
                fields.append({
                    'name': '🔍 Traceback',
                    'value': f'```{traceback[:500]}...```',
                    'inline': False
                })
            
            return self.send_embed(
                title=f"{emoji} {severity}",
                description=f"An error occurred in {module}",
                color=color,
                fields=fields
            )
            
        except Exception as e:
            self.logger.error(f"Error sending error notification: {e}")
            return False
    
    def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Send daily performance summary
        
        Args:
            summary_data: Summary statistics
            
        Returns:
            True if notification sent successfully
        """
        if not self.enabled or 'daily_summary' not in self.events:
            return False
        
        try:
            date = summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            total_trades = summary_data.get('total_trades', 0)
            winning_trades = summary_data.get('winning_trades', 0)
            losing_trades = summary_data.get('losing_trades', 0)
            win_rate = summary_data.get('win_rate', 0)
            total_pnl = summary_data.get('total_pnl', 0)
            total_pnl_pct = summary_data.get('total_pnl_pct', 0)
            best_trade = summary_data.get('best_trade', 0)
            worst_trade = summary_data.get('worst_trade', 0)
            sharpe_ratio = summary_data.get('sharpe_ratio', 0)
            
            # Color based on overall performance
            if total_pnl > 0:
                color = "2ecc71"  # Green
                emoji = "📈"
            elif total_pnl < 0:
                color = "e74c3c"  # Red
                emoji = "📉"
            else:
                color = "95a5a6"  # Gray
                emoji = "➖"
            
            fields = [
                {
                    'name': '📊 Total Trades',
                    'value': f'**{total_trades}**',
                    'inline': True
                },
                {
                    'name': '✅ Winning',
                    'value': f'**{winning_trades}**',
                    'inline': True
                },
                {
                    'name': '❌ Losing',
                    'value': f'**{losing_trades}**',
                    'inline': True
                },
                {
                    'name': '🎯 Win Rate',
                    'value': f'**{win_rate:.1f}%**',
                    'inline': True
                },
                {
                    'name': f'{"💵" if total_pnl >= 0 else "💸"} Total P&L',
                    'value': f'**${total_pnl:,.2f}** ({total_pnl_pct:+.2f}%)',
                    'inline': True
                },
                {
                    'name': '📊 Sharpe Ratio',
                    'value': f'{sharpe_ratio:.2f}',
                    'inline': True
                },
                {
                    'name': '🏆 Best Trade',
                    'value': f'${best_trade:,.2f}',
                    'inline': True
                },
                {
                    'name': '📉 Worst Trade',
                    'value': f'${worst_trade:,.2f}',
                    'inline': True
                }
            ]
            
            return self.send_embed(
                title=f"{emoji} Daily Summary - {date}",
                description="Trading performance for today",
                color=color,
                fields=fields
            )
            
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {e}")
            return False
    
    def send_portfolio_update(self, portfolio_data: Dict[str, Any]) -> bool:
        """
        Send portfolio update
        
        Args:
            portfolio_data: Portfolio information
            
        Returns:
            True if notification sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            total_value = portfolio_data.get('total_value', 0)
            cash = portfolio_data.get('cash', 0)
            positions_value = portfolio_data.get('positions_value', 0)
            total_pnl = portfolio_data.get('total_pnl', 0)
            total_pnl_pct = portfolio_data.get('total_pnl_pct', 0)
            active_positions = portfolio_data.get('active_positions', 0)
            
            # Color based on P&L
            color = "2ecc71" if total_pnl >= 0 else "e74c3c"
            
            fields = [
                {
                    'name': '💰 Total Value',
                    'value': f'**${total_value:,.2f}**',
                    'inline': True
                },
                {
                    'name': '💵 Cash',
                    'value': f'${cash:,.2f}',
                    'inline': True
                },
                {
                    'name': '📦 Positions Value',
                    'value': f'${positions_value:,.2f}',
                    'inline': True
                },
                {
                    'name': f'{"💵" if total_pnl >= 0 else "💸"} Total P&L',
                    'value': f'**${total_pnl:,.2f}** ({total_pnl_pct:+.2f}%)',
                    'inline': True
                },
                {
                    'name': '📊 Active Positions',
                    'value': f'{active_positions}',
                    'inline': True
                }
            ]
            
            return self.send_embed(
                title="💼 Portfolio Update",
                description="Current portfolio status",
                color=color,
                fields=fields
            )
            
        except Exception as e:
            self.logger.error(f"Error sending portfolio update: {e}")
            return False
    
    def send_performance_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """
        Send performance metrics
        
        Args:
            metrics_data: Performance metrics
            
        Returns:
            True if notification sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            total_return = metrics_data.get('total_return', 0)
            sharpe_ratio = metrics_data.get('sharpe_ratio', 0)
            sortino_ratio = metrics_data.get('sortino_ratio', 0)
            max_drawdown = metrics_data.get('max_drawdown', 0)
            win_rate = metrics_data.get('win_rate', 0)
            profit_factor = metrics_data.get('profit_factor', 0)
            total_trades = metrics_data.get('total_trades', 0)
            
            # Color based on overall performance
            color = "2ecc71" if total_return > 0 else "e74c3c"
            
            fields = [
                {
                    'name': '📈 Total Return',
                    'value': f'**{total_return:+.2f}%**',
                    'inline': True
                },
                {
                    'name': '📊 Sharpe Ratio',
                    'value': f'{sharpe_ratio:.2f}',
                    'inline': True
                },
                {
                    'name': '📉 Sortino Ratio',
                    'value': f'{sortino_ratio:.2f}',
                    'inline': True
                },
                {
                    'name': '⚠️ Max Drawdown',
                    'value': f'{max_drawdown:.2f}%',
                    'inline': True
                },
                {
                    'name': '🎯 Win Rate',
                    'value': f'{win_rate:.1f}%',
                    'inline': True
                },
                {
                    'name': '💰 Profit Factor',
                    'value': f'{profit_factor:.2f}',
                    'inline': True
                },
                {
                    'name': '📊 Total Trades',
                    'value': f'{total_trades}',
                    'inline': True
                }
            ]
            
            return self.send_embed(
                title="📈 Performance Metrics",
                description="Comprehensive performance analysis",
                color=color,
                fields=fields
            )
            
        except Exception as e:
            self.logger.error(f"Error sending performance metrics: {e}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get bot statistics
        
        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()
