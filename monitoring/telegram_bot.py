"""
Telegram Bot Module for Trading Bot Monitoring and Alerts

This module provides real-time notifications and interactive commands
for monitoring trading activities via Telegram.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from core.logger import get_logger
from core.exceptions import BotTradingException

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        CallbackQueryHandler,
        ContextTypes,
        MessageHandler,
        filters
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramBot:
    """
    Telegram bot for trading notifications and interactive monitoring
    
    Features:
    - Trade notifications (entry/exit)
    - Error alerts
    - Daily performance summary
    - Interactive commands (/status, /portfolio, /help)
    - Real-time updates
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Telegram bot
        
        Args:
            config: Configuration dictionary with telegram settings
        """
        self.logger = get_logger()
        self.config = config.get('monitoring', {}).get('telegram', {})
        self.bot_token = self.config.get('bot_token', '')
        self.chat_id = self.config.get('chat_id', '')
        self.enabled = self.config.get('enabled', False)
        self.events = self.config.get('events', [])
        
        # State tracking
        self.application: Optional[Application] = None
        self.is_running = False
        self.stats = {
            'messages_sent': 0,
            'commands_received': 0,
            'errors': 0
        }
        
        # External callbacks for data
        self.get_portfolio_callback = None
        self.get_performance_callback = None
        self.get_positions_callback = None
        
        if not TELEGRAM_AVAILABLE:
            self.logger.warning("python-telegram-bot not installed. Telegram features disabled.")
            self.enabled = False
        
        if self.enabled and not self.bot_token:
            self.logger.error("Telegram bot token not configured")
            self.enabled = False
    
    async def initialize(self):
        """Initialize the Telegram bot application"""
        if not self.enabled:
            self.logger.info("Telegram bot disabled")
            return
        
        try:
            # Create application
            self.application = Application.builder().token(self.bot_token).build()
            
            # Register command handlers
            self.application.add_handler(CommandHandler("start", self._cmd_start))
            self.application.add_handler(CommandHandler("help", self._cmd_help))
            self.application.add_handler(CommandHandler("status", self._cmd_status))
            self.application.add_handler(CommandHandler("portfolio", self._cmd_portfolio))
            self.application.add_handler(CommandHandler("performance", self._cmd_performance))
            self.application.add_handler(CommandHandler("positions", self._cmd_positions))
            self.application.add_handler(CommandHandler("stats", self._cmd_stats))
            
            # Callback query handler for inline buttons
            self.application.add_handler(CallbackQueryHandler(self._handle_callback))
            
            self.logger.info("Telegram bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            self.enabled = False
    
    async def start(self):
        """Start the Telegram bot"""
        if not self.enabled or not self.application:
            return
        
        try:
            await self.application.initialize()
            await self.application.start()
            self.is_running = True
            self.logger.info("Telegram bot started")
            
            # Send startup message
            await self.send_message(
                "🤖 *Trading Bot Started*\n\n"
                "Bot is now online and monitoring trades.\n"
                "Use /help to see available commands.",
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start Telegram bot: {e}")
            self.enabled = False
    
    async def stop(self):
        """Stop the Telegram bot"""
        if not self.is_running or not self.application:
            return
        
        try:
            await self.send_message(
                "🛑 *Trading Bot Stopped*\n\n"
                "Bot is shutting down.",
                parse_mode='Markdown'
            )
            
            await self.application.stop()
            await self.application.shutdown()
            self.is_running = False
            self.logger.info("Telegram bot stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping Telegram bot: {e}")
    
    async def send_message(
        self,
        message: str,
        parse_mode: str = 'HTML',
        disable_notification: bool = False,
        reply_markup=None
    ) -> bool:
        """
        Send a message to the configured chat
        
        Args:
            message: Message text
            parse_mode: Parse mode (HTML, Markdown, MarkdownV2)
            disable_notification: Whether to send silently
            reply_markup: Inline keyboard markup
            
        Returns:
            True if message sent successfully
        """
        if not self.enabled or not self.application:
            return False
        
        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=disable_notification,
                reply_markup=reply_markup
            )
            self.stats['messages_sent'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            self.stats['errors'] += 1
            return False
    
    async def notify_trade_opened(self, trade_data: Dict[str, Any]):
        """
        Send notification when a trade is opened
        
        Args:
            trade_data: Trade information
        """
        if not self.enabled or 'trade_opened' not in self.events:
            return
        
        try:
            symbol = trade_data.get('symbol', 'N/A')
            side = trade_data.get('side', 'N/A')
            entry_price = trade_data.get('entry_price', 0)
            quantity = trade_data.get('quantity', 0)
            strategy = trade_data.get('strategy', 'N/A')
            stop_loss = trade_data.get('stop_loss', 'N/A')
            take_profit = trade_data.get('take_profit', 'N/A')
            
            # Emoji based on side
            emoji = "🟢" if side.upper() == "BUY" else "🔴"
            
            message = (
                f"{emoji} <b>Trade Opened</b>\n\n"
                f"📊 Symbol: <b>{symbol}</b>\n"
                f"📈 Side: <b>{side}</b>\n"
                f"💰 Entry Price: <code>${entry_price:,.2f}</code>\n"
                f"📦 Quantity: <code>{quantity:.4f}</code>\n"
                f"💵 Total Value: <code>${entry_price * quantity:,.2f}</code>\n"
                f"🎯 Strategy: <i>{strategy}</i>\n"
                f"🛡️ Stop Loss: <code>{stop_loss}</code>\n"
                f"🎯 Take Profit: <code>{take_profit}</code>\n"
                f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending trade opened notification: {e}")
    
    async def notify_trade_closed(self, trade_data: Dict[str, Any]):
        """
        Send notification when a trade is closed
        
        Args:
            trade_data: Trade information
        """
        if not self.enabled or 'trade_closed' not in self.events:
            return
        
        try:
            symbol = trade_data.get('symbol', 'N/A')
            side = trade_data.get('side', 'N/A')
            entry_price = trade_data.get('entry_price', 0)
            exit_price = trade_data.get('exit_price', 0)
            quantity = trade_data.get('quantity', 0)
            pnl = trade_data.get('pnl', 0)
            pnl_pct = trade_data.get('pnl_pct', 0)
            reason = trade_data.get('reason', 'Manual')
            
            # Emoji based on profit/loss
            if pnl > 0:
                emoji = "✅"
                color = "green"
            elif pnl < 0:
                emoji = "❌"
                color = "red"
            else:
                emoji = "➖"
                color = "gray"
            
            message = (
                f"{emoji} <b>Trade Closed</b>\n\n"
                f"📊 Symbol: <b>{symbol}</b>\n"
                f"📈 Side: <b>{side}</b>\n"
                f"💰 Entry: <code>${entry_price:,.2f}</code>\n"
                f"💰 Exit: <code>${exit_price:,.2f}</code>\n"
                f"📦 Quantity: <code>{quantity:.4f}</code>\n"
                f"{'💵' if pnl >= 0 else '💸'} P&L: <b>${pnl:,.2f} ({pnl_pct:+.2f}%)</b>\n"
                f"📝 Reason: <i>{reason}</i>\n"
                f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending trade closed notification: {e}")
    
    async def notify_error(self, error_data: Dict[str, Any]):
        """
        Send error notification
        
        Args:
            error_data: Error information
        """
        if not self.enabled or 'error' not in self.events:
            return
        
        try:
            error_type = error_data.get('type', 'Unknown')
            error_msg = error_data.get('message', 'No details')
            module = error_data.get('module', 'Unknown')
            severity = error_data.get('severity', 'ERROR')
            
            # Emoji based on severity
            if severity == 'CRITICAL':
                emoji = "🚨"
            elif severity == 'ERROR':
                emoji = "⚠️"
            else:
                emoji = "ℹ️"
            
            message = (
                f"{emoji} <b>{severity}</b>\n\n"
                f"🔧 Module: <code>{module}</code>\n"
                f"🏷️ Type: <code>{error_type}</code>\n"
                f"📝 Message:\n<pre>{error_msg}</pre>\n"
                f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await self.send_message(message, disable_notification=(severity == 'WARNING'))
            
        except Exception as e:
            self.logger.error(f"Error sending error notification: {e}")
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]):
        """
        Send daily performance summary
        
        Args:
            summary_data: Summary statistics
        """
        if not self.enabled or 'daily_summary' not in self.events:
            return
        
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
            
            message = (
                f"📊 <b>Daily Summary - {date}</b>\n\n"
                f"📈 Total Trades: <b>{total_trades}</b>\n"
                f"✅ Winning: <b>{winning_trades}</b>\n"
                f"❌ Losing: <b>{losing_trades}</b>\n"
                f"🎯 Win Rate: <b>{win_rate:.1f}%</b>\n\n"
                f"{'💵' if total_pnl >= 0 else '💸'} Total P&L: <b>${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)</b>\n"
                f"🏆 Best Trade: <code>${best_trade:,.2f}</code>\n"
                f"📉 Worst Trade: <code>${worst_trade:,.2f}</code>\n\n"
                f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {e}")
    
    # Command Handlers
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        self.stats['commands_received'] += 1
        
        welcome_message = (
            "🤖 <b>Welcome to Trading Bot</b>\n\n"
            "I will keep you updated on all trading activities.\n\n"
            "Available commands:\n"
            "/help - Show all commands\n"
            "/status - Bot status\n"
            "/portfolio - Portfolio overview\n"
            "/performance - Performance metrics\n"
            "/positions - Active positions\n"
            "/stats - Bot statistics"
        )
        
        await update.message.reply_text(welcome_message, parse_mode='HTML')
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        self.stats['commands_received'] += 1
        
        help_message = (
            "📚 <b>Available Commands</b>\n\n"
            "/start - Welcome message\n"
            "/help - This help message\n"
            "/status - Show bot status\n"
            "/portfolio - Portfolio overview\n"
            "/performance - Performance metrics\n"
            "/positions - List active positions\n"
            "/stats - Bot statistics\n\n"
            "🔔 <b>Notifications</b>\n"
            "You will receive notifications for:\n"
            "• Trade entries and exits\n"
            "• Errors and warnings\n"
            "• Daily summaries"
        )
        
        await update.message.reply_text(help_message, parse_mode='HTML')
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        self.stats['commands_received'] += 1
        
        status_message = (
            "🟢 <b>Bot Status</b>\n\n"
            f"Status: <b>{'Running' if self.is_running else 'Stopped'}</b>\n"
            f"Messages Sent: <code>{self.stats['messages_sent']}</code>\n"
            f"Commands Received: <code>{self.stats['commands_received']}</code>\n"
            f"Errors: <code>{self.stats['errors']}</code>\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        await update.message.reply_text(status_message, parse_mode='HTML')
    
    async def _cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command"""
        self.stats['commands_received'] += 1
        
        if self.get_portfolio_callback:
            try:
                portfolio_data = await self.get_portfolio_callback()
                
                total_value = portfolio_data.get('total_value', 0)
                cash = portfolio_data.get('cash', 0)
                positions_value = portfolio_data.get('positions_value', 0)
                total_pnl = portfolio_data.get('total_pnl', 0)
                total_pnl_pct = portfolio_data.get('total_pnl_pct', 0)
                
                message = (
                    "💼 <b>Portfolio Overview</b>\n\n"
                    f"💰 Total Value: <b>${total_value:,.2f}</b>\n"
                    f"💵 Cash: <code>${cash:,.2f}</code>\n"
                    f"📦 Positions: <code>${positions_value:,.2f}</code>\n"
                    f"{'💵' if total_pnl >= 0 else '💸'} Total P&L: <b>${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)</b>\n"
                    f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                await update.message.reply_text(message, parse_mode='HTML')
                
            except Exception as e:
                await update.message.reply_text(
                    f"❌ Error fetching portfolio data: {str(e)}",
                    parse_mode='HTML'
                )
        else:
            await update.message.reply_text(
                "⚠️ Portfolio callback not configured",
                parse_mode='HTML'
            )
    
    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command"""
        self.stats['commands_received'] += 1
        
        if self.get_performance_callback:
            try:
                perf_data = await self.get_performance_callback()
                
                message = (
                    "📈 <b>Performance Metrics</b>\n\n"
                    f"Total Trades: <code>{perf_data.get('total_trades', 0)}</code>\n"
                    f"Win Rate: <code>{perf_data.get('win_rate', 0):.1f}%</code>\n"
                    f"Sharpe Ratio: <code>{perf_data.get('sharpe_ratio', 0):.2f}</code>\n"
                    f"Max Drawdown: <code>{perf_data.get('max_drawdown', 0):.2f}%</code>\n"
                    f"Profit Factor: <code>{perf_data.get('profit_factor', 0):.2f}</code>\n"
                    f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                await update.message.reply_text(message, parse_mode='HTML')
                
            except Exception as e:
                await update.message.reply_text(
                    f"❌ Error fetching performance data: {str(e)}",
                    parse_mode='HTML'
                )
        else:
            await update.message.reply_text(
                "⚠️ Performance callback not configured",
                parse_mode='HTML'
            )
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        self.stats['commands_received'] += 1
        
        if self.get_positions_callback:
            try:
                positions = await self.get_positions_callback()
                
                if not positions:
                    await update.message.reply_text(
                        "📭 No active positions",
                        parse_mode='HTML'
                    )
                    return
                
                message = "📊 <b>Active Positions</b>\n\n"
                
                for pos in positions:
                    symbol = pos.get('symbol', 'N/A')
                    side = pos.get('side', 'N/A')
                    quantity = pos.get('quantity', 0)
                    entry_price = pos.get('entry_price', 0)
                    current_price = pos.get('current_price', 0)
                    pnl = pos.get('pnl', 0)
                    pnl_pct = pos.get('pnl_pct', 0)
                    
                    emoji = "🟢" if side.upper() == "LONG" else "🔴"
                    
                    message += (
                        f"{emoji} <b>{symbol}</b> ({side})\n"
                        f"Qty: <code>{quantity:.4f}</code>\n"
                        f"Entry: <code>${entry_price:,.2f}</code>\n"
                        f"Current: <code>${current_price:,.2f}</code>\n"
                        f"P&L: <b>${pnl:,.2f} ({pnl_pct:+.2f}%)</b>\n\n"
                    )
                
                await update.message.reply_text(message, parse_mode='HTML')
                
            except Exception as e:
                await update.message.reply_text(
                    f"❌ Error fetching positions: {str(e)}",
                    parse_mode='HTML'
                )
        else:
            await update.message.reply_text(
                "⚠️ Positions callback not configured",
                parse_mode='HTML'
            )
    
    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        self.stats['commands_received'] += 1
        
        stats_message = (
            "📊 <b>Bot Statistics</b>\n\n"
            f"Messages Sent: <code>{self.stats['messages_sent']}</code>\n"
            f"Commands Received: <code>{self.stats['commands_received']}</code>\n"
            f"Errors: <code>{self.stats['errors']}</code>\n"
            f"Status: <b>{'🟢 Running' if self.is_running else '🔴 Stopped'}</b>\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        await update.message.reply_text(stats_message, parse_mode='HTML')
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        query = update.callback_query
        await query.answer()
        
        # Handle different callback data
        # Can be extended for interactive features
        
    def set_portfolio_callback(self, callback):
        """Set callback function to get portfolio data"""
        self.get_portfolio_callback = callback
    
    def set_performance_callback(self, callback):
        """Set callback function to get performance data"""
        self.get_performance_callback = callback
    
    def set_positions_callback(self, callback):
        """Set callback function to get positions data"""
        self.get_positions_callback = callback
