"""
Performance Metrics Module
Comprehensive trading performance calculations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.logger import get_logger


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    daily_returns_mean: float
    daily_returns_std: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade: float
    average_trade_duration: timedelta
    
    # Consecutive statistics
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Additional metrics
    expectancy: float
    kelly_criterion: float
    recovery_factor: float
    risk_reward_ratio: float


class MetricsCalculator:
    """
    Calculate comprehensive trading performance metrics
    
    Features:
    - Multiple risk-adjusted return metrics (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Trade statistics
    - Win/loss analysis
    - Kelly Criterion
    - Expectancy calculation
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.logger = get_logger()
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, equity_curve: List[Tuple[datetime, float]],
                             trades: List[any], initial_capital: float) -> PerformanceMetrics:
        """
        Calculate all performance metrics
        
        Args:
            equity_curve: List of (timestamp, equity) tuples
            trades: List of BacktestTrade objects
            initial_capital: Starting capital
            
        Returns:
            PerformanceMetrics dataclass with all metrics
        """
        self.logger.info("Calculating comprehensive performance metrics...")
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate daily returns
        daily_returns = equity_df['equity'].pct_change().dropna()
        
        # Returns metrics
        total_return = self._calculate_total_return(equity_df, initial_capital)
        annualized_return = self._calculate_annualized_return(equity_df, initial_capital)
        daily_returns_mean = daily_returns.mean()
        daily_returns_std = daily_returns.std()
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity_df)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_dd)
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades)
        
        # Additional metrics
        expectancy = self._calculate_expectancy(trades)
        kelly = self._calculate_kelly_criterion(trade_stats)
        recovery_factor = self._calculate_recovery_factor(total_return, max_dd)
        risk_reward = trade_stats['average_win'] / abs(trade_stats['average_loss']) if trade_stats['average_loss'] != 0 else 0
        
        return PerformanceMetrics(
            # Returns
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns_mean=daily_returns_mean * 100,  # as percentage
            daily_returns_std=daily_returns_std * 100,
            
            # Risk metrics
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            
            # Trade statistics
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            average_win=trade_stats['average_win'],
            average_loss=trade_stats['average_loss'],
            largest_win=trade_stats['largest_win'],
            largest_loss=trade_stats['largest_loss'],
            average_trade=trade_stats['average_trade'],
            average_trade_duration=trade_stats['average_duration'],
            
            # Consecutive statistics
            max_consecutive_wins=trade_stats['max_consecutive_wins'],
            max_consecutive_losses=trade_stats['max_consecutive_losses'],
            
            # Additional metrics
            expectancy=expectancy,
            kelly_criterion=kelly,
            recovery_factor=recovery_factor,
            risk_reward_ratio=risk_reward
        )
    
    def _calculate_total_return(self, equity_df: pd.DataFrame, initial_capital: float) -> float:
        """Calculate total return percentage"""
        final_equity = equity_df['equity'].iloc[-1]
        return ((final_equity - initial_capital) / initial_capital) * 100
    
    def _calculate_annualized_return(self, equity_df: pd.DataFrame, initial_capital: float) -> float:
        """Calculate annualized return"""
        final_equity = equity_df['equity'].iloc[-1]
        total_days = (equity_df.index[-1] - equity_df.index[0]).days
        
        if total_days == 0:
            return 0.0
        
        years = total_days / 365.25
        annualized = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
        
        return annualized
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio (annualized)
        
        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0.0
        
        # Convert annual risk-free rate to daily
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        
        excess_returns = daily_returns - daily_rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino_ratio(self, daily_returns: pd.Series) -> float:
        """
        Calculate Sortino ratio (only penalizes downside volatility)
        
        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        """
        if len(daily_returns) == 0:
            return 0.0
        
        # Convert annual risk-free rate to daily
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        
        excess_returns = daily_returns - daily_rf
        
        # Only negative returns for downside deviation
        downside_returns = daily_returns[daily_returns < daily_rf]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        
        return sortino
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio
        
        Calmar = Annualized Return / Max Drawdown
        """
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / abs(max_drawdown)
    
    def _calculate_max_drawdown(self, equity_df: pd.DataFrame) -> Tuple[float, timedelta]:
        """
        Calculate maximum drawdown and its duration
        
        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_duration)
        """
        equity = equity_df['equity']
        
        # Calculate running maximum
        running_max = equity.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max * 100
        
        max_dd = drawdown.min()
        
        # Calculate drawdown duration
        dd_duration = timedelta(0)
        current_dd_start = None
        max_dd_duration = timedelta(0)
        
        for timestamp, dd in drawdown.items():
            if dd < -0.01:  # In drawdown (more than 0.01%)
                if current_dd_start is None:
                    current_dd_start = timestamp
            else:  # Out of drawdown
                if current_dd_start is not None:
                    duration = timestamp - current_dd_start
                    if duration > max_dd_duration:
                        max_dd_duration = duration
                    current_dd_start = None
        
        # Check if still in drawdown at end
        if current_dd_start is not None:
            duration = equity_df.index[-1] - current_dd_start
            if duration > max_dd_duration:
                max_dd_duration = duration
        
        return max_dd, max_dd_duration
    
    def _calculate_trade_statistics(self, trades: List[any]) -> Dict[str, any]:
        """Calculate detailed trade statistics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'average_trade': 0.0,
                'average_duration': timedelta(0),
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        # Calculate statistics
        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = (num_winning / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor = Gross Profit / Gross Loss
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Averages
        average_win = gross_profit / num_winning if num_winning > 0 else 0
        average_loss = -gross_loss / num_losing if num_losing > 0 else 0
        average_trade = sum(t.pnl for t in trades) / total_trades
        
        # Extremes
        largest_win = max((t.pnl for t in winning_trades), default=0)
        largest_loss = min((t.pnl for t in losing_trades), default=0)
        
        # Average duration
        average_duration = sum((t.duration for t in trades), timedelta(0)) / total_trades
        
        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = self._calculate_consecutive_trades(trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'average_trade': average_trade,
            'average_duration': average_duration,
            'max_consecutive_wins': max_consec_wins,
            'max_consecutive_losses': max_consec_losses
        }
    
    def _calculate_consecutive_trades(self, trades: List[any]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_expectancy(self, trades: List[any]) -> float:
        """
        Calculate trade expectancy
        
        Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
        """
        if not trades:
            return 0.0
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades)
        loss_rate = len(losing_trades) / len(trades)
        
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        
        return expectancy
    
    def _calculate_kelly_criterion(self, trade_stats: Dict[str, any]) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Kelly % = W - [(1 - W) / R]
        Where: W = Win rate, R = Win/Loss ratio
        """
        win_rate = trade_stats['win_rate'] / 100  # Convert to decimal
        
        if trade_stats['average_loss'] == 0:
            return 0.0
        
        win_loss_ratio = abs(trade_stats['average_win'] / trade_stats['average_loss'])
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fractional Kelly (half Kelly is more practical)
        fractional_kelly = kelly * 0.5
        
        return max(0, min(fractional_kelly * 100, 100))  # Return as percentage, capped at 100%
    
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """
        Calculate recovery factor
        
        Recovery Factor = Total Return / Max Drawdown
        """
        if max_drawdown == 0:
            return 0.0
        
        return total_return / abs(max_drawdown)
    
    def generate_metrics_summary(self, metrics: PerformanceMetrics) -> str:
        """
        Generate formatted metrics summary
        
        Args:
            metrics: PerformanceMetrics object
            
        Returns:
            Formatted string summary
        """
        summary = f"""
{'=' * 80}
PERFORMANCE METRICS SUMMARY
{'=' * 80}

📈 RETURNS
   Total Return:              {metrics.total_return:>10.2f}%
   Annualized Return:         {metrics.annualized_return:>10.2f}%
   Daily Mean Return:         {metrics.daily_returns_mean:>10.4f}%
   Daily Std Deviation:       {metrics.daily_returns_std:>10.4f}%

📊 RISK-ADJUSTED RETURNS
   Sharpe Ratio:              {metrics.sharpe_ratio:>10.2f}
   Sortino Ratio:             {metrics.sortino_ratio:>10.2f}
   Calmar Ratio:              {metrics.calmar_ratio:>10.2f}
   Max Drawdown:              {metrics.max_drawdown:>10.2f}%
   Max DD Duration:           {str(metrics.max_drawdown_duration).split('.')[0]}

📋 TRADE STATISTICS
   Total Trades:              {metrics.total_trades:>10}
   Winning Trades:            {metrics.winning_trades:>10}
   Losing Trades:             {metrics.losing_trades:>10}
   Win Rate:                  {metrics.win_rate:>10.2f}%
   Profit Factor:             {metrics.profit_factor:>10.2f}

💰 TRADE ANALYSIS
   Average Win:               ${metrics.average_win:>10.2f}
   Average Loss:              ${metrics.average_loss:>10.2f}
   Largest Win:               ${metrics.largest_win:>10.2f}
   Largest Loss:              ${metrics.largest_loss:>10.2f}
   Average Trade:             ${metrics.average_trade:>10.2f}
   Avg Trade Duration:        {str(metrics.average_trade_duration).split('.')[0]}

🔄 CONSECUTIVE TRADES
   Max Consecutive Wins:      {metrics.max_consecutive_wins:>10}
   Max Consecutive Losses:    {metrics.max_consecutive_losses:>10}

💡 ADVANCED METRICS
   Expectancy:                ${metrics.expectancy:>10.2f}
   Kelly Criterion:           {metrics.kelly_criterion:>10.2f}%
   Recovery Factor:           {metrics.recovery_factor:>10.2f}
   Risk/Reward Ratio:         {metrics.risk_reward_ratio:>10.2f}

{'=' * 80}
"""
        return summary
