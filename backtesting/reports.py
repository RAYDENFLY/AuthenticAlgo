"""
Report Generator Module
Generate comprehensive backtest reports with visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Tuple, Optional
from datetime import datetime
import json

from core.logger import get_logger
from backtesting.metrics import PerformanceMetrics


class ReportGenerator:
    """
    Generate comprehensive backtest reports
    
    Features:
    - Summary reports (text and JSON)
    - Trade list with details
    - Equity curve visualization
    - Drawdown chart
    - Distribution plots
    - Monthly returns heatmap
    - HTML export
    """
    
    def __init__(self):
        """Initialize report generator"""
        self.logger = get_logger()
    
    def generate_summary_report(self, results: dict, metrics: PerformanceMetrics) -> str:
        """
        Generate text summary report
        
        Args:
            results: Backtest results dict
            metrics: Performance metrics
            
        Returns:
            Formatted summary string
        """
        summary = f"""
{'=' * 90}
                           BACKTEST SUMMARY REPORT
{'=' * 90}

📅 BACKTEST PERIOD
   Start Date:                {results.get('start_date', 'N/A')}
   End Date:                  {results.get('end_date', 'N/A')}
   Duration:                  {results.get('duration', 'N/A')}

💰 CAPITAL & RETURNS
   Initial Capital:           ${results['initial_capital']:>15,.2f}
   Final Equity:              ${results['final_equity']:>15,.2f}
   Total Return:              {metrics.total_return:>15.2f}%
   Annualized Return:         {metrics.annualized_return:>15.2f}%
   Total P&L:                 ${results.get('total_pnl', 0):>15,.2f}

📊 RISK METRICS
   Sharpe Ratio:              {metrics.sharpe_ratio:>15.2f}
   Sortino Ratio:             {metrics.sortino_ratio:>15.2f}
   Calmar Ratio:              {metrics.calmar_ratio:>15.2f}
   Max Drawdown:              {metrics.max_drawdown:>15.2f}%
   Recovery Factor:           {metrics.recovery_factor:>15.2f}

📋 TRADE STATISTICS
   Total Trades:              {metrics.total_trades:>15}
   Winning Trades:            {metrics.winning_trades:>15} ({metrics.win_rate:.1f}%)
   Losing Trades:             {metrics.losing_trades:>15} ({100-metrics.win_rate:.1f}%)
   Profit Factor:             {metrics.profit_factor:>15.2f}
   Expectancy:                ${metrics.expectancy:>15.2f}

💵 TRADE ANALYSIS
   Average Win:               ${metrics.average_win:>15,.2f}
   Average Loss:              ${metrics.average_loss:>15,.2f}
   Largest Win:               ${metrics.largest_win:>15,.2f}
   Largest Loss:              ${metrics.largest_loss:>15,.2f}
   Risk/Reward Ratio:         {metrics.risk_reward_ratio:>15.2f}

📈 CONSISTENCY
   Max Consecutive Wins:      {metrics.max_consecutive_wins:>15}
   Max Consecutive Losses:    {metrics.max_consecutive_losses:>15}
   Avg Trade Duration:        {str(metrics.average_trade_duration).split('.')[0]:>15}

💡 POSITION SIZING
   Kelly Criterion:           {metrics.kelly_criterion:>15.2f}%
   Total Commission Paid:     ${results.get('total_commission', 0):>15,.2f}

{'=' * 90}
"""
        return summary
    
    def generate_trade_list(self, trades: List[any]) -> pd.DataFrame:
        """
        Generate detailed trade list DataFrame
        
        Args:
            trades: List of BacktestTrade objects
            
        Returns:
            DataFrame with trade details
        """
        if not trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in trades:
            trade_data.append({
                'Trade ID': trade.trade_id,
                'Symbol': trade.symbol,
                'Side': trade.side,
                'Quantity': trade.quantity,
                'Entry Price': trade.entry_price,
                'Exit Price': trade.exit_price,
                'Entry Time': trade.entry_time,
                'Exit Time': trade.exit_time,
                'P&L': trade.pnl,
                'P&L %': trade.pnl_pct,
                'Commission': trade.commission,
                'Duration': str(trade.duration).split('.')[0],
                'Exit Reason': trade.exit_reason
            })
        
        df = pd.DataFrame(trade_data)
        return df
    
    def plot_equity_curve(self, equity_curve: List[Tuple[datetime, float]], 
                         save_path: Optional[str] = None):
        """
        Plot equity curve
        
        Args:
            equity_curve: List of (timestamp, equity) tuples
            save_path: Optional path to save figure
        """
        if not equity_curve:
            self.logger.warning("No equity curve data to plot")
            return
        
        timestamps, equity = zip(*equity_curve)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(timestamps, equity, linewidth=2, color='#2E86AB', label='Equity')
        ax.fill_between(timestamps, equity, alpha=0.3, color='#2E86AB')
        
        ax.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add annotations
        initial_equity = equity[0]
        final_equity = equity[-1]
        total_return = ((final_equity - initial_equity) / initial_equity) * 100
        
        ax.text(0.02, 0.98, f'Initial: ${initial_equity:,.2f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.text(0.02, 0.93, f'Final: ${final_equity:,.2f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.text(0.02, 0.88, f'Return: {total_return:+.2f}%', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Equity curve saved to {save_path}")
        
        plt.show()
    
    def plot_drawdown(self, equity_curve: List[Tuple[datetime, float]], 
                     save_path: Optional[str] = None):
        """
        Plot drawdown chart
        
        Args:
            equity_curve: List of (timestamp, equity) tuples
            save_path: Optional path to save figure
        """
        if not equity_curve:
            self.logger.warning("No equity curve data to plot")
            return
        
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate drawdown
        running_max = df['equity'].expanding().max()
        drawdown = (df['equity'] - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.5, color='#E63946', label='Drawdown')
        ax.plot(drawdown.index, drawdown, linewidth=2, color='#C1121F')
        
        ax.set_title('Portfolio Drawdown', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', fontsize=10)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add max drawdown annotation
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        ax.annotate(f'Max DD: {max_dd:.2f}%', 
                   xy=(max_dd_date, max_dd), 
                   xytext=(max_dd_date, max_dd - 5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Drawdown chart saved to {save_path}")
        
        plt.show()
    
    def plot_trade_distribution(self, trades: List[any], save_path: Optional[str] = None):
        """
        Plot trade P&L distribution
        
        Args:
            trades: List of BacktestTrade objects
            save_path: Optional path to save figure
        """
        if not trades:
            self.logger.warning("No trades to plot")
            return
        
        pnls = [t.pnl for t in trades]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1.hist(pnls, bins=30, alpha=0.7, color='#457B9D', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('P&L ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(pnls, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#A8DADC', alpha=0.7),
                   medianprops=dict(color='#E63946', linewidth=2))
        ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax2.set_title('Trade P&L Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel('P&L ($)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_returns_heatmap(self, equity_curve: List[Tuple[datetime, float]], 
                            save_path: Optional[str] = None):
        """
        Plot monthly returns heatmap
        
        Args:
            equity_curve: List of (timestamp, equity) tuples
            save_path: Optional path to save figure
        """
        if not equity_curve:
            self.logger.warning("No equity curve data to plot")
            return
        
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate daily returns
        daily_returns = df['equity'].pct_change().dropna()
        
        # Resample to monthly returns
        monthly_returns = (1 + daily_returns).resample('M').prod() - 1
        monthly_returns = monthly_returns * 100  # Convert to percentage
        
        if len(monthly_returns) < 2:
            self.logger.warning("Insufficient data for monthly returns heatmap")
            return
        
        # Create pivot table for heatmap
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='returns')
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        # Set ticks
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Monthly Return (%)', fontsize=12)
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.values[i, j]
                if not pd.isna(value):
                    text_color = 'white' if abs(value) > 5 else 'black'
                    ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                           color=text_color, fontsize=9)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Returns heatmap saved to {save_path}")
        
        plt.show()
    
    def export_to_json(self, results: dict, metrics: PerformanceMetrics, 
                      file_path: str):
        """
        Export results to JSON file
        
        Args:
            results: Backtest results dict
            metrics: Performance metrics
            file_path: Path to save JSON file
        """
        export_data = {
            'backtest_results': {
                'initial_capital': results['initial_capital'],
                'final_equity': results['final_equity'],
                'total_return': results['total_return'],
                'total_trades': results['total_trades'],
                'winning_trades': results['winning_trades'],
                'losing_trades': results['losing_trades'],
                'win_rate': results['win_rate'],
                'total_pnl': results.get('total_pnl', 0),
                'total_commission': results.get('total_commission', 0)
            },
            'performance_metrics': {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'max_drawdown': metrics.max_drawdown,
                'profit_factor': metrics.profit_factor,
                'expectancy': metrics.expectancy,
                'kelly_criterion': metrics.kelly_criterion,
                'recovery_factor': metrics.recovery_factor,
                'risk_reward_ratio': metrics.risk_reward_ratio
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=4, default=str)
        
        self.logger.info(f"Results exported to {file_path}")
    
    def generate_html_report(self, results: dict, metrics: PerformanceMetrics, 
                           trades_df: pd.DataFrame, file_path: str):
        """
        Generate HTML report
        
        Args:
            results: Backtest results dict
            metrics: Performance metrics
            trades_df: Trade list DataFrame
            file_path: Path to save HTML file
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2E86AB; text-align: center; }}
        h2 {{ color: #457B9D; border-bottom: 2px solid #A8DADC; padding-bottom: 10px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .metric-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #2E86AB; }}
        .metric-label {{ font-size: 14px; color: #6c757d; margin-bottom: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background-color: #2E86AB; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Report</h1>
        
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if metrics.total_return > 0 else 'negative'}">{metrics.total_return:+.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{metrics.max_drawdown:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics.win_rate:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{metrics.profit_factor:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{metrics.total_trades}</div>
            </div>
        </div>
        
        <h2>Trade History</h2>
        {trades_df.to_html(index=False, classes='table', border=0)}
    </div>
</body>
</html>
"""
        
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to {file_path}")
