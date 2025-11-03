"""
Streamlit Dashboard for Trading Bot Monitoring

This module provides a real-time web-based dashboard for monitoring
trading activities, performance metrics, and portfolio status.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time

from core.logger import get_logger
from core.exceptions import BotTradingException


class TradingDashboard:
    """
    Interactive Streamlit dashboard for trading bot monitoring
    
    Features:
    - Real-time portfolio overview
    - Performance charts
    - Trade history table
    - Position tracking
    - Risk metrics
    - Interactive controls
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dashboard
        
        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger()
        self.config = config.get('monitoring', {}).get('dashboard', {})
        self.refresh_interval = self.config.get('refresh_interval', 5)
        
        # Callbacks for data
        self.get_portfolio_callback = None
        self.get_positions_callback = None
        self.get_trades_callback = None
        self.get_performance_callback = None
        self.get_equity_curve_callback = None
        
        # Configure Streamlit page
        st.set_page_config(
            page_title="Trading Bot Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the dashboard"""
        
        # Title and header
        st.title("🤖 Trading Bot Dashboard")
        st.markdown("---")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        col1, col2, col3, col4 = st.columns(4)
        
        # Fetch data
        portfolio_data = self._get_portfolio_data()
        performance_data = self._get_performance_data()
        
        # Key metrics
        with col1:
            self._render_metric_card(
                "💰 Total Value",
                f"${portfolio_data.get('total_value', 0):,.2f}",
                portfolio_data.get('total_pnl_pct', 0)
            )
        
        with col2:
            self._render_metric_card(
                "💵 Total P&L",
                f"${portfolio_data.get('total_pnl', 0):,.2f}",
                portfolio_data.get('total_pnl_pct', 0)
            )
        
        with col3:
            self._render_metric_card(
                "📊 Win Rate",
                f"{performance_data.get('win_rate', 0):.1f}%",
                None
            )
        
        with col4:
            self._render_metric_card(
                "📈 Sharpe Ratio",
                f"{performance_data.get('sharpe_ratio', 0):.2f}",
                None
            )
        
        st.markdown("---")
        
        # Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📈 Equity Curve")
            self._render_equity_curve()
        
        with col_right:
            st.subheader("📊 Portfolio Distribution")
            self._render_portfolio_distribution()
        
        st.markdown("---")
        
        # Active positions
        st.subheader("📊 Active Positions")
        self._render_positions_table()
        
        st.markdown("---")
        
        # Trade history
        st.subheader("📋 Recent Trades")
        self._render_trades_table()
        
        st.markdown("---")
        
        # Performance metrics
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.subheader("📈 Performance Metrics")
            self._render_performance_metrics()
        
        with col_perf2:
            st.subheader("⚠️ Risk Metrics")
            self._render_risk_metrics()
        
        # Auto-refresh
        if st.session_state.get('auto_refresh', False):
            time.sleep(self.refresh_interval)
            st.rerun()
    
    def _render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.title("⚙️ Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox(
            "Auto Refresh",
            value=st.session_state.get('auto_refresh', False),
            key='auto_refresh'
        )
        
        if auto_refresh:
            st.sidebar.info(f"Refreshing every {self.refresh_interval} seconds")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Filters
        st.sidebar.subheader("🔍 Filters")
        
        # Time range
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Today", "Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            index=1
        )
        st.session_state['time_range'] = time_range
        
        # Strategy filter
        strategies = st.sidebar.multiselect(
            "Strategies",
            ["All", "RSI_MACD", "Bollinger", "ML Strategy"],
            default=["All"]
        )
        st.session_state['strategies'] = strategies
        
        st.sidebar.markdown("---")
        
        # Status
        st.sidebar.subheader("📊 Status")
        st.sidebar.success("🟢 Bot Running")
        st.sidebar.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    def _render_metric_card(self, label: str, value: str, change: Optional[float]):
        """Render a metric card"""
        if change is not None:
            delta_color = "normal" if change >= 0 else "inverse"
            st.metric(label, value, f"{change:+.2f}%", delta_color=delta_color)
        else:
            st.metric(label, value)
    
    def _render_equity_curve(self):
        """Render equity curve chart"""
        equity_data = self._get_equity_curve_data()
        
        if not equity_data:
            st.info("No equity data available")
            return
        
        df = pd.DataFrame(equity_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#00d4aa', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_portfolio_distribution(self):
        """Render portfolio distribution pie chart"""
        positions = self._get_positions_data()
        
        if not positions:
            st.info("No active positions")
            return
        
        # Aggregate by symbol
        symbols = [p.get('symbol', 'N/A') for p in positions]
        values = [abs(p.get('value', 0)) for p in positions]
        
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_positions_table(self):
        """Render active positions table"""
        positions = self._get_positions_data()
        
        if not positions:
            st.info("No active positions")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(positions)
        
        # Format columns
        if 'entry_price' in df.columns:
            df['entry_price'] = df['entry_price'].apply(lambda x: f"${x:,.2f}")
        if 'current_price' in df.columns:
            df['current_price'] = df['current_price'].apply(lambda x: f"${x:,.2f}")
        if 'pnl' in df.columns:
            df['pnl'] = df['pnl'].apply(lambda x: f"${x:,.2f}")
        if 'pnl_pct' in df.columns:
            df['pnl_pct'] = df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
    
    def _render_trades_table(self):
        """Render recent trades table"""
        trades = self._get_trades_data()
        
        if not trades:
            st.info("No recent trades")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Limit to recent trades
        df = df.head(20)
        
        # Format columns
        if 'entry_price' in df.columns:
            df['entry_price'] = df['entry_price'].apply(lambda x: f"${x:,.2f}")
        if 'exit_price' in df.columns:
            df['exit_price'] = df['exit_price'].apply(lambda x: f"${x:,.2f}")
        if 'pnl' in df.columns:
            df['pnl'] = df['pnl'].apply(lambda x: f"${x:,.2f}")
        if 'pnl_pct' in df.columns:
            df['pnl_pct'] = df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
    
    def _render_performance_metrics(self):
        """Render performance metrics"""
        performance = self._get_performance_data()
        
        metrics_data = {
            "Metric": [
                "Total Return",
                "Annual Return",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Win Rate",
                "Profit Factor",
                "Total Trades"
            ],
            "Value": [
                f"{performance.get('total_return', 0):+.2f}%",
                f"{performance.get('annual_return', 0):+.2f}%",
                f"{performance.get('sharpe_ratio', 0):.2f}",
                f"{performance.get('sortino_ratio', 0):.2f}",
                f"{performance.get('calmar_ratio', 0):.2f}",
                f"{performance.get('win_rate', 0):.1f}%",
                f"{performance.get('profit_factor', 0):.2f}",
                f"{performance.get('total_trades', 0)}"
            ]
        }
        
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _render_risk_metrics(self):
        """Render risk metrics"""
        performance = self._get_performance_data()
        
        risk_data = {
            "Metric": [
                "Max Drawdown",
                "Avg Drawdown",
                "Current Drawdown",
                "Volatility (Ann.)",
                "Value at Risk (95%)",
                "Expected Shortfall",
                "Beta",
                "Alpha"
            ],
            "Value": [
                f"{performance.get('max_drawdown', 0):.2f}%",
                f"{performance.get('avg_drawdown', 0):.2f}%",
                f"{performance.get('current_drawdown', 0):.2f}%",
                f"{performance.get('volatility', 0):.2f}%",
                f"{performance.get('var_95', 0):.2f}%",
                f"{performance.get('expected_shortfall', 0):.2f}%",
                f"{performance.get('beta', 0):.2f}",
                f"{performance.get('alpha', 0):.2f}%"
            ]
        }
        
        df = pd.DataFrame(risk_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Data fetching methods
    
    def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data"""
        if self.get_portfolio_callback:
            try:
                return self.get_portfolio_callback()
            except Exception as e:
                self.logger.error(f"Error fetching portfolio data: {e}")
        
        # Return mock data
        return {
            'total_value': 10000.00,
            'cash': 5000.00,
            'positions_value': 5000.00,
            'total_pnl': 0.00,
            'total_pnl_pct': 0.00
        }
    
    def _get_positions_data(self) -> List[Dict[str, Any]]:
        """Get positions data"""
        if self.get_positions_callback:
            try:
                return self.get_positions_callback()
            except Exception as e:
                self.logger.error(f"Error fetching positions data: {e}")
        
        # Return empty list
        return []
    
    def _get_trades_data(self) -> List[Dict[str, Any]]:
        """Get trades data"""
        if self.get_trades_callback:
            try:
                return self.get_trades_callback()
            except Exception as e:
                self.logger.error(f"Error fetching trades data: {e}")
        
        # Return empty list
        return []
    
    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data"""
        if self.get_performance_callback:
            try:
                return self.get_performance_callback()
            except Exception as e:
                self.logger.error(f"Error fetching performance data: {e}")
        
        # Return mock data
        return {
            'total_return': 0.00,
            'annual_return': 0.00,
            'sharpe_ratio': 0.00,
            'sortino_ratio': 0.00,
            'calmar_ratio': 0.00,
            'win_rate': 0.00,
            'profit_factor': 0.00,
            'total_trades': 0,
            'max_drawdown': 0.00,
            'avg_drawdown': 0.00,
            'current_drawdown': 0.00,
            'volatility': 0.00,
            'var_95': 0.00,
            'expected_shortfall': 0.00,
            'beta': 0.00,
            'alpha': 0.00
        }
    
    def _get_equity_curve_data(self) -> List[Dict[str, Any]]:
        """Get equity curve data"""
        if self.get_equity_curve_callback:
            try:
                return self.get_equity_curve_callback()
            except Exception as e:
                self.logger.error(f"Error fetching equity curve data: {e}")
        
        # Return mock data
        return []
    
    # Callback setters
    
    def set_portfolio_callback(self, callback):
        """Set callback for portfolio data"""
        self.get_portfolio_callback = callback
    
    def set_positions_callback(self, callback):
        """Set callback for positions data"""
        self.get_positions_callback = callback
    
    def set_trades_callback(self, callback):
        """Set callback for trades data"""
        self.get_trades_callback = callback
    
    def set_performance_callback(self, callback):
        """Set callback for performance data"""
        self.get_performance_callback = callback
    
    def set_equity_curve_callback(self, callback):
        """Set callback for equity curve data"""
        self.get_equity_curve_callback = callback


def main():
    """Main function to run dashboard"""
    # Load config (you would normally load from config file)
    config = {
        'monitoring': {
            'dashboard': {
                'refresh_interval': 5
            }
        }
    }
    
    # Create and run dashboard
    dashboard = TradingDashboard(config)
    dashboard.run()


if __name__ == "__main__":
    main()
