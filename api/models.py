"""
Pydantic Models for API Request/Response
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class PositionType(str, Enum):
    LONG = "long"
    SHORT = "short"


class QualityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ============================================================================
# HEALTH & STATUS
# ============================================================================

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status (online/offline)")
    api_version: str = Field(..., description="Current API version")
    database: bool = Field(..., description="Database connection status")
    trading_bot: bool = Field(..., description="Trading bot status")
    ml_models: bool = Field(..., description="ML models loaded status")
    exchange_connection: bool = Field(..., description="Exchange API status")
    timestamp: datetime = Field(..., description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "online",
                "api_version": "1.0.0",
                "database": True,
                "trading_bot": True,
                "ml_models": True,
                "exchange_connection": True,
                "timestamp": "2025-11-03T15:00:00"
            }
        }


# ============================================================================
# PUBLIC STATS
# ============================================================================

class PublicStatsResponse(BaseModel):
    current_pnl: float = Field(..., description="Current PnL in USD")
    total_trades: int = Field(..., description="Total number of trades")
    win_rate: float = Field(..., description="Win rate percentage")
    roi: float = Field(..., description="Return on Investment percentage")
    active_positions: int = Field(..., description="Number of open positions")
    initial_capital: float = Field(..., description="Starting capital")
    current_capital: float = Field(..., description="Current total capital")
    
    class Config:
        json_schema_extra = {
            "example": {
                "current_pnl": 45.23,
                "total_trades": 156,
                "win_rate": 85.5,
                "roi": 452.3,
                "active_positions": 3,
                "initial_capital": 10.0,
                "current_capital": 55.23
            }
        }


# ============================================================================
# TRADING SIGNALS
# ============================================================================

class TPSLLevels(BaseModel):
    tp1: float = Field(..., description="Take Profit 1")
    tp2: float = Field(..., description="Take Profit 2")
    tp3: float = Field(..., description="Take Profit 3")
    sl: float = Field(..., description="Stop Loss")
    risk_reward: float = Field(..., description="Risk/Reward ratio")


class SignalResponse(BaseModel):
    id: str = Field(..., description="Signal ID")
    symbol: str = Field(..., description="Trading pair")
    direction: SignalDirection = Field(..., description="Signal direction")
    entry_price: float = Field(..., description="Recommended entry price")
    confidence: float = Field(..., description="Confidence score (0-100)")
    quality: QualityLevel = Field(..., description="Signal quality")
    tp_sl: TPSLLevels = Field(..., description="TP/SL levels")
    reasoning: str = Field(..., description="Signal reasoning")
    timestamp: datetime = Field(..., description="Signal generation time")
    source: str = Field(..., description="Signal source (ML/TA/Hybrid)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "sig_12345",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 68450.0,
                "confidence": 84.5,
                "quality": "HIGH",
                "tp_sl": {
                    "tp1": 68720.0,
                    "tp2": 69100.0,
                    "tp3": 69550.0,
                    "sl": 68200.0,
                    "risk_reward": 4.4
                },
                "reasoning": "Strong bullish momentum, RSI oversold, MACD golden cross",
                "timestamp": "2025-11-03T15:00:00",
                "source": "ML"
            }
        }


# ============================================================================
# POSITIONS
# ============================================================================

class PositionResponse(BaseModel):
    position_id: str = Field(..., description="Position ID")
    symbol: str = Field(..., description="Trading pair")
    direction: PositionType = Field(..., description="Position direction")
    entry_price: float = Field(..., description="Entry price")
    current_price: float = Field(..., description="Current market price")
    size: float = Field(..., description="Position size (contracts)")
    leverage: float = Field(..., description="Leverage used")
    unrealized_pnl: float = Field(..., description="Unrealized PnL in USD")
    unrealized_pnl_pct: float = Field(..., description="Unrealized PnL percentage")
    tp_sl: TPSLLevels = Field(..., description="TP/SL levels")
    opened_at: datetime = Field(..., description="Position open time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "position_id": "pos_12345",
                "symbol": "BTCUSDT",
                "direction": "long",
                "entry_price": 68450.0,
                "current_price": 68720.0,
                "size": 0.01,
                "leverage": 10.0,
                "unrealized_pnl": 2.70,
                "unrealized_pnl_pct": 0.39,
                "tp_sl": {
                    "tp1": 68720.0,
                    "tp2": 69100.0,
                    "tp3": 69550.0,
                    "sl": 68200.0,
                    "risk_reward": 4.4
                },
                "opened_at": "2025-11-03T14:00:00"
            }
        }


# ============================================================================
# TRADES
# ============================================================================

class TradeResponse(BaseModel):
    trade_id: str = Field(..., description="Trade ID")
    symbol: str = Field(..., description="Trading pair")
    direction: PositionType = Field(..., description="Trade direction")
    entry_price: float = Field(..., description="Entry price")
    exit_price: float = Field(..., description="Exit price")
    size: float = Field(..., description="Position size")
    leverage: float = Field(..., description="Leverage used")
    pnl: float = Field(..., description="Realized PnL in USD")
    pnl_pct: float = Field(..., description="PnL percentage")
    exit_reason: str = Field(..., description="Exit reason (TP1/TP2/TP3/SL)")
    opened_at: datetime = Field(..., description="Trade open time")
    closed_at: datetime = Field(..., description="Trade close time")
    duration: str = Field(..., description="Trade duration")


class TradeHistoryResponse(BaseModel):
    total: int = Field(..., description="Total number of trades")
    trades: List[TradeResponse] = Field(..., description="List of trades")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


# ============================================================================
# PERFORMANCE
# ============================================================================

class LivePerformanceResponse(BaseModel):
    current_balance: float = Field(..., description="Current account balance")
    today_pnl: float = Field(..., description="Today's PnL")
    today_pnl_pct: float = Field(..., description="Today's PnL percentage")
    open_positions: int = Field(..., description="Number of open positions")
    win_rate: float = Field(..., description="Overall win rate")
    total_trades: int = Field(..., description="Total trades executed")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    average_win: float = Field(..., description="Average winning trade")
    average_loss: float = Field(..., description="Average losing trade")
    largest_win: float = Field(..., description="Largest winning trade")
    largest_loss: float = Field(..., description="Largest losing trade")
    equity_curve: List[Dict] = Field(..., description="Equity curve data points")


# ============================================================================
# PORTFOLIO
# ============================================================================

class AssetAllocation(BaseModel):
    symbol: str
    percentage: float
    value: float
    unrealized_pnl: float


class PortfolioResponse(BaseModel):
    total_value: float = Field(..., description="Total portfolio value")
    cash_balance: float = Field(..., description="Available cash")
    margin_used: float = Field(..., description="Margin in use")
    margin_available: float = Field(..., description="Available margin")
    leverage_avg: float = Field(..., description="Average leverage")
    allocations: List[AssetAllocation] = Field(..., description="Asset breakdown")


# ============================================================================
# MACHINE LEARNING
# ============================================================================

class MLModelResponse(BaseModel):
    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (XGBoost, etc)")
    symbol: str = Field(..., description="Trading pair")
    timeframe: str = Field(..., description="Timeframe")
    training_accuracy: float = Field(..., description="Training accuracy %")
    live_accuracy: float = Field(..., description="Live trading accuracy %")
    total_predictions: int = Field(..., description="Total predictions made")
    last_updated: datetime = Field(..., description="Last training date")
    status: str = Field(..., description="Model status (active/inactive)")


class FeatureImportance(BaseModel):
    feature_name: str
    importance: float


class MLPerformanceResponse(BaseModel):
    overall_accuracy: float = Field(..., description="Overall model accuracy")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    confusion_matrix: Dict[str, int] = Field(..., description="Confusion matrix")
    feature_importance: List[FeatureImportance] = Field(..., description="Top features")
    prediction_distribution: Dict[str, int] = Field(..., description="Prediction counts")


class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair to predict")
    timeframe: str = Field(default="1h", description="Timeframe")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "timeframe": "1h"
            }
        }


class PredictionResponse(BaseModel):
    symbol: str
    direction: SignalDirection
    confidence: float
    probability_long: float
    probability_short: float
    features_used: int
    model_used: str
    timestamp: datetime


class BacktestResponse(BaseModel):
    model_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: str
    best_trade: float
    worst_trade: float


# ============================================================================
# COMPETITION / PAPER TRADING
# ============================================================================

class StrategyPerformance(BaseModel):
    strategy_name: str
    current_capital: float
    roi: float
    trades_completed: int
    win_rate: float
    average_leverage: float
    largest_win: float
    largest_loss: float


class CompetitionResponse(BaseModel):
    competition_id: str
    status: str = Field(..., description="running/completed")
    started_at: datetime
    strategies: List[StrategyPerformance]
    leader: str = Field(..., description="Current leader")


class CompetitionConfig(BaseModel):
    initial_capital: float = Field(default=10.0, description="Starting capital")
    max_trades: int = Field(default=10, description="Max trades per strategy")
    symbols: Optional[List[str]] = Field(default=None, description="Allowed symbols")
    leverage_range: tuple = Field(default=(5, 125), description="Min/max leverage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "initial_capital": 10.0,
                "max_trades": 10,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "leverage_range": [5, 125]
            }
        }


class CompetitionStartResponse(BaseModel):
    competition_id: str
    status: str
    message: str


class CompetitionResultsResponse(BaseModel):
    competition_id: str
    winner: str
    strategies: List[StrategyPerformance]
    summary: Dict[str, Any]
    completed_at: datetime


# ============================================================================
# ADMIN MODELS
# ============================================================================

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    account_status: str
    current_balance: float
    total_pnl: float
    total_trades: int
    win_rate: float
    created_at: datetime
    last_active: datetime


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    username: str
    roi: float
    total_pnl: float
    win_rate: float
    sharpe_ratio: float


class LeaderboardResponse(BaseModel):
    leaderboard: List[LeaderboardEntry]
    updated_at: datetime


class ClosePositionRequest(BaseModel):
    position_id: str = Field(..., description="Position ID to close")
    reason: str = Field(..., description="Reason for manual close")
    
    class Config:
        json_schema_extra = {
            "example": {
                "position_id": "pos_12345",
                "reason": "Admin intervention - market risk"
            }
        }


class LeverageAdjustRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User ID (None = global)")
    min_leverage: float = Field(..., description="Minimum leverage")
    max_leverage: float = Field(..., description="Maximum leverage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": None,
                "min_leverage": 5.0,
                "max_leverage": 50.0
            }
        }


class SystemMonitorResponse(BaseModel):
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    network_latency: float = Field(..., description="Network latency ms")
    api_response_time: float = Field(..., description="Avg API response time ms")
    database_connections: int = Field(..., description="Active DB connections")
    active_websockets: int = Field(..., description="Active WebSocket connections")
    ml_models_loaded: int = Field(..., description="Number of loaded ML models")
    timestamp: datetime


class RiskExposureResponse(BaseModel):
    total_exposure: float = Field(..., description="Total USD exposure")
    leverage_weighted: float = Field(..., description="Leverage-weighted exposure")
    margin_used: float = Field(..., description="Total margin used")
    margin_available: float = Field(..., description="Available margin")
    position_count: int = Field(..., description="Total open positions")
    exposure_by_symbol: Dict[str, float] = Field(..., description="Exposure breakdown")
    correlation_risk: float = Field(..., description="Portfolio correlation")


class AuditLogResponse(BaseModel):
    log_id: str
    timestamp: datetime
    action_type: str = Field(..., description="Type of action")
    user_id: Optional[str] = Field(None, description="User who performed action")
    details: Dict[str, Any] = Field(..., description="Action details")
    ip_address: Optional[str] = Field(None, description="IP address")


# ============================================================================
# CHARTS
# ============================================================================

class EquityCurvePoint(BaseModel):
    timestamp: datetime
    balance: float
    pnl: float
    drawdown: float


class EquityCurveResponse(BaseModel):
    timeframe: str
    data_points: List[EquityCurvePoint]
    summary: Dict[str, Any]


# ============================================================================
# REPORTS
# ============================================================================

class TradeBreakdown(BaseModel):
    wins: int
    losses: int
    breakeven: int


class DailyReportResponse(BaseModel):
    date: str
    trades_executed: int
    trade_breakdown: TradeBreakdown
    total_pnl: float
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]
    win_rate: float
    average_trade_duration: str
    ml_accuracy: float
    signals_generated: int
    risk_metrics: Dict[str, float]
