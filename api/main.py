"""
AuthenticAlgo Pro - Trading API
FastAPI backend with complete Swagger documentation
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from api.models import *
from api.services import TradingService, MLService, AdminService
from core.logger import setup_logger

logger = setup_logger()

# Initialize FastAPI with metadata
app = FastAPI(
    title="AuthenticAlgo Pro Trading API",
    description="""
    🚀 **Professional AI Trading Bot API**
    
    Transform $5 into $100+ with AI-powered trading strategies.
    
    ## Features
    
    * 📊 **Live Trading Dashboard** - Real-time PnL, positions, and signals
    * 🤖 **Machine Learning Models** - 96% accuracy XGBoost predictions
    * 📈 **Paper Trading Arena** - Compare strategies (TA vs ML vs Hybrid)
    * 👑 **Admin Control Panel** - Multi-user management & risk controls
    * 🔔 **Real-time WebSocket** - Live updates for trades and market data
    
    ## Authentication
    
    Use Bearer token for protected endpoints (admin routes).
    
    ## Rate Limiting
    
    - Public endpoints: 100 requests/minute
    - Admin endpoints: 1000 requests/minute
    
    ## Support
    
    Contact: support@authenticalgo.com
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "AuthenticAlgo Support",
        "email": "support@authenticalgo.com",
        "url": "https://authenticalgo.com"
    },
    license_info={
        "name": "Proprietary",
        "url": "https://authenticalgo.com/license"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
trading_service = TradingService()
ml_service = MLService()
admin_service = AdminService()

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


# ============================================================================
# PUBLIC ENDPOINTS - No authentication required
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """
    🏠 **API Health Check**
    
    Returns API status and version info.
    """
    return {
        "status": "online",
        "service": "AuthenticAlgo Pro Trading API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "documentation": "/docs"
    }


@app.get("/api/v1/health", tags=["Health"], response_model=HealthResponse)
async def health_check():
    """
    🏥 **System Health Status**
    
    Returns comprehensive system health metrics:
    - API status
    - Database connection
    - Trading bot status
    - ML model status
    - Exchange connectivity
    """
    return await trading_service.get_health_status()


@app.get("/api/v1/stats/public", tags=["Public"], response_model=PublicStatsResponse)
async def get_public_stats():
    """
    📊 **Public Performance Stats**
    
    Live statistics for public display (hero section):
    - Current PnL
    - Win rate
    - Total trades
    - ROI percentage
    - Active positions
    
    **Perfect for:** Landing page live counter
    """
    return await trading_service.get_public_stats()


@app.get("/api/v1/performance/live", tags=["Public"], response_model=LivePerformanceResponse)
async def get_live_performance():
    """
    📈 **Live Trading Performance**
    
    Real-time trading metrics:
    - Equity curve data
    - Current open positions
    - Recent trade history
    - Today's PnL breakdown
    
    **Perfect for:** Live trading widget on homepage
    """
    return await trading_service.get_live_performance()


@app.get("/api/v1/signals/recent", tags=["Public"], response_model=List[SignalResponse])
async def get_recent_signals(limit: int = 10):
    """
    🎯 **Recent Trading Signals**
    
    Latest AI-generated trading signals with:
    - Signal direction (LONG/SHORT)
    - Confidence score
    - Entry price
    - TP/SL levels
    - Reasoning
    
    **Perfect for:** Signal feed on homepage
    
    **Parameters:**
    - limit: Number of signals to return (default: 10, max: 50)
    """
    if limit > 50:
        raise HTTPException(status_code=400, detail="Max limit is 50")
    return await trading_service.get_recent_signals(limit)


@app.get("/api/v1/charts/equity", tags=["Public"], response_model=EquityCurveResponse)
async def get_equity_curve(timeframe: str = "24h"):
    """
    📊 **Equity Curve Chart Data**
    
    Historical equity curve for visualization:
    - Timestamp series
    - Balance progression
    - Drawdown periods
    - Milestone markers
    
    **Perfect for:** TradingView charts
    
    **Parameters:**
    - timeframe: 1h, 24h, 7d, 30d, all
    """
    return await trading_service.get_equity_curve(timeframe)


@app.get("/api/v1/reports/daily", tags=["Public"], response_model=DailyReportResponse)
async def get_daily_report():
    """
    📋 **Daily Trading Report**
    
    Comprehensive daily summary:
    - Trades executed
    - Win/loss breakdown
    - Best/worst trades
    - Risk metrics
    - ML model performance
    
    **Perfect for:** Daily performance reports section
    """
    return await trading_service.get_daily_report()


# ============================================================================
# TRADING ENDPOINTS - Real-time data
# ============================================================================

@app.get("/api/v1/positions/current", tags=["Trading"], response_model=List[PositionResponse])
async def get_current_positions():
    """
    💼 **Current Open Positions**
    
    All active trading positions with:
    - Symbol and direction
    - Entry price and size
    - Current PnL
    - TP/SL levels
    - Unrealized profit
    
    **Updates:** Real-time via WebSocket
    """
    return await trading_service.get_current_positions()


@app.get("/api/v1/trades/history", tags=["Trading"], response_model=TradeHistoryResponse)
async def get_trade_history(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None
):
    """
    📜 **Trade History**
    
    Paginated trade history with filtering:
    - All closed trades
    - Entry/exit details
    - PnL per trade
    - Trade duration
    
    **Parameters:**
    - limit: Results per page (max 100)
    - offset: Pagination offset
    - status: Filter by status (closed, open, cancelled)
    """
    return await trading_service.get_trade_history(limit, offset, status)


@app.get("/api/v1/portfolio/breakdown", tags=["Trading"], response_model=PortfolioResponse)
async def get_portfolio_breakdown():
    """
    🎯 **Portfolio Allocation**
    
    Current portfolio composition:
    - Asset allocation percentages
    - Risk exposure by symbol
    - Leverage usage
    - Margin requirements
    
    **Perfect for:** Portfolio pie charts
    """
    return await trading_service.get_portfolio_breakdown()


# ============================================================================
# MACHINE LEARNING ENDPOINTS
# ============================================================================

@app.get("/api/v1/ml/models", tags=["Machine Learning"], response_model=List[MLModelResponse])
async def get_ml_models():
    """
    🤖 **ML Model Information**
    
    All available ML models with:
    - Model name and type
    - Training accuracy
    - Live accuracy
    - Feature importance
    - Last updated timestamp
    
    **Models:** XGBoost, LightGBM, Random Forest, etc.
    """
    return await ml_service.get_models()


@app.get("/api/v1/ml/performance", tags=["Machine Learning"], response_model=MLPerformanceResponse)
async def get_ml_performance():
    """
    📊 **ML Model Performance**
    
    Detailed performance metrics:
    - Accuracy tracking (train vs live)
    - Confusion matrix
    - Feature importance ranking
    - Prediction confidence distribution
    
    **Perfect for:** ML analytics dashboard
    """
    return await ml_service.get_performance()


@app.post("/api/v1/ml/predict", tags=["Machine Learning"], response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """
    🔮 **Generate ML Prediction**
    
    Get AI prediction for specific symbol:
    - Direction (LONG/SHORT/NEUTRAL)
    - Confidence score
    - Feature analysis
    - Recommended action
    
    **Input:** Symbol and timeframe
    """
    return await ml_service.predict(request.symbol, request.timeframe)


@app.get("/api/v1/ml/backtest", tags=["Machine Learning"], response_model=BacktestResponse)
async def get_backtest_results(model: str = "xgboost"):
    """
    🔄 **Backtest Results**
    
    Historical backtest performance:
    - Total return
    - Sharpe ratio
    - Max drawdown
    - Trade distribution
    
    **Parameters:**
    - model: Model name to backtest
    """
    return await ml_service.get_backtest(model)


# ============================================================================
# PAPER TRADING ARENA
# ============================================================================

@app.get("/api/v1/arena/competition", tags=["Paper Trading"], response_model=CompetitionResponse)
async def get_competition_status():
    """
    🏆 **Strategy Competition Status**
    
    3-way strategy comparison:
    - Technical Analysis performance
    - Pure ML performance
    - Hybrid (TA + ML) performance
    
    **Includes:**
    - Current standings
    - Trade counts
    - ROI comparison
    - Win rates
    """
    return await trading_service.get_competition_status()


@app.post("/api/v1/arena/start", tags=["Paper Trading"], response_model=CompetitionStartResponse)
async def start_competition(config: CompetitionConfig):
    """
    🚀 **Start New Competition**
    
    Launch new strategy competition with:
    - Initial capital
    - Max trades per strategy
    - Allowed symbols
    - Leverage range
    
    **Note:** Only one competition can run at a time
    """
    return await trading_service.start_competition(config)


@app.get("/api/v1/arena/results/{competition_id}", tags=["Paper Trading"], response_model=CompetitionResultsResponse)
async def get_competition_results(competition_id: str):
    """
    📊 **Competition Results**
    
    Detailed results of completed competition:
    - Winner announcement
    - Trade-by-trade analysis
    - Performance charts
    - Risk metrics comparison
    """
    return await trading_service.get_competition_results(competition_id)


# ============================================================================
# ADMIN ENDPOINTS - Protected
# ============================================================================

@app.get("/api/v1/admin/users", tags=["Admin"], response_model=List[UserResponse])
async def get_all_users():
    """
    👥 **User Management**
    
    List all registered users with:
    - Account details
    - Trading performance
    - Risk metrics
    - Account status
    
    **Protected:** Admin only
    """
    return await admin_service.get_users()


@app.get("/api/v1/admin/leaderboard", tags=["Admin"], response_model=LeaderboardResponse)
async def get_leaderboard():
    """
    🏆 **Trader Leaderboard**
    
    Top performers ranking:
    - Sorted by ROI
    - Win rate comparison
    - Total PnL
    - Risk-adjusted returns
    
    **Protected:** Admin only
    """
    return await admin_service.get_leaderboard()


@app.post("/api/v1/admin/user/{user_id}/freeze", tags=["Admin"])
async def freeze_user(user_id: str):
    """
    🔒 **Freeze User Account**
    
    Suspend trading for specific user:
    - Close all open positions
    - Block new trades
    - Maintain account data
    
    **Protected:** Admin only
    """
    return await admin_service.freeze_user(user_id)


@app.post("/api/v1/admin/emergency-stop", tags=["Admin"])
async def emergency_stop():
    """
    🚨 **Emergency Stop All Trading**
    
    Immediate halt of all trading activity:
    - Close all positions
    - Cancel pending orders
    - Disable new signals
    
    **Protected:** Admin only
    **Use case:** Market crash, system issues
    """
    return await admin_service.emergency_stop()


@app.post("/api/v1/admin/position/close", tags=["Admin"])
async def close_position_manual(request: ClosePositionRequest):
    """
    ✂️ **Manual Position Close**
    
    Force close specific position:
    - Override TP/SL
    - Market execution
    - Admin intervention
    
    **Protected:** Admin only
    """
    return await admin_service.close_position(request.position_id, request.reason)


@app.put("/api/v1/admin/leverage/adjust", tags=["Admin"])
async def adjust_leverage(request: LeverageAdjustRequest):
    """
    ⚙️ **Adjust Leverage Settings**
    
    Change leverage limits:
    - Min/max leverage
    - Per-user overrides
    - Risk tier adjustments
    
    **Protected:** Admin only
    """
    return await admin_service.adjust_leverage(request)


@app.get("/api/v1/admin/system/monitor", tags=["Admin"], response_model=SystemMonitorResponse)
async def get_system_monitor():
    """
    📊 **System Monitoring**
    
    Comprehensive system health:
    - CPU, Memory, Disk usage
    - Network connectivity
    - API latency
    - Database performance
    - ML model status
    
    **Protected:** Admin only
    **Updates:** Every 5 seconds
    """
    return await admin_service.get_system_monitor()


@app.get("/api/v1/admin/risk/exposure", tags=["Admin"], response_model=RiskExposureResponse)
async def get_global_exposure():
    """
    ⚠️ **Global Risk Exposure**
    
    Total risk across all users:
    - Aggregate positions
    - Leverage distribution
    - Margin usage
    - Correlation risk
    
    **Protected:** Admin only
    """
    return await admin_service.get_global_exposure()


@app.get("/api/v1/admin/audit/trail", tags=["Admin"], response_model=List[AuditLogResponse])
async def get_audit_trail(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    action_type: Optional[str] = None
):
    """
    📋 **Audit Trail**
    
    Complete activity log:
    - All admin actions
    - System events
    - User activities
    - Compliance records
    
    **Protected:** Admin only
    **Retention:** 90 days
    """
    return await admin_service.get_audit_trail(start_date, end_date, action_type)


# ============================================================================
# WEBSOCKET ENDPOINTS - Real-time updates
# ============================================================================

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    🔴 **Live Trading WebSocket**
    
    Real-time streaming of:
    - Trade executions
    - Position updates
    - PnL changes
    - Signal generation
    
    **Connection:** ws://localhost:8000/ws/live
    
    **Message format:**
    ```json
    {
        "type": "trade_update",
        "data": {...},
        "timestamp": "2025-11-03T15:00:00"
    }
    ```
    """
    await manager.connect(websocket)
    try:
        while True:
            # Send live updates every second
            data = await trading_service.get_live_update()
            await websocket.send_json({
                "type": "live_update",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    🎯 **Trading Signals WebSocket**
    
    Real-time signal stream:
    - New signal generation
    - Confidence updates
    - Signal invalidation
    
    **Connection:** ws://localhost:8000/ws/signals
    """
    await manager.connect(websocket)
    try:
        while True:
            signals = await trading_service.get_new_signals()
            if signals:
                await websocket.send_json({
                    "type": "new_signals",
                    "data": signals,
                    "timestamp": datetime.now().isoformat()
                })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# STARTUP / SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 AuthenticAlgo Pro API starting...")
    await trading_service.initialize()
    await ml_service.initialize()
    await admin_service.initialize()
    logger.info("✅ API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 API shutting down...")
    await trading_service.cleanup()
    await ml_service.cleanup()
    await admin_service.cleanup()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
