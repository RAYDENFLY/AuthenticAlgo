# 🚀 AuthenticAlgo Pro - Trading API

Professional AI-powered trading bot backend API with complete Swagger documentation.

## 🎯 Features

### Public Endpoints
- **Live Performance Stats** - Real-time PnL, win rate, ROI
- **Trading Signals Feed** - AI-generated signals with TP/SL levels
- **Equity Curve Charts** - Portfolio growth visualization
- **Daily Reports** - Comprehensive trading summaries

### Trading Dashboard
- **Current Positions** - Real-time open positions with unrealized PnL
- **Trade History** - Paginated historical trades
- **Portfolio Breakdown** - Asset allocation and risk exposure

### Machine Learning
- **Model Information** - All available ML models and accuracy
- **Performance Metrics** - Confusion matrix, feature importance
- **Live Predictions** - Get AI predictions for any symbol
- **Backtest Results** - Historical performance analysis

### Paper Trading Arena
- **Strategy Competition** - Compare TA vs ML vs Hybrid
- **Start Competition** - Launch new strategy battles
- **Results Analysis** - Detailed competition outcomes

### Admin Panel (Protected)
- **User Management** - Multi-user accounts and permissions
- **Leaderboard** - Top trader rankings
- **Emergency Controls** - Stop trading, close positions
- **System Monitoring** - CPU, memory, network stats
- **Risk Management** - Global exposure and leverage control
- **Audit Trail** - Complete activity logging

### Real-time WebSocket
- **Live Trading Updates** - Trade executions, PnL changes
- **Signal Stream** - New signals as they're generated

## 📖 Documentation

### Swagger UI (Interactive)
```
http://localhost:8000/docs
```

### ReDoc (Alternative)
```
http://localhost:8000/redoc
```

### OpenAPI JSON
```
http://localhost:8000/openapi.json
```

## 🚦 Quick Start

### 1. Install Dependencies
```bash
cd api
pip install -r requirements.txt
```

### 2. Start API Server
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access Swagger
Open browser: http://localhost:8000/docs

## 📡 API Endpoints Overview

### Health Check
```http
GET /api/v1/health
```

### Public Stats
```http
GET /api/v1/stats/public
```

### Recent Signals
```http
GET /api/v1/signals/recent?limit=10
```

### Current Positions
```http
GET /api/v1/positions/current
```

### ML Prediction
```http
POST /api/v1/ml/predict
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "timeframe": "1h"
}
```

### Competition Status
```http
GET /api/v1/arena/competition
```

### WebSocket - Live Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Live update:', data);
};
```

## 🔐 Authentication (Future)

Currently all endpoints are open. To add authentication:

1. Add JWT token generation
2. Protect admin routes with `Depends(get_current_user)`
3. Use Bearer token in headers:

```http
Authorization: Bearer <your_token_here>
```

## 📊 Response Examples

### Public Stats Response
```json
{
  "current_pnl": 45.23,
  "total_trades": 156,
  "win_rate": 85.5,
  "roi": 452.3,
  "active_positions": 3,
  "initial_capital": 10.0,
  "current_capital": 55.23
}
```

### Trading Signal Response
```json
{
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
```

### Position Response
```json
{
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
```

## 🔌 Integration with Trading Bot

### Connect API to Bot
```python
from api.services import TradingService

trading_service = TradingService()

# Get live stats
stats = await trading_service.get_public_stats()

# Get current positions
positions = await trading_service.get_current_positions()

# Get recent signals
signals = await trading_service.get_recent_signals(limit=10)
```

### Database Integration (TODO)
Currently using mock data. To connect to actual trading bot:

1. Update `services.py` to read from database
2. Connect to AsterDEX competition results
3. Load ML model predictions from `ml/` module
4. Stream live data via WebSocket

## 📈 Rate Limiting

- **Public endpoints**: 100 requests/minute
- **Admin endpoints**: 1000 requests/minute
- **WebSocket**: Unlimited (1 update/second)

## 🛠️ Development

### Project Structure
```
api/
├── main.py           # FastAPI app & routes
├── models.py         # Pydantic schemas
├── services.py       # Business logic
├── requirements.txt  # Dependencies
└── README.md         # This file
```

### Adding New Endpoints

1. Define Pydantic models in `models.py`:
```python
class NewFeatureResponse(BaseModel):
    field1: str
    field2: float
```

2. Add service method in `services.py`:
```python
async def get_new_feature(self) -> NewFeatureResponse:
    return NewFeatureResponse(field1="value", field2=123.45)
```

3. Create route in `main.py`:
```python
@app.get("/api/v1/new-feature", tags=["Feature"])
async def new_feature():
    """Feature description"""
    return await trading_service.get_new_feature()
```

### Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## 🚀 Deployment

### Docker
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t authenticalgo-api .
docker run -p 8000:8000 authenticalgo-api
```

### Production Settings
```python
# Use production ASGI server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Or with Gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🔒 Security Checklist

- [ ] Add JWT authentication
- [ ] Enable HTTPS only
- [ ] Set proper CORS origins
- [ ] Add rate limiting (Redis)
- [ ] Implement API keys for admin
- [ ] Add request validation
- [ ] Enable audit logging
- [ ] Use environment variables for secrets

## 📞 Support

- **Email**: support@authenticalgo.com
- **Documentation**: http://localhost:8000/docs
- **Issues**: GitHub Issues

## 📄 License

Proprietary - AuthenticAlgo Pro
