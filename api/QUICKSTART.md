# 🚀 API Quick Start Guide

## 1. Start API Server

```bash
cd api
python main.py
```

Output:
```
✅ API ready!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 2. Open Swagger Documentation

**Browser:** http://localhost:8000/docs

You'll see interactive API documentation with all endpoints!

## 3. Test Endpoints

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Public Stats
```bash
curl http://localhost:8000/api/v1/stats/public
```

### Recent Signals
```bash
curl http://localhost:8000/api/v1/signals/recent?limit=5
```

### Current Positions
```bash
curl http://localhost:8000/api/v1/positions/current
```

### ML Prediction
```bash
curl -X POST http://localhost:8000/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "1h"}'
```

### Competition Status
```bash
curl http://localhost:8000/api/v1/arena/competition
```

## 4. WebSocket Test (JavaScript)

```javascript
// Live trading updates
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onopen = () => {
  console.log('✅ Connected to live feed');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('📊 Live update:', data);
};

ws.onerror = (error) => {
  console.error('❌ WebSocket error:', error);
};
```

## 5. Frontend Integration Example

### React
```javascript
import { useState, useEffect } from 'react';

function TradingDashboard() {
  const [stats, setStats] = useState(null);
  const [positions, setPositions] = useState([]);

  useEffect(() => {
    // Fetch public stats
    fetch('http://localhost:8000/api/v1/stats/public')
      .then(res => res.json())
      .then(data => setStats(data));

    // Fetch current positions
    fetch('http://localhost:8000/api/v1/positions/current')
      .then(res => res.json())
      .then(data => setPositions(data));

    // WebSocket for live updates
    const ws = new WebSocket('ws://localhost:8000/ws/live');
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      // Update UI with live data
      console.log('Live:', update);
    };

    return () => ws.close();
  }, []);

  return (
    <div>
      <h1>Trading Dashboard</h1>
      
      {stats && (
        <div>
          <h2>Current PnL: ${stats.current_pnl}</h2>
          <p>Win Rate: {stats.win_rate}%</p>
          <p>ROI: {stats.roi}%</p>
        </div>
      )}

      <h3>Open Positions: {positions.length}</h3>
      {positions.map(pos => (
        <div key={pos.position_id}>
          <strong>{pos.symbol}</strong>: {pos.direction}
          <br />
          Entry: ${pos.entry_price} | Current: ${pos.current_price}
          <br />
          Unrealized PnL: ${pos.unrealized_pnl} ({pos.unrealized_pnl_pct}%)
        </div>
      ))}
    </div>
  );
}
```

### Vue.js
```javascript
<template>
  <div>
    <h1>Trading Dashboard</h1>
    
    <div v-if="stats">
      <h2>Current PnL: ${{ stats.current_pnl }}</h2>
      <p>Win Rate: {{ stats.win_rate }}%</p>
      <p>ROI: {{ stats.roi }}%</p>
    </div>

    <h3>Open Positions: {{ positions.length }}</h3>
    <div v-for="pos in positions" :key="pos.position_id">
      <strong>{{ pos.symbol }}</strong>: {{ pos.direction }}
      <br />
      Entry: ${{ pos.entry_price }} | Current: ${{ pos.current_price }}
      <br />
      Unrealized PnL: ${{ pos.unrealized_pnl }} ({{ pos.unrealized_pnl_pct }}%)
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      stats: null,
      positions: [],
      ws: null
    };
  },
  
  async mounted() {
    // Fetch public stats
    const statsRes = await fetch('http://localhost:8000/api/v1/stats/public');
    this.stats = await statsRes.json();

    // Fetch positions
    const posRes = await fetch('http://localhost:8000/api/v1/positions/current');
    this.positions = await posRes.json();

    // WebSocket live updates
    this.ws = new WebSocket('ws://localhost:8000/ws/live');
    this.ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      console.log('Live:', update);
      // Update reactive data here
    };
  },
  
  beforeUnmount() {
    if (this.ws) this.ws.close();
  }
};
</script>
```

## 6. Python Client Example

```python
import requests
import asyncio
import websockets
import json

# Base URL
BASE_URL = "http://localhost:8000"

def get_public_stats():
    """Get public trading stats"""
    response = requests.get(f"{BASE_URL}/api/v1/stats/public")
    return response.json()

def get_recent_signals(limit=10):
    """Get recent trading signals"""
    response = requests.get(f"{BASE_URL}/api/v1/signals/recent", params={"limit": limit})
    return response.json()

def get_current_positions():
    """Get current open positions"""
    response = requests.get(f"{BASE_URL}/api/v1/positions/current")
    return response.json()

def get_ml_prediction(symbol, timeframe="1h"):
    """Get ML prediction for symbol"""
    response = requests.post(
        f"{BASE_URL}/api/v1/ml/predict",
        json={"symbol": symbol, "timeframe": timeframe}
    )
    return response.json()

async def listen_live_updates():
    """Listen to live trading updates via WebSocket"""
    uri = "ws://localhost:8000/ws/live"
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to live feed")
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📊 Live update: {data}")

# Example usage
if __name__ == "__main__":
    print("📊 Public Stats:")
    stats = get_public_stats()
    print(f"  PnL: ${stats['current_pnl']}")
    print(f"  Win Rate: {stats['win_rate']}%")
    print(f"  ROI: {stats['roi']}%")
    
    print("\n🎯 Recent Signals:")
    signals = get_recent_signals(limit=3)
    for signal in signals:
        print(f"  {signal['symbol']}: {signal['direction']} @ ${signal['entry_price']}")
    
    print("\n💼 Current Positions:")
    positions = get_current_positions()
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['direction']} | PnL: ${pos['unrealized_pnl']}")
    
    print("\n🔮 ML Prediction:")
    prediction = get_ml_prediction("BTCUSDT", "1h")
    print(f"  {prediction['symbol']}: {prediction['direction']} ({prediction['confidence']}%)")
    
    print("\n🔴 Starting live feed (Press Ctrl+C to stop)...")
    asyncio.run(listen_live_updates())
```

## 7. Postman Collection

Import this JSON into Postman:

```json
{
  "info": {
    "name": "AuthenticAlgo Pro API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/api/v1/health"
      }
    },
    {
      "name": "Public Stats",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/api/v1/stats/public"
      }
    },
    {
      "name": "Recent Signals",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/api/v1/signals/recent?limit=10"
      }
    },
    {
      "name": "ML Prediction",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"symbol\": \"BTCUSDT\",\n  \"timeframe\": \"1h\"\n}"
        },
        "url": "http://localhost:8000/api/v1/ml/predict"
      }
    }
  ]
}
```

## 8. Common Issues

### Port Already in Use
```bash
# Kill process on port 8000 (Windows PowerShell)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force

# Or use different port
uvicorn main:app --port 8001
```

### CORS Errors
API already has CORS enabled for all origins. For production, update `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### WebSocket Connection Failed
Make sure using `ws://` (not `wss://`) for local development.

## 9. Next Steps

1. ✅ API is running with Swagger docs
2. 🔄 Connect to actual trading bot data (update `services.py`)
3. 🔄 Add database connection (PostgreSQL/SQLite)
4. 🔄 Implement authentication (JWT tokens)
5. 🔄 Add rate limiting (Redis)
6. 🔄 Deploy to production (Docker/Cloud)

## 10. Production Deployment

### Environment Variables
Create `.env` file:
```env
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
JWT_SECRET=your_secret_key_here
CORS_ORIGINS=https://yourdomain.com
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

Run:
```bash
docker-compose up -d
```

---

**🎉 That's it! Your API is ready to power your trading dashboard!**

For full documentation, visit: http://localhost:8000/docs
