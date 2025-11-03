# 🤖 Bot Trading V2

Professional Python trading bot dengan arsitektur modular, machine learning integration, dan comprehensive risk management.

## 🌟 Features

### Core Features
- ✅ **Multi-Exchange Support**: Binance, AsterDEX
- ✅ **Real-time Data Streaming**: WebSocket & REST API
- ✅ **Technical Analysis Engine**: 30+ indicators
- ✅ **Multiple Trading Strategies**: Modular & customizable
- ✅ **Advanced Risk Management**: Stop-loss, trailing stop, position sizing
- ✅ **Paper Trading Mode**: Test strategies safely

### Advanced Features
- 🧠 **Machine Learning Integration**: XGBoost, LSTM models
- 📊 **Professional Backtesting**: Walk-forward, Monte Carlo simulation
- 📈 **Real-time Monitoring**: Telegram & Discord alerts
- 💾 **Data Management**: SQLite/PostgreSQL storage
- 🎯 **Portfolio Management**: Multi-asset, correlation analysis

## 📁 Project Structure

```
Bot Trading V2/
├── core/                   # Core utilities & base classes
│   ├── __init__.py
│   ├── config.py          # Configuration loader
│   ├── exceptions.py      # Custom exceptions
│   ├── logger.py          # Logging setup
│   └── utils.py           # Utility functions
│
├── data/                   # Data management
│   ├── __init__.py
│   ├── collector.py       # Data collection
│   ├── streamer.py        # Real-time streaming
│   ├── storage.py         # Database operations
│   └── preprocessor.py    # Data cleaning & preprocessing
│
├── indicators/             # Technical indicators
│   ├── __init__.py
│   ├── trend.py           # Trend indicators (MA, MACD, ADX)
│   ├── momentum.py        # Momentum indicators (RSI, Stochastic)
│   ├── volatility.py      # Volatility indicators (BB, ATR)
│   ├── volume.py          # Volume indicators (VWAP, OBV)
│   └── custom.py          # Custom indicators
│
├── strategies/             # Trading strategies
│   ├── __init__.py
│   ├── base_strategy.py   # Base strategy class
│   ├── rsi_macd.py        # RSI + MACD strategy
│   ├── bollinger.py       # Bollinger Bands strategy
│   └── ml_strategy.py     # ML-based strategy
│
├── execution/              # Order execution
│   ├── __init__.py
│   ├── exchange.py        # Exchange interface
│   ├── order_manager.py   # Order management
│   └── position_sizer.py  # Position sizing logic
│
├── ml/                     # Machine Learning
│   ├── __init__.py
│   ├── feature_engine.py  # Feature engineering
│   ├── model_trainer.py   # Model training
│   ├── predictor.py       # Real-time prediction
│   └── models/            # Saved models
│
├── risk/                   # Risk management
│   ├── __init__.py
│   ├── risk_manager.py    # Main risk manager
│   ├── stop_loss.py       # Stop-loss logic
│   └── portfolio.py       # Portfolio management
│
├── backtesting/            # Backtesting engine
│   ├── __init__.py
│   ├── backtest_engine.py # Main backtesting engine
│   ├── metrics.py         # Performance metrics
│   └── reports.py         # Report generation
│
├── monitoring/             # Monitoring & alerts
│   ├── __init__.py
│   ├── telegram_bot.py    # Telegram notifications
│   ├── discord_bot.py     # Discord notifications
│   └── dashboard.py       # Streamlit dashboard
│
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
│
├── tests/                  # Unit tests
│   ├── test_indicators.py
│   ├── test_strategies.py
│   └── test_risk.py
│
├── logs/                   # Log files
├── database/               # Database files
├── .env.example           # Environment variables template
├── .gitignore
├── requirements.txt       # Python dependencies
├── main.py               # Main entry point
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### 2. Installation

```powershell
# Clone or navigate to project directory
cd "C:\Users\Administrator\Documents\Bot Trading V2"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```powershell
# Copy environment template
Copy-Item .env.example .env

# Edit .env file with your API keys
notepad .env
```

**Important**: Update the following in `.env`:
- `BINANCE_API_KEY` and `BINANCE_API_SECRET`
- Set `TRADING_MODE=paper` for safe testing
- Configure notifications (optional)

### 4. Run the Bot

```powershell
# Activate virtual environment (if not already)
.\venv\Scripts\Activate.ps1

# Run in paper trading mode
python main.py --mode paper

# Run backtest
python main.py --mode backtest --start 2023-01-01 --end 2024-12-31

# Run live trading (be careful!)
python main.py --mode live
```

## 📊 Trading Strategies

### Available Strategies

1. **RSI + MACD Strategy** (`strategies/rsi_macd.py`)
   - Entry: RSI oversold + MACD bullish crossover
   - Exit: RSI overbought + MACD bearish crossover
   - Best for: Trending markets

2. **Bollinger Bands Strategy** (`strategies/bollinger.py`)
   - Entry: Price touches lower band + volume spike
   - Exit: Price reaches middle/upper band
   - Best for: Range-bound markets

3. **ML Strategy** (`strategies/ml_strategy.py`)
   - Uses machine learning models for prediction
   - Features: RSI, MACD, volume, volatility
   - Best for: Complex pattern recognition

### Creating Custom Strategy

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        
    def generate_signal(self, data):
        # Your strategy logic
        if condition_for_buy:
            return "BUY"
        elif condition_for_sell:
            return "SELL"
        return "HOLD"
```

## 🛡️ Risk Management

### Built-in Risk Controls
- **Position Sizing**: Fixed %, Kelly Criterion, Volatility-based
- **Stop Loss**: Percentage, ATR-based, Trailing
- **Max Drawdown Protection**: Auto-stop at threshold
- **Daily Loss Limit**: Circuit breaker
- **Correlation Check**: Avoid correlated positions

### Configuration (config.yaml)
```yaml
risk_management:
  max_position_size_pct: 10
  max_daily_loss_pct: 5
  stop_loss:
    enabled: true
    value: 2  # 2%
```

## 📈 Backtesting

```powershell
# Run backtest with specific strategy
python main.py --mode backtest --strategy RSI_MACD_Strategy --start 2023-01-01 --end 2024-12-31

# Generate performance report
python backtesting/generate_report.py --results results/backtest_20241103.json
```

### Metrics Provided
- Total Return, Annual Return
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown, Win Rate
- Profit Factor, Average Trade

## 📱 Monitoring & Alerts

### Telegram Setup
1. Create bot via [@BotFather](https://t.me/botfather)
2. Get bot token
3. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
4. Update `.env`:
   ```
   TELEGRAM_ENABLED=true
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

### Discord Setup
1. Create webhook in your Discord channel
2. Update `.env`:
   ```
   DISCORD_ENABLED=true
   DISCORD_WEBHOOK_URL=your_webhook_url
   ```

### Dashboard (Optional)
```powershell
# Run Streamlit dashboard
streamlit run monitoring/dashboard.py
```

## 🧪 Testing

```powershell
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_strategies.py -v

# Run with coverage
pytest --cov=. tests/
```

## 📦 Database

### SQLite (Default)
- Automatic setup
- File: `database/trading_bot.db`
- Good for: Development, small-scale

### PostgreSQL (Production)
```powershell
# Update .env
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_bot
DB_USER=your_user
DB_PASSWORD=your_password
```

## 🔧 Development

### Adding New Exchange
1. Create new file in `execution/exchanges/your_exchange.py`
2. Implement `BaseExchange` interface
3. Register in `execution/exchange.py`

### Adding New Indicator
1. Add to appropriate file in `indicators/`
2. Follow existing pattern
3. Add unit tests

### Code Style
- Follow PEP 8
- Use type hints
- Document all functions
- Keep functions < 50 lines

## ⚠️ Important Notes

### Security
- **NEVER** commit `.env` file
- **NEVER** share API keys
- Use testnet for development
- Start with paper trading

### Performance
- Use async for I/O operations
- Enable caching for repeated calculations
- Monitor memory usage with large datasets

### Risk Disclaimer
- This software is for educational purposes
- Trading carries risk of financial loss
- Always test strategies thoroughly
- Start with small amounts
- Never trade more than you can afford to lose

## 📝 Configuration

### Main Config (`config/config.yaml`)
- Exchange settings
- Strategy parameters
- Risk management rules
- Indicator settings

### Environment (`.env`)
- API keys
- Database credentials
- Notification tokens
- Trading mode

## 🤝 Contributing

1. Fork the project
2. Create feature branch
3. Write tests
4. Submit pull request

## 📄 License

MIT License - feel free to use for personal/commercial projects

## 📞 Support

- GitHub Issues: [Create an issue]
- Documentation: See `/docs` folder (coming soon)

## 🗺️ Roadmap

- [ ] Add more exchanges (Bybit, OKX)
- [ ] Implement advanced ML models (Transformers)
- [ ] Add sentiment analysis
- [ ] Create web UI
- [ ] Add multi-timeframe analysis
- [ ] Implement strategy optimization

## ✅ Checklist Before Live Trading

- [ ] Backtested strategy (>1 year data)
- [ ] Paper traded successfully (>1 month)
- [ ] Configured risk management
- [ ] Set up monitoring & alerts
- [ ] Tested with small amounts
- [ ] Understood all risks
- [ ] Have stop-loss rules
- [ ] Regular monitoring plan

---

**Built with ❤️ for smart trading**

**Remember**: Past performance does not guarantee future results. Trade responsibly! 🚀
