# 🚀 Quick Start Guide

## Setup dalam 5 Menit

### 1. Aktivasi Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Test Bot
```powershell
# Test run (akan menampilkan banner saja karena implementasi belum lengkap)
python main.py --mode paper
```

## ✅ Yang Sudah Dibuat

### Struktur Project
- ✅ **13 Folder Modular** - Terorganisir dengan sangat rapi
- ✅ **Core Module** - Config, Logger, Exceptions, Utils
- ✅ **Base Strategy Class** - Template untuk semua strategi
- ✅ **Configuration System** - YAML + Environment Variables
- ✅ **Main Entry Point** - CLI dengan argparse

### File-File Penting
- ✅ `main.py` - Entry point utama
- ✅ `requirements.txt` - Semua dependencies
- ✅ `config/config.yaml` - Konfigurasi lengkap
- ✅ `.env` - Environment variables (SUDAH ADA API KEYS!)
- ✅ `.gitignore` - Protect sensitive files
- ✅ `README.md` - Dokumentasi komprehensif

### Core Components
- ✅ **Config Management** - Load YAML + env variables
- ✅ **Logger** - Loguru dengan file rotation
- ✅ **Exception Handling** - Custom exceptions
- ✅ **Utilities** - Helper functions

## 📋 Next Steps - Yang Perlu Dibangun

### Phase 1: Data Module (PRIORITAS TINGGI)
```
data/
├── collector.py      # Ambil data dari exchange
├── streamer.py       # Real-time websocket
├── storage.py        # Save ke database
└── preprocessor.py   # Clean & prepare data
```

### Phase 2: Indicators (PRIORITAS TINGGI)
```
indicators/
├── trend.py          # MA, EMA, MACD, ADX
├── momentum.py       # RSI, Stochastic, CCI
├── volatility.py     # Bollinger Bands, ATR
├── volume.py         # VWAP, OBV
└── custom.py         # Custom kombinasi
```

### Phase 3: Execution Module
```
execution/
├── exchange.py       # CCXT wrapper
├── order_manager.py  # Handle orders
└── position_sizer.py # Calculate lot size
```

### Phase 4: Strategies
```
strategies/
├── rsi_macd.py       # RSI + MACD strategy
├── bollinger.py      # Bollinger Bands strategy
└── ml_strategy.py    # ML-based strategy
```

### Phase 5: Risk Management
```
risk/
├── risk_manager.py   # Main risk controller
├── stop_loss.py      # Stop loss logic
└── portfolio.py      # Portfolio management
```

### Phase 6: Backtesting
```
backtesting/
├── backtest_engine.py # Main backtesting
├── metrics.py         # Performance metrics
└── reports.py         # Generate reports
```

### Phase 7: Machine Learning
```
ml/
├── feature_engine.py  # Feature engineering
├── model_trainer.py   # Train models
└── predictor.py       # Real-time prediction
```

### Phase 8: Monitoring
```
monitoring/
├── telegram_bot.py    # Telegram alerts
├── discord_bot.py     # Discord alerts
└── dashboard.py       # Streamlit dashboard
```

## 🎯 Recommended Build Order

1. **Data Collector** - Tanpa data, ga bisa apa-apa
2. **Basic Indicators** - RSI, MACD, Bollinger Bands
3. **Exchange Interface** - Connect ke Binance
4. **Simple Strategy** - RSI + MACD
5. **Risk Manager** - Stop-loss, position sizing
6. **Backtest Engine** - Test strategy
7. **Paper Trading** - Live simulation
8. **Monitoring** - Telegram alerts
9. **ML Models** - Advanced features
10. **Live Trading** - Production ready!

## 💡 Tips untuk Development

### Clean Code Practices
```python
# ✅ GOOD - Type hints, docstring
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI indicator
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values
    """
    pass

# ❌ BAD - No types, no docs
def calc_rsi(p, per=14):
    pass
```

### Testing Pattern
```python
# Selalu test setiap function
def test_calculate_rsi():
    prices = pd.Series([100, 102, 101, 103, 105])
    rsi = calculate_rsi(prices)
    assert not rsi.empty
    assert rsi.iloc[-1] > 0
    assert rsi.iloc[-1] < 100
```

### Error Handling
```python
# Selalu handle errors dengan proper exceptions
try:
    data = exchange.fetch_ohlcv(symbol)
except ccxt.NetworkError as e:
    logger.error(f"Network error: {e}")
    raise ExchangeError(f"Failed to fetch data: {e}")
```

## 🔥 Quick Commands

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt

# Run paper trading
python main.py --mode paper

# Run backtest
python main.py --mode backtest --start 2023-01-01 --end 2024-12-31

# Run specific strategy
python main.py --mode paper --strategy RSI_MACD_Strategy

# Run tests
pytest tests/ -v

# Check code style
black . --check
flake8 .
```

## 📚 Learning Resources

### CCXT (Exchange Integration)
- Docs: https://docs.ccxt.com/
- Examples: `pip install ccxt` → check examples

### Pandas TA (Technical Analysis)
- Docs: https://github.com/twopirllc/pandas-ta
- Usage: `df.ta.rsi()`, `df.ta.macd()`

### Backtrader (Backtesting)
- Docs: https://www.backtrader.com/docu/
- Tutorial: Comprehensive backtesting framework

## 🎨 Project Philosophy

1. **Modular** - Setiap module independent
2. **Clean** - Code mudah dibaca dan maintain
3. **Tested** - Semua logic punya unit test
4. **Documented** - Setiap function ada docstring
5. **Type-Safe** - Gunakan type hints
6. **Async-First** - Untuk I/O operations
7. **Config-Driven** - Semua settings di config
8. **Secure** - Never hardcode secrets

## ⚠️ Important Reminders

- **NEVER commit .env file**
- **ALWAYS start with paper trading**
- **ALWAYS backtest before live**
- **ALWAYS use proper risk management**
- **NEVER risk more than you can afford to lose**

---

**Ready to build? Let's start with Data Collector! 🚀**
