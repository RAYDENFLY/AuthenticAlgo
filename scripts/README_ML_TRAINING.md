# ML Continuous Learning & Competition Suite

## 📚 Overview

Suite lengkap untuk training ML models dengan real market data dan testing 3 strategy secara bersamaan.

## 🎯 Features

### 1. **3-Way Strategy Competition** (`demo/AsterDEX/run_competition.py`)
- Test 3 strategy sekaligus: Technical, ML, Hybrid
- Real-time AsterDEX market data
- 10 trades per strategy dengan $10 capital
- Automatic leverage optimization (5x-125x)
- Comprehensive comparison reports (JSON + Markdown)

### 2. **ML Continuous Learning** (`scripts/ml_continuous_learning.py`)
- Fetch real-time market data dari AsterDEX
- Train multiple ML models (XGBoost, RandomForest, LightGBM)
- Calculate 18+ technical indicators
- Save training history ke SQLite database
- Generate CSV reports untuk tracking
- Analyze model improvement over time

### 3. **Run All Training** (`scripts/run_all_training.py`)
- Run semua script sekaligus (parallel execution)
- Monitor semua process secara real-time
- Automatic error handling

## 🚀 Quick Start

### Option 1: Run Competition Only (Test 3 Strategy)
```bash
# Langsung test 3 strategy sekaligus
python demo/AsterDEX/run_competition.py
```

**Output:**
- Console logs dengan ranking real-time
- JSON report: `Reports/benchmark/AsterDEX/competition_results_YYYYMMDD_HHMMSS.json`
- Markdown report: `Reports/benchmark/AsterDEX/competition_report_YYYYMMDD_HHMMSS.md`

### Option 2: Run ML Training Only
```bash
# Train ML models dengan real market data
python scripts/ml_continuous_learning.py
```

**Output:**
- SQLite database: `database/ml_training.db` (persistent history)
- CSV reports: `Reports/ml_training/training_report_SYMBOL_YYYYMMDD_HHMMSS.csv`
- Summary: `Reports/ml_training/training_summary_SYMBOL.csv`

### Option 3: Run Everything (Recommended)
```bash
# Run semua script sekaligus untuk maximum efficiency
python scripts/run_all_training.py
```

**Output:**
- Semua output dari Option 1 & 2
- Real-time monitoring di console
- Parallel execution untuk speed

## 📊 Database Schema

### SQLite Database: `database/ml_training.db`

#### Table: `training_sessions`
```sql
- id: Primary key
- timestamp: Training time
- symbol: Trading pair
- timeframe: Candle interval
- data_points: Number of samples
- training_duration: Total time (seconds)
- models_trained: Number of models
- status: completed/failed
```

#### Table: `model_performance`
```sql
- id: Primary key
- session_id: Foreign key
- model_name: XGBoost/RandomForest/LightGBM
- model_type: Classifier type
- accuracy: 0-100%
- precision_score: 0-100%
- recall: 0-100%
- f1_score: 0-100%
- training_time: Seconds
- feature_count: Number of features
- timestamp: Model trained time
```

#### Table: `market_conditions`
```sql
- session_id: Foreign key
- symbol: Trading pair
- price: Current price
- volume: Trading volume
- volatility: ATR value
- trend: bullish/bearish
- rsi: RSI value
- macd: MACD value
```

## 📈 CSV Reports

### Training Report Format
```csv
model_name,model_type,accuracy,precision,recall,f1_score,training_time,feature_count,timestamp,symbol
XGBoost Classifier,xgboost,96.50,94.20,95.80,95.00,2.34,18,2025-11-03T10:30:00,BTCUSDT
Random Forest,random_forest,94.20,92.10,93.50,92.80,3.12,18,2025-11-03T10:30:00,BTCUSDT
LightGBM,lightgbm,95.80,93.50,94.20,93.85,1.89,18,2025-11-03T10:30:00,BTCUSDT
```

### Training Summary Format
```csv
timestamp,symbol,models_trained,best_accuracy,avg_accuracy,total_samples
2025-11-03T10:30:00,BTCUSDT,3,96.50,95.50,800
2025-11-03T11:45:00,BTCUSDT,3,97.20,96.10,800
2025-11-03T13:00:00,BTCUSDT,3,96.80,95.90,800
```

## 🔧 Technical Indicators Used

ML training menggunakan 18+ indicators:

**Trend Indicators:**
- SMA (20, 50)
- EMA (12, 26)
- MACD (12, 26, 9)

**Momentum Indicators:**
- RSI (14)
- Stochastic RSI (K, D)

**Volatility Indicators:**
- Bollinger Bands (upper, middle, lower, width)
- ATR (14)

**Volume Indicators:**
- Volume SMA (20)
- Volume Ratio

**Price Action:**
- Price Change (%)
- High-Low Ratio

## 📊 Analysis Examples

### Check Training History
```python
from scripts.ml_continuous_learning import MLContinuousLearner

learner = MLContinuousLearner()
history = learner.get_training_history(limit=10)
print(history)
```

### Query Database Directly
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('database/ml_training.db')

# Get best models
query = """
SELECT model_name, AVG(accuracy) as avg_acc, MAX(accuracy) as max_acc
FROM model_performance
GROUP BY model_name
ORDER BY avg_acc DESC
"""

df = pd.read_sql_query(query, conn)
print(df)
```

### Analyze CSV Reports
```python
import pandas as pd

# Load all training summaries
df = pd.read_csv('Reports/ml_training/training_summary_BTCUSDT.csv')

# Plot improvement over time
import matplotlib.pyplot as plt

df['timestamp'] = pd.to_datetime(df['timestamp'])
plt.plot(df['timestamp'], df['best_accuracy'])
plt.title('ML Model Improvement Over Time')
plt.xlabel('Time')
plt.ylabel('Best Accuracy (%)')
plt.show()
```

## 🎮 Usage Examples

### 1. Single Training Session
```bash
# Train on BTC with 1000 candles
python scripts/ml_continuous_learning.py
```

### 2. Custom Symbol Training
Edit `ml_continuous_learning.py`:
```python
# In main() function
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]  # Add more
```

### 3. Scheduled Training (Cron/Task Scheduler)
```bash
# Run training every 6 hours
0 */6 * * * cd /path/to/Bot\ Trading\ V2 && python scripts/ml_continuous_learning.py
```

### 4. Competition Testing
```bash
# Quick test
python demo/AsterDEX/run_competition.py

# Check results
cat Reports/benchmark/AsterDEX/competition_report_*.md
```

## 📁 File Structure

```
Bot Trading V2/
├── demo/
│   └── AsterDEX/
│       ├── run_competition.py          # 3-way competition
│       ├── trader_technical.py
│       ├── trader_ml.py
│       └── trader_hybrid.py
├── scripts/
│   ├── ml_continuous_learning.py       # ML training script
│   ├── run_all_training.py             # Run everything
│   └── README_ML_TRAINING.md           # This file
├── database/
│   └── ml_training.db                  # Training history (SQLite)
└── Reports/
    ├── benchmark/
    │   └── AsterDEX/                   # Competition results
    │       ├── competition_results_*.json
    │       └── competition_report_*.md
    └── ml_training/                     # ML training reports
        ├── training_report_*.csv
        └── training_summary_*.csv
```

## 🔍 Monitoring & Debugging

### Check Database Size
```bash
ls -lh database/ml_training.db
```

### View Recent Training Logs
```bash
tail -f logs/trading_bot.log
```

### Check CSV Reports
```bash
# Latest training report
ls -lt Reports/ml_training/training_report_*.csv | head -1

# View summary
cat Reports/ml_training/training_summary_BTCUSDT.csv
```

## 💡 Tips & Best Practices

1. **Regular Training**: Run training setiap 6-12 jam untuk keep models updated
2. **Multiple Symbols**: Train pada berbagai pairs untuk robustness
3. **Data Size**: 1000 candles optimal (tidak terlalu besar, tidak terlalu kecil)
4. **Check Reports**: Review CSV summaries regularly untuk track improvement
5. **Database Backup**: Backup `ml_training.db` secara berkala
6. **Competition Testing**: Run competition setelah major training untuk validate

## 🐛 Troubleshooting

### Error: "No data fetched"
```bash
# Check AsterDEX connection
python -c "from data.asterdex_collector import AsterDEXCollector; import asyncio; asyncio.run(AsterDEXCollector().test_connection())"
```

### Error: "Database locked"
```bash
# Close all processes using the database
pkill -f ml_continuous_learning.py
```

### Error: "Model training failed"
```bash
# Check if all dependencies installed
pip install xgboost lightgbm scikit-learn pandas numpy
```

## 📞 Support

Jika ada issues:
1. Check logs: `logs/trading_bot.log`
2. Verify database: `sqlite3 database/ml_training.db ".schema"`
3. Test AsterDEX connection
4. Check CSV reports for errors

## 🎉 Success Metrics

**Good Results:**
- Competition ROI: > 20%
- ML Accuracy: > 90%
- Win Rate: > 65%

**Excellent Results:**
- Competition ROI: > 50%
- ML Accuracy: > 95%
- Win Rate: > 70%

Happy Trading! 🚀
