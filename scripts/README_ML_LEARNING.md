# ML Continuous Learning - Multi-Coin Auto-Screening

## 🎯 Overview

Sistem ML training yang **OTOMATIS** screening ratusan coin dari AsterDEX, filter yang paling potensial, dan train ML models untuk prediksi trading.

### Key Features

✅ **Mass Screening** - Scan 100+ coins automatically  
✅ **Smart Filtering** - Rank berdasarkan volume, volatility, trend  
✅ **Multi-Model Training** - XGBoost, RandomForest, LightGBM  
✅ **Database Storage** - SQLite untuk tracking  
✅ **CSV Reports** - Easy analysis & monitoring  
✅ **Real Market Data** - Langsung dari AsterDEX API  

---

## 📦 Files

```
scripts/
├── ml_continuous_learner.py   # Main screening & training engine
├── run_learning.py             # Simple runner script
└── README_ML_LEARNING.md       # This file
```

---

## 🚀 Quick Start

### 1. Basic Usage (Default)

```bash
cd scripts
python run_learning.py
```

**Ini akan:**
- Screen ALL available coins dari AsterDEX
- Train ML models untuk top 15 coins
- Save results ke database & CSV

### 2. Screening Only (No Training)

```bash
python run_learning.py --screening-only
```

Hanya screening tanpa training. Bagus untuk explore market potential.

### 3. Train More Coins

```bash
python run_learning.py --top 25
```

Train top 25 coins instead of default 15.

### 4. Faster Screening

```bash
python run_learning.py --max-concurrent 25
```

Increase concurrent requests untuk screening lebih cepat.

---

## 📊 Output Files

### 1. Database: `database/ml_training_enhanced.db`

**Tables:**

**`coin_screening`** - Hasil screening semua coins
- symbol, price, volume_24h, price_change_24h
- volatility_score, volume_score, trend_score
- overall_score, screening_rank, recommendation

**`training_results`** - Hasil training ML models
- symbol, best_model, best_accuracy
- xgb_accuracy, rf_accuracy, lgb_accuracy
- data_points, coin_score

**`pattern_detection`** - Technical patterns detected
- symbol, pattern_type, confidence
- timeframe, description, potential_move

**`market_regime`** - Overall market conditions
- regime_type, btc_dominance, total_volume
- volatility_index, trending/ranging counts

### 2. CSV Reports: `Reports/ml_screening/`

**Screening Results:**
```
coin_screening_20251103_143025.csv
```
Contains: All screened coins dengan scores & recommendations

**Training Results:**
```
training_BTCUSDT_20251103_143530.csv
training_ETHUSDT_20251103_143545.csv
...
```
Individual training results per coin.

---

## 📈 Screening Metrics

### Overall Score (0-100)

Weighted combination of:
- **Volatility Score (30%)** - Optimal 2-10% range
- **Volume Score (30%)** - Higher volume = more liquidity
- **Trend Score (20%)** - Uptrend/downtrend strength
- **Pattern Score (20%)** - Technical patterns detected

### Recommendations

| Score | Recommendation | Meaning |
|-------|---------------|---------|
| 80-100 | STRONG_BUY | Excellent trading potential |
| 70-79 | BUY | Good opportunity |
| 60-69 | WATCH | Monitor for entry |
| 40-59 | NEUTRAL | No clear signal |
| 0-39 | AVOID | Poor conditions |

---

## 🎓 ML Training Process

### 1. Data Collection
- Fetch 1000 candles (1h timeframe = ~40 days)
- Minimum 500 candles required

### 2. Feature Engineering
- **Technical Indicators:**
  - SMA (5, 10, 20, 50, 200)
  - EMA (12, 26)
  - MACD (12, 26, 9)
  - RSI (14)
  - Stochastic RSI
  - Bollinger Bands (20, 2)
  - ATR (14)
  - Volume indicators

- **Target Variable:**
  - Binary: 1 = price up next candle, 0 = down

### 3. Model Training
- **XGBoost** - Gradient boosting
- **Random Forest** - Ensemble learning
- **LightGBM** - Fast gradient boosting

### 4. Evaluation
- 80/20 train-test split
- Accuracy metric
- Best model selection

---

## 📊 Example Output

```
🔍 MASS COIN SCREENING STARTED
================================================================================
📊 Screening 105 coins...
   Processing batch 1/7
   Processing batch 2/7
   ...
✅ Screening completed: 98/105 coins analyzed

📊 SCREENING SUMMARY REPORT
================================================================================

📈 Screening Statistics:
   Total coins analyzed: 98
   Average score: 62.3
   Best score: 87.5
   Worst score: 31.2

🎯 Recommendations:
   STRONG_BUY  :  12 coins
   BUY         :  23 coins
   WATCH       :  31 coins
   NEUTRAL     :  24 coins
   AVOID       :   8 coins

🏆 TOP 20 COINS:
--------------------------------------------------------------------------------
Rank | Symbol      | Score | Recommendation | Volatility | Trend
--------------------------------------------------------------------------------
   1 | BTCUSDT    |  87.5 | STRONG_BUY    |        6.2 |  0.72
   2 | ETHUSDT    |  84.3 | STRONG_BUY    |        5.8 |  0.68
   3 | SOLUSDT    |  81.7 | STRONG_BUY    |        7.1 |  0.65
   ...

🎓 SMART TRAINING SELECTION
================================================================================
🏆 Selected 15 coins for ML training:
    1. BTCUSDT     | Score:  87.5 | Rec: STRONG_BUY  | Vol:  6.2%
    2. ETHUSDT     | Score:  84.3 | Rec: STRONG_BUY  | Vol:  5.8%
   ...

🎓 Training ML models for BTCUSDT...
   Training XGBoost...
   Training RandomForest...
   Training LightGBM...
   ✅ Best: xgboost (78.3%)

📊 TRAINING SUMMARY
================================================================================
✅ Trained 15 coins successfully
   Average accuracy: 76.8%
   Best accuracy: 82.1%
   Worst accuracy: 68.5%

🏆 Best Model Distribution:
   xgboost        :   8 times
   lightgbm       :   5 times
   random_forest  :   2 times

🌟 Top 5 Best Trained Coins:
   1. ETHUSDT    - xgboost    - 82.1%
   2. BTCUSDT    - xgboost    - 78.3%
   3. SOLUSDT    - lightgbm   - 77.9%
   ...
```

---

## ⚙️ Configuration

### Screening Settings

Edit `ml_continuous_learner.py`:

```python
# Screening batch size
max_concurrent = 15  # Concurrent API calls

# Minimum requirements
min_score = 65.0     # Minimum score untuk training
top_n = 15           # Top N coins to train

# Data requirements
min_candles = 500    # Minimum candles for training
data_limit = 1000    # Candles to fetch
```

### Model Training

```python
# Train-test split
train_ratio = 0.8    # 80% training, 20% testing

# Models
models = ['xgboost', 'random_forest', 'lightgbm']
```

---

## 🔧 Troubleshooting

### "No symbols found"

**Problem:** API tidak return symbols  
**Solution:**
- Check AsterDEX API credentials (.env)
- Verify API endpoint available
- Fallback symbols akan digunakan

### "Insufficient data"

**Problem:** Coin tidak punya cukup historical data  
**Solution:**
- Skip coin tersebut
- Lower `min_candles` requirement
- Try different timeframe

### "Failed to screen coin"

**Problem:** Individual coin screening failed  
**Solution:**
- Normal, bisa karena delisted/suspended
- Script akan continue dengan coins lain
- Check logs untuk details

### Rate Limit Errors

**Problem:** Too many API requests  
**Solution:**
- Reduce `max_concurrent`
- Increase delay between batches
- Check AsterDEX rate limits

---

## 📚 Advanced Usage

### Custom Symbol List

Edit `ml_continuous_learner.py`:

```python
async def get_all_symbols(self):
    # Your custom symbols
    return ["BTCUSDT", "ETHUSDT", "YOUR_FAVORITE_COINS"]
```

### Custom Scoring

Edit `_calculate_overall_score()`:

```python
# Adjust weights
volatility_weight = 0.3  # 30%
volume_weight = 0.3      # 30%
trend_weight = 0.2       # 20%
pattern_weight = 0.2     # 20%
```

### Add More Indicators

Edit `_calculate_indicators_for_training()`:

```python
# Add your indicators
df['your_indicator'] = calculate_your_indicator(df)
```

---

## 💡 Best Practices

### 1. Run Daily
```bash
# Cron job example (Linux)
0 0 * * * cd /path/to/bot && python scripts/run_learning.py
```

Market conditions change, re-train regularly!

### 2. Start Small
```bash
# Test dengan top 5 dulu
python run_learning.py --top 5
```

### 3. Monitor Results

Check database untuk trend:
```sql
SELECT 
    symbol,
    best_accuracy,
    coin_score,
    timestamp
FROM training_results
ORDER BY timestamp DESC
LIMIT 20;
```

### 4. Compare Models

Lihat model mana yang paling sering menang:
```sql
SELECT 
    best_model,
    COUNT(*) as wins,
    AVG(best_accuracy) as avg_accuracy
FROM training_results
GROUP BY best_model
ORDER BY wins DESC;
```

---

## 📞 Support

**Issues?**
- Check logs: `logs/trading_bot.log`
- Verify API credentials: `.env`
- Test single coin first

**Questions?**
- Read this README carefully
- Check example outputs
- Review code comments

---

## 🎯 Next Steps

1. ✅ Run initial screening
2. ✅ Review top coins
3. ✅ Train ML models
4. ✅ Check accuracy results
5. 🚀 Integrate best models into trading bot
6. 📊 Monitor performance
7. 🔄 Re-train regularly

---

**Happy Trading! 🚀📈**
