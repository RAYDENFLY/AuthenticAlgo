# ML Continuous Learning - Quick Reference

## 🚀 Quick Commands

### Basic Run (Recommended)
```bash
cd scripts
python run_learning.py
```
**Output:** Screen ALL coins + Train top 15

### Screening Only
```bash
python run_learning.py --screening-only
```
**Output:** Screening report tanpa training

### Train More Coins
```bash
python run_learning.py --top 25
```
**Output:** Train top 25 coins

### Faster Screening
```bash
python run_learning.py --max-concurrent 25
```
**Output:** 25 concurrent requests

---

## 📊 What It Does

1. **Fetch ALL symbols** dari AsterDEX (100+ coins)
2. **Screen each coin** - Volume, volatility, trend analysis
3. **Rank by score** (0-100)
4. **Train ML models** on top performers
5. **Save to database** + CSV reports

---

## 📁 Output Locations

### Database
```
database/ml_training_enhanced.db
```
**Tables:**
- `coin_screening` - All screening results
- `training_results` - ML training results
- `pattern_detection` - Technical patterns
- `market_regime` - Market conditions

### CSV Reports
```
Reports/ml_screening/
├── coin_screening_20251103_143025.csv
├── training_BTCUSDT_20251103_143530.csv
├── training_ETHUSDT_20251103_143545.csv
└── ...
```

---

## 🏆 Scoring System

| Score | Recommendation | Action |
|-------|---------------|--------|
| 80-100 | STRONG_BUY | High potential, prioritize |
| 70-79 | BUY | Good opportunity |
| 60-69 | WATCH | Monitor for entry |
| 40-59 | NEUTRAL | No clear edge |
| 0-39 | AVOID | Poor conditions |

**Score Components:**
- Volatility (30%) - Optimal 2-10% range
- Volume (30%) - Liquidity measure
- Trend (20%) - Direction strength
- Patterns (20%) - Technical setups

---

## 🎓 ML Models Trained

1. **XGBoost** - Gradient boosting (usually best)
2. **Random Forest** - Ensemble learning
3. **LightGBM** - Fast gradient boosting

**Target:** Predict next candle direction (UP/DOWN)  
**Accuracy:** Typically 65-85%  
**Data:** 1000 candles (1h = ~40 days)

---

## ⏱️ Estimated Time

- **Screening:** 3-5 minutes (100 coins)
- **Training per coin:** 20-30 seconds
- **Total (15 coins):** 8-12 minutes

---

## 💡 Pro Tips

### 1. Run Daily
Market changes → Models need retraining!

### 2. Start Small
```bash
python run_learning.py --top 5
```
Test system dengan 5 coins dulu

### 3. Check Results
```bash
# View screening summary
sqlite3 database/ml_training_enhanced.db
> SELECT * FROM coin_screening ORDER BY overall_score DESC LIMIT 10;
```

### 4. Monitor Accuracy
```bash
# View training results
> SELECT symbol, best_model, best_accuracy 
  FROM training_results 
  ORDER BY best_accuracy DESC 
  LIMIT 10;
```

---

## 🔥 Expected Results

### Screening
```
✅ 98/105 coins analyzed
   12 STRONG_BUY
   23 BUY
   31 WATCH
```

### Training
```
✅ 15 coins trained
   Average accuracy: 76.8%
   Best: ETHUSDT (82.1%)
   Models: 8x XGBoost, 5x LightGBM, 2x RF
```

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| "No symbols found" | Check .env API credentials |
| "Insufficient data" | Normal, coin skipped |
| "Rate limit" | Reduce --max-concurrent |
| "Import error" | `pip install -r requirements.txt` |

---

## 📞 Need Help?

1. Read **README_ML_LEARNING.md** (full docs)
2. Check logs: `logs/trading_bot.log`
3. Test with `--screening-only` first
4. Verify .env has AsterDEX credentials

---

## 🎯 Next Steps

After running:

1. ✅ Check `coin_screening` table - Which coins scored highest?
2. ✅ Review training accuracy - Which models performed best?
3. ✅ Integrate top models into your trading strategies
4. ✅ Monitor live performance
5. 🔄 Re-run weekly untuk updates

---

**Remember:** 
- Higher score ≠ guaranteed profit
- ML predictions are probabilistic
- Always use risk management
- Paper trade first!

---

## 🚀 Integration with Bot

```python
# Use trained models in your strategy
from ml.predictor import MLPredictor

predictor = MLPredictor()
prediction = await predictor.predict(symbol="BTCUSDT")

if prediction == 1 and score > 0.7:  # High confidence UP
    # Enter LONG position
    pass
```

See `ml/predictor.py` for details.

---

**Happy Trading! 📈🤖**
