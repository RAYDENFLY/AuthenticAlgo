"""
V6 Paper Trading Demo
- Trains V6 model (quick train using scripts/quantum_ml_trainer_v6_0.py internals)
- Runs up to 10 simulated paper trades on AsterDEX using live data
- Starts with $5 account balance; stops if balance exhausted
- Uses screening + entry + TP/SL rules from strategy markdowns (simplified)
- Saves a PDF summary report to results/v6_demo_report.pdf

Notes:
- This demo uses the exchange adapter in testnet mode.
- For speed and safety this demo uses market orders and simplified sizing.
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Add workspace root to path (if needed)
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.quantum_ml_trainer_v6_0 import QuantumMLTrainerV60
from execution.asterdex import AsterDEXFutures
from execution.position_sizer import PositionSizer, SizingMethod
from core.logger import setup_logger

logger = setup_logger()

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Simplified screening: pick top 1 coin by model confidence
SCREEN_COINS = ['BTCUSDT', 'ETHUSDT', 'TRUMPUSDT']
MAX_TRADES = 10
START_BALANCE = 5.0  # $5 starting balance

# Simplified entry/confidence thresholds
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MED = 0.75

# TP/SL params (from entrysslStrategy)
TP_SL_PARAMS = {
    'high': {'tp1': 0.02, 'tp2': 0.04, 'sl': 0.015, 'trailing_activation': 0.035, 'trailing_distance': 0.01},
    'medium': {'tp1': 0.015, 'tp2': 0.03, 'sl': 0.01, 'trailing_activation': 0.025, 'trailing_distance': 0.008},
    'low': {'tp1': 0.01, 'tp2': 0.02, 'sl': 0.008, 'trailing_activation': 0.02, 'trailing_distance': 0.006}
}

class TradeRecord:
    def __init__(self, symbol, side, entry_price, size, tp_sl_params, confidence):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.size = size
        self.tp_sl = tp_sl_params
        self.confidence = confidence
        self.status = 'OPEN'
        self.exit_price = None
        self.pnl = 0.0
        self.timestamps = {'entry': datetime.now(), 'exit': None}

    def close(self, exit_price):
        self.exit_price = exit_price
        # compute pnl (long only simplified)
        self.pnl = (exit_price - self.entry_price) / self.entry_price * self.size
        self.status = 'CLOSED'
        self.timestamps['exit'] = datetime.now()


async def run_demo():
    # 1. Train V6 quickly to get models (this may take time)
    config = {'exchange': 'asterdex', 'api_key': 'dummy', 'api_secret': 'dummy', 'testnet': True}
    trainer = QuantumMLTrainerV60(config)

    # Train model for each coin to get calibrated classifiers and selector
    trained_models = {}

    for symbol in SCREEN_COINS:
        res = await trainer.train_v60(symbol)
        if res is None:
            logger.error(f"Training failed for {symbol}, skipping")
            continue
        trained_models[symbol] = res

    if not trained_models:
        logger.error("No trained models available. Exiting.")
        return

    # 2. Initialize exchange (testnet)
    exchange = AsterDEXFutures({'api_key': 'dummy', 'api_secret': 'dummy', 'testnet': True})

    # 3. Paper trading loop
    balance = START_BALANCE
    trades_executed = 0
    trade_log = []

    # Use fixed-percentage sizing to keep it simple
    sizer = PositionSizer(account_balance=balance, max_risk_percent=2.0, max_position_percent=50.0, leverage=1)

    # For demo: iterate through coins and try to open trades until MAX_TRADES or balance exhausted
    for symbol, res in trained_models.items():
        if trades_executed >= MAX_TRADES or balance <= 0:
            break

        # Use the ensemble probabilities on the latest features as "confidence"
        # For demo, compute a synthetic confidence from ensemble AUC and last probability
        ensemble = res['probabilities'] if 'probabilities' in res else None
        if ensemble is None:
            confidence = 0.7
        else:
            confidence = float(res['auc'])  # simplified proxy

        # Determine confidence bucket
        if confidence >= CONFIDENCE_HIGH:
            bucket = 'high'
        elif confidence >= CONFIDENCE_MED:
            bucket = 'medium'
        else:
            bucket = 'low'

        tp_sl = TP_SL_PARAMS[bucket]

        # Get current price
        ticker = await exchange.get_ticker_price(symbol)
        price = float(ticker.get('price', 0)) if isinstance(ticker, dict) else 0
        if price <= 0:
            logger.warning(f"Could not fetch price for {symbol}, skipping")
            continue

        # Determine position size (fixed percentage of balance)
        # Use 10% for high, 5% for medium, 2% for low
        pos_pct = {'high': 10.0, 'medium': 5.0, 'low': 2.0}[bucket]
        sizer.update_balance(balance)
        ps = sizer.fixed_percentage(current_price=price, position_percent=pos_pct)

        # If position value is less than $0.01 skip
        if ps.position_value < 0.01:
            logger.warning(f"Position too small for {symbol} (value=${ps.position_value:.4f}), skipping")
            continue

        # Place market buy (paper): simulate by deducting position_value
        entry_price = price
        size_value = ps.position_value

        if size_value > balance:
            # reduce to remaining balance
            size_value = balance
            quantity = size_value / price
        else:
            quantity = ps.quantity

        # Simulate entry
        balance -= size_value
        trades_executed += 1

        trade = TradeRecord(symbol=symbol, side='LONG', entry_price=entry_price, size=size_value, tp_sl_params=tp_sl, confidence=confidence)
        trade_log.append(trade)

        logger.info(f"Executed paper trade #{trades_executed}: {symbol} size=${size_value:.4f} entry={entry_price:.2f} conf={confidence:.3f}")

        # For demo simplicity: immediately simulate exit at TP1 price
        exit_price = entry_price * (1 + tp_sl['tp1'])
        trade.close(exit_price)
        # Credit PnL back
        pnl_value = trade.pnl * trade.size
        balance += trade.size + pnl_value  # return principal + pnl

        logger.info(f"Closed trade #{trades_executed}: PnL=${pnl_value:.4f} new_balance=${balance:.4f}")

        # Stop if balance exhausted or reached trade limit
        if balance <= 0 or trades_executed >= MAX_TRADES:
            break

    # 4. Generate PDF report
    report_path = RESULTS_DIR / f"v6_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    generate_pdf_report(report_path, trade_log, START_BALANCE, balance)

    logger.info(f"Demo complete. Trades executed: {trades_executed}. Final balance=${balance:.4f}")
    await exchange.close()


def generate_pdf_report(path, trade_log, starting_balance, final_balance):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    c.setFont('Helvetica-Bold', 14)
    c.drawString(40, height - 40, 'QUANTUM V6.0 Paper Trading Demo Report')
    c.setFont('Helvetica', 10)
    c.drawString(40, height - 60, f'Date: {datetime.now().isoformat()}')
    c.drawString(40, height - 75, f'Starting Balance: ${starting_balance:.4f}')
    c.drawString(40, height - 90, f'Final Balance: ${final_balance:.4f}')

    y = height - 120
    c.setFont('Helvetica-Bold', 11)
    c.drawString(40, y, 'Trade Log:')
    y -= 18
    c.setFont('Helvetica', 9)

    for t in trade_log:
        if y < 80:
            c.showPage()
            y = height - 60
        c.drawString(40, y, f"{t.timestamps['entry'].isoformat()} | {t.symbol} | {t.side} | Entry={t.entry_price:.4f} | Exit={t.exit_price:.4f} | PnL=${t.pnl* t.size:.4f} | BalanceAfter=${starting_balance:.4f}")
        y -= 14

    c.save()
    logger.info(f"PDF report saved: {path}")


if __name__ == '__main__':
    asyncio.run(run_demo())
