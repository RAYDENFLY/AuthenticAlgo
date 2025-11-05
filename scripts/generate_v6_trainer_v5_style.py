#!/usr/bin/env python3
"""Generate a V5-style one-page summary PDF for V6 trainer results.

This script reads the latest `results/v6_enhanced_eval_log_*.csv`, computes
aggregate metrics and renders a single-page PDF similar to the provided
V5 example (title, generated timestamp, metrics table, and a small bar chart).
"""
import os
import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'results')
OUTDIR = os.path.join(ROOT, 'Reports', 'v6_models')
os.makedirs(OUTDIR, exist_ok=True)


def find_latest_eval():
    pattern = os.path.join(RESULTS, 'v6_enhanced_eval_log_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return sorted(files)[-1]


def compute_summary(csv_path):
    df = pd.read_csv(csv_path)
    # Expect columns: symbol, accuracy/acc, auc/AUC etc. Try multiple names.
    if 'symbol' not in df.columns:
        # try to infer symbol by filename or group
        df['symbol'] = df.get('symbol', '_unknown')

    # pick latest row per symbol
    grouped = df.groupby('symbol')
    symbols = []
    accs = []
    aucs = []
    for s, g in grouped:
        last = g.iloc[-1]
        symbols.append(s)
        a = None
        for col in ('accuracy', 'acc'):
            if col in last.index:
                a = last[col]
                break
        auc = None
        for col in ('auc', 'AUC'):
            if col in last.index:
                auc = last[col]
                break
        accs.append(float(a) if a is not None else None)
        aucs.append(float(auc) if auc is not None else None)

    coins_analyzed = len(symbols)
    avg_acc = (sum(x for x in accs if x is not None) / max(1, sum(1 for x in accs if x is not None))) if accs else None
    avg_auc = (sum(x for x in aucs if x is not None) / max(1, sum(1 for x in aucs if x is not None))) if aucs else None

    return {
        'symbols': symbols,
        'accs': accs,
        'aucs': aucs,
        'coins_analyzed': coins_analyzed,
        'avg_acc': avg_acc,
        'avg_auc': avg_auc,
        'generated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    }


def render_pdf(summary, outpath):
    c = canvas.Canvas(outpath, pagesize=letter)
    w, h = letter
    margin = 40
    y = h - margin

    # Title
    c.setFont('Helvetica-Bold', 20)
    c.drawString(margin, y, 'QUANTUM LEAP V6.0')
    y -= 28
    c.setFont('Helvetica', 12)
    c.drawString(margin, y, 'Model Training Summary - Deep Learning + Ensemble')
    y -= 18
    c.setFont('Helvetica', 9)
    c.drawString(margin, y, f'Generated: {summary.get("generated")} UTC')
    y -= 24

    # Metrics table
    c.setFont('Helvetica-Bold', 10)
    tx = margin
    ty = y
    cell_h = 18
    c.drawString(tx, ty, 'Metric')
    c.drawString(tx + 250, ty, 'Value')
    ty -= cell_h
    c.setFont('Helvetica', 10)
    rows = [
        ('Coins Analyzed', str(summary.get('coins_analyzed'))),
        ('Average Accuracy', f"{summary.get('avg_acc')*100:.2f}%" if summary.get('avg_acc') is not None else 'N/A'),
        ('Average AUC', f"{summary.get('avg_auc'):.3f}" if summary.get('avg_auc') is not None else 'N/A'),
        ('Training Method', 'TCN + Attention + Ensemble'),
        ('Deep Learning', 'Temporal CNNs + Transformers'),
        ('Optimization', 'Calibrated Ensemble (XGB/LGBM/CatBoost)'),
        ('Uncertainty', 'MC Dropout + Ensemble'),
        ('Version', 'Quantum Leap 6.0'),
    ]
    for k, v in rows:
        c.drawString(tx, ty, k)
        c.drawString(tx + 250, ty, v)
        ty -= cell_h

    # Draw small bar chart image using matplotlib
    chart_path = os.path.join(OUTDIR, 'v6_tmp_chart.png')
    try:
        symbols = summary.get('symbols', [])
        accs = [x * 100 if x is not None else 0 for x in summary.get('accs', [])]
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.bar(symbols, accs, color='steelblue')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Quantum V6 Performance vs Targets')
        ax.axhline(77, color='green', linestyle='--', label='Target 77%')
        ax.axhline(80, color='red', linestyle='--', label='Ultimate 80%')
        for i, v in enumerate(accs):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        c.drawImage(chart_path, margin, ty - 160, width=6*inch, height=2.5*inch)
    except Exception:
        pass

    c.save()
    try:
        if os.path.exists(chart_path):
            os.remove(chart_path)
    except Exception:
        pass


def main():
    csv = find_latest_eval()
    if not csv:
        print('No v6 eval CSV found in results/.')
        return
    summary = compute_summary(csv)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(OUTDIR, f'v6_trainer_v5style_{timestamp}.pdf')
    render_pdf(summary, outpath)
    print('V5-style trainer report written to', outpath)


if __name__ == '__main__':
    main()
