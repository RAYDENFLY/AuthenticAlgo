#!/usr/bin/env python3
"""Generate consolidated V6 trainer reports (models + eval CSVs) into Reports/v6_models

This script is non-invasive: it only reads files from `results/` and writes a PDF
summary to `Reports/v6_models/`.
"""
import io
import os
import sys
from datetime import datetime
import joblib
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'results')
OUTDIR = os.path.join(ROOT, 'Reports', 'v6_models')
os.makedirs(OUTDIR, exist_ok=True)


def find_joblibs(path):
    files = []
    for fn in os.listdir(path):
        if fn.startswith('v6_') and fn.endswith('_models.joblib'):
            files.append(os.path.join(path, fn))
    return sorted(files)


def find_eval_csvs(path):
    files = []
    for fn in os.listdir(path):
        if 'v6' in fn and fn.endswith('.csv') and 'eval' in fn:
            files.append(os.path.join(path, fn))
    return sorted(files)


def summarize_joblib(path):
    info = {'file': os.path.basename(path)}
    try:
        st = os.stat(path)
        info['size_kb'] = int(st.st_size / 1024)
        info['modified'] = datetime.fromtimestamp(st.st_mtime).isoformat()
    except Exception:
        info['size_kb'] = None
        info['modified'] = None

    try:
        obj = joblib.load(path)
        # joblib may store a dict, tuple or estimator. Try to inspect lightly.
        if isinstance(obj, dict):
            info['keys'] = list(obj.keys())[:10]
            # try to pull metrics if present
            for k in ['metrics', 'eval', 'score', 'scores']:
                if k in obj:
                    info['contained_metrics'] = str(type(obj[k]))
                    break
        else:
            info['object_type'] = type(obj).__name__
            # try common attributes
            for attr in ('feature_importances_', 'best_score_', 'cv_results_'):
                if hasattr(obj, attr):
                    try:
                        val = getattr(obj, attr)
                        info.setdefault('attrs', {})[attr] = str(type(val))
                    except Exception:
                        info.setdefault('attrs', {})[attr] = 'error'
    except Exception as e:
        info['load_error'] = repr(e)

    return info


def collect_eval_metrics(csv_files):
    # Read evaluation CSVs and compute per-symbol summary metrics (accuracy, auc, last rows)
    metrics = {}
    for p in csv_files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # try common column names
        if 'symbol' in df.columns:
            grouped = df.groupby('symbol')
            for symbol, g in grouped:
                # pick last row metrics if available
                last = g.iloc[-1]
                row = {}
                for col in ('accuracy', 'acc', 'auc', 'AUC', 'precision', 'recall'):
                    if col in last.index:
                        row[col] = float(last[col]) if pd.notna(last[col]) else None
                row['source_csv'] = os.path.basename(p)
                metrics.setdefault(symbol, []).append(row)
        else:
            # If no symbol column, try to infer overall metrics from the CSV last row
            last = df.iloc[-1]
            summary = {}
            for col in ('accuracy', 'acc', 'auc', 'AUC', 'precision', 'recall'):
                if col in last.index:
                    summary[col] = float(last[col]) if pd.notna(last[col]) else None
            metrics.setdefault('_overall', []).append({'summary': summary, 'source_csv': os.path.basename(p)})

    return metrics


def write_pdf(joblib_infos, eval_metrics, outpath):
    c = canvas.Canvas(outpath, pagesize=A4)
    w, h = A4
    margin = 40
    y = h - margin
    c.setFont('Helvetica-Bold', 14)
    c.drawString(margin, y, 'V6 Trainer Models Summary')
    c.setFont('Helvetica', 9)
    y -= 18
    c.drawString(margin, y, f'Generated: {datetime.utcnow().isoformat()} UTC')
    y -= 18

    for info in joblib_infos:
        if y < 120:
            c.showPage()
            y = h - margin
        c.setFont('Helvetica-Bold', 11)
        c.drawString(margin, y, info.get('file', 'unknown'))
        y -= 14
        c.setFont('Helvetica', 9)
        c.drawString(margin + 8, y, f"Size (KB): {info.get('size_kb')}")
        y -= 12
        c.drawString(margin + 8, y, f"Modified: {info.get('modified')}")
        y -= 12
        if 'object_type' in info:
            c.drawString(margin + 8, y, f"Object: {info['object_type']}")
            y -= 12
        if 'keys' in info:
            c.drawString(margin + 8, y, f"Keys: {', '.join(map(str, info['keys']))}")
            y -= 12
        if 'attrs' in info:
            for k, v in info['attrs'].items():
                c.drawString(margin + 8, y, f"{k}: {v}")
                y -= 12
        if 'load_error' in info:
            c.setFillColorRGB(0.6, 0, 0)
            c.drawString(margin + 8, y, f"Load error: {info['load_error']}")
            c.setFillColorRGB(0, 0, 0)
            y -= 12

        # add any eval metrics associated with the symbol inferred from filename
        # filename format is v6_<SYMBOL>_models.joblib
        fn = info.get('file', '')
        parts = fn.split('_')
        symbol = None
        if len(parts) >= 3:
            symbol = parts[1]
        if symbol and symbol in eval_metrics:
            c.setFont('Helvetica-Oblique', 9)
            c.drawString(margin + 8, y, f"Related eval metrics (from CSV):")
            y -= 12
            for m in eval_metrics[symbol]:
                line = ', '.join(f"{k}:{v:.3f}" for k, v in m.items() if isinstance(v, (int, float)))
                c.drawString(margin + 12, y, f"{m.get('source_csv','')}: {line}")
                y -= 12

        y -= 6

    # write a short appendix of CSVs processed
    if eval_metrics:
        if y < 160:
            c.showPage()
            y = h - margin
        c.setFont('Helvetica-Bold', 12)
        c.drawString(margin, y, 'Evaluation CSVs processed (per-symbol snippets)')
        y -= 16
        c.setFont('Helvetica', 9)
        for symbol, rows in eval_metrics.items():
            if y < 80:
                c.showPage()
                y = h - margin
            c.drawString(margin + 4, y, str(symbol))
            y -= 12
            for r in rows[:3]:
                c.drawString(margin + 12, y, str(r))
                y -= 10
            y -= 6

    c.save()


def main():
    joblibs = find_joblibs(RESULTS)
    csvs = find_eval_csvs(RESULTS)

    joblib_infos = [summarize_joblib(p) for p in joblibs]
    eval_metrics = collect_eval_metrics(csvs)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(OUTDIR, f'v6_trainer_models_summary_{timestamp}.pdf')
    write_pdf(joblib_infos, eval_metrics, outpath)
    print('Trainer report written to', outpath)


if __name__ == '__main__':
    main()
