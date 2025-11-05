"""
QUANTUM LEAP ML TRAINING SYSTEM
Advanced ML training dengan multi-timeframe, alternative data, dan ensemble models

Inspired by: DeepSeekAI Quantum Leap Strategy
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for PDF generation
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logger
from data.asterdex_collector import AsterDEXDataCollector
from execution.asterdex import AsterDEXFutures

logger = setup_logger()


class QuantumPDFReporter:
    """Generate professional PDF reports for Quantum ML results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_quantum_report(self, all_results: List[Dict]) -> Path:
        """Generate comprehensive Quantum ML PDF report"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = self.output_dir / f"Quantum_ML_Report_{timestamp}.pdf"
            
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'QuantumTitle',
                parent=styles['Heading1'],
                fontSize=28,
                textColor=colors.HexColor('#2e4053'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph("🚀 QUANTUM LEAP ML TRAINING", title_style))
            story.append(Paragraph(f"Advanced Multi-Model Analysis Report", styles['Heading3']))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 0.5*inch))
            
            # Executive Summary
            story.append(Paragraph("📊 Executive Summary", styles['Heading2']))
            
            total_coins = len(all_results)
            avg_accuracy = np.mean([r['best_accuracy'] for r in all_results])
            avg_auc = np.mean([r['best_auc'] for r in all_results])
            total_features = all_results[0]['features'] if all_results else 0
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Coins Analyzed', str(total_coins)],
                ['Average Accuracy', f"{avg_accuracy*100:.2f}%"],
                ['Average AUC Score', f"{avg_auc:.3f}"],
                ['Quantum Features', str(total_features)],
                ['Training Method', 'Multi-Timeframe Ensemble'],
                ['Confidence Filtering', 'High-Confidence Only (>1.0σ)']
            ]
            
            t = Table(summary_data, colWidths=[3*inch, 3*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')])
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4*inch))
            
            # Performance Charts
            story.append(Paragraph("📈 Performance Analysis", styles['Heading2']))
            
            # Accuracy comparison chart
            acc_chart = self._create_accuracy_chart(all_results)
            if acc_chart:
                story.append(Image(str(acc_chart), width=6.5*inch, height=4*inch))
            story.append(Spacer(1, 0.3*inch))
            
            # Model comparison chart
            model_chart = self._create_model_comparison_chart(all_results)
            if model_chart:
                story.append(Image(str(model_chart), width=6.5*inch, height=4*inch))
            
            # Detailed Results
            story.append(PageBreak())
            story.append(Paragraph("🎯 Detailed Results by Coin", styles['Heading2']))
            
            for result in sorted(all_results, key=lambda x: x['best_accuracy'], reverse=True):
                story.append(self._create_coin_section(result, styles))
                story.append(Spacer(1, 0.2*inch))
            
            # Model Benchmarks
            story.append(PageBreak())
            story.append(Paragraph("🤖 Model Algorithm Benchmark", styles['Heading2']))
            
            benchmark_data = [['Model', 'Avg Accuracy', 'Avg AUC', 'Wins']]
            
            # Aggregate model performance
            model_stats = {}
            for result in all_results:
                for model_name, metrics in result['models'].items():
                    if model_name not in model_stats:
                        model_stats[model_name] = {'accuracies': [], 'aucs': [], 'wins': 0}
                    model_stats[model_name]['accuracies'].append(metrics['accuracy'])
                    model_stats[model_name]['aucs'].append(metrics['auc'])
                    if model_name == result['best_model']:
                        model_stats[model_name]['wins'] += 1
            
            for model_name, stats in sorted(model_stats.items(), key=lambda x: np.mean(x[1]['accuracies']), reverse=True):
                avg_acc = np.mean(stats['accuracies'])
                avg_auc = np.mean(stats['aucs'])
                wins = stats['wins']
                benchmark_data.append([
                    model_name.replace('_', ' ').title(),
                    f"{avg_acc*100:.1f}%",
                    f"{avg_auc:.3f}",
                    f"{wins}/{total_coins}"
                ])
            
            t2 = Table(benchmark_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            story.append(t2)
            
            # Quantum Features Section
            story.append(PageBreak())
            story.append(Paragraph("🔬 Quantum Feature Engineering", styles['Heading2']))
            
            features_text = """
            <b>Advanced Features Implemented:</b><br/>
            <br/>
            • <b>Market Regime Detection:</b> Trending, Ranging, Volatile classification<br/>
            • <b>Volatility Regime:</b> Low, Normal, High percentile-based classification<br/>
            • <b>Fibonacci Retracement:</b> 7 key levels (23.6%, 38.2%, 50%, 61.8%, etc.)<br/>
            • <b>Harmonic Patterns:</b> Pivot detection and pattern strength scoring<br/>
            • <b>Order Flow Metrics:</b> Delta volume, order imbalance analysis<br/>
            • <b>Momentum Clusters:</b> Multi-period alignment detection<br/>
            <br/>
            <b>Smart Target Engineering:</b><br/>
            • Multi-class targets: STRONG_BUY (+2), WEAK_BUY (+1), WEAK_SELL (-1), STRONG_SELL (-2)<br/>
            • Confidence scoring: Based on normalized volatility (0-3σ)<br/>
            • High-confidence filtering: Only train on samples with confidence > 1.0σ<br/>
            """
            story.append(Paragraph(features_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            logger.info(f"📄 Quantum PDF Report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_coin_section(self, result: Dict, styles):
        """Create detailed section for one coin"""
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        symbol = result['symbol']
        best_model = result['best_model'].replace('_', ' ').title()
        best_acc = result['best_accuracy'] * 100
        best_auc = result['best_auc']
        train_samples = result['train_samples']
        test_samples = result['test_samples']
        
        # Get top 3 models
        sorted_models = sorted(result['models'].items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
        
        data = [
            ['Coin', symbol],
            ['Best Model', best_model],
            ['Best Accuracy', f"{best_acc:.2f}%"],
            ['Best AUC', f"{best_auc:.3f}"],
            ['Train/Test Samples', f"{train_samples}/{test_samples}"],
            ['', ''],
            ['Top 3 Models:', '']
        ]
        
        for i, (model_name, metrics) in enumerate(sorted_models, 1):
            data.append([
                f"{i}. {model_name.replace('_', ' ').title()}",
                f"Acc: {metrics['accuracy']*100:.1f}% | AUC: {metrics['auc']:.3f}"
            ])
        
        t = Table(data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, 5), 0.5, colors.grey),
            ('BACKGROUND', (0, 1), (-1, 5), colors.HexColor('#ebf5fb')),
            ('SPAN', (0, 6), (-1, 6)),
            ('FONTNAME', (0, 6), (-1, 6), 'Helvetica-Bold'),
        ]))
        
        return t
    
    def _create_accuracy_chart(self, all_results: List[Dict]) -> Path:
        """Create accuracy comparison bar chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            symbols = [r['symbol'] for r in all_results]
            accuracies = [r['best_accuracy'] * 100 for r in all_results]
            aucs = [r['best_auc'] * 100 for r in all_results]
            
            x = np.arange(len(symbols))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x + width/2, aucs, width, label='AUC Score (x100)', color='#e74c3c', alpha=0.8)
            
            ax.set_xlabel('Cryptocurrency', fontsize=12, fontweight='bold')
            ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
            ax.set_title('Quantum ML Performance by Coin', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(symbols, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            chart_path = self.output_dir / f"accuracy_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating accuracy chart: {e}")
            plt.close()
            return None
    
    def _create_model_comparison_chart(self, all_results: List[Dict]) -> Path:
        """Create model win rate pie chart"""
        try:
            # Count wins per model
            model_wins = {}
            for result in all_results:
                best = result['best_model']
                model_wins[best] = model_wins.get(best, 0) + 1
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart
            colors_pie = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            labels = [name.replace('_', ' ').title() for name in model_wins.keys()]
            values = list(model_wins.values())
            
            ax1.pie(values, labels=labels, autopct='%1.1f%%', colors=colors_pie[:len(values)], 
                   startangle=90, textprops={'fontsize': 10})
            ax1.set_title('Best Model Distribution', fontsize=14, fontweight='bold')
            
            # Bar chart
            ax2.bar(range(len(model_wins)), values, color=colors_pie[:len(values)], alpha=0.8)
            ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Number of Wins', fontsize=11, fontweight='bold')
            ax2.set_title('Model Win Count', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(model_wins)))
            ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                ax2.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            chart_path = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating model comparison chart: {e}")
            plt.close()
            return None


class QuantumMLTrainer:
    """Advanced ML Trainer dengan Quantum Leap Strategy"""
    
    def __init__(self, config: Dict):
        self.collector = AsterDEXDataCollector(config)
        self.config = config
        
        # PDF Reporter setup
        self.pdf_dir = Path(__file__).parent.parent / "reports" / "quantum_pdf"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_reporter = QuantumPDFReporter(self.pdf_dir)
        
        logger.info("🚀 QUANTUM LEAP ML TRAINER INITIALIZED")
        logger.info("   Features: Multi-TF, Regime Detection, Super Ensemble")
        logger.info(f"   PDF Reports: {self.pdf_dir}")
    
    async def get_multi_timeframe_data(self, symbol: str, limit: int = 1000) -> Dict:
        """
        A. DATA REVOLUTION - Multi-Timeframe Data Fusion
        Collect data dari multiple timeframes dan merge
        """
        logger.info(f"📊 Collecting Multi-Timeframe Data for {symbol}...")
        
        timeframes = {
            '15m': 500,   # ~5 days
            '1h': 1000,   # ~40 days
            '4h': 500,    # ~80 days
        }
        
        multi_tf_data = {}
        
        for tf, limit_per_tf in timeframes.items():
            try:
                klines = await self.collector.exchange.get_klines(
                    symbol=symbol,
                    interval=tf,
                    limit=limit_per_tf
                )
                
                if klines and len(klines) > 0:
                    df = self._klines_to_dataframe(klines)
                    multi_tf_data[tf] = df
                    logger.info(f"   ✅ {tf}: {len(df)} candles")
                else:
                    logger.warning(f"   ⚠️ {tf}: No data")
                    
            except Exception as e:
                logger.warning(f"   ❌ {tf}: {e}")
        
        return multi_tf_data
    
    def _klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """Convert klines to DataFrame"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def quantum_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        B. ADVANCED FEATURE ENGINEERING
        Implement Harmonic Patterns, Market Regimes, Advanced Technicals
        """
        logger.info("🔬 Quantum Feature Engineering...")
        
        try:
            df = df.copy()
            
            # 1. MARKET REGIME DETECTION
            df = self._detect_market_regime(df)
            
            # 2. VOLATILITY REGIME
            df = self._detect_volatility_regime(df)
            
            # 3. FIBONACCI LEVELS
            df = self._calculate_fibonacci_levels(df)
            
            # 4. HARMONIC PATTERNS (Simplified)
            df = self._detect_harmonic_patterns(df)
            
            # 5. ORDER FLOW APPROXIMATION
            df = self._calculate_order_flow_metrics(df)
            
            # 6. MOMENTUM CLUSTERS
            df = self._calculate_momentum_clusters(df)
            
            logger.info(f"   ✅ Generated {len(df.columns)} quantum features")
            return df
            
        except Exception as e:
            logger.error(f"Error in quantum feature engineering: {e}")
            return df
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market Regime Detection: TRENDING, RANGING, VOLATILE
        """
        # ADX untuk trend strength
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        df['adx'] = adx
        
        # Market Regime Classification
        df['regime_trending'] = (adx > 25).astype(int)
        df['regime_ranging'] = (adx < 20).astype(int)
        df['regime_volatile'] = (atr / df['close'] > df['close'].pct_change().rolling(20).std() * 2).astype(int)
        
        # Regime numeric encoding
        conditions = [
            (df['regime_trending'] == 1),
            (df['regime_ranging'] == 1),
            (df['regime_volatile'] == 1)
        ]
        choices = [2, 0, 1]  # 2=Trending, 1=Volatile, 0=Ranging
        df['market_regime'] = np.select(conditions, choices, default=0)
        
        return df
    
    def _detect_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility Regime: LOW, NORMAL, HIGH
        """
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        
        # Percentile-based classification
        vol_25 = volatility.quantile(0.25)
        vol_75 = volatility.quantile(0.75)
        
        df['volatility'] = volatility
        df['vol_regime_low'] = (volatility < vol_25).astype(int)
        df['vol_regime_normal'] = ((volatility >= vol_25) & (volatility <= vol_75)).astype(int)
        df['vol_regime_high'] = (volatility > vol_75).astype(int)
        
        return df
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Fibonacci Retracement Levels
        """
        rolling_high = df['high'].rolling(window).max()
        rolling_low = df['low'].rolling(window).min()
        
        diff = rolling_high - rolling_low
        
        # Fibonacci levels
        df['fib_0'] = rolling_high
        df['fib_236'] = rolling_high - (diff * 0.236)
        df['fib_382'] = rolling_high - (diff * 0.382)
        df['fib_500'] = rolling_high - (diff * 0.500)
        df['fib_618'] = rolling_high - (diff * 0.618)
        df['fib_786'] = rolling_high - (diff * 0.786)
        df['fib_100'] = rolling_low
        
        # Distance from current price to key levels
        df['dist_to_fib_382'] = (df['close'] - df['fib_382']) / df['close']
        df['dist_to_fib_618'] = (df['close'] - df['fib_618']) / df['close']
        
        return df
    
    def _detect_harmonic_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified Harmonic Pattern Detection (Gartley, Butterfly, Bat)
        """
        try:
            # Simplified using pivot points with safe indexing
            def is_high_pivot(window):
                if len(window) == 5:
                    return 1.0 if window[2] == max(window) else 0.0
                return 0.0
            
            def is_low_pivot(window):
                if len(window) == 5:
                    return 1.0 if window[2] == min(window) else 0.0
                return 0.0
            
            high_pivots = df['high'].rolling(5, center=True).apply(is_high_pivot, raw=True)
            low_pivots = df['low'].rolling(5, center=True).apply(is_low_pivot, raw=True)
            
            df['high_pivot'] = high_pivots.fillna(0)
            df['low_pivot'] = low_pivots.fillna(0)
            
            # Pattern strength (simplified)
            df['pattern_strength'] = (df['high_pivot'] + df['low_pivot']).rolling(20).sum().fillna(0)
        except Exception as e:
            logger.warning(f"Harmonic pattern detection failed: {e}, using fallback")
            df['high_pivot'] = 0
            df['low_pivot'] = 0
            df['pattern_strength'] = 0
        
        return df
    
    def _calculate_order_flow_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order Flow Approximation (Volume-based)
        """
        try:
            # Delta Volume (approximation using taker buy)
            if 'taker_buy_base' in df.columns:
                df['buy_volume'] = pd.to_numeric(df['taker_buy_base'], errors='coerce').fillna(0)
                df['sell_volume'] = df['volume'] - df['buy_volume']
                df['delta_volume'] = df['buy_volume'] - df['sell_volume']
                df['delta_volume_cumsum'] = df['delta_volume'].cumsum()
            else:
                # Fallback: Approximation using price movement
                df['buy_volume_approx'] = df['volume'] * (df['close'] > df['open']).astype(int)
                df['sell_volume_approx'] = df['volume'] * (df['close'] < df['open']).astype(int)
                df['delta_volume'] = df['buy_volume_approx'] - df['sell_volume_approx']
            
            # Order Imbalance (avoid division by zero)
            df['order_imbalance'] = df['delta_volume'] / df['volume'].replace(0, np.nan)
            df['order_imbalance'] = df['order_imbalance'].fillna(0)
            df['order_imbalance_ma'] = df['order_imbalance'].rolling(20).mean().fillna(0)
        except Exception as e:
            logger.warning(f"Order flow calculation failed: {e}, using fallback")
            df['delta_volume'] = 0
            df['order_imbalance'] = 0
            df['order_imbalance_ma'] = 0
        
        return df
    
    def _calculate_momentum_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum Clustering using multiple periods
        """
        try:
            periods = [5, 10, 20, 50]
            
            for period in periods:
                df[f'momentum_{period}'] = df['close'].pct_change(periods=period).fillna(0)
            
            # Momentum alignment (all pointing same direction = strong signal)
            momentum_cols = [f'momentum_{p}' for p in periods]
            
            def calculate_alignment(row):
                try:
                    if all(row > 0):
                        return 1
                    elif all(row < 0):
                        return -1
                    else:
                        return 0
                except:
                    return 0
            
            df['momentum_alignment'] = df[momentum_cols].apply(calculate_alignment, axis=1)
        except Exception as e:
            logger.warning(f"Momentum cluster calculation failed: {e}, using fallback")
            for period in [5, 10, 20, 50]:
                df[f'momentum_{period}'] = 0
            df['momentum_alignment'] = 0
        
        return df
    
    def create_smart_target(self, df: pd.DataFrame, forward_periods: int = 4) -> pd.DataFrame:
        """
        D. ADVANCED TARGET ENGINEERING
        Regime-aware, multi-class target dengan confidence scoring
        """
        logger.info("🎯 Creating Smart Targets...")
        
        # Future returns
        future_returns = df['close'].pct_change(periods=forward_periods).shift(-forward_periods)
        
        # Adaptive threshold based on volatility
        volatility = df['close'].pct_change().rolling(20).std()
        dynamic_threshold = volatility * 2  # 2 sigma move
        
        # Multi-class target
        conditions = [
            (future_returns > dynamic_threshold),       # STRONG_BUY (2)
            (future_returns > 0),                       # WEAK_BUY (1)
            (future_returns < -dynamic_threshold),      # STRONG_SELL (-2)
            (future_returns < 0)                        # WEAK_SELL (-1)
        ]
        
        choices = [2, 1, -2, -1]
        df['smart_target'] = np.select(conditions, choices, default=0)
        
        # Confidence score (normalized by volatility)
        df['target_confidence'] = np.abs(future_returns) / (volatility + 1e-8)
        df['target_confidence'] = df['target_confidence'].clip(0, 3)  # Cap at 3 sigma
        
        # Binary target for traditional models
        df['target'] = (future_returns > 0).astype(int)
        
        logger.info(f"   ✅ Target distribution: {df['smart_target'].value_counts().to_dict()}")
        
        return df
    
    async def train_quantum_models(self, symbol: str, top_n: int = 5):
        """
        Train models menggunakan Quantum Leap approach
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 QUANTUM LEAP TRAINING: {symbol}")
        logger.info(f"{'='*80}")
        
        # 1. Multi-Timeframe Data
        multi_tf_data = await self.get_multi_timeframe_data(symbol)
        
        if not multi_tf_data or '1h' not in multi_tf_data:
            logger.error("❌ Insufficient data")
            return None
        
        # Use 1h as primary timeframe
        df = multi_tf_data['1h']
        
        # 2. Quantum Feature Engineering
        df = self.quantum_feature_engineering(df)
        
        # 3. Smart Target Creation
        df = self.create_smart_target(df)
        
        # 4. Remove NaN and prepare data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Drop first/last rows with missing targets
        df = df.dropna(subset=['target', 'smart_target'])
        
        if len(df) < 200:
            logger.warning("⚠️ Insufficient clean data")
            return None
        
        # 5. Feature Selection
        exclude_cols = ['timestamp', 'close_time', 'target', 'smart_target', 
                       'target_confidence', 'quote_volume', 'trades', 
                       'taker_buy_quote', 'ignore', 'taker_buy_base']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        
        # Convert all features to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(0)
        
        y = df['target']
        y_smart = df['smart_target']
        confidence = df['target_confidence']
        
        # 6. Filter high-confidence samples
        high_conf_mask = confidence > 1.0  # Only train on confident samples
        X_high_conf = X[high_conf_mask]
        y_high_conf = y[high_conf_mask]
        
        logger.info(f"   📊 Total samples: {len(X)}")
        logger.info(f"   🎯 High-confidence samples: {len(X_high_conf)} ({len(X_high_conf)/len(X)*100:.1f}%)")
        logger.info(f"   📈 Features: {len(feature_cols)}")
        
        # 7. Train-Test Split (time-based)
        split_idx = int(len(X_high_conf) * 0.8)
        X_train, X_test = X_high_conf.iloc[:split_idx], X_high_conf.iloc[split_idx:]
        y_train, y_test = y_high_conf.iloc[:split_idx], y_high_conf.iloc[split_idx:]
        
        if len(X_train) < 100 or len(X_test) < 20:
            logger.warning("⚠️ Insufficient samples after filtering")
            return None
        
        logger.info(f"   🔄 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 8. Train Ensemble Models
        results = self._train_super_ensemble(X_train, y_train, X_test, y_test, symbol)
        
        return results
    
    def _train_super_ensemble(self, X_train, y_train, X_test, y_test, symbol: str):
        """
        C. ENSEMBLE OF ENSEMBLES
        Train multiple advanced models and ensemble them
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        logger.info("\n   🤖 SUPER ENSEMBLE TRAINING")
        logger.info("   " + "="*70)
        
        models = {}
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1]
        }
        
        # 1. Random Forest (Optimized)
        try:
            logger.info("   [1/5] Random Forest (Quantum Optimized)")
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_proba = rf.predict_proba(X_test)[:, 1]
            
            models['random_forest_quantum'] = {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred, zero_division=0),
                'recall': recall_score(y_test, rf_pred, zero_division=0),
                'f1': f1_score(y_test, rf_pred, zero_division=0),
                'auc': roc_auc_score(y_test, rf_proba)
            }
            logger.info(f"        ✅ Acc: {models['random_forest_quantum']['accuracy']*100:.1f}% | "
                       f"AUC: {models['random_forest_quantum']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ RF Error: {e}")
        
        # 2. XGBoost (Hyperoptimized)
        try:
            import xgboost as xgb
            logger.info("   [2/5] XGBoost (Quantum Hyperoptimized)")
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            models['xgboost_quantum'] = {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred, zero_division=0),
                'recall': recall_score(y_test, xgb_pred, zero_division=0),
                'f1': f1_score(y_test, xgb_pred, zero_division=0),
                'auc': roc_auc_score(y_test, xgb_proba)
            }
            logger.info(f"        ✅ Acc: {models['xgboost_quantum']['accuracy']*100:.1f}% | "
                       f"AUC: {models['xgboost_quantum']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ XGB Error: {e}")
        
        # 3. LightGBM (Speed Optimized)
        try:
            import lightgbm as lgb
            logger.info("   [3/5] LightGBM (Quantum Speed)")
            
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.01,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
            
            models['lightgbm_quantum'] = {
                'accuracy': accuracy_score(y_test, lgb_pred),
                'precision': precision_score(y_test, lgb_pred, zero_division=0),
                'recall': recall_score(y_test, lgb_pred, zero_division=0),
                'f1': f1_score(y_test, lgb_pred, zero_division=0),
                'auc': roc_auc_score(y_test, lgb_proba)
            }
            logger.info(f"        ✅ Acc: {models['lightgbm_quantum']['accuracy']*100:.1f}% | "
                       f"AUC: {models['lightgbm_quantum']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ LGB Error: {e}")
        
        # 4. Gradient Boosting (Sklearn)
        try:
            logger.info("   [4/5] Gradient Boosting (Ensemble)")
            gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            gb.fit(X_train, y_train)
            gb_pred = gb.predict(X_test)
            gb_proba = gb.predict_proba(X_test)[:, 1]
            
            models['gradient_boosting'] = {
                'accuracy': accuracy_score(y_test, gb_pred),
                'precision': precision_score(y_test, gb_pred, zero_division=0),
                'recall': recall_score(y_test, gb_pred, zero_division=0),
                'f1': f1_score(y_test, gb_pred, zero_division=0),
                'auc': roc_auc_score(y_test, gb_proba)
            }
            logger.info(f"        ✅ Acc: {models['gradient_boosting']['accuracy']*100:.1f}% | "
                       f"AUC: {models['gradient_boosting']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ GB Error: {e}")
        
        # 5. Meta-Ensemble (Voting)
        try:
            from sklearn.ensemble import VotingClassifier
            logger.info("   [5/5] Meta-Ensemble (Voting)")
            
            # Only use successfully trained models
            estimators = []
            if 'random_forest_quantum' in models:
                estimators.append(('rf', rf))
            if 'xgboost_quantum' in models:
                estimators.append(('xgb', xgb_model))
            if 'lightgbm_quantum' in models:
                estimators.append(('lgb', lgb_model))
            if 'gradient_boosting' in models:
                estimators.append(('gb', gb))
            
            if len(estimators) >= 2:
                voting = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    n_jobs=-1
                )
                voting.fit(X_train, y_train)
                voting_pred = voting.predict(X_test)
                voting_proba = voting.predict_proba(X_test)[:, 1]
                
                models['meta_ensemble'] = {
                    'accuracy': accuracy_score(y_test, voting_pred),
                    'precision': precision_score(y_test, voting_pred, zero_division=0),
                    'recall': recall_score(y_test, voting_pred, zero_division=0),
                    'f1': f1_score(y_test, voting_pred, zero_division=0),
                    'auc': roc_auc_score(y_test, voting_proba)
                }
                logger.info(f"        ✅ Acc: {models['meta_ensemble']['accuracy']*100:.1f}% | "
                           f"AUC: {models['meta_ensemble']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ Ensemble Error: {e}")
        
        # Find best model
        if models:
            best_model = max(models.items(), key=lambda x: x[1]['accuracy'])
            results['best_model'] = best_model[0]
            results['best_accuracy'] = best_model[1]['accuracy']
            results['best_auc'] = best_model[1]['auc']
            results['models'] = models
            
            # Display comparison
            logger.info("\n   📊 QUANTUM MODEL BENCHMARK:")
            logger.info("   " + "="*70)
            logger.info(f"   {'Model':<25} | {'Accuracy':<10} | {'AUC':<8} | {'F1':<8}")
            logger.info("   " + "-"*70)
            
            sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for i, (name, metrics) in enumerate(sorted_models):
                winner = "🏆" if name == best_model[0] else "  "
                logger.info(f"   {winner} {name:<23} | {metrics['accuracy']*100:>8.1f}% | "
                           f"{metrics['auc']:>6.3f} | {metrics['f1']*100:>6.1f}%")
            
            logger.info("   " + "="*70)
            logger.info(f"   🏆 CHAMPION: {best_model[0].upper()} "
                       f"(Acc: {best_model[1]['accuracy']*100:.1f}%, AUC: {best_model[1]['auc']:.3f})")
        
        return results


async def main():
    """Demo Quantum ML Training"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    trainer = QuantumMLTrainer(config)
    
    # Test pada beberapa coins
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'TRUMPUSDT']
    all_results = []
    
    for symbol in test_symbols:
        results = await trainer.train_quantum_models(symbol)
        if results:
            all_results.append(results)
        await asyncio.sleep(2)
    
    # Generate PDF Report
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("📄 GENERATING QUANTUM PDF REPORT...")
        logger.info("="*80)
        
        pdf_path = trainer.pdf_reporter.generate_quantum_report(all_results)
        
        if pdf_path:
            logger.info(f"\n✅ QUANTUM PDF REPORT GENERATED!")
            logger.info(f"   Location: {pdf_path}")
            logger.info(f"   Coins Analyzed: {len(all_results)}")
            avg_acc = np.mean([r['best_accuracy'] for r in all_results])
            logger.info(f"   Average Accuracy: {avg_acc*100:.2f}%")
        else:
            logger.warning("⚠️ PDF Report generation failed")
    else:
        logger.warning("⚠️ No results to generate PDF report")
    
    logger.info("\n✅ QUANTUM LEAP TRAINING COMPLETE!")


if __name__ == "__main__":
    asyncio.run(main())
