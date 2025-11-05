"""
QUANTUM LEAP ML TRAINING SYSTEM V2.0
Enhanced with advanced target engineering, feature selection, and meta-learning

Based on: DeepSeekAI Quantum Leap Fix V2.0
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
            pdf_path = self.output_dir / f"Quantum_ML_V2_Report_{timestamp}.pdf"
            
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
            story.append(Paragraph("🚀 QUANTUM LEAP V2.0 ML TRAINING", title_style))
            story.append(Paragraph(f"Advanced Meta-Learning Analysis Report", styles['Heading3']))
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
                ['Selected Features', str(total_features)],
                ['Training Method', 'Meta-Learning Ensemble V2.0'],
                ['Confidence Filtering', 'Adaptive (0.8σ training, 0.6σ test)'],
                ['Version', 'Quantum Leap 2.0']
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
            
            # Build PDF
            doc.build(story)
            logger.info(f"📄 Quantum V2 PDF Report generated: {pdf_path}")
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
        
        data = [
            ['Coin', symbol],
            ['Best Model', best_model],
            ['Best Accuracy', f"{best_acc:.2f}%"],
            ['Best AUC', f"{best_auc:.3f}"],
            ['Selected Features', str(result.get('features', 'N/A'))],
        ]
        
        t = Table(data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ebf5fb')),
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
            ax.set_title('Quantum ML V2.0 Performance by Coin', fontsize=14, fontweight='bold')
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
            ax1.set_title('Best Model Distribution (V2.0)', fontsize=14, fontweight='bold')
            
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


class QuantumMLTrainerV2:
    """Advanced ML Trainer dengan Quantum Leap V2.0 Strategy"""
    
    def __init__(self, config: Dict):
        self.collector = AsterDEXDataCollector(config)
        self.config = config
        
        # PDF Reporter setup
        self.pdf_dir = Path(__file__).parent.parent / "reports" / "quantum_v2_pdf"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_reporter = QuantumPDFReporter(self.pdf_dir)
        
        logger.info("🚀 QUANTUM LEAP V2.0 ML TRAINER INITIALIZED")
        logger.info("   Features: Meta-Learning, Smart Features, Adaptive Targets")
        logger.info(f"   PDF Reports: {self.pdf_dir}")
    
    async def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        A. MULTI-TIMEFRAME DATA COLLECTION
        Collect data from multiple timeframes for comprehensive analysis
        """
        logger.info(f"📊 Collecting Multi-Timeframe Data for {symbol}...")
        
        timeframes = {
            '15m': 500,   # ~5 days
            '1h': 1000,   # ~40 days (primary)
            '4h': 500,    # ~80 days
        }
        
        data = {}
        tasks = []
        
        for interval, limit in timeframes.items():
            task = self.collector.exchange.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            tasks.append((interval, task))
        
        # Collect all timeframes concurrently
        for interval, task in tasks:
            try:
                klines = await task
                if klines is not None and len(klines) > 0:
                    # Convert to DataFrame using collector's method
                    df = self.collector._klines_to_dataframe(klines)
                    data[interval] = df
                    logger.info(f"   ✅ {interval}: {len(df)} candles")
                else:
                    logger.warning(f"   ⚠️ {interval}: No data")
            except Exception as e:
                logger.error(f"   ❌ {interval}: Error - {e}")
        
        return data
    
    def detect_market_regime_v2(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime: high_vol, low_vol, normal
        """
        volatility = df['close'].pct_change().rolling(20).std()
        vol_25 = volatility.quantile(0.25)
        vol_75 = volatility.quantile(0.75)
        
        conditions = [
            volatility > vol_75,  # High volatility
            volatility < vol_25,  # Low volatility
        ]
        choices = ['high_vol', 'low_vol']
        
        regime = np.select(conditions, choices, default='normal')
        return pd.Series(regime, index=df.index)
    
    def quantum_target_v2(self, df: pd.DataFrame, prediction_horizon: int = 6) -> pd.DataFrame:
        """
        Quantum Leap Target 2.0 - Adaptive and robust target engineering
        Using binary targets for better accuracy
        """
        logger.info("🎯 Creating Quantum Targets V2.0...")
        
        # 1. ADAPTIVE VOLATILITY THRESHOLD
        returns = df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        volatility = df['close'].pct_change().rolling(20).std()
        
        # Dynamic threshold based on market regime
        regime = self.detect_market_regime_v2(df)
        
        # Initialize thresholds
        threshold_multiplier = pd.Series(index=df.index, dtype=float)
        
        for idx in df.index:
            if regime[idx] == 'high_vol':
                threshold_multiplier[idx] = 1.5  # More conservative in high vol
            elif regime[idx] == 'low_vol':
                threshold_multiplier[idx] = 0.75  # More aggressive in low vol
            else:  # normal
                threshold_multiplier[idx] = 1.0
        
        # 2. BINARY TARGET WITH ADAPTIVE THRESHOLD
        threshold = volatility * threshold_multiplier
        
        # BUY (1) if returns > threshold, SELL (0) otherwise
        df['quantum_target'] = (returns > threshold).astype(int)
        
        # 3. ADVANCED CONFIDENCE SCORE
        base_confidence = np.abs(returns) / volatility.replace(0, np.nan)
        base_confidence = base_confidence.fillna(0)
        
        # Volume confirmation boost
        volume_ma = df['volume'].rolling(20).mean()
        volume_confirmation = (df['volume'] > volume_ma * 1.5).astype(float) * 0.3
        
        # Technical alignment boost (simplified)
        rsi = self._calculate_rsi(df['close'], 14)
        technical_alignment = ((rsi > 70) | (rsi < 30)).astype(float) * 0.2
        
        total_confidence = base_confidence + volume_confirmation + technical_alignment
        df['quantum_confidence'] = total_confidence.clip(0, 3)
        
        # Log target distribution
        target_dist = df['quantum_target'].value_counts().to_dict()
        logger.info(f"   ✅ Target distribution: BUY={target_dist.get(1, 0)}, SELL={target_dist.get(0, 0)}")
        logger.info(f"   📊 Average confidence: {df['quantum_confidence'].mean():.2f}σ")
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def quantum_feature_engineering_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantum Feature Engineering V2.0 - Smart feature selection
        """
        logger.info("🔬 Quantum Feature Engineering V2.0...")
        
        df = df.copy()
        
        # 1. CORE PRICE FEATURES (5 features)
        df['momentum_5'] = df['close'].pct_change(5).fillna(0)
        df['momentum_10'] = df['close'].pct_change(10).fillna(0)
        df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean().fillna(0)
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # 2. QUANTUM MOMENTUM FEATURES (5 features)
        df['momentum_accel'] = df['momentum_5'] - df['momentum_10']
        df['rsi_momentum'] = df['rsi_14'].diff(3).fillna(0)
        df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].quantile(0.7)).astype(int)
        df['price_volume_trend'] = (df['volume'] * df['close'].pct_change()).rolling(5).mean().fillna(0)
        
        # Support/Resistance strength (simplified)
        rolling_high = df['high'].rolling(20).max()
        rolling_low = df['low'].rolling(20).min()
        df['sr_strength'] = ((df['close'] - rolling_low) / (rolling_high - rolling_low).replace(0, 1)).fillna(0.5)
        
        # 3. MARKET STRUCTURE FEATURES (3 features)
        regime = self.detect_market_regime_v2(df)
        df['market_regime'] = pd.Categorical(regime).codes
        
        # Trend strength (ADX-like)
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = -df['low'].diff().clip(lower=0)
        df['trend_strength'] = np.abs(plus_dm - minus_dm).rolling(14).mean().fillna(0)
        
        # Mean reversion probability
        sma_20 = df['close'].rolling(20).mean()
        deviation = (df['close'] - sma_20) / sma_20.replace(0, 1)
        df['mean_reversion_score'] = np.abs(deviation).fillna(0)
        
        logger.info(f"   ✅ Generated {len(df.columns)} features")
        
        return df
    
    def smart_feature_selection(self, df: pd.DataFrame, target_col: str = 'quantum_target', keep_features: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Smart feature selection using statistical methods
        """
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Exclude non-feature columns
        exclude_cols = ['timestamp', 'close_time', 'quantum_target', 'quantum_confidence', 
                       'quote_volume', 'trades', 'taker_buy_quote', 'taker_buy_base', 'ignore']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Remove NaN
        X = X.fillna(0)
        
        # Select K best features
        selector = SelectKBest(score_func=f_classif, k=min(keep_features, len(feature_cols)))
        selector.fit(X, y)
        
        # Get selected feature names
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        best_features = feature_scores.head(keep_features)['feature'].tolist()
        
        logger.info(f"   📊 Selected top {len(best_features)} features:")
        for i, row in feature_scores.head(keep_features).iterrows():
            logger.info(f"      {row['feature']}: {row['score']:.2f}")
        
        return df[best_features + [target_col, 'quantum_confidence']], best_features
    
    async def train_quantum_models_v2(self, symbol: str, keep_features: int = 10):
        """
        Train models using Quantum Leap V2.0 approach
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 QUANTUM LEAP V2.0 TRAINING: {symbol}")
        logger.info(f"{'='*80}")
        
        # 1. Multi-Timeframe Data
        multi_tf_data = await self.get_multi_timeframe_data(symbol)
        
        if not multi_tf_data or '1h' not in multi_tf_data:
            logger.error("❌ Insufficient data")
            return None
        
        # Use 1h as primary timeframe
        df = multi_tf_data['1h']
        
        # 2. Quantum Feature Engineering V2
        df = self.quantum_feature_engineering_v2(df)
        
        # 3. Quantum Target Creation V2
        df = self.quantum_target_v2(df)
        
        # 4. Smart Feature Selection
        df_selected, best_features = self.smart_feature_selection(df, keep_features=keep_features)
        
        # 5. Remove NaN
        df_selected = df_selected.replace([np.inf, -np.inf], np.nan)
        df_selected = df_selected.fillna(method='ffill').fillna(method='bfill').fillna(0)
        df_selected = df_selected.dropna(subset=['quantum_target'])
        
        if len(df_selected) < 200:
            logger.warning("⚠️ Insufficient clean data")
            return None
        
        X = df_selected[best_features]
        y = df_selected['quantum_target']
        confidence = df_selected['quantum_confidence']
        
        # 6. Filter high-confidence samples (V2.0: Lower threshold for better coverage)
        high_conf_mask = confidence > 0.6  # V2.0: More inclusive
        X_high_conf = X[high_conf_mask]
        y_high_conf = y[high_conf_mask]
        
        logger.info(f"   📊 Total samples: {len(X)}")
        logger.info(f"   🎯 High-confidence samples (>0.6σ): {len(X_high_conf)} ({len(X_high_conf)/len(X)*100:.1f}%)")
        logger.info(f"   📈 Selected features: {len(best_features)}")
        
        if len(X_high_conf) < 100:
            logger.warning("   ⚠️ Not enough high-conf samples, using threshold 0.4σ")
            high_conf_mask = confidence > 0.4
            X_high_conf = X[high_conf_mask]
            y_high_conf = y[high_conf_mask]
        
        # 7. Train-Test Split (time-based)
        split_idx = int(len(X_high_conf) * 0.8)
        X_train, X_test = X_high_conf.iloc[:split_idx], X_high_conf.iloc[split_idx:]
        y_train, y_test = y_high_conf.iloc[:split_idx], y_high_conf.iloc[split_idx:]
        
        if len(X_train) < 50 or len(X_test) < 10:
            logger.warning("⚠️ Insufficient samples after filtering")
            return None
        
        logger.info(f"   🔄 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 8. Train Quantum Ensemble V2
        results = self._train_quantum_ensemble_v2(X_train, y_train, X_test, y_test, symbol, best_features)
        
        return results
    
    def _train_quantum_ensemble_v2(self, X_train, y_train, X_test, y_test, symbol: str, feature_names: List[str]):
        """
        Train Quantum Ensemble V2.0 with meta-learning
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from sklearn.preprocessing import label_binarize
        
        logger.info("\n   🤖 QUANTUM ENSEMBLE V2.0 TRAINING")
        logger.info("   " + "="*70)
        
        models = {}
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(feature_names),
            'feature_names': feature_names
        }
        
        # Prepare for multi-class AUC
        classes = np.unique(y_train)
        n_classes = len(classes)
        
        # 1. XGBoost Quantum
        try:
            import xgboost as xgb
            logger.info("   [1/5] XGBoost Quantum V2 (n=1000, depth=8)")
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_proba = xgb_model.predict_proba(X_test)
            
            # Calculate AUC for multi-class
            if n_classes > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, xgb_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, xgb_proba[:, 1])
            
            models['xgboost_quantum_v2'] = {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, xgb_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, xgb_pred, average='weighted', zero_division=0),
                'auc': auc
            }
            logger.info(f"        ✅ Acc: {models['xgboost_quantum_v2']['accuracy']*100:.1f}% | "
                       f"AUC: {models['xgboost_quantum_v2']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ XGB Error: {e}")
        
        # 2. LightGBM Quantum
        try:
            import lightgbm as lgb
            logger.info("   [2/5] LightGBM Quantum V2 (n=1000, depth=10)")
            
            lgb_model = lgb.LGBMClassifier(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.01,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            lgb_proba = lgb_model.predict_proba(X_test)
            
            if n_classes > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, lgb_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, lgb_proba[:, 1])
            
            models['lightgbm_quantum_v2'] = {
                'accuracy': accuracy_score(y_test, lgb_pred),
                'precision': precision_score(y_test, lgb_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, lgb_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, lgb_pred, average='weighted', zero_division=0),
                'auc': auc
            }
            logger.info(f"        ✅ Acc: {models['lightgbm_quantum_v2']['accuracy']*100:.1f}% | "
                       f"AUC: {models['lightgbm_quantum_v2']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ LGB Error: {e}")
        
        # 3. Random Forest Quantum V2
        try:
            logger.info("   [3/5] Random Forest Quantum V2 (n=500, depth=20)")
            rf = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_proba = rf.predict_proba(X_test)
            
            if n_classes > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, rf_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, rf_proba[:, 1])
            
            models['random_forest_quantum_v2'] = {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, rf_pred, average='weighted', zero_division=0),
                'auc': auc
            }
            logger.info(f"        ✅ Acc: {models['random_forest_quantum_v2']['accuracy']*100:.1f}% | "
                       f"AUC: {models['random_forest_quantum_v2']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ RF Error: {e}")
        
        # 4. Gradient Boosting
        try:
            logger.info("   [4/5] Gradient Boosting V2 (n=500, depth=8)")
            gb = GradientBoostingClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                random_state=42
            )
            gb.fit(X_train, y_train)
            gb_pred = gb.predict(X_test)
            gb_proba = gb.predict_proba(X_test)
            
            if n_classes > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, gb_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, gb_proba[:, 1])
            
            models['gradient_boosting_v2'] = {
                'accuracy': accuracy_score(y_test, gb_pred),
                'precision': precision_score(y_test, gb_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, gb_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, gb_pred, average='weighted', zero_division=0),
                'auc': auc
            }
            logger.info(f"        ✅ Acc: {models['gradient_boosting_v2']['accuracy']*100:.1f}% | "
                       f"AUC: {models['gradient_boosting_v2']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ GB Error: {e}")
        
        # 5. Meta-Learner (Neural Network)
        try:
            logger.info("   [5/5] Meta-Learner (Neural Network)")
            
            # Stack predictions from all models
            meta_features_train = []
            meta_features_test = []
            
            if 'xgboost_quantum_v2' in models:
                xgb_model_temp = xgb.XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.01, random_state=42, n_jobs=-1)
                xgb_model_temp.fit(X_train, y_train)
                meta_features_train.append(xgb_model_temp.predict_proba(X_train))
                meta_features_test.append(xgb_model_temp.predict_proba(X_test))
            
            if 'lightgbm_quantum_v2' in models:
                lgb_model_temp = lgb.LGBMClassifier(n_estimators=1000, max_depth=10, learning_rate=0.01, random_state=42, n_jobs=-1, verbose=-1)
                lgb_model_temp.fit(X_train, y_train)
                meta_features_train.append(lgb_model_temp.predict_proba(X_train))
                meta_features_test.append(lgb_model_temp.predict_proba(X_test))
            
            if len(meta_features_train) >= 2:
                # Combine with original features
                meta_X_train = np.hstack(meta_features_train + [X_train.values])
                meta_X_test = np.hstack(meta_features_test + [X_test.values])
                
                # Train meta-learner
                meta_learner = MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    learning_rate='adaptive',
                    early_stopping=True,
                    random_state=42,
                    max_iter=500
                )
                meta_learner.fit(meta_X_train, y_train)
                meta_pred = meta_learner.predict(meta_X_test)
                meta_proba = meta_learner.predict_proba(meta_X_test)
                
                if n_classes > 2:
                    y_test_bin = label_binarize(y_test, classes=classes)
                    auc = roc_auc_score(y_test_bin, meta_proba, multi_class='ovr', average='weighted')
                else:
                    auc = roc_auc_score(y_test, meta_proba[:, 1])
                
                models['meta_learner_v2'] = {
                    'accuracy': accuracy_score(y_test, meta_pred),
                    'precision': precision_score(y_test, meta_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, meta_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, meta_pred, average='weighted', zero_division=0),
                    'auc': auc
                }
                logger.info(f"        ✅ Acc: {models['meta_learner_v2']['accuracy']*100:.1f}% | "
                           f"AUC: {models['meta_learner_v2']['auc']:.3f}")
        except Exception as e:
            logger.error(f"        ❌ Meta-Learner Error: {e}")
        
        # Find best model
        if models:
            best_model = max(models.items(), key=lambda x: x[1]['accuracy'])
            results['best_model'] = best_model[0]
            results['best_accuracy'] = best_model[1]['accuracy']
            results['best_auc'] = best_model[1]['auc']
            results['models'] = models
            
            # Display comparison
            logger.info("\n   📊 QUANTUM V2.0 MODEL BENCHMARK:")
            logger.info("   " + "="*70)
            logger.info(f"   {'Model':<30} | {'Accuracy':<10} | {'AUC':<8} | {'F1':<8}")
            logger.info("   " + "-"*70)
            
            sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for i, (name, metrics) in enumerate(sorted_models):
                winner = "🏆" if name == best_model[0] else "  "
                logger.info(f"   {winner} {name:<28} | {metrics['accuracy']*100:>8.1f}% | "
                           f"{metrics['auc']:>6.3f} | {metrics['f1']*100:>6.1f}%")
            
            logger.info("   " + "="*70)
            logger.info(f"   🏆 CHAMPION: {best_model[0].upper()} "
                       f"(Acc: {best_model[1]['accuracy']*100:.1f}%, AUC: {best_model[1]['auc']:.3f})")
        
        return results


async def main():
    """Demo Quantum ML Training V2.0"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    trainer = QuantumMLTrainerV2(config)
    
    # Test pada beberapa coins
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'TRUMPUSDT']
    all_results = []
    
    for symbol in test_symbols:
        results = await trainer.train_quantum_models_v2(symbol, keep_features=10)
        if results:
            all_results.append(results)
        await asyncio.sleep(2)
    
    # Generate PDF Report
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("📄 GENERATING QUANTUM V2.0 PDF REPORT...")
        logger.info("="*80)
        
        pdf_path = trainer.pdf_reporter.generate_quantum_report(all_results)
        
        if pdf_path:
            logger.info(f"\n✅ QUANTUM V2.0 PDF REPORT GENERATED!")
            logger.info(f"   Location: {pdf_path}")
            logger.info(f"   Coins Analyzed: {len(all_results)}")
            avg_acc = np.mean([r['best_accuracy'] for r in all_results])
            avg_auc = np.mean([r['best_auc'] for r in all_results])
            logger.info(f"   Average Accuracy: {avg_acc*100:.2f}%")
            logger.info(f"   Average AUC: {avg_auc:.3f}")
        else:
            logger.warning("⚠️ PDF Report generation failed")
    else:
        logger.warning("⚠️ No results to generate PDF report")
    
    logger.info("\n✅ QUANTUM LEAP V2.0 TRAINING COMPLETE!")


if __name__ == "__main__":
    asyncio.run(main())
