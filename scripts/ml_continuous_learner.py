"""
ML Continuous Learning dengan Auto-Screening
Belajar dari SEMUA coins di AsterDEX, bukan cuma top 5

Features:
- Auto-detect semua available symbols
- Smart filtering berdasarkan volume & volatility
- Priority training untuk coins dengan patterns menarik
- Adaptive learning berdasarkan market conditions
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import sqlite3
import json
from typing import Dict, List, Tuple
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logger

logger = setup_logger()

from data.asterdex_collector import AsterDEXDataCollector
from execution.asterdex import AsterDEXFutures

# Simplified imports - avoid complex dependencies
try:
    from ml.model_trainer import ModelTrainer
    HAS_MODEL_TRAINER = True
except Exception as e:
    HAS_MODEL_TRAINER = False
    logger.warning(f"ModelTrainer not available: {e}")


class PDFReportGenerator:
    """Generate professional PDF reports for ML training results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_training_report(self, training_results: List[Dict], screening_results: List[Dict] = None):
        """Generate comprehensive PDF report with charts"""
        try:
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = self.output_dir / f"ML_Training_Report_{timestamp}.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph("ML Training Report", title_style))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 0.5*inch))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            df = pd.DataFrame(training_results)
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Coins Trained', str(len(df))],
                ['Average Accuracy', f"{df['best_accuracy'].mean()*100:.2f}%"],
                ['Best Accuracy', f"{df['best_accuracy'].max()*100:.2f}%"],
                ['Worst Accuracy', f"{df['best_accuracy'].min()*100:.2f}%"],
                ['Total Data Points', f"{df['data_points'].sum():,}"]
            ]
            
            t = Table(summary_data, colWidths=[3*inch, 3*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3*inch))
            
            # Model Benchmark Chart
            story.append(Paragraph("Model Performance Benchmark", styles['Heading2']))
            chart_path = self._create_model_benchmark_chart(training_results)
            if chart_path:
                story.append(Image(str(chart_path), width=6*inch, height=4*inch))
            story.append(Spacer(1, 0.3*inch))
            
            # Detailed Results Table
            story.append(PageBreak())
            story.append(Paragraph("Detailed Training Results", styles['Heading2']))
            
            # Prepare detailed data
            detailed_data = [['Rank', 'Symbol', 'Best Model', 'Accuracy', 'Precision', 'Recall', 'F1']]
            
            for i, result in enumerate(sorted(training_results, key=lambda x: x['best_accuracy'], reverse=True), 1):
                models = result.get('models', {})
                best_model_name = result['best_model']
                best_metrics = models.get(best_model_name, {})
                
                detailed_data.append([
                    str(i),
                    result['symbol'],
                    best_model_name[:10],
                    f"{result['best_accuracy']*100:.1f}%",
                    f"{best_metrics.get('precision', 0)*100:.1f}%",
                    f"{best_metrics.get('recall', 0)*100:.1f}%",
                    f"{best_metrics.get('f1_score', 0)*100:.1f}%"
                ])
            
            t2 = Table(detailed_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a90e2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            story.append(t2)
            
            # Model Comparison
            story.append(PageBreak())
            story.append(Paragraph("Model Algorithm Comparison", styles['Heading2']))
            comparison_chart = self._create_model_comparison_chart(training_results)
            if comparison_chart:
                story.append(Image(str(comparison_chart), width=6*inch, height=4*inch))
            
            # Build PDF
            doc.build(story)
            logger.info(f"📄 PDF Report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_model_benchmark_chart(self, training_results: List[Dict]) -> Path:
        """Create model benchmark comparison chart"""
        try:
            # Collect model accuracies
            model_data = {'RandomForest': [], 'XGBoost': [], 'LightGBM': []}
            
            for result in training_results:
                models = result.get('models', {})
                if 'random_forest' in models:
                    model_data['RandomForest'].append(models['random_forest'].get('accuracy', 0) * 100)
                if 'xgboost' in models:
                    model_data['XGBoost'].append(models['xgboost'].get('accuracy', 0) * 100)
                if 'lightgbm' in models:
                    model_data['LightGBM'].append(models['lightgbm'].get('accuracy', 0) * 100)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Box plot
            data_to_plot = [v for v in model_data.values() if v]
            labels_to_plot = [k for k, v in model_data.items() if v]
            
            if data_to_plot:
                bp = plt.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
                
                # Customize colors
                colors_list = ['#4a90e2', '#e24a4a', '#4ae290']
                for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                plt.ylabel('Accuracy (%)', fontsize=12)
                plt.title('Model Performance Distribution', fontsize=14, fontweight='bold')
                plt.grid(axis='y', alpha=0.3)
                
                chart_path = self.output_dir / f"benchmark_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.tight_layout()
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                return chart_path
            
        except Exception as e:
            logger.error(f"Error creating benchmark chart: {e}")
            plt.close()
        return None
    
    def _create_model_comparison_chart(self, training_results: List[Dict]) -> Path:
        """Create model win rate comparison chart"""
        try:
            # Count wins for each model
            model_wins = {'RandomForest': 0, 'XGBoost': 0, 'LightGBM': 0}
            
            for result in training_results:
                best_model = result.get('best_model', '')
                if 'random_forest' in best_model:
                    model_wins['RandomForest'] += 1
                elif 'xgboost' in best_model:
                    model_wins['XGBoost'] += 1
                elif 'lightgbm' in best_model:
                    model_wins['LightGBM'] += 1
            
            # Filter out models with 0 wins
            model_wins = {k: v for k, v in model_wins.items() if v > 0}
            
            if model_wins:
                # Create pie chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Pie chart
                colors_pie = ['#4a90e2', '#e24a4a', '#4ae290']
                ax1.pie(model_wins.values(), labels=model_wins.keys(), autopct='%1.1f%%',
                       colors=colors_pie[:len(model_wins)], startangle=90)
                ax1.set_title('Best Model Distribution', fontsize=14, fontweight='bold')
                
                # Bar chart
                ax2.bar(model_wins.keys(), model_wins.values(), color=colors_pie[:len(model_wins)], alpha=0.7)
                ax2.set_ylabel('Number of Wins', fontsize=12)
                ax2.set_title('Model Win Count', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                chart_path = self.output_dir / f"comparison_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.tight_layout()
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                return chart_path
                
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            plt.close()
        return None


class MLContinuousLearnerEnhanced:
    """Enhanced ML learner dengan auto-screening semua coins"""
    
    def __init__(self, db_path: str = None):
        # Initialize collector with config
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
            'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
            'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
        }
        
        self.collector = AsterDEXDataCollector(config)
        
        # Simplified model trainer config
        if HAS_MODEL_TRAINER:
            trainer_config = {
                'models': ['xgboost', 'random_forest', 'lightgbm'],
                'optimization': 'speed',
                'use_gpu': True
            }
            self.trainer = ModelTrainer(trainer_config)
        else:
            self.trainer = None
            logger.warning("Running in screening-only mode (no ML trainer available)")
        
        # Database setup
        if db_path is None:
            db_dir = Path(__file__).parent.parent / "database"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "ml_training_enhanced.db")
        
        self.db_path = db_path
        self.setup_enhanced_database()
        
        # CSV reports directory
        self.reports_dir = Path(__file__).parent.parent / "reports" / "ml_screening"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF reports directory
        self.pdf_dir = Path(__file__).parent.parent / "reports" / "ml_pdf"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_generator = PDFReportGenerator(self.pdf_dir)
        
        # Screening results
        self.screening_results = []
        
        # Adaptive learning state
        self.learning_history = self._load_learning_history()
        
        logger.info(f"🎯 ML Enhanced Learner dengan Auto-Screening & Adaptive Learning")
        logger.info(f"   Database: {self.db_path}")
        logger.info(f"   Reports: {self.reports_dir}")
        logger.info(f"   PDF Reports: {self.pdf_dir}")
        logger.info(f"   Learning History: {len(self.learning_history)} sessions")
    
    def setup_enhanced_database(self):
        """Setup enhanced database untuk multi-coin screening"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Coin screening table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coin_screening (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL UNIQUE,
                current_price REAL,
                volume_24h REAL,
                price_change_24h REAL,
                volatility_score REAL,
                volume_score REAL,
                trend_score REAL,
                overall_score REAL,
                screening_rank INTEGER,
                recommendation TEXT,
                last_updated TEXT
            )
        """)
        
        # Pattern detection table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL,
                timeframe TEXT,
                description TEXT,
                potential_move REAL
            )
        """)
        
        # Market regime table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_regime (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime_type TEXT NOT NULL,
                btc_dominance REAL,
                total_volume REAL,
                volatility_index REAL,
                trending_coins_count INTEGER,
                ranging_coins_count INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ Enhanced database tables created")
    
    def _load_learning_history(self) -> List[Dict]:
        """Load previous training sessions for adaptive learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get last 10 training sessions
            cursor.execute("""
                SELECT symbol, best_model, best_accuracy, timestamp, data_points
                FROM training_results 
                ORDER BY timestamp DESC 
                LIMIT 100
            """)
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'symbol': row[0],
                    'best_model': row[1],
                    'best_accuracy': row[2],
                    'timestamp': row[3],
                    'data_points': row[4]
                })
            
            conn.close()
            return history
            
        except Exception as e:
            logger.warning(f"Could not load learning history: {e}")
            return []
    
    def get_adaptive_recommendations(self, symbol: str) -> Dict:
        """Get adaptive recommendations based on learning history"""
        # Check if this coin was trained before
        coin_history = [h for h in self.learning_history if h['symbol'] == symbol]
        
        if not coin_history:
            return {
                'is_new': True,
                'recommended_model': 'random_forest',  # Default for new coins
                'reason': 'New coin - using balanced RandomForest'
            }
        
        # Analyze previous performance
        recent = coin_history[0]
        avg_accuracy = sum(h['best_accuracy'] for h in coin_history) / len(coin_history)
        
        recommendations = {
            'is_new': False,
            'previous_best': recent['best_model'],
            'previous_accuracy': recent['best_accuracy'] * 100,
            'average_accuracy': avg_accuracy * 100,
            'training_count': len(coin_history),
            'last_trained': recent['timestamp']
        }
        
        # Recommend model based on history
        model_performance = {}
        for h in coin_history:
            model = h['best_model']
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(h['best_accuracy'])
        
        # Best performing model for this coin
        best_model = max(model_performance.items(), key=lambda x: sum(x[1])/len(x[1]))[0]
        recommendations['recommended_model'] = best_model
        recommendations['reason'] = f"Best historical performance ({sum(model_performance[best_model])/len(model_performance[best_model])*100:.1f}%)"
        
        return recommendations
    
    async def get_all_symbols(self) -> List[str]:
        """Get semua available symbols dari AsterDEX"""
        try:
            logger.info("📡 Fetching all available symbols from AsterDEX...")
            
            # Get dari AsterDEX API
            try:
                exchange_info = await self.collector.exchange.get_exchange_info()
                
                # Extract symbols yang aktif
                symbols = []
                if 'symbols' in exchange_info:
                    for symbol_info in exchange_info['symbols']:
                        if symbol_info.get('status') == 'TRADING':
                            symbol = symbol_info['symbol']
                            # Filter hanya USDT pairs
                            if symbol.endswith('USDT'):
                                symbols.append(symbol)
                
                if symbols:
                    logger.info(f"✅ Found {len(symbols)} active USDT pairs from AsterDEX")
                    return symbols
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch from exchange API: {e}")
            
            # Fallback: predefined symbols + auto-generate
            symbols = self._get_fallback_symbols()
            logger.info(f"✅ Using fallback symbols: {len(symbols)} symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"❌ Error fetching symbols: {e}")
            return self._get_fallback_symbols()
    
    def _get_fallback_symbols(self) -> List[str]:
        """Fallback symbols jika API tidak available"""
        # Top 100 crypto by market cap + popular trading pairs
        symbols = [
            # Top 10 Major coins
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
            
            # Top 11-30
            "LINKUSDT", "MATICUSDT", "SHIBUSDT", "UNIUSDT", "LTCUSDT",
            "BCHUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT",
            "OPUSDT", "LDOUSDT", "ETCUSDT", "XMRUSDT", "FILUSDT",
            "ALGOUSDT", "VETUSDT", "ICPUSDT", "HBARUSDT", "INJUSDT",
            
            # DeFi tokens (31-50)
            "AAVEUSDT", "MKRUSDT", "COMPUSDT", "CRVUSDT", "SNXUSDT",
            "YFIUSDT", "1INCHUSDT", "SUSHIUSDT", "RUNEUSDT", "ZILUSDT",
            
            # Layer 1/2 (51-70)
            "SUIUSDT", "STXUSDT", "FTMUSDT", "MANAUSDT", "SANDUSDT",
            "AXSUSDT", "GALAUSDT", "ENJUSDT", "CHZUSDT", "FLOWUSDT",
            
            # Gaming & Metaverse (71-85)
            "IMXUSDT", "GRTUSDT", "RENDERUSDT", "ROSEUSDT", "GMTUSDT",
            "APEUSDT", "BLZUSDT", "AUDIOUSDT", "THETAUSDT", "ICXUSDT",
            
            # Trending & Meme (86-100)
            "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT", "ORDIUSDT",
            "OCEANUSDT", "FETUSDT", "AGIXUSDT", "RLCUSDT", "NMRUSDT",
            
            # Additional high volume pairs
            "LEVERUSDT", "IDEXUSDT", "TLMUSDT", "DARUSDT", "BELUSDT"
        ]
        
        return symbols
    
    async def screen_coin(self, symbol: str) -> Dict:
        """Screen individual coin untuk potential"""
        try:
            logger.debug(f"   Screening {symbol}...")
            
            # Fetch recent data (100 candles = ~4 days on 1h timeframe)
            klines = await self.collector.exchange.get_klines(
                symbol=symbol,
                interval='1h',
                limit=100
            )
            
            if not klines or len(klines) < 50:
                return None
            
            # Convert to DataFrame
            data = self._klines_to_dataframe(klines)
            
            if data is None or len(data) < 50:
                return None
            
            # Calculate screening metrics
            metrics = self._calculate_screening_metrics(data, symbol)
            
            return metrics
            
        except Exception as e:
            logger.debug(f"   ❌ Failed to screen {symbol}: {e}")
            return None
    
    def _klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """Convert klines to DataFrame"""
        try:
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
        except Exception as e:
            logger.error(f"Error converting klines: {e}")
            return None
    
    def _calculate_screening_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate screening metrics untuk coin"""
        try:
            df = data.copy()
            
            # Basic metrics
            current_price = df['close'].iloc[-1]
            volume_24h = df['volume'].tail(24).sum() if len(df) >= 24 else df['volume'].sum()
            price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[-24] - 1) * 100 if len(df) >= 24 else 0
            
            # Volatility score (ATR based)
            high_low_range = (df['high'] - df['low']) / df['close']
            volatility_score = high_low_range.rolling(window=20).mean().iloc[-1] * 100
            
            # Volume score (relative to average)
            volume_avg = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_score = (df['volume'].iloc[-1] / volume_avg) if volume_avg > 0 else 1
            
            # Trend score (multiple timeframe)
            trend_1h = self._calculate_trend_score(df, periods=[5, 10, 20])
            
            # Pattern detection
            patterns = self._detect_patterns(df)
            
            # Overall score (0-100)
            overall_score = self._calculate_overall_score(
                volatility_score, volume_score, trend_1h, patterns
            )
            
            # Recommendation
            recommendation = self._generate_recommendation(overall_score, patterns)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'volume_24h': volume_24h,
                'price_change_24h': price_change_24h,
                'volatility_score': volatility_score,
                'volume_score': volume_score,
                'trend_score': trend_1h,
                'overall_score': overall_score,
                'patterns': patterns,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            return None
    
    def _calculate_trend_score(self, df: pd.DataFrame, periods: List[int]) -> float:
        """Calculate trend strength score"""
        try:
            scores = []
            for period in periods:
                if len(df) >= period:
                    sma = df['close'].rolling(window=period).mean()
                    current_sma = sma.iloc[-1]
                    prev_sma = sma.iloc[-2] if len(sma) > 1 else current_sma
                    
                    # Score: 0.5 = sideways, >0.5 = uptrend, <0.5 = downtrend
                    trend_strength = (current_sma / prev_sma - 1) * 100
                    normalized_score = 0.5 + (trend_strength / 10)  # Normalize
                    scores.append(max(0, min(1, normalized_score)))
            
            return np.mean(scores) if scores else 0.5
            
        except:
            return 0.5
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect technical patterns"""
        patterns = []
        
        try:
            # Simple pattern detection
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Support/Resistance breakout
            if len(close) >= 20:
                resistance = np.max(high[-20:-5])  # Resistance dari 20 candles sebelumnya
                current_high = high[-1]
                
                if current_high > resistance:
                    patterns.append({
                        'type': 'RESISTANCE_BREAKOUT',
                        'confidence': 0.7,
                        'potential_move': (current_high / resistance - 1) * 100
                    })
            
            # Trend detection
            if len(close) >= 10:
                short_ma = np.mean(close[-5:])
                long_ma = np.mean(close[-10:])
                
                if short_ma > long_ma:
                    patterns.append({
                        'type': 'UPTREND',
                        'confidence': 0.6,
                        'potential_move': 5.0  # Conservative estimate
                    })
                else:
                    patterns.append({
                        'type': 'DOWNTREND', 
                        'confidence': 0.6,
                        'potential_move': -3.0
                    })
            
        except Exception as e:
            logger.debug(f"Pattern detection error: {e}")
        
        return patterns
    
    def _calculate_overall_score(self, volatility: float, volume: float, 
                               trend: float, patterns: List[Dict]) -> float:
        """Calculate overall screening score (0-100)"""
        try:
            # Weighted scoring
            volatility_weight = 0.3
            volume_weight = 0.3
            trend_weight = 0.2
            pattern_weight = 0.2
            
            # Normalize volatility (optimal range 2-10%)
            vol_score = max(0, 1 - abs(volatility - 6) / 10)
            
            # Volume score (higher is better, but not extreme)
            vol_mult_score = min(volume, 3) / 3  # Cap at 3x average
            
            # Trend score (already normalized)
            trend_score = trend
            
            # Pattern score
            pattern_score = 0
            if patterns:
                pattern_score = sum(p['confidence'] for p in patterns) / len(patterns)
            
            overall = (vol_score * volatility_weight + 
                      vol_mult_score * volume_weight + 
                      trend_score * trend_weight + 
                      pattern_score * pattern_weight)
            
            return overall * 100
            
        except:
            return 50.0
    
    def _generate_recommendation(self, score: float, patterns: List[Dict]) -> str:
        """Generate trading recommendation"""
        if score >= 80:
            return "STRONG_BUY"
        elif score >= 70:
            return "BUY"
        elif score >= 60:
            return "WATCH"
        elif score >= 40:
            return "NEUTRAL"
        else:
            return "AVOID"
    
    async def run_mass_screening(self, max_concurrent: int = 10):
        """Run screening untuk semua coins"""
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MASS COIN SCREENING STARTED")
        logger.info("=" * 80)
        
        # Get all symbols
        symbols = await self.get_all_symbols()
        logger.info(f"📊 Screening {len(symbols)} coins...")
        
        # Run screening concurrently
        screening_results = []
        
        # Batch processing untuk avoid rate limits
        batch_size = max_concurrent
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"   Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            tasks = [self.screen_coin(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            successful_results = [r for r in batch_results if r is not None and not isinstance(r, Exception)]
            screening_results.extend(successful_results)
            
            # Small delay antara batches
            await asyncio.sleep(1)
        
        # Sort by overall score
        screening_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Save results
        self.screening_results = screening_results
        await self.save_screening_results(screening_results)
        
        logger.info(f"✅ Screening completed: {len(screening_results)}/{len(symbols)} coins analyzed")
        
        return screening_results
    
    async def save_screening_results(self, results: List[Dict]):
        """Save screening results ke database dan CSV"""
        try:
            # Save to database
            conn = sqlite3.connect(self.db_path)
            
            # Clear old results
            cursor = conn.cursor()
            cursor.execute("DELETE FROM coin_screening")
            
            # Insert new results
            for i, result in enumerate(results):
                cursor.execute("""
                    INSERT INTO coin_screening 
                    (timestamp, symbol, current_price, volume_24h, price_change_24h,
                     volatility_score, volume_score, trend_score, overall_score,
                     screening_rank, recommendation, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['timestamp'],
                    result['symbol'],
                    result['current_price'],
                    result['volume_24h'],
                    result['price_change_24h'],
                    result['volatility_score'],
                    result['volume_score'],
                    result['trend_score'],
                    result['overall_score'],
                    i + 1,  # rank
                    result['recommendation'],
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            # Save to CSV
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.reports_dir / f"coin_screening_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"💾 Screening results saved:")
            logger.info(f"   • Database: {len(results)} coins")
            logger.info(f"   • CSV: {csv_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Error saving screening results: {e}")
    
    def get_top_coins(self, top_n: int = 20, min_score: float = 60.0) -> List[Dict]:
        """Get top coins berdasarkan screening results"""
        if not self.screening_results:
            return []
        
        filtered = [coin for coin in self.screening_results 
                   if coin['overall_score'] >= min_score]
        
        return filtered[:top_n]
    
    async def smart_training_selection(self, top_n: int = 10):
        """Pilih coins terbaik untuk training berdasarkan screening + adaptive learning"""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 SMART TRAINING SELECTION WITH ADAPTIVE LEARNING")
        logger.info("=" * 80)
        
        # Lower minimum score to 45 (more realistic for crypto)
        top_coins = self.get_top_coins(top_n=top_n, min_score=45.0)
        
        if not top_coins:
            logger.warning("❌ No suitable coins found for training")
            logger.warning("   Try lowering minimum score threshold or increasing data collection period")
            return
        
        logger.info(f"🏆 Selected {len(top_coins)} coins for ML training:")
        for i, coin in enumerate(top_coins, 1):
            # Get adaptive recommendations
            adaptive = self.get_adaptive_recommendations(coin['symbol'])
            status = "🆕 NEW" if adaptive['is_new'] else f"📈 Trained {adaptive['training_count']}x"
            
            logger.info(f"   {i:2d}. {coin['symbol']:12} | Score: {coin['overall_score']:5.1f} | "
                       f"{status} | Rec Model: {adaptive['recommended_model'][:10]}")
        
        # Train models untuk top coins
        training_results = []
        
        for coin in top_coins:
            symbol = coin['symbol']
            adaptive = self.get_adaptive_recommendations(symbol)
            
            logger.info(f"\n🎓 Training ML models for {symbol}...")
            if not adaptive['is_new']:
                logger.info(f"   📊 Historical Performance: {adaptive['previous_accuracy']:.1f}% ({adaptive['previous_best']})")
                logger.info(f"   🎯 Recommended: {adaptive['recommended_model']} - {adaptive['reason']}")
            
            try:
                # Fetch full training data (1000 candles = ~40 days on 1h)
                klines = await self.collector.exchange.get_klines(
                    symbol=symbol,
                    interval='1h',
                    limit=1000
                )
                
                if not klines or len(klines) < 500:
                    logger.warning(f"⚠️ Insufficient data for {symbol}")
                    continue
                
                # Convert and process
                data = self._klines_to_dataframe(klines)
                processed_data = self._calculate_indicators_for_training(data)
                
                if processed_data is None or len(processed_data) < 100:
                    logger.warning(f"⚠️ Failed to process {symbol}")
                    continue
                
                # Train models
                results = await self._train_models_for_symbol(processed_data, symbol)
                
                if results:
                    # Add adaptive info to results
                    results['adaptive_info'] = adaptive
                    training_results.append(results)
                    self._save_training_to_db(results, symbol, coin)
                    self._save_training_csv(results, symbol)
                
            except Exception as e:
                logger.error(f"❌ Failed to train {symbol}: {e}")
            
            # Small delay antara training
            await asyncio.sleep(2)
        
        # Summary report & PDF generation
        if training_results:
            self._generate_training_summary(training_results)
            
            # Generate PDF Report
            logger.info("\n📄 Generating PDF Report...")
            pdf_path = self.pdf_generator.generate_training_report(
                training_results, 
                self.screening_results
            )
            if pdf_path:
                logger.info(f"✅ PDF Report saved: {pdf_path}")
            
            # Update learning history
            self.learning_history = self._load_learning_history()
    
    def _calculate_indicators_for_training(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ENHANCED technical indicators untuk ML training"""
        try:
            df = data.copy()
            
            # ===== TREND INDICATORS =====
            # Multiple SMAs for trend detection
            for period in [5, 10, 20, 50, 100, 200]:
                if len(df) >= period:
                    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                    df[f'sma_{period}_distance'] = (df['close'] - df[f'sma_{period}']) / df['close']
            
            # EMAs
            for period in [9, 12, 26, 50]:
                if len(df) >= period:
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # ===== MOMENTUM INDICATORS =====
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            df['macd_cross'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
            
            # CCI (Commodity Channel Index)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # ===== VOLATILITY INDICATORS =====
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            df['atr_percent'] = df['atr'] / df['close']
            
            # Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_middle'] = sma20
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ===== VOLUME INDICATORS =====
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # OBV (On-Balance Volume)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_sma'] = df['obv'].rolling(window=20).mean()
            
            # Volume-Price Trend
            df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
            
            # ===== PRICE ACTION FEATURES =====
            # Candle patterns
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            
            # Price changes (multiple periods)
            for period in [1, 3, 5, 10, 20]:
                df[f'price_change_{period}'] = df['close'].pct_change(periods=period)
                df[f'volume_change_{period}'] = df['volume'].pct_change(periods=period)
            
            # Returns and volatility
            df['returns'] = df['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            
            # ===== TREND STRENGTH =====
            # ADX (Average Directional Index) - simplified
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr14 = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr14)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr14)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
            
            # ===== SUPPORT/RESISTANCE =====
            # Recent highs and lows
            df['highest_20'] = df['high'].rolling(window=20).max()
            df['lowest_20'] = df['low'].rolling(window=20).min()
            df['distance_to_high'] = (df['highest_20'] - df['close']) / df['close']
            df['distance_to_low'] = (df['close'] - df['lowest_20']) / df['close']
            
            # ===== TARGET VARIABLE =====
            # Multi-period targets for better learning
            df['target_1'] = (df['close'].shift(-1) > df['close']).astype(int)
            df['target_3'] = (df['close'].shift(-3) > df['close']).astype(int)
            df['target_5'] = (df['close'].shift(-5) > df['close']).astype(int)
            
            # Use 1-period target as main
            df['target'] = df['target_1']
            
            # Drop last rows (no target)
            df = df[:-5]
            
            # Drop NaN values
            df = df.dropna()
            
            # Validate data quality
            if len(df) < 100:
                logger.warning(f"After indicators: only {len(df)} rows remaining")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _train_models_for_symbol(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Train OPTIMIZED ML models for a symbol"""
        try:
            # Split features and target
            exclude_cols = ['target', 'target_1', 'target_3', 'target_5', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            X = data[feature_cols]
            y = data['target']
            
            # Data quality validation
            if len(data) < 200:
                logger.warning(f"   ⚠️ Insufficient data: {len(data)} rows (minimum 200)")
                return None
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Train-test split
            split_idx = int(len(data) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"   Data: {len(X_train)} train, {len(X_test)} test | Features: {len(feature_cols)}")
            
            # Results dict
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(data),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': len(feature_cols),
                'models': {}
            }
            
            if HAS_MODEL_TRAINER and self.trainer:
                # Use ModelTrainer if available
                logger.info(f"   Training XGBoost...")
                xgb_model = self.trainer.train_xgboost(X_train, y_train)
                xgb_acc = self.trainer.evaluate_model(xgb_model, X_test, y_test)
                results['models']['xgboost'] = {'accuracy': xgb_acc}
                
                logger.info(f"   Training RandomForest...")
                rf_model = self.trainer.train_random_forest(X_train, y_train)
                rf_acc = self.trainer.evaluate_model(rf_model, X_test, y_test)
                results['models']['random_forest'] = {'accuracy': rf_acc}
                
                logger.info(f"   Training LightGBM...")
                lgb_model = self.trainer.train_lightgbm(X_train, y_train)
                lgb_acc = self.trainer.evaluate_model(lgb_model, X_test, y_test)
                results['models']['lightgbm'] = {'accuracy': lgb_acc}
            else:
                # Train all models with enhanced configuration and detailed output
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                logger.info(f"   🤖 Training Multiple Models...")
                logger.info(f"      Data: {len(X_train)} train, {len(X_test)} test | Features: {len(feature_cols)}")
                
                # 1. ENHANCED RandomForest
                logger.info(f"\n      [1/3] RandomForest (Optimized)")
                n_estimators = 200 if len(X_train) > 500 else 100
                max_depth = 20 if len(X_train) > 500 else 15
                min_samples_split = 10 if len(X_train) > 500 else 5
                
                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                
                rf_acc = accuracy_score(y_test, rf_pred)
                rf_precision = precision_score(y_test, rf_pred, zero_division=0)
                rf_recall = recall_score(y_test, rf_pred, zero_division=0)
                rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
                
                results['models']['random_forest'] = {
                    'accuracy': rf_acc,
                    'precision': rf_precision,
                    'recall': rf_recall,
                    'f1_score': rf_f1,
                    'params': f"n_est={n_estimators}, depth={max_depth}"
                }
                logger.info(f"         ✅ Acc: {rf_acc*100:.1f}% | Precision: {rf_precision*100:.1f}% | Recall: {rf_recall*100:.1f}% | F1: {rf_f1*100:.1f}%")
                
                # 2. XGBoost
                try:
                    import xgboost as xgb
                    logger.info(f"\n      [2/3] XGBoost")
                    
                    n_estimators_xgb = 200 if len(X_train) > 500 else 100
                    max_depth_xgb = 8 if len(X_train) > 500 else 6
                    
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=n_estimators_xgb,
                        max_depth=max_depth_xgb,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    
                    xgb_model.fit(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_test)
                    
                    xgb_acc = accuracy_score(y_test, xgb_pred)
                    xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
                    xgb_recall = recall_score(y_test, xgb_pred, zero_division=0)
                    xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
                    
                    results['models']['xgboost'] = {
                        'accuracy': xgb_acc,
                        'precision': xgb_precision,
                        'recall': xgb_recall,
                        'f1_score': xgb_f1,
                        'params': f"n_est={n_estimators_xgb}, depth={max_depth_xgb}, lr=0.05"
                    }
                    logger.info(f"         ✅ Acc: {xgb_acc*100:.1f}% | Precision: {xgb_precision*100:.1f}% | Recall: {xgb_recall*100:.1f}% | F1: {xgb_f1*100:.1f}%")
                except ImportError:
                    logger.warning(f"\n      [2/3] XGBoost - ⚠️ Not installed (pip install xgboost)")
                except Exception as e:
                    logger.warning(f"\n      [2/3] XGBoost - ⚠️ Training failed: {e}")
                
                # 3. LightGBM
                try:
                    import lightgbm as lgb
                    logger.info(f"\n      [3/3] LightGBM")
                    
                    n_estimators_lgb = 200 if len(X_train) > 500 else 100
                    max_depth_lgb = 8 if len(X_train) > 500 else 6
                    
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=n_estimators_lgb,
                        max_depth=max_depth_lgb,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1
                    )
                    
                    lgb_model.fit(X_train, y_train)
                    lgb_pred = lgb_model.predict(X_test)
                    
                    lgb_acc = accuracy_score(y_test, lgb_pred)
                    lgb_precision = precision_score(y_test, lgb_pred, zero_division=0)
                    lgb_recall = recall_score(y_test, lgb_pred, zero_division=0)
                    lgb_f1 = f1_score(y_test, lgb_pred, zero_division=0)
                    
                    results['models']['lightgbm'] = {
                        'accuracy': lgb_acc,
                        'precision': lgb_precision,
                        'recall': lgb_recall,
                        'f1_score': lgb_f1,
                        'params': f"n_est={n_estimators_lgb}, depth={max_depth_lgb}, lr=0.05"
                    }
                    logger.info(f"         ✅ Acc: {lgb_acc*100:.1f}% | Precision: {lgb_precision*100:.1f}% | Recall: {lgb_recall*100:.1f}% | F1: {lgb_f1*100:.1f}%")
                except ImportError:
                    logger.warning(f"\n      [3/3] LightGBM - ⚠️ Not installed (pip install lightgbm)")
                except Exception as e:
                    logger.warning(f"\n      [3/3] LightGBM - ⚠️ Training failed: {e}")
            
            # Best model selection and summary
            if results['models']:
                best_model = max(results['models'].items(), key=lambda x: x[1]['accuracy'])
                results['best_model'] = best_model[0]
                results['best_accuracy'] = best_model[1]['accuracy']
                
                # Quality score
                data_quality = min(1.0, len(data) / 1000)
                feature_quality = min(1.0, len(feature_cols) / 30)
                results['quality_score'] = (
                    results['best_accuracy'] * 0.6 +
                    data_quality * 0.3 +
                    feature_quality * 0.1
                )
                
                # Display benchmark comparison
                logger.info(f"\n   📊 MODEL BENCHMARK COMPARISON:")
                logger.info(f"   {'='*70}")
                logger.info(f"   {'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
                logger.info(f"   {'-'*70}")
                
                # Sort models by accuracy
                sorted_models = sorted(results['models'].items(), key=lambda x: x[1]['accuracy'], reverse=True)
                for model_name, metrics in sorted_models:
                    winner = "🏆" if model_name == results['best_model'] else "  "
                    logger.info(f"   {winner} {model_name:<13} | {metrics['accuracy']*100:>8.1f}% | {metrics.get('precision', 0)*100:>8.1f}% | {metrics.get('recall', 0)*100:>8.1f}% | {metrics.get('f1_score', 0)*100:>8.1f}%")
                
                logger.info(f"   {'-'*70}")
                logger.info(f"   🏆 BEST MODEL: {results['best_model'].upper()} ({results['best_accuracy']*100:.1f}%)")
                logger.info(f"   📈 Quality Score: {results['quality_score']*100:.1f}%")
                logger.info(f"   {'='*70}")
            else:
                logger.warning(f"   ⚠️ No models trained successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_training_to_db(self, results: Dict, symbol: str, coin_info: Dict):
        """Save training results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create training results table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    best_model TEXT,
                    best_accuracy REAL,
                    xgb_accuracy REAL,
                    rf_accuracy REAL,
                    lgb_accuracy REAL,
                    data_points INTEGER,
                    coin_score REAL,
                    coin_recommendation TEXT
                )
            """)
            
            # Get accuracies with defaults
            xgb_acc = results['models'].get('xgboost', {}).get('accuracy', None)
            rf_acc = results['models'].get('random_forest', {}).get('accuracy', None)
            lgb_acc = results['models'].get('lightgbm', {}).get('accuracy', None)
            
            cursor.execute("""
                INSERT INTO training_results 
                (timestamp, symbol, best_model, best_accuracy, xgb_accuracy, 
                 rf_accuracy, lgb_accuracy, data_points, coin_score, coin_recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                results['timestamp'],
                symbol,
                results['best_model'],
                results['best_accuracy'],
                xgb_acc,
                rf_acc,
                lgb_acc,
                results['data_points'],
                coin_info['overall_score'],
                coin_info['recommendation']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_training_csv(self, results: Dict, symbol: str):
        """Save training results to CSV"""
        try:
            csv_file = self.reports_dir / f"training_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Create summary with safe gets
            summary = {
                'Symbol': symbol,
                'Timestamp': results['timestamp'],
                'Best_Model': results['best_model'],
                'Best_Accuracy': results['best_accuracy'],
                'XGBoost_Accuracy': results['models'].get('xgboost', {}).get('accuracy', None),
                'RandomForest_Accuracy': results['models'].get('random_forest', {}).get('accuracy', None),
                'LightGBM_Accuracy': results['models'].get('lightgbm', {}).get('accuracy', None),
                'Data_Points': results['data_points']
            }
            
            df = pd.DataFrame([summary])
            df.to_csv(csv_file, index=False)
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_training_summary(self, training_results: List[Dict]):
        """Generate enhanced training summary report with model benchmarks"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("=" * 80)
        
        df = pd.DataFrame(training_results)
        
        logger.info(f"\n✅ Trained {len(df)} coins successfully")
        logger.info(f"   Average accuracy: {df['best_accuracy'].mean()*100:.1f}%")
        logger.info(f"   Best accuracy: {df['best_accuracy'].max()*100:.1f}%")
        logger.info(f"   Worst accuracy: {df['best_accuracy'].min()*100:.1f}%")
        
        # Model comparison - count how many times each model won
        model_counts = df['best_model'].value_counts()
        logger.info(f"\n🏆 Best Model Distribution (Winner Count):")
        for model, count in model_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"   {model:15}: {count:3d} times ({percentage:.1f}%)")
        
        # Average accuracy per model (if data available)
        logger.info(f"\n📊 Average Accuracy Per Model:")
        try:
            # Collect all model accuracies
            model_accuracies = {'random_forest': [], 'xgboost': [], 'lightgbm': []}
            
            for result in training_results:
                models = result.get('models', {})
                for model_name in ['random_forest', 'xgboost', 'lightgbm']:
                    if model_name in models and models[model_name]:
                        acc = models[model_name].get('accuracy')
                        if acc is not None:
                            model_accuracies[model_name].append(acc)
            
            # Display averages
            for model_name in ['random_forest', 'xgboost', 'lightgbm']:
                if model_accuracies[model_name]:
                    avg_acc = sum(model_accuracies[model_name]) / len(model_accuracies[model_name])
                    count = len(model_accuracies[model_name])
                    logger.info(f"   {model_name:15}: {avg_acc*100:>6.1f}% (tested on {count} coins)")
                else:
                    logger.info(f"   {model_name:15}: N/A (not available)")
        except Exception as e:
            logger.warning(f"   Could not calculate model averages: {e}")
        
        # Top performers
        logger.info(f"\n🌟 Top 5 Best Trained Coins:")
        top_5 = df.nlargest(5, 'best_accuracy')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            logger.info(f"   {i}. {row['symbol']:10} - {row['best_model']:15} - {row['best_accuracy']*100:.1f}%")
    
    def generate_screening_report(self):
        """Generate comprehensive screening report"""
        if not self.screening_results:
            logger.warning("No screening results available")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 SCREENING SUMMARY REPORT")
        logger.info("=" * 80)
        
        df = pd.DataFrame(self.screening_results)
        
        # Statistics
        logger.info(f"\n📈 Screening Statistics:")
        logger.info(f"   Total coins analyzed: {len(df)}")
        logger.info(f"   Average score: {df['overall_score'].mean():.1f}")
        logger.info(f"   Best score: {df['overall_score'].max():.1f}")
        logger.info(f"   Worst score: {df['overall_score'].min():.1f}")
        
        # Recommendations breakdown
        rec_counts = df['recommendation'].value_counts()
        logger.info(f"\n🎯 Recommendations:")
        for rec, count in rec_counts.items():
            logger.info(f"   {rec:12}: {count:3d} coins")
        
        # Top 20 coins
        top_20 = df.head(20)
        logger.info(f"\n🏆 TOP 20 COINS:")
        logger.info("-" * 80)
        logger.info("Rank | Symbol      | Score | Recommendation | Volatility | Trend")
        logger.info("-" * 80)
        
        for i, (_, coin) in enumerate(top_20.iterrows(), 1):
            logger.info(f"{i:4} | {coin['symbol']:10} | {coin['overall_score']:5.1f} | "
                       f"{coin['recommendation']:13} | {coin['volatility_score']:10.1f} | "
                       f"{coin['trend_score']:5.1f}")


async def main_enhanced():
    """Enhanced main function dengan auto-screening"""
    learner = MLContinuousLearnerEnhanced()
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 ML CONTINUOUS LEARNING - MULTI-COIN SCREENING")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📍 This will:")
    logger.info("   1. Screen ALL available coins from AsterDEX")
    logger.info("   2. Rank them by trading potential")
    logger.info("   3. Train ML models on top performers")
    logger.info("   4. Save results to database & CSV")
    logger.info("")
    
    try:
        # Step 1: Run mass screening
        logger.info("🔍 STEP 1: Mass Coin Screening")
        await learner.run_mass_screening(max_concurrent=15)
        
        # Step 2: Generate screening report
        logger.info("\n📊 STEP 2: Generate Screening Report")
        learner.generate_screening_report()
        
        # Step 3: Smart training selection
        logger.info("\n🎓 STEP 3: Train ML Models on Top Coins")
        await learner.smart_training_selection(top_n=15)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ENHANCED ML SCREENING COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📁 Check your reports:")
        logger.info(f"   • Database: {learner.db_path}")
        logger.info(f"   • CSV Reports: {learner.reports_dir}")
        logger.info("")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main_enhanced())