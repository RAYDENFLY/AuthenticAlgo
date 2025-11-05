"""
QuantumMLTrainerV60 Online Continuous Learning Script - FIXED

Deskripsi:
Script yang sudah diperbaiki untuk online learning dengan AsterDEX yang benar.

Perbaikan:
1. Ganti get_tickers() dengan get_ticker_24h()
2. Tambah error handling yang lebih robust
3. Simplify untuk fokus ke 1-2 symbol dulu
"""

import os
import sys
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import traceback

# Ensure project root is in sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.logger import get_logger
from core.config import Config
from data.asterdex_collector import AsterDEXDataCollector

# Try multiple import paths for the online learner
try:
    from ml.traine.example.build_example import QuantumOnlineLearner
except ImportError:
    try:
        from ml.traine.example.build_example import QuantumOnlineLearner
    except ImportError:
        logger.warning("QuantumOnlineLearner tidak ditemukan, menggunakan fallback")
        # Fallback implementation
        class QuantumOnlineLearner:
            def __init__(self, symbol, initial_model, config):
                self.symbol = symbol
                self.model = initial_model
                self.config = config
                self.performance_history = []
            
            async def run_continuous_learning(self, duration_days=3):
                logger.info(f"Simulasi online learning untuk {self.symbol} selama {duration_days} hari")
                for i in range(duration_days * 24):
                    await asyncio.sleep(3600)  # Simulasi 1 jam
                    logger.info(f"Update {i+1}/{duration_days * 24} untuk {self.symbol}")


# Import QuantumMLTrainerV60 directly from scripts.quantum_ml_trainer_v6_0
from scripts.quantum_ml_trainer_v6_0 import QuantumMLTrainerV60

logger = get_logger()

# Konfigurasi
CONFIG_PATH = "config/ml_config_1050ti.yaml"
OUTPUT_DIR = "results/online_v6_reports"

async def get_top_symbols(collector, top_n=5):
    """Ambil top N coin/symbol dari AsterDEX dengan method yang benar."""
    try:
        ticker_24h = await collector.get_ticker_24h_safe()
        if not ticker_24h:
            logger.warning("❌ Tidak dapat mengambil data ticker, menggunakan fallback symbols")
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'][:top_n]
        logger.info(f"📊 Data ticker_24h type: {type(ticker_24h)}")
        symbols_data = []
        if isinstance(ticker_24h, list):
            for ticker in ticker_24h:
                symbol = ticker.get('symbol')
                volume = float(ticker.get('volume', 0))
                if symbol and volume > 0:
                    symbols_data.append((symbol, volume))
        elif isinstance(ticker_24h, dict) and 'symbol' in ticker_24h:
            symbol = ticker_24h.get('symbol')
            volume = float(ticker_24h.get('volume', 0))
            if symbol and volume > 0:
                symbols_data.append((symbol, volume))
        else:
            logger.warning(f"📋 Format data tidak dikenali: {type(ticker_24h)}")
        symbols_data.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, volume in symbols_data[:top_n]]
        logger.info(f"🏆 Top {top_n} symbols by volume: {top_symbols}")
        return top_symbols
    except Exception as e:
        logger.error(f"❌ Error getting top symbols: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'][:top_n]

async def run_online_learning_for_symbol(symbol, config):
    """Jalankan online learning untuk satu symbol."""
    try:
        logger.info(f"🚀 MEMULAI ONLINE LEARNING UNTUK {symbol}")
        # 1. Training model awal V6.0 dengan DEEP FEATURES
        logger.info(f"📊 Training model awal V6.0 untuk {symbol}...")
        trainer = QuantumMLTrainerV60(config)
        initial_results = await trainer.train_v60(symbol)
        if not initial_results:
            logger.error(f"❌ Gagal training model awal untuk {symbol}")
            return None
        # ✅ PASTIKAN MODEL PUNYA DEEP FEATURES
        if initial_results.get('total_features', 0) < 200:
            logger.warning(f"⚠️ Model hanya punya {initial_results.get('total_features')} features, perlu deep features!")
            return None
        # 2. Setup online learning dengan COMPLETE FEATURE PIPELINE
        online_config = {
            'learning_rate': 0.02,
            'update_interval': 3600,
            'min_samples': 4,
            'expected_feature_count': initial_results['total_features'],
            'base_feature_cols': initial_results['base_feature_cols'],
            'selector': initial_results['selector'],
            'trainer': trainer
        }
        online_learner = QuantumOnlineLearner(
            symbol=symbol,
            initial_model=initial_results,
            config=online_config
        )
        # 3. Jalankan continuous learning (durasi jam dari config)
        duration_hours = config.get('duration_hours', 1)
        logger.info(f"🔄 Memulai continuous learning untuk {symbol} ({duration_hours} jam)...")
        await online_learner.run_continuous_learning(duration_hours=duration_hours)
        return online_learner.performance_history
    except Exception as e:
        logger.error(f"❌ Error dalam online learning untuk {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_simple_report(symbol, performance_history, output_dir):
    """Generate laporan PDF sederhana dalam Bahasa Indonesia."""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Halaman cover
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 40, 'LAPORAN ONLINE LEARNING QUANTUM V6.0', 0, 1, 'C')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Symbol: {symbol}', 0, 1, 'C')
        pdf.cell(0, 10, f'Tanggal: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        
        # Halaman hasil
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'HASIL ONLINE LEARNING', 0, 1)
        pdf.ln(10)
        
        if performance_history:
            # Ringkasan performa
            initial_acc = performance_history[0]['accuracy']
            final_acc = performance_history[-1]['accuracy']
            initial_auc = performance_history[0]['auc']
            final_auc = performance_history[-1]['auc']
            
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f'Akurasi Awal: {initial_acc:.3f}', 0, 1)
            pdf.cell(0, 8, f'Akurasi Akhir: {final_acc:.3f}', 0, 1)
            pdf.cell(0, 8, f'Improvement Akurasi: {final_acc - initial_acc:+.3f}', 0, 1)
            pdf.ln(5)
            
            pdf.cell(0, 8, f'AUC Awal: {initial_auc:.3f}', 0, 1)
            pdf.cell(0, 8, f'AUC Akhir: {final_auc:.3f}', 0, 1)
            pdf.cell(0, 8, f'Improvement AUC: {final_auc - initial_auc:+.3f}', 0, 1)
            pdf.ln(10)
            
            # Rekomendasi
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'REKOMENDASI TRADING:', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            if final_auc > 0.82:
                pdf.cell(0, 8, '✅ MODEL SIAP UNTUK PRODUCTION', 0, 1)
            else:
                pdf.cell(0, 8, '⚠️ MODEL PERLU OPTIMASI LEBIH LANJUT', 0, 1)
                
            if final_acc > 0.75:
                pdf.cell(0, 8, '✅ AKURASI MEMENUHI TARGET', 0, 1)
            else:
                pdf.cell(0, 8, '⚠️ AKURASI DI BAWAH TARGET', 0, 1)
        
        else:
            pdf.cell(0, 10, 'Tidak ada data performa yang tersedia', 0, 1)
        
        # Simpan PDF
        report_file = os.path.join(output_dir, f"online_learning_report_{symbol}.pdf")
        pdf.output(report_file)
        logger.info(f"📄 Laporan PDF disimpan: {report_file}")
        
        return report_file
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None

async def main():
    """Main function yang sudah diperbaiki."""
    try:
        # Load konfigurasi
        config = Config(config_path=CONFIG_PATH)
        # Tambahkan duration_hours ke config jika ada ENV atau default
        import os
    # duration_hours is now only handled in run_online_learning_for_symbol
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Inisialisasi data collector
        collector = AsterDEXDataCollector(config=config)
        
        # Test koneksi AsterDEX
        logger.info("🧪 Testing koneksi AsterDEX...")
        try:
            ticker = await collector.exchange.get_ticker_24h_safe()
            logger.info("✅ Koneksi AsterDEX berhasil")
        except Exception as e:
            logger.error(f"❌ Koneksi AsterDEX gagal: {e}")
            return

        # Ambil top 3 symbols saja untuk testing
        top_symbols = await get_top_symbols(collector, top_n=3)
        logger.info(f"Symbol yang akan diproses: {top_symbols}")

        results = {}
        
        # Loop untuk setiap symbol
        for symbol in top_symbols:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"PROSESING SYMBOL: {symbol}")
                logger.info(f"{'='*50}")
                
                # Jalankan online learning
                performance_history = await run_online_learning_for_symbol(symbol, config)
                
                # Generate report
                if performance_history:
                    report_path = generate_simple_report(symbol, performance_history, OUTPUT_DIR)
                    results[symbol] = {
                        'performance': performance_history,
                        'report_path': report_path,
                        'final_accuracy': performance_history[-1]['accuracy'],
                        'final_auc': performance_history[-1]['auc']
                    }
                else:
                    logger.error(f"❌ Gagal online learning untuk {symbol}")
                    
                # Jeda antar symbol
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("LAPORAN AKHIR ONLINE LEARNING")
        logger.info(f"{'='*60}")
        
        for symbol, result in results.items():
            logger.info(f"📊 {symbol}:")
            logger.info(f"   Akurasi: {result['final_accuracy']:.3f}")
            logger.info(f"   AUC: {result['final_auc']:.3f}")
            logger.info(f"   Laporan: {result['report_path']}")
        
        logger.info("✅ ONLINE LEARNING SELESAI!")

    except Exception as e:
        logger.error(f"❌ Error utama: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run dengan timeout 4 hari (3 hari learning + buffer)
    try:
        import asyncio
        asyncio.run(asyncio.wait_for(main(), timeout=345600))  # 4 days in seconds
    except asyncio.TimeoutError:
        logger.error("⏰ Online learning timeout setelah 4 hari")
    except KeyboardInterrupt:
        logger.info("🛑 Online learning dihentikan oleh user")