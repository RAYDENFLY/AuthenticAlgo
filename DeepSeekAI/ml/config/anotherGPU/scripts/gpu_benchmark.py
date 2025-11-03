"""
GPU Benchmark Script for Trading Bot
Tests performance with different configurations
"""

import time
import logging
from pathlib import Path

def run_gpu_benchmark():
    """Run benchmark to determine optimal settings"""
    print("=== GPU TRADING BOT BENCHMARK ===")
    
    # Test configurations
    configs = [
        ("GTX 1050 Ti 4GB", "config/gtx_1050ti_config.yaml"),
        ("GTX 1660 6GB", "config/gtx_1660_6gb.yaml"),
        ("RTX 2060 Super 8GB", "config/rtx_2060s_8gb.yaml"),
        ("RX 6600 8GB", "config/rx_6600_8gb.yaml")
    ]
    
    results = {}
    
    for gpu_name, config_path in configs:
        if not Path(config_path).exists():
            print(f"Config not found: {config_path}")
            continue
            
        print(f"\n--- Testing {gpu_name} ---")
        
        try:
            # Load config
            from core.config import load_config
            config = load_config(config_path)
            
            # Test feature engineering
            start_time = time.time()
            
            from ml.feature_engine import FeatureEngine
            from data.collector import DataCollector
            
            # Generate sample data
            data_collector = DataCollector(config)
            data = data_collector.get_historical_data('BTC/USDT', '1h', 1000)
            
            feature_engine = FeatureEngine(config)
            features = feature_engine.transform(data, fit_scaler=True)
            
            feature_time = time.time() - start_time
            
            print(f"Feature engineering: {feature_time:.2f}s")
            print(f"Features generated: {features.shape[1]}")
            
            results[gpu_name] = {
                'feature_time': feature_time,
                'feature_count': features.shape[1],
                'config_file': config_path
            }
            
        except Exception as e:
            print(f"Error testing {gpu_name}: {e}")
            results[gpu_name] = {'error': str(e)}
    
    # Print benchmark results
    print("\n=== BENCHMARK RESULTS ===")
    for gpu_name, result in results.items():
        if 'error' in result:
            print(f"{gpu_name}: ERROR - {result['error']}")
        else:
            print(f"{gpu_name}: {result['feature_time']:.2f}s for {result['feature_count']} features")

if __name__ == "__main__":
    run_gpu_benchmark()