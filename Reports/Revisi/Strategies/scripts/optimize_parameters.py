"""
Parameter Optimization untuk AuthenticAlgo
Optimize strategy parameters berdasarkan benchmark results
"""

import optuna
import pandas as pd
import logging
from datetime import datetime
import json

class ParameterOptimizer:
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger("optimizer")
    
    def optimize_rsi_macd(self, n_trials=100):
        """Optimize RSI+MACD parameters menggunakan Optuna"""
        
        def objective(trial):
            # Define parameter ranges
            rsi_period = trial.suggest_int('rsi_period', 10, 21)
            rsi_oversold = trial.suggest_int('rsi_oversold', 20, 35)
            rsi_overbought = trial.suggest_int('rsi_overbought', 65, 80)
            macd_fast = trial.suggest_int('macd_fast', 10, 15)
            macd_slow = trial.suggest_int('macd_slow', 20, 30)
            
            # Backtest dengan parameters ini
            result = self._backtest_rsi_macd_optimized(
                rsi_period, rsi_oversold, rsi_overbought, 
                macd_fast, macd_slow
            )
            
            # Objective: maximize Sharpe ratio
            return result['sharpe_ratio']
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"Best RSI+MACD parameters: {study.best_params}")
        return study.best_params
    
    def optimize_bollinger(self, n_trials=100):
        """Optimize Bollinger Bands parameters"""
        
        def objective(trial):
            bb_period = trial.suggest_int('bb_period', 15, 25)
            bb_std = trial.suggest_float('bb_std', 1.5, 2.5)
            use_volume = trial.suggest_categorical('use_volume', [True, False])
            
            result = self._backtest_bollinger_optimized(bb_period, bb_std, use_volume)
            return result['sharpe_ratio']
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"Best Bollinger parameters: {study.best_params}")
        return study.best_params

# Run optimization
def main():
    from benchmark_strategies import load_data
    
    data = load_data("BTCUSDT", "1h")
    optimizer = ParameterOptimizer(data)
    
    print("Optimizing RSI+MACD parameters...")
    best_rsi_params = optimizer.optimize_rsi_macd(n_trials=50)
    
    print("Optimizing Bollinger Bands parameters...") 
    best_bb_params = optimizer.optimize_bollinger(n_trials=50)
    
    # Save optimized parameters
    optimized_params = {
        'timestamp': datetime.now().isoformat(),
        'rsi_macd': best_rsi_params,
        'bollinger_bands': best_bb_params
    }
    
    with open('configs/optimized_parameters.json', 'w') as f:
        json.dump(optimized_params, f, indent=2)
    
    print("Optimization complete! Parameters saved to configs/optimized_parameters.json")

if __name__ == "__main__":
    main()