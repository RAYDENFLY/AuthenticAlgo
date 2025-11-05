"""
Test V5.0 Components
Quick test for TCN, Attention, RL, and Uncertainty modules
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logger
import numpy as np

logger = setup_logger()

def test_v5_components():
    """Test all V5.0 components"""
    
    logger.info("🧪 TESTING QUANTUM LEAP V5.0 COMPONENTS")
    logger.info("="*70)
    
    # Generate dummy data
    n_samples = 200
    n_features = 22
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # 1. Test TCN
    logger.info("\n1️⃣ Testing Temporal CNN...")
    try:
        from ml.temporal_cnn import TCNFeatureExtractor
        
        tcn = TCNFeatureExtractor(
            input_size=n_features,
            sequence_length=20,
            num_channels=[32, 64, 128],  # Smaller for quick test
            batch_size=16
        )
        
        tcn_features = tcn.transform(X)
        logger.info(f"   ✅ TCN output shape: {tcn_features.shape}")
        assert tcn_features.shape[0] == n_samples
        logger.info("   ✅ TCN test PASSED!")
        
    except Exception as e:
        logger.error(f"   ❌ TCN test FAILED: {e}")
        return False
    
    # 2. Test Attention
    logger.info("\n2️⃣ Testing Multi-Head Attention...")
    try:
        from ml.attention import AttentionFeatureExtractor
        
        attention = AttentionFeatureExtractor(
            input_size=n_features,
            sequence_length=20,
            d_model=128,  # Smaller for quick test
            batch_size=16
        )
        
        regime_ids = np.random.randint(0, 4, n_samples)
        attention_features = attention.transform(X, regime_ids)
        logger.info(f"   ✅ Attention output shape: {attention_features.shape}")
        assert attention_features.shape[0] == n_samples
        logger.info("   ✅ Attention test PASSED!")
        
    except Exception as e:
        logger.error(f"   ❌ Attention test FAILED: {e}")
        return False
    
    # 3. Test RL Optimizer
    logger.info("\n3️⃣ Testing RL Threshold Optimizer...")
    try:
        from ml.rl_optimizer import RLThresholdOptimizer
        
        predictions = np.random.randint(0, 2, n_samples)
        confidences = np.random.uniform(0.5, 1.0, n_samples)
        returns = np.random.randn(n_samples) * 0.02
        regimes = np.random.randint(0, 4, n_samples)
        volatilities = np.random.uniform(0.01, 0.05, n_samples)
        momentums = np.random.randn(n_samples)
        
        optimizer = RLThresholdOptimizer(n_episodes=5, window_size=30)  # Quick test
        best_thresholds = optimizer.optimize(
            predictions, confidences, y, returns,
            regimes, volatilities, momentums
        )
        
        logger.info(f"   ✅ Optimized thresholds: {best_thresholds}")
        assert len(best_thresholds) == 4
        logger.info("   ✅ RL test PASSED!")
        
    except Exception as e:
        logger.error(f"   ❌ RL test FAILED: {e}")
        return False
    
    # 4. Test Uncertainty
    logger.info("\n4️⃣ Testing Uncertainty Quantification...")
    try:
        from ml.uncertainty import UncertaintyAwarePredictor
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        
        # Split data
        split = int(0.7 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        base_estimators = [
            RandomForestClassifier(n_estimators=50),
            XGBClassifier(n_estimators=50, eval_metric='logloss')
        ]
        
        predictor = UncertaintyAwarePredictor(
            base_estimators=base_estimators,
            n_models=2,  # Small for quick test
            mc_samples=10
        )
        
        predictor.fit(X_train, y_train)
        results = predictor.predict_with_uncertainty(X_test)
        
        logger.info(f"   ✅ Predictions: {results['predictions'].shape}")
        logger.info(f"   ✅ Confidence: {results['confidence'].mean():.3f}")
        logger.info(f"   ✅ Uncertainty: {results['total_uncertainty'].mean():.4f}")
        logger.info(f"   ✅ Reliable predictions: {results['reliable_mask'].sum()}/{len(results['reliable_mask'])}")
        
        # Calculate accuracy
        acc = (results['predictions'] == y_test).mean()
        logger.info(f"   ✅ Test accuracy: {acc:.3f}")
        
        logger.info("   ✅ Uncertainty test PASSED!")
        
    except Exception as e:
        logger.error(f"   ❌ Uncertainty test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All tests passed
    logger.info("\n" + "="*70)
    logger.info("🎉 ALL V5.0 COMPONENTS TESTED SUCCESSFULLY!")
    logger.info("✅ TCN: Temporal pattern extraction")
    logger.info("✅ Attention: Regime-aware weighting")
    logger.info("✅ RL: Dynamic threshold optimization")
    logger.info("✅ Uncertainty: Confidence quantification")
    logger.info("\n🚀 Ready for V5.0 integration!")
    
    return True


if __name__ == "__main__":
    success = test_v5_components()
    sys.exit(0 if success else 1)
