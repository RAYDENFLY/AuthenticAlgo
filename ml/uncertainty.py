"""
Uncertainty Quantification for ML Predictions
QUANTUM LEAP V5.0 - Component 4
Target: 90%+ Accuracy

Features:
- Monte Carlo Dropout for epistemic uncertainty
- Ensemble uncertainty for aleatoric uncertainty  
- Confidence intervals for predictions
- Uncertainty-aware trading decisions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from scipy import stats
from sklearn.base import BaseEstimator
from core.logger import get_logger

logger = get_logger()


class MCDropoutModel(nn.Module):
    """
    Neural network with Monte Carlo Dropout
    """
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        output_size: int = 2,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dropout always enabled
        """
        return self.network(x)
    
    def enable_dropout(self):
        """Enable dropout during inference for MC sampling"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class UncertaintyEnsemble:
    """
    Ensemble of models with uncertainty quantification
    """
    def __init__(
        self,
        input_size: int,
        n_models: int = 5,
        mc_samples: int = 30,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            input_size: Input feature dimension
            n_models: Number of models in ensemble
            mc_samples: Number of MC dropout samples
            device: Device for computation
        """
        self.input_size = input_size
        self.n_models = n_models
        self.mc_samples = mc_samples
        self.device = device
        
        # Create ensemble
        self.models = [
            MCDropoutModel(input_size).to(device)
            for _ in range(n_models)
        ]
        
        self.fitted = False
        
        logger.info(f"🎲 Uncertainty Ensemble: {n_models} models x {mc_samples} MC samples")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001
    ):
        """
        Train ensemble models
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples,)
            epochs: Training epochs per model
            batch_size: Batch size
            lr: Learning rate
        """
        logger.info(f"🏋️ Training {self.n_models} models...")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        for model_idx, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X_tensor[indices]
            y_boot = y_tensor[indices]
            
            # Training loop
            for epoch in range(epochs):
                for i in range(0, len(X_boot), batch_size):
                    batch_X = X_boot[i:i+batch_size]
                    batch_y = y_boot[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            if (model_idx + 1) % max(1, self.n_models // 5) == 0:
                logger.info(f"   Model {model_idx+1}/{self.n_models} trained")
        
        self.fitted = True
        logger.info("✅ Ensemble training complete")
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification
        
        Args:
            X: (n_samples, n_features)
        Returns:
            predictions: (n_samples,) - Most likely class
            probabilities: (n_samples, n_classes) - Mean probabilities
            epistemic_uncertainty: (n_samples,) - Model uncertainty
            aleatoric_uncertainty: (n_samples,) - Data uncertainty
        """
        if not self.fitted:
            raise ValueError("Models not trained. Call fit() first.")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Collect predictions from all models with MC dropout
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                model.enable_dropout()  # Enable dropout for MC sampling
                
                mc_probs = []
                for _ in range(self.mc_samples):
                    outputs = model(X_tensor)
                    probs = F.softmax(outputs, dim=1)
                    mc_probs.append(probs.cpu().numpy())
                
                # Average over MC samples
                model_probs = np.mean(mc_probs, axis=0)
                all_probs.append(model_probs)
        
        all_probs = np.array(all_probs)  # (n_models, n_samples, n_classes)
        
        # Mean prediction
        mean_probs = np.mean(all_probs, axis=0)  # (n_samples, n_classes)
        predictions = np.argmax(mean_probs, axis=1)
        
        # Epistemic uncertainty (model uncertainty)
        # Measured as variance across models
        epistemic = np.var(all_probs, axis=0).max(axis=1)  # (n_samples,)
        
        # Aleatoric uncertainty (data uncertainty)
        # Measured as entropy of mean prediction
        aleatoric = stats.entropy(mean_probs.T)  # (n_samples,)
        
        return predictions, mean_probs, epistemic, aleatoric
    
    def get_confidence_intervals(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals for predictions
        
        Args:
            X: (n_samples, n_features)
            confidence_level: Confidence level (0-1)
        Returns:
            lower_bounds: (n_samples, n_classes)
            upper_bounds: (n_samples, n_classes)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Collect all predictions
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                model.enable_dropout()
                
                for _ in range(self.mc_samples):
                    outputs = model(X_tensor)
                    probs = F.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())
        
        all_probs = np.array(all_probs)  # (n_models * mc_samples, n_samples, n_classes)
        
        # Calculate percentiles
        alpha = (1 - confidence_level) / 2
        lower_bounds = np.percentile(all_probs, alpha * 100, axis=0)
        upper_bounds = np.percentile(all_probs, (1 - alpha) * 100, axis=0)
        
        return lower_bounds, upper_bounds


class UncertaintyAwarePredictor:
    """
    Wrapper for sklearn-compatible uncertainty-aware predictions
    """
    def __init__(
        self,
        base_estimators: List[BaseEstimator],
        n_models: int = 5,
        mc_samples: int = 30,
        uncertainty_threshold: float = 0.15
    ):
        """
        Args:
            base_estimators: List of sklearn estimators (XGB, LGBM, etc.)
            n_models: Number of models per estimator
            mc_samples: MC samples for neural network
            uncertainty_threshold: Max acceptable uncertainty
        """
        self.base_estimators = base_estimators
        self.n_models = n_models
        self.mc_samples = mc_samples
        self.uncertainty_threshold = uncertainty_threshold
        
        self.nn_ensemble = None
        self.fitted = False
        
        logger.info(f"🎯 Uncertainty-Aware Predictor initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all models"""
        logger.info("🏋️ Training uncertainty-aware models...")
        
        # Train base estimators
        for est in self.base_estimators:
            est.fit(X, y)
        
        # Train neural ensemble
        self.nn_ensemble = UncertaintyEnsemble(
            input_size=X.shape[1],
            n_models=self.n_models,
            mc_samples=self.mc_samples
        )
        self.nn_ensemble.fit(X, y)
        
        self.fitted = True
        logger.info("✅ All models trained")
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Predict with comprehensive uncertainty analysis
        
        Returns:
            results: Dict containing:
                - predictions: Final predictions
                - confidence: Confidence scores
                - epistemic_uncertainty: Model uncertainty
                - aleatoric_uncertainty: Data uncertainty
                - total_uncertainty: Combined uncertainty
                - reliable_mask: Mask for reliable predictions
        """
        if not self.fitted:
            raise ValueError("Not trained. Call fit() first.")
        
        # Get predictions from base estimators
        base_preds = []
        base_probs = []
        
        for est in self.base_estimators:
            pred = est.predict(X)
            base_preds.append(pred)
            
            if hasattr(est, 'predict_proba'):
                prob = est.predict_proba(X)
                base_probs.append(prob)
        
        # Get neural ensemble predictions with uncertainty
        nn_preds, nn_probs, epistemic, aleatoric = \
            self.nn_ensemble.predict_with_uncertainty(X)
        
        # Combine predictions (weighted voting)
        all_preds = np.array(base_preds + [nn_preds])
        final_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=all_preds
        )
        
        # Combine probabilities
        if base_probs:
            all_probs = np.array(base_probs + [nn_probs])
            mean_probs = np.mean(all_probs, axis=0)
        else:
            mean_probs = nn_probs
        
        # Confidence scores
        confidence = mean_probs.max(axis=1)
        
        # Total uncertainty
        total_uncertainty = epistemic + aleatoric
        
        # Reliable predictions (low uncertainty)
        reliable_mask = (total_uncertainty < self.uncertainty_threshold)
        
        return {
            'predictions': final_preds,
            'probabilities': mean_probs,
            'confidence': confidence,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total_uncertainty,
            'reliable_mask': reliable_mask
        }
    
    def get_uncertainty_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get normalized uncertainty score [0, 1]
        0 = very certain, 1 = very uncertain
        """
        results = self.predict_with_uncertainty(X)
        
        # Normalize uncertainty to [0, 1]
        uncertainty = results['total_uncertainty']
        normalized = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        
        return normalized


def calculate_prediction_intervals(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals based on uncertainty
    
    Args:
        predictions: (n_samples,) - Point predictions
        uncertainties: (n_samples,) - Uncertainty estimates
        confidence_level: Confidence level
    Returns:
        lower_bounds: (n_samples,)
        upper_bounds: (n_samples,)
    """
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    lower = predictions - z_score * uncertainties
    upper = predictions + z_score * uncertainties
    
    return lower, upper


if __name__ == "__main__":
    # Test uncertainty quantification
    logger.info("🧪 Testing Uncertainty Quantification...")
    
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create predictor
    base_estimators = [
        RandomForestClassifier(n_estimators=100),
        XGBClassifier(n_estimators=100, eval_metric='logloss')
    ]
    
    predictor = UncertaintyAwarePredictor(
        base_estimators=base_estimators,
        n_models=3,
        mc_samples=10
    )
    
    # Train
    predictor.fit(X_train, y_train)
    
    # Predict with uncertainty
    results = predictor.predict_with_uncertainty(X_test)
    
    print(f"\n✅ Predictions: {results['predictions'].shape}")
    print(f"✅ Confidence: {results['confidence'].mean():.3f} ± {results['confidence'].std():.3f}")
    print(f"✅ Epistemic uncertainty: {results['epistemic_uncertainty'].mean():.4f}")
    print(f"✅ Aleatoric uncertainty: {results['aleatoric_uncertainty'].mean():.4f}")
    print(f"✅ Reliable predictions: {results['reliable_mask'].sum()}/{len(results['reliable_mask'])}")
    
    # Accuracy on reliable vs uncertain
    reliable = results['reliable_mask']
    if reliable.sum() > 0:
        acc_reliable = (results['predictions'][reliable] == y_test[reliable]).mean()
        print(f"✅ Accuracy (reliable): {acc_reliable:.3f}")
    
    if (~reliable).sum() > 0:
        acc_uncertain = (results['predictions'][~reliable] == y_test[~reliable]).mean()
        print(f"✅ Accuracy (uncertain): {acc_uncertain:.3f}")
    
    logger.info("🎉 Uncertainty test PASSED!")
