"""
Temporal Convolution Network (TCN) for Trading Patterns
QUANTUM LEAP V5.0 - Component 1
Target: 90%+ Accuracy

Features:
- Multi-scale temporal feature extraction
- Residual connections for gradient flow
- Causal convolution (no future leakage)
- Dilated convolution for larger receptive field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from core.logger import get_logger

logger = get_logger()


class TemporalBlock(nn.Module):
    """
    Basic TCN block with residual connection
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Causal padding to prevent future leakage
        self.padding = (kernel_size - 1) * dilation
        
        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            out: (batch, channels, seq_len)
        """
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal cropping
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TemporalConvolutionNetwork(nn.Module):
    """
    Multi-layer TCN for trading pattern recognition
    """
    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Number of input features
            num_channels: Number of channels in each layer
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.output_size = num_channels[-1]
        
        logger.info(f"🔬 TCN initialized: {input_size}→{num_channels}, RF={self._receptive_field(num_levels, kernel_size)}")
    
    def _receptive_field(self, num_levels: int, kernel_size: int) -> int:
        """Calculate receptive field size"""
        return sum([2**i * (kernel_size - 1) for i in range(num_levels)]) + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, features, seq_len)
        Returns:
            out: (batch, channels, seq_len)
        """
        return self.network(x)


class TradingTCN(nn.Module):
    """
    Complete TCN model for trading signal prediction
    """
    def __init__(
        self,
        input_size: int,
        sequence_length: int = 20,
        num_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.3,
        output_classes: int = 2  # BUY/SELL
    ):
        """
        Args:
            input_size: Number of features per timestep
            sequence_length: Length of input sequences
            num_channels: TCN layer sizes
            kernel_size: Convolution kernel
            dropout: Dropout rate
            output_classes: Number of output classes (2 for binary)
        """
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        # Temporal convolution backbone
        self.tcn = TemporalConvolutionNetwork(
            input_size,
            num_channels,
            kernel_size,
            dropout
        )
        
        # Global pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head
        pooled_size = num_channels[-1] * 2  # avg + max
        self.classifier = nn.Sequential(
            nn.Linear(pooled_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_classes)
        )
        
        # Feature extraction mode (for ensemble)
        self.feature_mode = False
        
        logger.info(f"📊 Trading TCN: seq={sequence_length}, in={input_size}, out={output_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features) or (batch, features, seq_len)
        Returns:
            logits: (batch, output_classes) or features: (batch, pooled_size)
        """
        # Ensure correct shape: (batch, features, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        if x.shape[1] != self.input_size:
            x = x.transpose(1, 2)
        
        # TCN feature extraction
        tcn_out = self.tcn(x)  # (batch, channels, seq_len)
        
        # Global pooling
        avg = self.avg_pool(tcn_out).squeeze(-1)  # (batch, channels)
        max_pool = self.max_pool(tcn_out).squeeze(-1)
        pooled = torch.cat([avg, max_pool], dim=1)  # (batch, channels*2)
        
        # Return features or predictions
        if self.feature_mode:
            return pooled
        
        # Classification
        logits = self.classifier(pooled)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal features without classification"""
        self.feature_mode = True
        features = self.forward(x)
        self.feature_mode = False
        return features


class TCNFeatureExtractor:
    """
    Scikit-learn compatible wrapper for TCN feature extraction
    """
    def __init__(
        self,
        input_size: int,
        sequence_length: int = 20,
        num_channels: List[int] = [64, 128, 256],
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        
        # Build model
        self.model = TradingTCN(
            input_size=input_size,
            sequence_length=sequence_length,
            num_channels=num_channels
        ).to(device)
        
        self.model.eval()
        
        logger.info(f"✅ TCN Extractor ready on {device}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from raw data
        
        Args:
            X: (n_samples, n_features) or (n_samples, seq_len, n_features)
        Returns:
            features: (n_samples, feature_dim)
        """
        # Reshape to sequences if needed
        if X.ndim == 2:
            X = self._create_sequences(X)
        
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i+self.batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                features = self.model.extract_features(batch_tensor)
                features_list.append(features.cpu().numpy())
        
        return np.vstack(features_list)
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Create sliding window sequences
        
        Args:
            X: (n_samples, n_features)
        Returns:
            sequences: (n_samples, seq_len, n_features)
        """
        sequences = []
        for i in range(len(X)):
            start = max(0, i - self.sequence_length + 1)
            seq = X[start:i+1]
            
            # Pad if needed
            if len(seq) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(seq), X.shape[1]))
                seq = np.vstack([padding, seq])
            
            sequences.append(seq)
        
        return np.array(sequences)


def create_temporal_features(
    X: np.ndarray,
    sequence_length: int = 20,
    num_channels: List[int] = [64, 128, 256]
) -> np.ndarray:
    """
    Convenient function to extract temporal features
    
    Args:
        X: (n_samples, n_features)
        sequence_length: Length of temporal sequences
        num_channels: TCN architecture
    Returns:
        features: (n_samples, feature_dim)
    """
    extractor = TCNFeatureExtractor(
        input_size=X.shape[1],
        sequence_length=sequence_length,
        num_channels=num_channels
    )
    
    return extractor.transform(X)


if __name__ == "__main__":
    # Test TCN
    logger.info("🧪 Testing Temporal CNN...")
    
    # Dummy data
    batch_size = 16
    seq_len = 20
    features = 22
    
    X = torch.randn(batch_size, seq_len, features)
    
    # Build model
    model = TradingTCN(
        input_size=features,
        sequence_length=seq_len,
        num_channels=[64, 128, 256]
    )
    
    # Forward pass
    logits = model(X)
    print(f"✅ Logits shape: {logits.shape}")  # (16, 2)
    
    # Feature extraction
    features_out = model.extract_features(X)
    print(f"✅ Features shape: {features_out.shape}")  # (16, 512)
    
    # Test with numpy
    X_np = np.random.randn(100, features)
    extractor = TCNFeatureExtractor(input_size=features)
    temporal_features = extractor.transform(X_np)
    print(f"✅ Extracted features: {temporal_features.shape}")  # (100, 512)
    
    logger.info("🎉 TCN test PASSED!")
