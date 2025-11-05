"""
Multi-Head Attention Mechanism for Market Regimes
QUANTUM LEAP V5.0 - Component 2
Target: 90%+ Accuracy

Features:
- Self-attention across temporal features
- Regime-aware attention weighting
- Cross-attention between market regimes
- Positional encoding for temporal data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
from core.logger import get_logger

logger = get_logger()


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal data
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: (batch, seq_len) or (batch, seq_len, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights


class RegimeAttentionLayer(nn.Module):
    """
    Attention layer with regime-aware weighting
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_regimes: int = 4,  # trending, ranging, high_vol, low_vol
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Regime embedding
        self.regime_embedding = nn.Embedding(num_regimes, d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            regime_ids: (batch,) - regime IDs for each sample
            mask: (batch, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Add regime information if provided
        if regime_ids is not None:
            regime_emb = self.regime_embedding(regime_ids).unsqueeze(1)  # (batch, 1, d_model)
            x = x + regime_emb
        
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x, attn_weights


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal trading data
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_regimes: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            RegimeAttentionLayer(d_model, num_heads, num_regimes, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"🔬 Transformer: d={d_model}, heads={num_heads}, layers={num_layers}")
    
    def forward(
        self,
        x: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (batch, seq_len, input_size)
            regime_ids: (batch,)
            mask: (batch, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: List of attention weights from each layer
        """
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        attention_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, regime_ids, mask)
            attention_weights_list.append(attn_weights)
        
        return x, attention_weights_list


class TradingTransformer(nn.Module):
    """
    Complete Transformer model for trading signal prediction
    """
    def __init__(
        self,
        input_size: int,
        sequence_length: int = 20,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_regimes: int = 4,
        dropout: float = 0.3,
        output_classes: int = 2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        # Transformer encoder
        self.transformer = TemporalTransformer(
            input_size,
            d_model,
            num_heads,
            num_layers,
            num_regimes,
            dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_classes)
        )
        
        # Feature extraction mode
        self.feature_mode = False
        
        logger.info(f"📊 Trading Transformer: seq={sequence_length}, d_model={d_model}")
    
    def forward(
        self,
        x: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
            regime_ids: (batch,)
            mask: (batch, seq_len)
        Returns:
            logits: (batch, output_classes) or features: (batch, d_model)
        """
        # Ensure correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Transformer encoding
        encoded, _ = self.transformer(x, regime_ids, mask)
        
        # Use last timestep for classification
        features = encoded[:, -1, :]  # (batch, d_model)
        
        if self.feature_mode:
            return features
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def extract_features(
        self,
        x: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract attention-weighted features"""
        self.feature_mode = True
        features = self.forward(x, regime_ids)
        self.feature_mode = False
        return features
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None
    ) -> list:
        """Get attention weights for visualization"""
        _, attention_weights = self.transformer(x, regime_ids)
        return attention_weights


class AttentionFeatureExtractor:
    """
    Scikit-learn compatible wrapper for attention feature extraction
    """
    def __init__(
        self,
        input_size: int,
        sequence_length: int = 20,
        d_model: int = 256,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        
        # Build model
        self.model = TradingTransformer(
            input_size=input_size,
            sequence_length=sequence_length,
            d_model=d_model
        ).to(device)
        
        self.model.eval()
        
        logger.info(f"✅ Attention Extractor ready on {device}")
    
    def transform(
        self,
        X: np.ndarray,
        regime_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract attention-weighted features
        
        Args:
            X: (n_samples, n_features) or (n_samples, seq_len, n_features)
            regime_ids: (n_samples,) - Optional regime IDs
        Returns:
            features: (n_samples, d_model)
        """
        if X.ndim == 2:
            X = self._create_sequences(X)
        
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i+self.batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                regime_tensor = None
                if regime_ids is not None:
                    regime_batch = regime_ids[i:i+self.batch_size]
                    regime_tensor = torch.LongTensor(regime_batch).to(self.device)
                
                features = self.model.extract_features(batch_tensor, regime_tensor)
                features_list.append(features.cpu().numpy())
        
        return np.vstack(features_list)
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sliding window sequences"""
        sequences = []
        for i in range(len(X)):
            start = max(0, i - self.sequence_length + 1)
            seq = X[start:i+1]
            
            if len(seq) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(seq), X.shape[1]))
                seq = np.vstack([padding, seq])
            
            sequences.append(seq)
        
        return np.array(sequences)


def create_attention_features(
    X: np.ndarray,
    sequence_length: int = 20,
    d_model: int = 256,
    regime_ids: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convenient function to extract attention features
    
    Args:
        X: (n_samples, n_features)
        sequence_length: Sequence length
        d_model: Model dimension
        regime_ids: Optional regime IDs
    Returns:
        features: (n_samples, d_model)
    """
    extractor = AttentionFeatureExtractor(
        input_size=X.shape[1],
        sequence_length=sequence_length,
        d_model=d_model
    )
    
    return extractor.transform(X, regime_ids)


if __name__ == "__main__":
    # Test Transformer
    logger.info("🧪 Testing Multi-Head Attention...")
    
    batch_size = 16
    seq_len = 20
    features = 22
    
    X = torch.randn(batch_size, seq_len, features)
    regime_ids = torch.randint(0, 4, (batch_size,))
    
    # Build model
    model = TradingTransformer(
        input_size=features,
        sequence_length=seq_len,
        d_model=256,
        num_heads=8,
        num_layers=4
    )
    
    # Forward pass
    logits = model(X, regime_ids)
    print(f"✅ Logits shape: {logits.shape}")  # (16, 2)
    
    # Feature extraction
    features_out = model.extract_features(X, regime_ids)
    print(f"✅ Features shape: {features_out.shape}")  # (16, 256)
    
    # Attention weights
    attn_weights = model.get_attention_weights(X, regime_ids)
    print(f"✅ Attention layers: {len(attn_weights)}")
    print(f"✅ Attention shape: {attn_weights[0].shape}")  # (16, 8, 20, 20)
    
    # Test with numpy
    X_np = np.random.randn(100, features)
    regime_np = np.random.randint(0, 4, 100)
    extractor = AttentionFeatureExtractor(input_size=features)
    attention_features = extractor.transform(X_np, regime_np)
    print(f"✅ Extracted features: {attention_features.shape}")  # (100, 256)
    
    logger.info("🎉 Attention test PASSED!")
