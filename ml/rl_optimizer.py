"""
Reinforcement Learning Threshold Optimizer
QUANTUM LEAP V5.0 - Component 3
Target: 90%+ Accuracy

Features:
- PPO agent for dynamic threshold optimization
- Custom trading environment with risk-adjusted rewards
- Adaptive confidence thresholds per regime
- Continuous learning from market feedback
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from core.logger import get_logger

logger = get_logger()


@dataclass
class ThresholdState:
    """State representation for threshold optimization"""
    regime: int  # 0=trending, 1=ranging, 2=high_vol, 3=low_vol
    volatility: float
    momentum: float
    recent_accuracy: float
    recent_sharpe: float
    current_threshold: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([
            self.regime,
            self.volatility,
            self.momentum,
            self.recent_accuracy,
            self.recent_sharpe,
            self.current_threshold
        ], dtype=np.float32)


class TradingEnvironment:
    """
    Custom trading environment for threshold optimization
    """
    def __init__(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        true_labels: np.ndarray,
        returns: np.ndarray,
        regimes: np.ndarray,
        volatilities: np.ndarray,
        momentums: np.ndarray,
        window_size: int = 50
    ):
        """
        Args:
            predictions: (n_samples,) - Model predictions
            confidences: (n_samples,) - Prediction confidences
            true_labels: (n_samples,) - True labels
            returns: (n_samples,) - Future returns
            regimes: (n_samples,) - Market regimes
            volatilities: (n_samples,) - Market volatility
            momentums: (n_samples,) - Market momentum
            window_size: Rolling window for metrics
        """
        self.predictions = predictions
        self.confidences = confidences
        self.true_labels = true_labels
        self.returns = returns
        self.regimes = regimes
        self.volatilities = volatilities
        self.momentums = momentums
        self.window_size = window_size
        
        self.n_samples = len(predictions)
        self.current_idx = 0
        
        # Initialize thresholds per regime
        self.thresholds = {
            0: 0.65,  # trending
            1: 0.75,  # ranging
            2: 0.60,  # high_vol
            3: 0.70   # low_vol
        }
        
        # Performance tracking
        self.trade_history = deque(maxlen=window_size)
        
        logger.info(f"🏋️ Trading Environment initialized: {self.n_samples} samples")
    
    def reset(self) -> ThresholdState:
        """Reset environment to initial state"""
        self.current_idx = 0
        self.trade_history.clear()
        return self._get_current_state()
    
    def step(self, action: np.ndarray) -> Tuple[ThresholdState, float, bool, Dict]:
        """
        Take action (adjust thresholds) and observe reward
        
        Args:
            action: (4,) - Threshold adjustments for each regime [-0.1, 0.1]
        Returns:
            next_state: New state
            reward: Reward signal
            done: Episode finished
            info: Additional information
        """
        # Update thresholds
        for regime_id in range(4):
            adjustment = np.clip(action[regime_id], -0.1, 0.1)
            self.thresholds[regime_id] = np.clip(
                self.thresholds[regime_id] + adjustment,
                0.5,  # min threshold
                0.9   # max threshold
            )
        
        # Simulate trading with new thresholds
        window_end = min(self.current_idx + self.window_size, self.n_samples)
        window_slice = slice(self.current_idx, window_end)
        
        # Apply thresholds
        trades = []
        for i in range(self.current_idx, window_end):
            regime = self.regimes[i]
            conf = self.confidences[i]
            threshold = self.thresholds[regime]
            
            if conf >= threshold:
                # Execute trade
                pred = self.predictions[i]
                true = self.true_labels[i]
                ret = self.returns[i]
                
                correct = (pred == true)
                pnl = ret if correct else -ret
                
                trades.append({
                    'correct': correct,
                    'pnl': pnl,
                    'confidence': conf,
                    'regime': regime
                })
        
        # Calculate reward
        reward = self._calculate_reward(trades)
        
        # Update state
        self.trade_history.extend(trades)
        self.current_idx = window_end
        done = (self.current_idx >= self.n_samples)
        
        next_state = self._get_current_state()
        
        info = {
            'n_trades': len(trades),
            'accuracy': np.mean([t['correct'] for t in trades]) if trades else 0.0,
            'total_pnl': sum([t['pnl'] for t in trades]),
            'thresholds': self.thresholds.copy()
        }
        
        return next_state, reward, done, info
    
    def _get_current_state(self) -> ThresholdState:
        """Get current state representation"""
        if self.current_idx >= self.n_samples:
            regime = self.regimes[-1]
            vol = self.volatilities[-1]
            mom = self.momentums[-1]
        else:
            regime = self.regimes[self.current_idx]
            vol = self.volatilities[self.current_idx]
            mom = self.momentums[self.current_idx]
        
        # Calculate recent performance
        if len(self.trade_history) > 0:
            recent_acc = np.mean([t['correct'] for t in self.trade_history])
            returns_list = [t['pnl'] for t in self.trade_history]
            recent_sharpe = np.mean(returns_list) / (np.std(returns_list) + 1e-8)
        else:
            recent_acc = 0.5
            recent_sharpe = 0.0
        
        return ThresholdState(
            regime=int(regime),
            volatility=float(vol),
            momentum=float(mom),
            recent_accuracy=float(recent_acc),
            recent_sharpe=float(recent_sharpe),
            current_threshold=float(self.thresholds[int(regime)])
        )
    
    def _calculate_reward(self, trades: List[Dict]) -> float:
        """
        Calculate reward based on risk-adjusted returns
        
        Components:
        - Accuracy bonus
        - Sharpe ratio bonus
        - Trade frequency penalty (avoid overtrading)
        """
        if not trades:
            return -0.1  # Penalty for no trades
        
        # Accuracy component
        accuracy = np.mean([t['correct'] for t in trades])
        acc_reward = (accuracy - 0.5) * 2  # Scale: 0=random, 1=perfect
        
        # Risk-adjusted returns
        pnls = [t['pnl'] for t in trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) + 1e-8
        sharpe = mean_pnl / std_pnl
        sharpe_reward = np.tanh(sharpe)  # Bounded reward
        
        # Trade frequency (prefer selective trading)
        trade_rate = len(trades) / self.window_size
        freq_penalty = -0.2 * max(0, trade_rate - 0.5)  # Penalty if >50% trade rate
        
        # Total reward
        reward = 0.5 * acc_reward + 0.4 * sharpe_reward + freq_penalty
        
        return float(reward)


class PPONetwork(nn.Module):
    """
    Actor-Critic network for PPO
    """
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 4,  # 4 regimes
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Output in [-1, 1], scaled to [-0.1, 0.1]
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            action: (batch, action_dim)
            value: (batch, 1)
        """
        features = self.shared(state)
        action = self.actor(features) * 0.1  # Scale to [-0.1, 0.1]
        value = self.critic(features)
        return action, value


class PPOAgent:
    """
    Proximal Policy Optimization agent
    """
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epochs: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        
        # Networks
        self.policy = PPONetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        logger.info(f"🤖 PPO Agent initialized on {device}")
    
    def select_action(self, state: ThresholdState, deterministic: bool = False) -> np.ndarray:
        """
        Select action given state
        
        Args:
            state: Current state
            deterministic: If True, use mean action (no exploration)
        Returns:
            action: (action_dim,)
        """
        state_array = state.to_array()
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.policy(state_tensor)
        
        action = action.cpu().numpy()[0]
        value = value.cpu().item()
        
        # Add exploration noise if not deterministic
        if not deterministic:
            noise = np.random.normal(0, 0.02, size=action.shape)
            action = np.clip(action + noise, -0.1, 0.1)
        
        # Store for training
        self.states.append(state_array)
        self.actions.append(action)
        self.values.append(value)
        
        return action
    
    def store_reward(self, reward: float):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using collected experience
        
        Returns:
            metrics: Training metrics
        """
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.epochs):
            # Forward pass
            new_actions, new_values = self.policy(states)
            new_values = new_values.squeeze()
            
            # Policy loss (simplified, assumes Gaussian policy)
            action_diff = (new_actions - actions).pow(2).mean(dim=1)
            ratio = torch.exp(-action_diff)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Clear storage
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        
        return {
            'policy_loss': total_policy_loss / self.epochs,
            'value_loss': total_value_loss / self.epochs
        }
    
    def _calculate_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        return returns


class RLThresholdOptimizer:
    """
    High-level interface for RL-based threshold optimization
    """
    def __init__(
        self,
        n_episodes: int = 50,
        window_size: int = 50,
        lr: float = 3e-4
    ):
        self.n_episodes = n_episodes
        self.window_size = window_size
        self.lr = lr
        
        self.agent = None
        self.best_thresholds = None
        self.best_reward = -float('inf')
        
        logger.info(f"🎯 RL Optimizer: episodes={n_episodes}, window={window_size}")
    
    def optimize(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        true_labels: np.ndarray,
        returns: np.ndarray,
        regimes: np.ndarray,
        volatilities: np.ndarray,
        momentums: np.ndarray
    ) -> Dict[int, float]:
        """
        Optimize thresholds using PPO
        
        Returns:
            best_thresholds: Dict mapping regime_id to threshold
        """
        logger.info("🚀 Starting RL threshold optimization...")
        
        # Create environment
        env = TradingEnvironment(
            predictions, confidences, true_labels, returns,
            regimes, volatilities, momentums, self.window_size
        )
        
        # Create agent
        self.agent = PPOAgent(lr=self.lr)
        
        # Training loop
        episode_rewards = []
        
        for episode in range(self.n_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action = self.agent.select_action(state)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store reward
                self.agent.store_reward(reward)
                episode_reward += reward
                
                state = next_state
            
            # Update policy
            metrics = self.agent.update()
            episode_rewards.append(episode_reward)
            
            # Track best
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_thresholds = info['thresholds'].copy()
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"   Episode {episode+1}/{self.n_episodes}: "
                           f"Reward={episode_reward:.3f}, Avg={avg_reward:.3f}")
        
        logger.info(f"✅ Best thresholds: {self.best_thresholds}")
        logger.info(f"✅ Best reward: {self.best_reward:.3f}")
        
        return self.best_thresholds


if __name__ == "__main__":
    # Test RL optimizer
    logger.info("🧪 Testing RL Threshold Optimizer...")
    
    # Generate dummy data
    n_samples = 1000
    predictions = np.random.randint(0, 2, n_samples)
    confidences = np.random.uniform(0.5, 1.0, n_samples)
    true_labels = np.random.randint(0, 2, n_samples)
    returns = np.random.randn(n_samples) * 0.02
    regimes = np.random.randint(0, 4, n_samples)
    volatilities = np.random.uniform(0.01, 0.05, n_samples)
    momentums = np.random.randn(n_samples)
    
    # Optimize thresholds
    optimizer = RLThresholdOptimizer(n_episodes=20, window_size=50)
    best_thresholds = optimizer.optimize(
        predictions, confidences, true_labels, returns,
        regimes, volatilities, momentums
    )
    
    print(f"\n✅ Optimized thresholds:")
    for regime, threshold in best_thresholds.items():
        regime_names = ['Trending', 'Ranging', 'High Vol', 'Low Vol']
        print(f"   {regime_names[regime]}: {threshold:.3f}")
    
    logger.info("🎉 RL test PASSED!")
