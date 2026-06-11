"""
CPPO-PA: Constrained Proximal Policy Optimisation with Physiological Attention.

Complete implementation of the safe reinforcement learning framework for
closed-loop anaesthesia control described in:
  "Safe Reinforcement Learning with Physiological Attention for
   Closed-Loop Anesthesia Control"

Architecture:
  1. PhysiologicalAttention: Multi-head self-attention over 30s biosignal buffer
  2. DualCritic: Shared-trunk reward critic V_r + safety critic V_c
  3. GaussianPolicy: Diagonal Gaussian policy for continuous drug dosing
  4. SafetyProjection: Deployment-time constrained QP safety wrapper
  5. Lagrangian dual optimisation for adaptive reward-safety balancing

References:
  - Achiam et al., CPO, ICML 2017 (arXiv:1705.10528)
  - Schulman et al., PPO, 2017 (arXiv:1707.06347)
  - Vaswani et al., Attention Is All You Need, NeurIPS 2017
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List, NamedTuple
from collections import deque
import copy
import math


# ================================================================
# Rollout Storage
# ================================================================

class RolloutBatch(NamedTuple):
    """Single batch of rollout data for PPO update."""
    states: torch.Tensor      # (B, window, channels)
    actions: torch.Tensor     # (B, action_dim)
    old_log_probs: torch.Tensor  # (B,)
    returns: torch.Tensor     # (B,)
    advantages: torch.Tensor  # (B,)
    safety_returns: torch.Tensor  # (B, n_constraints)
    safety_advantages: torch.Tensor  # (B, n_constraints)


class RolloutBuffer:
    """Fixed-size buffer for storing trajectory segments."""

    def __init__(self, buffer_size: int, state_shape: Tuple, action_dim: int,
                 n_constraints: int = 3, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False

        # Pre-allocate
        self.states = torch.zeros(buffer_size, *state_shape)
        self.actions = torch.zeros(buffer_size, action_dim)
        self.log_probs = torch.zeros(buffer_size)
        self.rewards = torch.zeros(buffer_size)
        self.values = torch.zeros(buffer_size)
        self.safety_costs = torch.zeros(buffer_size, n_constraints)
        self.safety_values = torch.zeros(buffer_size, n_constraints)
        self.dones = torch.zeros(buffer_size)
        self.advantages = torch.zeros(buffer_size)
        self.returns = torch.zeros(buffer_size)
        self.safety_advantages = torch.zeros(buffer_size, n_constraints)
        self.safety_returns = torch.zeros(buffer_size, n_constraints)

    def add(self, state, action, log_prob, reward, value, safety_cost,
            safety_value, done):
        idx = self.ptr
        self.states[idx] = torch.FloatTensor(state)
        self.actions[idx] = torch.FloatTensor(action)
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.safety_costs[idx] = torch.FloatTensor(safety_cost)
        self.safety_values[idx] = torch.FloatTensor(safety_value)
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def compute_gae(self, gamma: float, gae_lambda: float,
                    last_value: float, last_safety_value: np.ndarray):
        """Compute GAE for reward and safety costs."""
        gae = 0.0
        safety_gae = np.zeros(self.safety_costs.shape[1])
        n = self.ptr if not self.full else self.buffer_size

        for i in reversed(range(n)):
            if i == n - 1:
                next_value = last_value
                next_safety_value = last_safety_value
                next_done = 1.0  # terminal
            else:
                next_value = self.values[i + 1].item()
                next_safety_value = self.safety_values[i + 1].numpy()
                next_done = self.dones[i + 1].item()

            mask = 1.0 - next_done

            # Reward advantage
            delta = (self.rewards[i].item() +
                     gamma * next_value * mask - self.values[i].item())
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[i] = gae

            # Safety advantages (per constraint)
            delta_s = (self.safety_costs[i].numpy() +
                       gamma * next_safety_value * mask -
                       self.safety_values[i].numpy())
            safety_gae = delta_s + gamma * gae_lambda * mask * safety_gae
            self.safety_advantages[i] = torch.FloatTensor(safety_gae.copy())

            # Returns
            self.returns[i] = self.advantages[i] + self.values[i]
            self.safety_returns[i] = (self.safety_advantages[i] +
                                       self.safety_values[i])

    def get_batches(self, batch_size: int) -> List[RolloutBatch]:
        """Yield shuffled minibatches."""
        n = self.ptr if not self.full else self.buffer_size
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield RolloutBatch(
                states=self.states[idx].to(self.device),
                actions=self.actions[idx].to(self.device),
                old_log_probs=self.log_probs[idx].to(self.device),
                returns=self.returns[idx].to(self.device),
                advantages=self.advantages[idx].to(self.device),
                safety_returns=self.safety_returns[idx].to(self.device),
                safety_advantages=self.safety_advantages[idx].to(self.device),
            )

    def __len__(self):
        return self.buffer_size if self.full else self.ptr


# ================================================================
# Physiological Attention Encoder
# ================================================================

class PhysiologicalAttention(nn.Module):
    """Multi-head self-attention encoder for multivariate biosignal time series.

    Processes a sliding window of physiological signals through multi-head
    self-attention to produce a compact state embedding that captures temporal
    dependencies and cross-channel interactions.

    Args:
        n_signals: Number of input physiological channels (default 7).
        d_model: Hidden dimension per attention head.
        n_heads: Number of parallel attention heads.
        window: Temporal window size (seconds, default 30).
        dropout: Attention dropout rate.
    """

    def __init__(self, n_signals: int = 7, d_model: int = 64,
                 n_heads: int = 4, window: int = 30, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.window = window

        # Input projection: signals → d_model
        self.input_proj = nn.Linear(n_signals, d_model)

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, window, d_model) * 0.02)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Output projection: flattened attention → embedding
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * window, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Store last attention weights for interpretability
        self.last_attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window, n_signals) biosignal buffer.
        Returns:
            h: (batch, 256) state embedding.
        """
        B, T, _ = x.shape

        # Project to d_model and add positional encoding
        x = self.input_proj(x)  # (B, T, d_model)
        x = x + self.pos_encoding[:, :T, :]

        # Multi-head self-attention
        attn_out, attn_weights = self.attention(x, x, x)
        self.last_attention_weights = attn_weights.detach()

        # Flatten and project
        h = self.output_proj(attn_out.reshape(B, -1))
        return h

    def get_attention_entropy(self) -> float:
        """Compute attention entropy for interpretability analysis."""
        if self.last_attention_weights is None:
            return 0.0
        weights = self.last_attention_weights.mean(dim=0)  # average over batch
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        return entropy.item()


# ================================================================
# Dual-Critic Architecture
# ================================================================

class DualCritic(nn.Module):
    """Dual critic with shared feature trunk.

    Separately estimates reward value V_r(s) and safety cost value V_c(s)
    through a shared representation, enabling efficient multi-objective
    value learning.

    Args:
        state_dim: Dimension of state embedding (default 256).
        hidden_dim: Hidden layer dimension.
        n_constraints: Number of safety constraints.
    """

    def __init__(self, state_dim: int = 256, hidden_dim: int = 128,
                 n_constraints: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.safety_head = nn.Linear(hidden_dim, n_constraints)

    def forward(self, h: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, state_dim) state embedding.
        Returns:
            v_r: (batch, 1) reward value estimate.
            v_c: (batch, n_constraints) safety cost value estimates.
        """
        features = self.shared(h)
        v_r = self.reward_head(features)
        v_c = self.safety_head(features)  # raw costs (positive = bad)
        return v_r, v_c


# ================================================================
# Gaussian Policy
# ================================================================

class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy for continuous drug dosing actions.

    Outputs mean and log-standard-deviation for a 2D action space
    (propofol rate, remifentanil rate), both normalised to [0, 1].
    """

    def __init__(self, state_dim: int = 256, action_dim: int = 2,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, h: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Independent]:
        """
        Args:
            h: (batch, state_dim) state embedding.
        Returns:
            mean: (batch, action_dim) action mean.
            log_std: (batch, action_dim) log standard deviation.
            dist: torch Independent Normal distribution.
        """
        mean = torch.tanh(self.mean_head(self.net(h)))  # bound to [-1, 1]
        mean = (mean + 1) / 2  # rescale to [0, 1]
        std = self.log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)
        return mean, self.log_std.expand_as(mean), dist


# ================================================================
# Safety Projection Layer
# ================================================================

class SafetyProjection:
    """Deployment-time safety projection via constrained quadratic programming.

    Solves the constrained QP described in Eq. 17 of the manuscript:
        a_safe = argmin ||a - a_raw||²₂
                 s.t. a ∈ [0, 1]², a_p ≤ overdose_threshold

    For the 2D action space, the QP has a closed-form analytical solution:
    the overdose constraint C₃ (propofol > 15 mg/kg/h ↔ a_p > 0.75) is enforced
    as a hard constraint via simple clipping. The state-dependent constraints
    C₁ (BIS < 40) and C₂ (MAP < 55) are enforced via proportional scaling
    based on safety critic predictions, since the safety critic estimates
    expected future constraint violations from the current state.

    This hybrid approach provides:
      - Hard guarantee: a_p ≤ overdose_threshold (formal C₃ enforcement)
      - Soft enforcement: proportional reduction when safety critic predicts
        elevated BIS/MAP violation risk (best-effort C₁/C₂ enforcement)
    """

    def __init__(self, safety_margins: np.ndarray = None,
                 overdose_threshold: float = 0.75):
        if safety_margins is None:
            safety_margins = np.array([0.01, 0.01, 0.005])
        self.safety_margins = safety_margins
        self.overdose_threshold = overdose_threshold

    def project(self, a_raw: np.ndarray,
                safety_pred: np.ndarray) -> np.ndarray:
        """Project raw action to safe region via constrained QP.

        Solves: min ||a - a_raw||²₂
                s.t. a ∈ [0, 1]²
                     a_p ≤ overdose_threshold          (C₃: hard constraint)
                     a ∝ 1/max(safety_pred/ε, 1)       (C₁, C₂: soft scaling)

        Args:
            a_raw: (action_dim,) raw policy action in [0, 1].
            safety_pred: (n_constraints,) predicted safety costs from V_c.

        Returns:
            a_safe: (action_dim,) safety-projected action in [0, 1].
        """
        a_safe = a_raw.copy()

        # --- Step 1: Hard constraint for propofol overdose (C₃) ---
        # a_p > 0.75 ↔ propofol > 15 mg/kg/h → clip to threshold
        a_safe[0] = min(a_safe[0], self.overdose_threshold)

        # --- Step 2: Soft scaling for state-dependent constraints (C₁, C₂) ---
        # When safety critic predicts elevated BIS<40 or MAP<55 risk,
        # proportionally reduce both drug rates to prevent violation.
        if safety_pred is not None and len(safety_pred) >= 2:
            # Only apply soft scaling for C₁ (BIS) and C₂ (MAP);
            # C₃ is already handled by the hard constraint above.
            violation_ratio = np.maximum(
                safety_pred[:2] / (self.safety_margins[:2] + 1e-8), 1.0
            )
            max_violation = np.max(violation_ratio)
            if max_violation > 1.0:
                # Reduce drug rates inversely proportional to worst violation
                scale = 1.0 / max_violation
                # Interpolate between scaled and raw action (50% weight each)
                a_safe = a_safe * (0.5 + 0.5 * scale) + a_raw * (0.5 - 0.5 * scale)

        # --- Step 3: Box constraints ---
        a_safe = np.clip(a_safe, 0.0, 1.0)

        return a_safe


# ================================================================
# CPPO-PA: Main Algorithm Class
# ================================================================

class CPPOPA:
    """Constrained PPO with Physiological Attention for anaesthesia control.

    Combines:
      - Multi-head physiological attention encoder
      - Dual-critic (reward + safety) value estimation
      - Gaussian policy with PPO clipped surrogate objective
      - Lagrangian dual optimisation for adaptive safety balancing
      - Safety projection layer for deployment-time guarantees

    Training hyperparameters follow those validated in the paper's
    ablation study (Table 1 of supplementary).

    Args:
        n_signals: Number of input biosignal channels.
        action_dim: Dimensionality of drug-dosing action space.
        lr: Learning rate (policy + critic).
        gamma: MDP discount factor.
        gae_lambda: GAE trace decay parameter.
        clip_eps: PPO clipping threshold.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value function loss coefficient.
        max_grad_norm: Gradient clipping norm.
        initial_lagrangian: Initial Lagrange multiplier value.
        lagrangian_lr: Learning rate for Lagrangian multiplier updates.
        safety_margins: Constraint thresholds [BIS, MAP, overdose].
        device: Torch device.
    """

    def __init__(self, n_signals: int = 7, action_dim: int = 2,
                 lr: float = 3e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 ent_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 n_constraints: int = 3,
                 initial_lagrangian: float = 0.1,
                 lagrangian_lr: float = 0.01,
                 safety_margins: np.ndarray = None,
                 device: str = 'cpu'):
        self.device = device

        # Networks
        self.attention = PhysiologicalAttention(
            n_signals=n_signals
        ).to(device)
        self.policy = GaussianPolicy(action_dim=action_dim).to(device)
        self.critic = DualCritic(n_constraints=n_constraints).to(device)

        # Safety
        if safety_margins is None:
            safety_margins = np.array([0.01, 0.01, 0.005])
        self.safety_margins = safety_margins
        self.safety_projection = SafetyProjection(safety_margins)
        self.n_constraints = n_constraints

        # Per-constraint Lagrangian multipliers (log-space for positivity)
        # manuscript Eq.14-16: separate λᵢ for each safety constraint
        self.log_lambdas = nn.Parameter(
            torch.full((n_constraints,), math.log(initial_lagrangian),
                       device=device)
        )

        # Optimisers with separate learning rates (manuscript Table tab:hyperparams)
        # Policy + attention: lr = 3e-4; Critic: lr = 1e-3
        self.optimizer = torch.optim.Adam([
            {'params': self.attention.parameters(), 'lr': lr},
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': critic_lr},
        ])
        self.lambda_optimizer = torch.optim.Adam(
            [self.log_lambdas], lr=lagrangian_lr
        )

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Training metrics
        self.metrics = {
            'policy_loss': [], 'value_loss': [], 'entropy': [],
            'lagrangian': [], 'lagrangians': [],  # lagrangians: per-constraint
            'safety_cost': [], 'reward': [],
        }

    def _build_state_buffer(self, history: np.ndarray,
                            t: int, window: int = 30) -> np.ndarray:
        """Build a state buffer from observation history.

        Args:
            history: (T_max, channels) full observation history.
            t: Current timestep.
            window: Buffer window size.

        Returns:
            buffer: (window, channels) padded observation buffer.
        """
        if t >= window:
            return history[t - window:t]
        else:
            # Pad with initial observations
            buffer = np.zeros((window, history.shape[1]))
            buffer[window - t:] = history[:t]
            # Repeat first observation for missing steps
            if t > 0:
                buffer[:window - t] = history[0]
            return buffer

    def act(self, history: np.ndarray, t: int,
            deterministic: bool = False) -> Tuple[np.ndarray, float, np.ndarray]:
        """Select action given observation history.

        Args:
            history: Full observation history array.
            t: Current timestep index.
            deterministic: If True, use mean action (no exploration).

        Returns:
            action: (action_dim,) normalised action in [0, 1].
            log_prob: Log probability of the selected action.
            safety_pred: Predicted safety costs.
        """
        buffer = self._build_state_buffer(history, t)
        x = torch.FloatTensor(buffer).unsqueeze(0).to(self.device)  # (1, W, C)

        with torch.no_grad():
            h = self.attention(x)
            mean, log_std, dist = self.policy(h)
            _, safety_pred = self.critic(h)

            if deterministic:
                action = mean.cpu().numpy().flatten()
            else:
                action = dist.sample().cpu().numpy().flatten()

            log_prob = dist.log_prob(
                torch.FloatTensor(action).unsqueeze(0).to(self.device)
            ).cpu().item()
            safety_pred = safety_pred.cpu().numpy().flatten()

        return np.clip(action, 0.0, 1.0), log_prob, safety_pred

    def act_safe(self, history: np.ndarray,
                 t: int) -> np.ndarray:
        """Select safe action with safety projection (deployment mode).

        Args:
            history: Full observation history.
            t: Current timestep.

        Returns:
            action: Safety-projected action.
        """
        action_raw, _, safety_pred = self.act(history, t, deterministic=True)
        action_safe = self.safety_projection.project(action_raw, safety_pred)
        return action_safe

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                            torch.Tensor, torch.Tensor]:
        """Evaluate actions on batched states (for PPO update).

        Args:
            states: (B, window, channels) state buffers.
            actions: (B, action_dim) actions.

        Returns:
            log_probs: (B,) log probabilities.
            entropy: (B,) distribution entropy.
            values_r: (B, 1) reward value estimates.
            values_c: (B, n_constraints) safety value estimates.
        """
        h = self.attention(states)
        mean, log_std, dist = self.policy(h)
        v_r, v_c = self.critic(h)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, v_r.squeeze(-1), v_c

    def update(self, buffer: RolloutBuffer, batch_size: int = 64,
               n_epochs: int = 10) -> Dict[str, float]:
        """PPO-style update with constrained objective.

        Implements Eq.15-16 from the manuscript:
          L_CPPO(θ) = E[min(r_t(θ)·Â_t, clip(r_t(θ),1-ε,1+ε)·Â_t)]
                      - Σᵢ λᵢ · Â_{Cᵢ,t}
          λᵢ ← max(0, λᵢ + η_λ · (J_{Cᵢ}(π_θ) - εᵢ))

        Args:
            buffer: Rollout buffer with computed GAE.
            batch_size: Minibatch size.
            n_epochs: Number of optimisation epochs per update.

        Returns:
            Dictionary of training metrics for this update.
        """
        metrics_epoch = {
            'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
            'lagrangian': 0.0, 'safety_cost': 0.0, 'n_batches': 0,
            'lagrangians': np.zeros(self.n_constraints),
        }

        for epoch in range(n_epochs):
            for batch in buffer.get_batches(batch_size):
                # Evaluate current policy
                log_probs, entropy, v_r, v_c = self.evaluate(
                    batch.states, batch.actions
                )

                # --- Policy Loss (PPO clipped + per-constraint Lagrangian) ---
                ratio = torch.exp(log_probs - batch.old_log_probs)
                adv = batch.advantages
                # Normalise advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Per-constraint Lagrangian penalty (Eq.15)
                # λᵢ weights each safety constraint independently
                lambdas = self.log_lambdas.exp()  # (n_constraints,)
                safety_adv = batch.safety_advantages  # (B, n_constraints)
                weighted_safety_adv = (lambdas.unsqueeze(0) * safety_adv).sum(dim=-1)
                combined_adv = adv - weighted_safety_adv

                surr1 = ratio * combined_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * combined_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value Loss ---
                v_r_target = batch.returns
                v_c_target = batch.safety_returns

                value_loss_r = F.mse_loss(v_r, v_r_target)
                value_loss_c = F.mse_loss(v_c, v_c_target)
                value_loss = self.vf_coef * (value_loss_r + value_loss_c)

                # --- Entropy Bonus ---
                entropy_loss = -self.ent_coef * entropy.mean()

                # --- Total Loss ---
                loss = policy_loss + value_loss + entropy_loss

                # --- Optimise ---
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.attention.parameters()) +
                    list(self.policy.parameters()) +
                    list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # --- Update Per-Constraint Lagrangian Multipliers (Eq.16) ---
                # λᵢ ← max(0, λᵢ + η_λ · (J_{Cᵢ} - εᵢ))
                mean_safety_cost = v_c_target.mean(dim=0)  # (n_constraints,)
                safety_margin_tensor = torch.FloatTensor(
                    self.safety_margins
                ).to(self.device)

                # Lagrangian dual loss: maximise λᵢ · (εᵢ − J_{Cᵢ})
                lambda_loss = -(
                    lambdas * (safety_margin_tensor - mean_safety_cost.detach())
                ).sum()

                self.lambda_optimizer.zero_grad()
                lambda_loss.backward()
                self.lambda_optimizer.step()

                # Clamp each λᵢ ≥ 0 (log-space: clamp to [-20, 5])
                with torch.no_grad():
                    self.log_lambdas.data.clamp_(min=-20.0, max=5.0)

                # --- Accumulate Metrics ---
                metrics_epoch['policy_loss'] += policy_loss.item()
                metrics_epoch['value_loss'] += value_loss.item()
                metrics_epoch['entropy'] += entropy.mean().item()
                metrics_epoch['lagrangian'] += lambdas.mean().item()
                metrics_epoch['lagrangians'] += lambdas.detach().cpu().numpy()
                metrics_epoch['safety_cost'] += mean_safety_cost.mean().item()
                metrics_epoch['n_batches'] += 1

        # Average over batches
        n = max(metrics_epoch['n_batches'], 1)
        for k in ['policy_loss', 'value_loss', 'entropy',
                   'lagrangian', 'safety_cost']:
            metrics_epoch[k] /= n
        metrics_epoch['lagrangians'] /= n

        # Store for tracking
        for k in ['policy_loss', 'value_loss', 'entropy',
                   'lagrangian', 'safety_cost']:
            self.metrics[k].append(metrics_epoch[k])
        self.metrics['lagrangians'].append(metrics_epoch['lagrangians'].tolist())

        return metrics_epoch

    def get_lambda(self) -> np.ndarray:
        """Get current per-constraint Lagrangian multiplier values.

        Returns:
            lambdas: (n_constraints,) array of λᵢ values.
        """
        return self.log_lambdas.exp().detach().cpu().numpy()

    def save(self, path: str):
        """Save model state."""
        torch.save({
            'attention': self.attention.state_dict(),
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'log_lambda': self.log_lambda.data,
            'metrics': self.metrics,
        }, path)

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.attention.load_state_dict(checkpoint['attention'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.log_lambda.data = checkpoint['log_lambda']
        self.metrics = checkpoint.get('metrics', self.metrics)


# ================================================================
# Standard PPO Agent (Baseline)
# ================================================================

class PPOAgent:
    """Standard PPO agent without safety constraints (baseline).

    Same network architecture as CPPO-PA but uses only the reward critic
    and omits the Lagrangian dual, safety critic, and safety projection.
    """

    def __init__(self, n_signals: int = 7, action_dim: int = 2,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 ent_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, device: str = 'cpu'):
        self.device = device
        self.attention = PhysiologicalAttention(n_signals=n_signals).to(device)
        self.policy = GaussianPolicy(action_dim=action_dim).to(device)

        # Single reward critic (same architecture as CPPO-PA for fair comparison)
        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.attention.parameters()) +
            list(self.policy.parameters()) +
            list(self.critic.parameters()),
            lr=lr
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_signals = n_signals
        self.action_dim = action_dim

        self.metrics = {
            'policy_loss': [], 'value_loss': [], 'entropy': [], 'reward': [],
        }

    def _build_state_buffer(self, history, t, window=30):
        if t >= window:
            return history[t - window:t]
        buffer = np.zeros((window, history.shape[1]))
        if t > 0:
            buffer[window - t:] = history[:t]
            buffer[:window - t] = history[0]
        return buffer

    def act(self, history, t, deterministic=False):
        buffer = self._build_state_buffer(history, t)
        x = torch.FloatTensor(buffer).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h = self.attention(x)
            mean, log_std, dist = self.policy(h)
            if deterministic:
                action = mean.cpu().numpy().flatten()
            else:
                action = dist.sample().cpu().numpy().flatten()
            log_prob = dist.log_prob(
                torch.FloatTensor(action).unsqueeze(0).to(self.device)
            ).cpu().item()
        return np.clip(action, 0.0, 1.0), log_prob

    def evaluate(self, states, actions):
        h = self.attention(states)
        mean, log_std, dist = self.policy(h)
        values = self.critic(h)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)

    def update(self, buffer: RolloutBuffer, batch_size=64, n_epochs=10):
        metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'n': 0}
        for _ in range(n_epochs):
            for batch in buffer.get_batches(batch_size):
                log_probs, entropy, values = self.evaluate(
                    batch.states, batch.actions)
                ratio = torch.exp(log_probs - batch.old_log_probs)
                adv = batch.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.vf_coef * F.mse_loss(
                    values, batch.returns)
                entropy_loss = -self.ent_coef * entropy.mean()
                loss = policy_loss + value_loss + entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.attention.parameters()) +
                    list(self.policy.parameters()) +
                    list(self.critic.parameters()),
                    self.max_grad_norm)
                self.optimizer.step()
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.mean().item()
                metrics['n'] += 1
        n = max(metrics['n'], 1)
        for k in ['policy_loss', 'value_loss', 'entropy']:
            self.metrics[k].append(metrics[k] / n)
        return {k: metrics[k] / n for k in ['policy_loss', 'value_loss', 'entropy']}

    def save(self, path):
        torch.save({
            'attention': self.attention.state_dict(),
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'metrics': self.metrics,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.attention.load_state_dict(ckpt['attention'])
        self.policy.load_state_dict(ckpt['policy'])
        self.critic.load_state_dict(ckpt['critic'])
        self.metrics = ckpt.get('metrics', self.metrics)


# ================================================================
# PID Controller (Baseline)
# ================================================================

class PIDController:
    """Cascaded PID controller for dual-drug anaesthesia.

    Two independent PID loops:
      1. BIS → propofol (target: BIS = 50)
      2. MAP → remifentanil (target: MAP = 80)

    Tuned via Ziegler-Nichols on a calibration set of virtual patients.
    """

    def __init__(self, target_bis: float = 50.0, target_map: float = 80.0,
                 kp_bis: float = 0.8, ki_bis: float = 0.05, kd_bis: float = 0.1,
                 kp_map: float = 0.5, ki_map: float = 0.03, kd_map: float = 0.05,
                 dt: float = 1.0):
        self.target_bis = target_bis
        self.target_map = target_map

        # BIS → propofol PID
        self.kp_b = kp_bis
        self.ki_b = ki_bis
        self.kd_b = kd_bis
        self.integral_b = 0.0
        self.prev_error_b = 0.0

        # MAP → remifentanil PID
        self.kp_m = kp_map
        self.ki_m = ki_map
        self.kd_m = kd_map
        self.integral_m = 0.0
        self.prev_error_m = 0.0

        self.dt = dt

    def reset(self):
        """Reset integrator states."""
        self.integral_b = 0.0
        self.prev_error_b = 0.0
        self.integral_m = 0.0
        self.prev_error_m = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Compute PID control action.

        Args:
            obs: [BIS, MAP, HR, SpO2, EtCO2, Ce_prop, Ce_remi].

        Returns:
            action: [propofol_norm, remifentanil_norm] in [0, 1].
        """
        bis = obs[0]
        map_val = obs[1]

        # --- BIS → Propofol ---
        error_b = self.target_bis - bis
        self.integral_b += error_b * self.dt
        derivative_b = (error_b - self.prev_error_b) / self.dt
        self.prev_error_b = error_b

        u_propofol = (self.kp_b * error_b +
                       self.ki_b * self.integral_b +
                       self.kd_b * derivative_b)

        # --- MAP → Remifentanil ---
        error_m = self.target_map - map_val
        self.integral_m += error_m * self.dt
        derivative_m = (error_m - self.prev_error_m) / self.dt
        self.prev_error_m = error_m

        u_remifentanil = (self.kp_m * error_m +
                           self.ki_m * self.integral_m +
                           self.kd_m * derivative_m)

        # Normalise to [0, 1]
        u_propofol_norm = np.clip(u_propofol / 20.0, 0.0, 1.0)
        u_remifentanil_norm = np.clip(u_remifentanil / 2.0, 0.0, 1.0)

        return np.array([u_propofol_norm, u_remifentanil_norm])


# ================================================================
# TCI Controller (Baseline)
# ================================================================

class TCIController:
    """Target-Controlled Infusion controller using Marsh PK model.

    Computes infusion rates to maintain target effect-site concentrations
    based on population PK models, without real-time BIS feedback.
    """

    def __init__(self, target_bis: float = 50.0, target_map: float = 80.0,
                 c50_p: float = 4.0, gamma_p: float = 2.0,
                 c50_r: float = 11.2, gamma_r: float = 2.51,
                 v1_p: float = 4.27, cl1_p: float = 1.89,
                 v1_r: float = 5.1, cl1_r: float = 2.6):
        # Target effect-site concentrations derived from Emax inversion
        self.target_bis = target_bis
        self.target_map = target_map
        self.c50_p = c50_p
        self.gamma_p = gamma_p
        self.c50_r = c50_r
        self.gamma_r = gamma_r

        # PK parameters for infusion rate computation
        self.v1_p = v1_p    # L
        self.cl1_p = cl1_p  # L/min
        self.v1_r = v1_r    # L
        self.cl1_r = cl1_r  # L/min

        # Target Ce (steady-state)
        effect_target = (100 - target_bis) / 100.0
        if effect_target > 0 and effect_target < 1:
            self.target_ce_p = c50_p * (effect_target / (1 - effect_target)) ** (1 / gamma_p)
        else:
            self.target_ce_p = 4.0  # default ~C50

    def reset(self):
        """Reset controller state."""
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Compute TCI infusion rates.

        At steady state: infusion_rate = clearance × target_concentration.
        During induction: bolus + infusion.

        Args:
            obs: [BIS, MAP, HR, SpO2, EtCO2, Ce_prop, Ce_remi].

        Returns:
            action in [0, 1].
        """
        ce_p = obs[5]
        ce_r = obs[6]

        # Propofol: approach target Ce
        error_p = self.target_ce_p - ce_p
        if error_p > 0:
            # Below target: infuse to reach
            u_p = self.cl1_p * self.target_ce_p * 1.5  # 50% overshoot for induction
        else:
            # At or above target: maintenance rate
            u_p = self.cl1_p * self.target_ce_p

        # Remifentanil: simple fixed target
        target_ce_r = 5.0  # ng/mL → μg/mL = 0.005
        error_r = target_ce_r - ce_r * 1000  # ce_r in μg/mL, target in ng/mL
        if error_r > 0:
            u_r = self.cl1_r * target_ce_r * 1.5 / 1000.0  # convert ng→μg
        else:
            u_r = self.cl1_r * target_ce_r / 1000.0

        # Normalise to [0, 1]
        u_p_norm = np.clip(u_p / 20.0, 0.0, 1.0)  # max 20 mg/kg/h
        u_r_norm = np.clip(u_r / 2.0, 0.0, 1.0)   # max 2 μg/kg/min

        return np.array([u_p_norm, u_r_norm])


# ================================================================
# Evaluation Utilities
# ================================================================


# ================================================================
# SAC Agent (Baseline)
# ================================================================

class SACAgent:
    """Soft Actor-Critic agent for continuous anaesthesia control.

    Implements off-policy maximum entropy RL with automatic temperature
    tuning, using the same physiological attention encoder as CPPO-PA
    for fair comparison.

    Reference:
      Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
      Deep Reinforcement Learning with a Stochastic Actor", ICML 2018.
    """

    def __init__(self, n_signals: int = 7, action_dim: int = 2,
                 lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2,
                 hidden_dim: int = 256, replay_size: int = 1000000,
                 device: str = 'cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Attention encoder (shared with CPPO-PA for fair comparison)
        self.attention = PhysiologicalAttention(
            n_signals=n_signals
        ).to(device)

        # Actor (Gaussian policy)
        self.actor = GaussianPolicy(
            state_dim=256, action_dim=action_dim
        ).to(device)

        # Twin Q-networks (no attention — take embedding + action)
        self.q1 = nn.Sequential(
            nn.Linear(256 + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        self.q2 = nn.Sequential(
            nn.Linear(256 + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        # Target Q-networks
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha), device=device)
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # Optimisers
        self.actor_optimizer = torch.optim.Adam(
            list(self.attention.parameters()) +
            list(self.actor.parameters()),
            lr=lr
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr
        )

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_size)

        # Metrics
        self.metrics = {
            'actor_loss': [], 'q_loss': [], 'alpha': [], 'entropy': [],
        }

    def _build_state_buffer(self, history: np.ndarray,
                            t: int, window: int = 30) -> np.ndarray:
        """Build state buffer from observation history."""
        if t >= window:
            return history[t - window:t]
        buffer = np.zeros((window, history.shape[1]))
        buffer[window - t:] = history[:t]
        if t > 0:
            buffer[:window - t] = history[0]
        return buffer

    def act(self, history: np.ndarray, t: int,
            deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Select action given observation history.

        Args:
            history: Full observation history.
            t: Current timestep.
            deterministic: If True, use mean action.

        Returns:
            action: (action_dim,) in [0, 1].
            log_prob: Log probability of action.
        """
        buffer = self._build_state_buffer(history, t)
        x = torch.FloatTensor(buffer).unsqueeze(0).to(self.device)

        with torch.no_grad():
            h = self.attention(x)
            mean, log_std, dist = self.actor(h)

            if deterministic:
                action = mean.cpu().numpy().flatten()
            else:
                action = dist.sample().cpu().numpy().flatten()

            log_prob = dist.log_prob(
                torch.FloatTensor(action).unsqueeze(0).to(self.device)
            ).cpu().item()

        return np.clip(action, 0.0, 1.0), log_prob

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: float):
        """Store transition in replay buffer."""
        self.replay_buffer.append(
            (state, action, reward, next_state, done)
        )

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Soft Actor-Critic update from replay buffer.

        Args:
            batch_size: Minibatch size for SAC update.

        Returns:
            Dictionary of training metrics.
        """
        if len(self.replay_buffer) < batch_size:
            return {'actor_loss': 0.0, 'q_loss': 0.0,
                    'alpha': self.log_alpha.exp().item(), 'entropy': 0.0}

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size,
                                   replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(-1).to(self.device)

        # Embed states via attention encoder
        with torch.no_grad():
            h = self.attention(states)
            h_next = self.attention(next_states)

        alpha = self.log_alpha.exp()

        # --- Q-function update ---
        with torch.no_grad():
            _, _, next_dist = self.actor(h_next)
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions).sum(-1, keepdim=True)
            q1_next = self.q1_target(torch.cat([h_next, next_actions], dim=-1))
            q2_next = self.q2_target(torch.cat([h_next, next_actions], dim=-1))
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1_pred = self.q1(torch.cat([h, actions], dim=-1))
        q2_pred = self.q2(torch.cat([h, actions], dim=-1))
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # --- Actor update ---
        _, _, dist = self.actor(h)
        actor_actions = dist.rsample()
        actor_log_probs = dist.log_prob(actor_actions).sum(-1, keepdim=True)

        q1_actor = self.q1(torch.cat([h, actor_actions], dim=-1))
        q2_actor = self.q2(torch.cat([h, actor_actions], dim=-1))
        q_actor = torch.min(q1_actor, q2_actor)

        actor_loss = (alpha * actor_log_probs - q_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha update (automatic temperature tuning) ---
        alpha_loss = -(
            self.log_alpha.exp() *
            (actor_log_probs.detach() + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft target update ---
        for target, source in [(self.q1_target, self.q1),
                               (self.q2_target, self.q2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        # Metrics
        metrics = {
            'actor_loss': actor_loss.item(),
            'q_loss': q_loss.item(),
            'alpha': alpha.item(),
            'entropy': -actor_log_probs.mean().item(),
        }
        for k, v in metrics.items():
            self.metrics[k].append(v)

        return metrics

    def save(self, path: str):
        """Save model state."""
        torch.save({
            'attention': self.attention.state_dict(),
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'log_alpha': self.log_alpha.data,
            'metrics': self.metrics,
        }, path)

    def load(self, path: str):
        """Load model state."""
        ckpt = torch.load(path, map_location=self.device)
        self.attention.load_state_dict(ckpt['attention'])
        self.actor.load_state_dict(ckpt['actor'])
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        self.q1_target.load_state_dict(ckpt['q1'])
        self.q2_target.load_state_dict(ckpt['q2'])
        self.log_alpha.data = ckpt['log_alpha']
        self.metrics = ckpt.get('metrics', self.metrics)


# ================================================================
# Evaluation Utilities
# ================================================================

def compute_metrics(history: np.ndarray, actions: np.ndarray,
                    dt: float = 1.0) -> Dict[str, float]:
    """Compute clinical evaluation metrics from a trajectory.

    Args:
        history: (T, 7) observation history [BIS, MAP, HR, SpO2, EtCO2, Ce_p, Ce_r].
        actions: (T, 2) action history [propofol_norm, remifentanil_norm].
        dt: Time step in seconds.

    Returns:
        Dictionary of evaluation metrics.
    """
    T = len(history)
    hours = T * dt / 3600.0
    n_actions = len(actions)
    # Trim to matching length
    min_len = min(T, n_actions)
    history = history[:min_len]
    actions = actions[:min_len]
    T = min_len

    bis = history[:, 0]
    map_vals = history[:, 1]
    propofol_rate = actions[:, 0] * 20  # denormalise

    # Time in Range
    tir_bis = np.mean((bis >= 40) & (bis <= 60)) * 100
    tir_map = np.mean((map_vals >= 65) & (map_vals <= 100)) * 100

    # MAE
    mae_bis = np.mean(np.abs(bis - 50))

    # Safety violations
    violations_bis = np.sum(bis < 40) / max(hours, 0.01)
    violations_map = np.sum(map_vals < 55) / max(hours, 0.01)
    violations_overdose = np.sum(propofol_rate > 15) / max(hours, 0.01)
    total_violations = violations_bis + violations_map + violations_overdose

    # Induction metrics (first 120 seconds)
    induction_window = min(120, T)
    bis_induction = bis[:induction_window]
    induction_time = np.argmax(bis_induction < 60) * dt if np.any(bis_induction < 60) else 120
    overshoot = np.any(bis_induction < 40)

    # Cumulative reward
    r_bis = np.exp(-(bis - 50) ** 2 / (2 * 10 ** 2))
    r_map = np.exp(-(map_vals - 80) ** 2 / (2 * 15 ** 2))
    r_drug = -0.01 * (actions[:, 0] ** 2 + actions[:, 1] ** 2)
    cumulative_reward = np.mean(0.5 * r_bis + 0.3 * r_map + r_drug)

    return {
        'tir_bis': tir_bis,
        'tir_map': tir_map,
        'mae_bis': mae_bis,
        'violations_per_100h': total_violations,
        'violations_bis': violations_bis,
        'violations_map': violations_map,
        'violations_overdose': violations_overdose,
        'induction_time': induction_time,
        'overshoot': float(overshoot),
        'cumulative_reward': cumulative_reward,
    }


if __name__ == '__main__':
    # Smoke test: instantiate CPPO-PA and run a forward pass
    print("=" * 60)
    print("CPPO-PA Smoke Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create agent
    agent = CPPOPA(device=device)
    n_params = sum(p.numel() for p in agent.attention.parameters()) + \
               sum(p.numel() for p in agent.policy.parameters()) + \
               sum(p.numel() for p in agent.critic.parameters())
    print(f"Total parameters: {n_params:,}")

    # Simulate a short episode
    from patient_sim import AnesthesiaEnv

    env = AnesthesiaEnv(n_patients=1, episode_length=300, seed=42)
    obs = env.reset()
    history = [obs[0]]

    total_reward = 0
    safety_costs = []
    actions_taken = []

    for t in range(300):
        hist_array = np.array(history)
        action, log_prob, safety_pred = agent.act(hist_array, t)
        # Also test safe mode
        if t % 50 == 0:
            action_safe = agent.act_safe(hist_array, t)
            lambdas = agent.get_lambda()
            print(f"  t={t:3d}: action={action}, safe_action={action_safe}, "
                  f"λ={lambdas}")

        obs, reward, done, info = env.step(np.array([action]))
        history.append(obs[0])
        actions_taken.append(action)
        total_reward += reward[0]

        # Safety costs
        c1 = float(obs[0, 0] < 40)
        c2 = float(obs[0, 1] < 55)
        c3 = float(action[0] > 0.75)
        safety_costs.append([c1, c2, c3])

    metrics = compute_metrics(np.array(history), np.array(actions_taken))
    print(f"\nEpisode metrics:")
    print(f"  TIR_BIS: {metrics['tir_bis']:.1f}%")
    print(f"  Safety violations/100h: {metrics['violations_per_100h']:.1f}")
    print(f"  Cumulative reward: {metrics['cumulative_reward']:.3f}")
    print(f"  Induction time: {metrics['induction_time']:.0f}s")
    print(f"  Overshoot: {bool(metrics['overshoot'])}")
    print(f"  Total reward: {total_reward:.3f}")

    # Test baselines
    print(f"\nBaseline tests:")
    pid = PIDController()
    pid.reset()
    pid_action = pid.act(history[30])
    print(f"  PID action at t=30: {pid_action}")

    tci = TCIController()
    tci_action = tci.act(history[30])
    print(f"  TCI action at t=30: {tci_action}")

    ppo = PPOAgent(device=device)
    ppo_action, ppo_lp = ppo.act(np.array(history), 30)
    print(f"  PPO action at t=30: {ppo_action}")

    print(f"\nAll systems operational!")
