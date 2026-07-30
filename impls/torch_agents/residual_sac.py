"""PyTorch residual SAC agent for SimLingo fine-tuning.

The residual actor and critics take the VLM driving features extracted from
the frozen SimLingo VLM backbone as their observation.  The actor produces
a Gaussian distribution over a (accel, steer) residual in [-1, 1]², which
is combined with the SimLingo base action as:

    final_action = clip(base_action + res_scale * residual, -1, 1)

Standard SAC with:
  - double Q-networks + target networks (tau soft-update)
  - automatic entropy coefficient tuning
  - no output normalization (matches SimLingo training convention)

VLM feature dim: 896  (InternVL2-1B Qwen2 backbone, 30 driving tokens mean-pooled)
Action dim: 2          [accel, steer]
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ── Constants ─────────────────────────────────────────────────────────────────
VLM_FEATURE_DIM = 896
ACTION_DIM = 2
EGO_STATE_DIM = 19  # obs["state"][6:] — drops world-frame position (x,y,z) and orientation (rpy)
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


# ── Running normalizer ────────────────────────────────────────────────────────

class RunningNormalizer:
    """Per-dimension online mean/variance normalizer (Welford's algorithm).

    Accepts raw state vectors one at a time via ``update()``, then
    ``normalize()`` / ``normalize_tensor()`` map them to approximately zero
    mean, unit variance, clipped to ``[-clip, clip]``.

    Statistics are serialisable so they travel with the SAC checkpoint.
    """

    def __init__(self, dim: int, clip: float = 5.0, epsilon: float = 1e-8):
        self.dim = dim
        self.clip = clip
        self.epsilon = epsilon
        self._count: float = 0.0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._M2 = np.zeros(dim, dtype=np.float64)  # sum of squared deviations

    # ── Online update ─────────────────────────────────────────────────────────

    def update(self, x: np.ndarray) -> None:
        """Add one observation to the running statistics."""
        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        self._count += 1.0
        delta = x - self._mean
        self._mean += delta / self._count
        self._M2 += delta * (x - self._mean)  # Welford update

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def mean(self) -> np.ndarray:
        return self._mean.astype(np.float32)

    @property
    def std(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(self.dim, dtype=np.float32)
        return np.sqrt(self._M2 / (self._count - 1) + self.epsilon).astype(np.float32)

    # ── Normalisation ─────────────────────────────────────────────────────────

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalise a single numpy state vector."""
        return np.clip((x.astype(np.float32) - self.mean) / self.std, -self.clip, self.clip)

    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise a batch of state tensors (in-place safe)."""
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=x.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=x.device)
        return torch.clamp((x - mean) / std, -self.clip, self.clip)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "count": self._count,
            "mean": self._mean.copy(),
            "M2": self._M2.copy(),
        }

    def load_state_dict(self, d: dict) -> None:
        self._count = float(d["count"])
        self._mean = np.asarray(d["mean"], dtype=np.float64)
        self._M2 = np.asarray(d["M2"], dtype=np.float64)


# ── Neural network modules ────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dims: List[int], out_dim: int, activate_last: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    dim = in_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(dim, h), nn.ReLU()])
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    if activate_last:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    """Gaussian actor: cat(vlm_features, base_action[, state]) → (mean, log_std) → tanh-squashed residual."""

    def __init__(
        self,
        vlm_feature_dim: int = VLM_FEATURE_DIM,
        hidden_dims: List[int] = (256, 256, 256),
        action_dim: int = ACTION_DIM,
        state_dim: int = EGO_STATE_DIM,
    ):
        super().__init__()
        self.state_dim = state_dim
        in_dim = vlm_feature_dim + action_dim + state_dim
        self.trunk = _mlp(in_dim, list(hidden_dims[:-1]), hidden_dims[-1], activate_last=True)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        # Small-weight init so the residual starts near zero rather than at the
        # tanh boundary.  VLM features can have std ~50–100 (InternVL2 hidden
        # states), which propagates through Kaiming-initialised trunk layers and
        # pushes |pre-tanh| >> 1 even with 3e-3 head weights.  1e-3 gives ~3×
        # smaller pre-tanh values; use 1e-4 if saturation persists.
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mean_head.bias, 0.0)
        # log_std bias = -2 → std ≈ 0.14 initially (low noise at start, grows
        # quickly as alpha tunes entropy; avoids noisy-exploration pushing tanh
        # into saturation before any learning has occurred)
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.log_std_head.bias, -2.0)

    def _cat_inputs(self, vlm_features: torch.Tensor, base_action: torch.Tensor, state: Optional[torch.Tensor]) -> torch.Tensor:
        parts = [vlm_features, base_action]
        if self.state_dim > 0 and state is not None:
            parts.append(state)
        return torch.cat(parts, dim=-1)

    def forward(self, vlm_features: torch.Tensor, base_action: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self._cat_inputs(vlm_features, base_action, state))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, vlm_features: torch.Tensor, base_action: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and log_prob using reparameterization + tanh squashing."""
        mean, log_std = self(vlm_features, base_action, state)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        x = mean + std * eps  # pre-tanh
        action = torch.tanh(x)
        # log_prob with tanh correction
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(x)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_mean_action(self, vlm_features: torch.Tensor, base_action: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Deterministic action at mean (for eval)."""
        mean, _ = self(vlm_features, base_action, state)
        return torch.tanh(mean)


class ResidualCritic(nn.Module):
    """Double Q-network: (vlm_features, base_action, final_action[, state][, coach_label]) → (Q1, Q2).

    coach_label_dim > 0 appends the Gemini coach's per-step delta-commentary BoW vector
    (from coaches/online_vlm_coach.py) to the critic input so the Q-function can be
    conditioned on retrospective language feedback about driving quality.
    """

    def __init__(
        self,
        vlm_feature_dim: int = VLM_FEATURE_DIM,
        hidden_dims: List[int] = (256, 256, 256),
        action_dim: int = ACTION_DIM,
        state_dim: int = EGO_STATE_DIM,
        coach_label_dim: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.coach_label_dim = coach_label_dim
        in_dim = vlm_feature_dim + action_dim + action_dim + state_dim + coach_label_dim
        self.q1 = _mlp(in_dim, list(hidden_dims), 1)
        self.q2 = _mlp(in_dim, list(hidden_dims), 1)

    def _cat_inputs(
        self,
        vlm_features: torch.Tensor,
        base_action: torch.Tensor,
        action: torch.Tensor,
        state: Optional[torch.Tensor],
        coach_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [vlm_features, base_action, action]
        if self.state_dim > 0 and state is not None:
            parts.append(state)
        if self.coach_label_dim > 0 and coach_label is not None:
            parts.append(coach_label)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        vlm_features: torch.Tensor,
        base_action: torch.Tensor,
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        coach_label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._cat_inputs(vlm_features, base_action, action, state, coach_label)
        return self.q1(x), self.q2(x)

    def min_q(
        self,
        vlm_features: torch.Tensor,
        base_action: torch.Tensor,
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        coach_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q1, q2 = self(vlm_features, base_action, action, state, coach_label)
        return torch.min(q1, q2)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Simple numpy ring buffer storing VLM features + composed final actions.

    Stores the composed final_action (base + res_scale * residual, clipped) rather than
    the raw residual so that the critic is trained on the actual action sent to the
    environment. base_action and next_base_action are also stored so that fresh actor
    samples can be composed during both the Bellman target and actor-loss computations.

    When ``coach_label_dim > 0``, each transition additionally stores a language label
    vector from the Gemini VLM coach (coaches/online_vlm_coach.py). Labels start as
    zeros and are retroactively filled by ``update_at()`` at episode end once Gemini
    has reviewed the rollout video.
    """

    def __init__(
        self,
        capacity: int,
        vlm_dim: int = VLM_FEATURE_DIM,
        action_dim: int = ACTION_DIM,
        state_dim: int = EGO_STATE_DIM,
        coach_label_dim: int = 0,
    ):
        self.capacity = capacity
        self.vlm_dim = vlm_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.coach_label_dim = coach_label_dim
        self._obs = np.zeros((capacity, vlm_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, vlm_dim), dtype=np.float32)
        self._final_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._next_base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)
        if coach_label_dim > 0:
            self._coach_labels = np.zeros((capacity, coach_label_dim), dtype=np.float32)
        else:
            self._coach_labels = None
        self._ptr = 0
        self._size = 0

    @property
    def last_ptr(self) -> int:
        """Index of the slot filled by the most recent ``add()`` call."""
        return (self._ptr - 1) % self.capacity

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        base_action: np.ndarray,
        next_base_action: np.ndarray,
        final_action: np.ndarray,
        state: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._base_actions[self._ptr] = base_action
        self._next_base_actions[self._ptr] = next_base_action
        self._final_actions[self._ptr] = final_action
        if self.state_dim > 0:
            self._states[self._ptr] = state
            self._next_states[self._ptr] = next_state
        self._rewards[self._ptr, 0] = reward
        self._dones[self._ptr, 0] = float(done)
        if self._coach_labels is not None:
            self._coach_labels[self._ptr] = 0.0  # zeroed until coach backfill
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update_at(self, idx: int, *, coach_label: np.ndarray) -> None:
        """Retroactively write a coach language label into an existing buffer slot.

        Called by ``OnlineVLMSession.backfill_buffer()`` at episode end once
        Gemini has returned per-chunk feedback for the just-completed episode.
        """
        if self._coach_labels is None:
            return
        if not (0 <= idx < self.capacity):
            return
        label = np.asarray(coach_label, dtype=np.float32)
        n = min(len(label), self.coach_label_dim)
        self._coach_labels[idx, :n] = label[:n]

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "obs": torch.from_numpy(self._obs[idx]).to(device),
            "next_obs": torch.from_numpy(self._next_obs[idx]).to(device),
            "base_actions": torch.from_numpy(self._base_actions[idx]).to(device),
            "next_base_actions": torch.from_numpy(self._next_base_actions[idx]).to(device),
            "final_actions": torch.from_numpy(self._final_actions[idx]).to(device),
            "rewards": torch.from_numpy(self._rewards[idx]).to(device),
            "dones": torch.from_numpy(self._dones[idx]).to(device),
        }
        if self.state_dim > 0:
            batch["states"] = torch.from_numpy(self._states[idx]).to(device)
            batch["next_states"] = torch.from_numpy(self._next_states[idx]).to(device)
        if self._coach_labels is not None:
            batch["coach_labels"] = torch.from_numpy(self._coach_labels[idx]).to(device)
        return batch

    def __len__(self) -> int:
        return self._size


class DaggerBuffer:
    """Ring buffer for DAgger BC training: stores (vlm_features, base_action, expert_action) triples.

    base_action and expert_action are both in the environment's (accel, steer) space so
    that the BC loss can be computed as MSE(clip(base + res_scale * tanh(mean), -1, 1), expert).
    """

    def __init__(self, capacity: int, vlm_dim: int = VLM_FEATURE_DIM, action_dim: int = ACTION_DIM, state_dim: int = EGO_STATE_DIM):
        self.capacity = capacity
        self.vlm_dim = vlm_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self._obs = np.zeros((capacity, vlm_dim), dtype=np.float32)
        self._base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._expert_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs: np.ndarray, base_action: np.ndarray, state: np.ndarray, expert_action: np.ndarray) -> None:
        self._obs[self._ptr] = obs
        self._base_actions[self._ptr] = base_action
        self._expert_actions[self._ptr] = expert_action
        if self.state_dim > 0:
            self._states[self._ptr] = state
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "obs": torch.from_numpy(self._obs[idx]).to(device),
            "base_actions": torch.from_numpy(self._base_actions[idx]).to(device),
            "expert_actions": torch.from_numpy(self._expert_actions[idx]).to(device),
        }
        if self.state_dim > 0:
            batch["states"] = torch.from_numpy(self._states[idx]).to(device)
        return batch

    def __len__(self) -> int:
        return self._size


# ── SAC agent ─────────────────────────────────────────────────────────────────

class ResidualSACAgent:
    """Residual SAC: trains a small MLP actor/critic on top of frozen SimLingo features.

    Args:
        vlm_feature_dim: Dimension of VLM features from SimLingo (896 for InternVL2-1B).
        action_dim:       Residual action dimension (2 = accel, steer).
        hidden_dims:      Hidden layer sizes for actor and critics.
        gamma:            Discount factor.
        tau:              Target network soft-update coefficient.
        actor_lr:         Actor learning rate.
        critic_lr:        Critic learning rate.
        target_entropy:   SAC target entropy (default = -action_dim).
        device:           Torch device.
    """

    def __init__(
        self,
        vlm_feature_dim: int = VLM_FEATURE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
        gamma: float = 0.97,
        tau: float = 0.01,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        target_entropy: Optional[float] = None,
        device: str = "cuda",
        actor_l2_reg: float = 0.0,
        res_scale=0.1,
        state_dim: int = EGO_STATE_DIM,
        ticks_per_wp: int = 1,
        coach_label_dim: int = 0,
    ):
        self.gamma = gamma
        # Each SAC "step" covers ticks_per_wp CARLA ticks.  The Bellman discount
        # for the meta-step is gamma^ticks_per_wp, not gamma^1.
        self.effective_gamma = gamma ** ticks_per_wp
        self.tau = tau
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.coach_label_dim = coach_label_dim
        self.device = torch.device(device)
        self.actor_l2_reg = actor_l2_reg

        self.actor = ResidualActor(vlm_feature_dim, list(hidden_dims), action_dim, state_dim).to(self.device)
        self.critic = ResidualCritic(vlm_feature_dim, list(hidden_dims), action_dim, state_dim, coach_label_dim).to(self.device)
        self.critic_target = ResidualCritic(vlm_feature_dim, list(hidden_dims), action_dim, state_dim, coach_label_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        self.res_scale = torch.as_tensor(res_scale, dtype=torch.float32).to(self.device)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

        # Automatic entropy tuning
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(math.log(0.1), requires_grad=True, device=self.device)
        self.alpha_opt = Adam([self.log_alpha], lr=actor_lr)

        # Running normalizer for the ego-state slice (None when state_dim == 0)
        self.state_normalizer: Optional[RunningNormalizer] = (
            RunningNormalizer(state_dim) if state_dim > 0 else None
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ── State normalisation ───────────────────────────────────────────────────

    def update_state_normalizer(self, raw_state: np.ndarray) -> None:
        """Feed a raw ego-state observation into the running statistics."""
        if self.state_normalizer is not None:
            self.state_normalizer.update(raw_state)

    # ── Action sampling ───────────────────────────────────────────────────────

    def _state_tensor(self, state: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if self.state_dim > 0 and state is not None:
            if self.state_normalizer is not None:
                state = self.state_normalizer.normalize(state)
            return torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        return None

    def sample_actions(self, vlm_features: np.ndarray, base_action: np.ndarray, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample stochastic residual action (for training rollouts)."""
        obs_t = torch.from_numpy(vlm_features).unsqueeze(0).float().to(self.device)
        base_t = torch.from_numpy(base_action).unsqueeze(0).float().to(self.device)
        state_t = self._state_tensor(state)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_t, base_t, state_t)
        return action.squeeze(0).cpu().numpy()

    def get_eval_action(self, vlm_features: np.ndarray, base_action: np.ndarray, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Deterministic mean action (for evaluation)."""
        obs_t = torch.from_numpy(vlm_features).unsqueeze(0).float().to(self.device)
        base_t = torch.from_numpy(base_action).unsqueeze(0).float().to(self.device)
        state_t = self._state_tensor(state)
        with torch.no_grad():
            action = self.actor.get_mean_action(obs_t, base_t, state_t)
        return action.squeeze(0).cpu().numpy()

    # ── SAC update ────────────────────────────────────────────────────────────

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        base_actions = batch["base_actions"]
        next_base_actions = batch["next_base_actions"]
        final_actions = batch["final_actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        states = batch.get("states")
        next_states = batch.get("next_states")
        # Coach language label: (B, coach_label_dim) or None when coach is disabled.
        # The same label is used for both current and next state since labels are
        # assigned per episode chunk and are consistent within a rollout window.
        coach_labels = batch.get("coach_labels")

        # Normalise raw ego-state tensors from the replay buffer.
        if self.state_normalizer is not None:
            if states is not None:
                states = self.state_normalizer.normalize_tensor(states)
            if next_states is not None:
                next_states = self.state_normalizer.normalize_tensor(next_states)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_residuals, next_log_probs = self.actor.sample(next_obs, next_base_actions, next_states)
            next_final = torch.clamp(next_base_actions + self.res_scale * next_residuals, -1.0, 1.0)
            q1_t, q2_t = self.critic_target(next_obs, next_base_actions, next_final, next_states, coach_labels)
            min_q_t = torch.min(q1_t, q2_t)
            y = rewards + self.effective_gamma * (1.0 - dones) * (min_q_t - self.alpha.detach() * next_log_probs)

        q1, q2 = self.critic(obs, base_actions, final_actions, states, coach_labels)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────────
        pi, log_probs = self.actor.sample(obs, base_actions, states)
        pi_final = torch.clamp(base_actions + self.res_scale * pi, -1.0, 1.0)
        q1_pi, q2_pi = self.critic(obs, base_actions, pi_final, states, coach_labels)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_probs - min_q_pi).mean()
        l2_loss = torch.tensor(0.0)
        if self.actor_l2_reg > 0.0:
            l2_loss = self.actor_l2_reg * pi.pow(2).mean()
            actor_loss = actor_loss + l2_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Entropy tuning ────────────────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Target network soft update ────────────────────────────────────────
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha_loss": float(alpha_loss),
            "alpha": float(self.alpha),
            "entropy": float(-log_probs.mean()),
            "q_mean": float(min_q_pi.mean()),
            "actor_l2_loss": float(l2_loss),
        }

    # ── DAgger BC update ──────────────────────────────────────────────────────

    def bc_update(self, batch: Dict[str, torch.Tensor], res_scale=0.1) -> Dict[str, float]:
        """DAgger residual BC update.

        Loss: MSE(clip(base + res_scale * tanh(mean), -1, 1), expert_action).
        Matches the celine-branch formulation: gradient is on the final action,
        not on the residual directly, so the loss correctly weights corrections
        relative to what the base policy already contributes.
        """
        obs = batch["obs"]
        base_actions = batch["base_actions"]
        expert_actions = batch["expert_actions"]
        states = batch.get("states")

        if self.state_normalizer is not None and states is not None:
            states = self.state_normalizer.normalize_tensor(states)

        mean, _ = self.actor(obs, base_actions, states)
        residual = torch.tanh(mean)
        _rs = torch.as_tensor(res_scale, dtype=torch.float32).to(base_actions.device)
        predicted = torch.clamp(base_actions + _rs * residual, -1.0, 1.0)
        bc_loss = F.mse_loss(predicted, expert_actions)

        self.actor_opt.zero_grad()
        bc_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            base_mse = F.mse_loss(base_actions, expert_actions)

        return {
            "bc_loss": float(bc_loss),
            "base_mse": float(base_mse),
            "residual_accel_mean": float(residual[:, 0].mean()),
            "residual_steer_mean": float(residual[:, 1].mean()),
            "residual_abs_mean": float(residual.abs().mean()),
            "residual_abs_max": float(residual.abs().max()),
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "log_alpha": self.log_alpha.data,
                "alpha_opt": self.alpha_opt.state_dict(),
                "state_normalizer": (
                    self.state_normalizer.state_dict()
                    if self.state_normalizer is not None else None
                ),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        if self.state_normalizer is not None and ckpt.get("state_normalizer") is not None:
            self.state_normalizer.load_state_dict(ckpt["state_normalizer"])
