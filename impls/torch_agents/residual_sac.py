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
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


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
    """Gaussian actor: vlm_features → (mean, log_std) → tanh-squashed action."""

    def __init__(
        self,
        vlm_feature_dim: int = VLM_FEATURE_DIM,
        hidden_dims: List[int] = (256, 256, 256),
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.trunk = _mlp(vlm_feature_dim, list(hidden_dims[:-1]), hidden_dims[-1], activate_last=True)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, vlm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(vlm_features)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, vlm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and log_prob using reparameterization + tanh squashing."""
        mean, log_std = self(vlm_features)
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

    def get_mean_action(self, vlm_features: torch.Tensor) -> torch.Tensor:
        """Deterministic action at mean (for eval)."""
        mean, _ = self(vlm_features)
        return torch.tanh(mean)


class ResidualCritic(nn.Module):
    """Double Q-network: (vlm_features, action) → (Q1, Q2)."""

    def __init__(
        self,
        vlm_feature_dim: int = VLM_FEATURE_DIM,
        hidden_dims: List[int] = (256, 256, 256),
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        in_dim = vlm_feature_dim + action_dim
        self.q1 = _mlp(in_dim, list(hidden_dims), 1)
        self.q2 = _mlp(in_dim, list(hidden_dims), 1)

    def forward(
        self, vlm_features: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([vlm_features, action], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, vlm_features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(vlm_features, action)
        return torch.min(q1, q2)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Simple numpy ring buffer storing VLM features + residual actions."""

    def __init__(self, capacity: int, vlm_dim: int = VLM_FEATURE_DIM, action_dim: int = ACTION_DIM):
        self.capacity = capacity
        self.vlm_dim = vlm_dim
        self.action_dim = action_dim
        self._obs = np.zeros((capacity, vlm_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, vlm_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr, 0] = reward
        self._dones[self._ptr, 0] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.from_numpy(self._obs[idx]).to(device),
            "next_obs": torch.from_numpy(self._next_obs[idx]).to(device),
            "actions": torch.from_numpy(self._actions[idx]).to(device),
            "rewards": torch.from_numpy(self._rewards[idx]).to(device),
            "dones": torch.from_numpy(self._dones[idx]).to(device),
        }

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
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.actor = ResidualActor(vlm_feature_dim, list(hidden_dims), action_dim).to(self.device)
        self.critic = ResidualCritic(vlm_feature_dim, list(hidden_dims), action_dim).to(self.device)
        self.critic_target = ResidualCritic(vlm_feature_dim, list(hidden_dims), action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

        # Automatic entropy tuning
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(math.log(0.1), requires_grad=True, device=self.device)
        self.alpha_opt = Adam([self.log_alpha], lr=actor_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ── Action sampling ───────────────────────────────────────────────────────

    def sample_actions(self, vlm_features: np.ndarray) -> np.ndarray:
        """Sample stochastic residual action (for training rollouts)."""
        obs_t = torch.from_numpy(vlm_features).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    def get_eval_action(self, vlm_features: np.ndarray) -> np.ndarray:
        """Deterministic mean action (for evaluation)."""
        obs_t = torch.from_numpy(vlm_features).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            action = self.actor.get_mean_action(obs_t)
        return action.squeeze(0).cpu().numpy()

    # ── SAC update ────────────────────────────────────────────────────────────

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_actions)
            min_q_t = torch.min(q1_t, q2_t)
            y = rewards + self.gamma * (1.0 - dones) * (min_q_t - self.alpha.detach() * next_log_probs)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────────
        pi, log_probs = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_probs - min_q_pi).mean()

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
