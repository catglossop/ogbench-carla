#!/usr/bin/env python3
"""Offline critic pretraining for the run_simlingo.sh InternVL2 setup.

Trains the PyTorch ResidualCritic from torch_agents/residual_sac.py using
InternVL2 encoder features (1920-dim, obs_mode="encoder") extracted offline
from stored CARLA datasets.  The saved checkpoint is compatible with
ResidualSACAgent.load(), so it can be passed via --pretrained_critic in
main_carla_simlingo.py to warm-start online SAC.

Observation: 1920-dim = L2-norm(InternVL2 vision CLS 1024) ++ L2-norm(prompt 896)
             from SimLingoBase.get_encoder_features()
Action:      [accel, steer] 2-dim
             noisy data -> noisy_throttle/steer/brake; non-noisy -> clean controls
Target:      MC returns (same reward scheme as ogbench/carla/carla_utils.py)

Run under the simlingo conda env:
  /home/celinet/miniconda3/envs/simlingo/bin/python impls/pretrain_critic_simlingo.py \\
      --simlingo_checkpoint /home/celinet/simlingo_checkpoints/simlingo/checkpoints/epoch=013.ckpt \\
      --noise_sweep_root /scratch/current/celinet/noise_sweep/data \\
      --total_steps 100000 --batch_size 256 \\
      --wandb_project carla_critic_pretrain_simlingo

To load the pretrained critic in main_carla_simlingo.py / run_simlingo.sh:
  bash run_simlingo.sh --pretrained-critic <ckpt_dir>/latest.pt ...
"""
from __future__ import annotations

import argparse
import datetime
import gc
import gzip
import json
import random
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# ── Path setup (mirrors main_carla_simlingo.py) ───────────────────────────────
_IMPLS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _IMPLS_ROOT.parent
_REBUTTAL_ROOT = _REPO_ROOT / "simlingo-rebuttal"

for _p in [str(_IMPLS_ROOT), str(_REPO_ROOT), str(_REBUTTAL_ROOT), str(_REBUTTAL_ROOT / "team_code")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────
# VLM_ENCODER_FEATURE_DIM from vlas/simlingo_base.py:
#   1024 (L2-norm InternVL2 vision CLS) + 896 (L2-norm prompt embed mean) = 1920
ENCODER_FEATURE_DIM = 1920
# Index of speed in the 25-dim ego state vector (from vlas/simlingo_base.py)
_EGO_STATE_IDX_SPEED = 15
ACTION_HORIZON = 10

PROGRESS_WEIGHT = 5.0
SPEED_LIMIT_PEN_WEIGHT = 0.1
COLLISION_PEN = -20.0
OUTSIDE_ROUTE_PEN = -20.0
TRAFFIC_PEN = -20.0
SUCCESS_BONUS = 50.0
FAILURE_BONUS = -20.0


# ── Reward ────────────────────────────────────────────────────────────────────

def _rewards_noisy(measurements: List[dict]) -> np.ndarray:
    rewards = np.zeros(len(measurements), dtype=np.float32)
    prev_outside = False
    prev_traffic = False
    for t, m in enumerate(measurements):
        r = (
            float(m.get("reward_component_progress", 0.0))
            + float(m.get("reward_component_speed_limit_pen", 0.0))
            + float(m.get("reward_component_crash_stuck_pen", 0.0))
        )
        r += COLLISION_PEN * float(m.get("reward_collision_active", False))
        outside = bool(m.get("reward_outside_road", False))
        if outside and not prev_outside:
            r += OUTSIDE_ROUTE_PEN
        prev_outside = outside
        traffic = bool(
            m.get("reward_traffic_light_violation", False)
            or m.get("reward_stop_sign_violation", False)
        )
        if traffic and not prev_traffic:
            r += TRAFFIC_PEN
        prev_traffic = traffic
        rewards[t] = r
    return rewards


def _step_reward_non_noisy(m: dict, m_next: dict, route_len_m: float) -> float:
    pos_c = np.array(m["ego_matrix"], dtype=np.float64)[:3, 3]
    pos_n = np.array(m_next["ego_matrix"], dtype=np.float64)[:3, 3]
    movement = float(np.linalg.norm(pos_n - pos_c))
    rc_delta = movement / max(route_len_m, 1.0) * 100.0
    speed = float(m.get("speed", 0.0))
    speed_limit = float(m.get("speed_limit", 13.9))
    overspeed = max(0.0, speed / max(speed_limit, 1e-3) - 1.0)
    return PROGRESS_WEIGHT * rc_delta - SPEED_LIMIT_PEN_WEIGHT * overspeed


def _route_length_m(m: dict) -> float:
    route = m.get("route_original") or m.get("route")
    if not route or len(route) < 2:
        return 200.0
    pts = np.array(route, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return 200.0
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def _mc_returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    G = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + discount * running
        G[t] = running
    return G


def _control_2d_action(m: dict, is_noisy: bool) -> np.ndarray:
    if is_noisy:
        throttle = float(m.get("noisy_throttle", m.get("throttle", 0.0)))
        steer = float(m.get("noisy_steer", m.get("steer", 0.0)))
        brake = float(m.get("noisy_brake", m.get("brake", 0.0)))
    else:
        throttle = float(m.get("throttle", 0.0))
        steer = float(m.get("steer", 0.0))
        brake = float(m.get("brake", 0.0))
    accel = float(np.clip(throttle - brake, -1.0, 1.0))
    steer = float(np.clip(steer, -1.0, 1.0))
    return np.array([accel, steer], dtype=np.float32)


def _target_points_ego(m: dict) -> np.ndarray:
    """Return 2 nearest route waypoints in ego frame, shape (2, 2).

    Used in get_encoder_features prompt construction.  Falls back to
    [5 m forward, 10 m forward] when route data is absent or malformed.
    """
    route = m.get("route_original") or m.get("route") or []
    fallback = np.array([[5.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    if len(route) < 2:
        return fallback
    try:
        ego = np.array(m.get("ego_matrix", np.eye(4)), dtype=np.float64)
        R, t_pos = ego[:3, :3], ego[:3, 3]
        pts_global = np.array(route[:4], dtype=np.float64)
        if pts_global.ndim != 2 or pts_global.shape[1] < 2:
            return fallback
        pts_ego = []
        for pt in pts_global:
            gp = np.array([pt[0], pt[1], 0.0])
            local = R.T @ (gp - t_pos)
            pts_ego.append([float(local[0]), float(local[1])])
        pts_ego_arr = np.array(pts_ego, dtype=np.float32)
        return pts_ego_arr[:2] if len(pts_ego_arr) >= 2 else fallback
    except Exception:
        return fallback


# ── Dataset ───────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    img_path: str
    action: np.ndarray          # (2,) [accel, steer]
    mc_return: float
    is_noisy: bool
    speed: float                # m/s, for encoder feature construction
    target_points: np.ndarray   # (2, 2) ego-frame route waypoints, for encoder prompt


def _load_measurements(meas_dir: Path) -> Optional[List[dict]]:
    files = sorted(meas_dir.glob("*.json.gz"))
    out: List[dict] = []
    for f in files:
        try:
            with gzip.open(str(f), "rt") as fp:
                out.append(json.load(fp))
        except Exception:
            return None
    return out or None


def load_route(
    route_dir: str,
    discount: float = 0.99,
    is_noisy: Optional[bool] = None,
) -> List[Transition]:
    p = Path(route_dir)
    meas_dir, rgb_dir = p / "measurements", p / "rgb"
    if not meas_dir.exists() or not rgb_dir.exists():
        return []
    measurements = _load_measurements(meas_dir)
    if measurements is None or len(measurements) < ACTION_HORIZON + 2:
        return []

    is_noisy = "reward_route_completion_delta" in measurements[0] if is_noisy is None else is_noisy
    n = len(measurements)

    if is_noisy:
        rewards = _rewards_noisy(measurements)
    else:
        route_len = _route_length_m(measurements[0])
        rewards = np.array(
            [
                _step_reward_non_noisy(
                    measurements[t],
                    measurements[t + 1] if t + 1 < n else measurements[t],
                    route_len,
                )
                for t in range(n)
            ],
            dtype=np.float32,
        )

    results_path = p / "results.json.gz"
    if results_path.exists():
        try:
            with gzip.open(str(results_path), "rt") as fp:
                res = json.load(fp)
            score = res["scores"].get("score_composed", 0.0)
            rewards[-1] += SUCCESS_BONUS if score >= 95.0 else FAILURE_BONUS
        except Exception:
            pass

    G = _mc_returns(rewards, discount)

    transitions: List[Transition] = []
    for t in range(n - 1):
        img_path = str(rgb_dir / f"{t:04d}.jpg")
        if not Path(img_path).exists():
            continue
        action = _control_2d_action(measurements[t], bool(is_noisy))
        speed = float(measurements[t].get("speed", 0.0))
        target_points = _target_points_ego(measurements[t])
        transitions.append(
            Transition(img_path, action, float(G[t]), bool(is_noisy), speed, target_points)
        )
    return transitions


def find_route_dirs(
    noisy_root: Optional[str],
    non_noisy_root: Optional[str],
    noise_sweep_root: Optional[str] = None,
) -> List[Tuple[str, bool]]:
    dirs: List[Tuple[str, bool]] = []
    if noisy_root:
        for worker in Path(noisy_root).iterdir():
            if not worker.name.startswith("worker"):
                continue
            for route in worker.iterdir():
                if (route / "measurements").exists() and (route / "rgb").exists():
                    dirs.append((str(route), True))
    if non_noisy_root:
        base = Path(non_noisy_root) / "database" / "simlingo" / "data" / "simlingo"
        if base.exists():
            for meas_path in base.rglob("measurements"):
                route = meas_path.parent
                if (route / "rgb").exists():
                    dirs.append((str(route), False))
    if noise_sweep_root:
        for task_dir in sorted(Path(noise_sweep_root).iterdir()):
            if not task_dir.is_dir():
                continue
            for route in sorted(task_dir.iterdir()):
                if (
                    route.is_dir()
                    and (route / "measurements").exists()
                    and (route / "rgb").exists()
                ):
                    dirs.append((str(route), True))
    return dirs


# ── Feature extraction ────────────────────────────────────────────────────────

def encode_transitions(
    transitions: List[Transition],
    simlingo_model,
    routing_command: str = "",
) -> np.ndarray:
    """Extract ENCODER_FEATURE_DIM-dim InternVL2 features for all transitions.

    Calls SimLingoBase.get_encoder_features(image, ego_state, target_points, routing_command).
    Speed is taken from each transition; target_points are estimated from route data.
    routing_command defaults to "" — appropriate for offline data with unknown routing.

    Returns (N, ENCODER_FEATURE_DIM) float32.
    """
    ego_state_buf = np.zeros(25, dtype=np.float32)
    features = []
    for tr in tqdm(transitions, desc="encode internvl2", unit="img"):
        img_bgr = cv2.imread(tr.img_path)
        if img_bgr is None:
            features.append(np.zeros(ENCODER_FEATURE_DIM, dtype=np.float32))
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ego_state_buf[:] = 0.0
        ego_state_buf[_EGO_STATE_IDX_SPEED] = tr.speed
        feat = simlingo_model.get_encoder_features(
            img_rgb, ego_state_buf, tr.target_points, routing_command
        )
        features.append(feat.astype(np.float32))
    return np.stack(features)


# ── Critic training ───────────────────────────────────────────────────────────

def build_agent(
    vlm_feature_dim: int,
    hidden_dims: Tuple[int, ...],
    lr: float,
    device: str,
):
    """Build ResidualSACAgent with state_dim=0 (no ego state) and no coach label.

    Only the critic is trained offline.  The actor starts at random init and
    is fine-tuned when online SAC begins.
    """
    from torch_agents.residual_sac import ResidualSACAgent  # type: ignore

    return ResidualSACAgent(
        vlm_feature_dim=vlm_feature_dim,
        action_dim=2,
        hidden_dims=hidden_dims,
        gamma=0.99,
        tau=0.005,
        actor_lr=lr,
        critic_lr=lr,
        device=device,
        state_dim=0,
        coach_label_dim=0,
        expert_action_dim=0,
    )


def critic_step(
    agent,
    obs_t: torch.Tensor,
    acts_t: torch.Tensor,
    targets_t: torch.Tensor,
    tau: float,
) -> dict:
    """One supervised MSE step: Q(obs, a, a) -> MC return.

    Both base_action and action are set to the stored action — there is no
    base/residual decomposition in offline data.  The target critic is
    soft-updated after every step.
    """
    dev = agent.device
    obs_t = obs_t.to(dev)
    acts_t = acts_t.to(dev)
    targets_t = targets_t.to(dev)

    q1, q2 = agent.critic(obs_t, acts_t, acts_t)
    q_min = torch.min(q1, q2).squeeze(-1)
    loss = F.mse_loss(q_min, targets_t)

    agent.critic_opt.zero_grad()
    loss.backward()
    agent.critic_opt.step()

    for tp, sp in zip(agent.critic_target.parameters(), agent.critic.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    with torch.no_grad():
        q1_t, q2_t = agent.critic_target(obs_t, acts_t, acts_t)
        q_t = torch.min(q1_t, q2_t).squeeze(-1)

    return {
        "critic_loss": float(loss),
        "q_mean": float(q_min.mean()),
        "q_std": float(q_min.std()),
        "q_target_mean": float(q_t.mean()),
        "target_mean": float(targets_t.mean()),
        "target_std": float(targets_t.std()),
    }


def sample_batch(
    transitions: List[Transition],
    embeddings: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idxs = np.random.randint(len(transitions), size=batch_size)
    obs = embeddings[idxs]
    acts = np.stack([transitions[i].action for i in idxs])
    rets = np.array([transitions[i].mc_return for i in idxs], dtype=np.float32)
    return obs, acts, rets


# ── Visualization ─────────────────────────────────────────────────────────────

def log_q_vis(
    agent,
    val_transitions: List[Transition],
    val_embeddings: np.ndarray,
    noisy_cache: np.ndarray,
    clean_cache: np.ndarray,
    step: int,
    n_frames: int = 8,
    n_each: int = 3,
    n_random: int = 3,
) -> None:
    dev = agent.device
    action_std = np.std(
        np.concatenate([noisy_cache, clean_cache], axis=0), axis=0
    ).clip(1e-3)

    frame_idxs = random.sample(range(len(val_transitions)), min(n_frames, len(val_transitions)))
    images = []
    for fi in frame_idxs:
        tr = val_transitions[fi]
        obs_np = val_embeddings[fi : fi + 1]

        ni = np.random.choice(len(noisy_cache), size=n_each, replace=len(noisy_cache) < n_each)
        ci = np.random.choice(len(clean_cache), size=n_each, replace=len(clean_cache) < n_each)
        rand_a = (np.random.randn(n_random, 2) * action_std).astype(np.float32)

        sampled = np.concatenate([tr.action[None], noisy_cache[ni], clean_cache[ci], rand_a])
        labels = (
            ["GT"]
            + [f"N{i+1}" for i in range(n_each)]
            + [f"C{i+1}" for i in range(n_each)]
            + [f"R{i+1}" for i in range(n_random)]
        )

        K = len(sampled)
        obs_t = torch.from_numpy(obs_np).float().to(dev).expand(K, -1)
        acts_t = torch.from_numpy(sampled).float().to(dev)
        with torch.no_grad():
            q1, q2 = agent.critic(obs_t, acts_t, acts_t)
        q_vals = torch.min(q1, q2).squeeze(-1).cpu().numpy()

        legend_h = max(280, 28 * K)
        panel = np.zeros((legend_h, 200, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        colors = [
            (255, 255, 255), (255, 87, 34), (33, 150, 243), (76, 175, 80),
            (156, 39, 176), (255, 193, 7), (0, 188, 212), (255, 152, 0),
        ]
        order = np.argsort(-q_vals)
        cv2.putText(panel, "Q-values", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for rank, idx in enumerate(order):
            y = 32 + rank * 24
            if y + 16 > legend_h:
                break
            color = colors[idx % len(colors)]
            cv2.rectangle(panel, (8, y), (22, y + 14), color, -1)
            cv2.putText(
                panel, f"{labels[idx]}  {q_vals[idx]:+.2f}", (28, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1,
            )

        img_bgr = cv2.imread(tr.img_path)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h = min(img_rgb.shape[0], legend_h)
            vis = np.concatenate([img_rgb[:h], panel[:h]], axis=1)
        else:
            vis = panel

        images.append(
            wandb.Image(
                vis,
                caption=(
                    f"step={step}  mc={tr.mc_return:.2f}  "
                    f"Q_min={q_vals.min():.2f}  Q_max={q_vals.max():.2f}"
                ),
            )
        )
    wandb.log({"val/q_vis": images}, step=step)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline critic pretraining (InternVL2 / run_simlingo.sh setup)"
    )
    # Required
    p.add_argument(
        "--simlingo_checkpoint", required=True,
        help="Path to SimLingo checkpoint dir (epoch=013.ckpt). Used to load InternVL2 "
             "for offline feature extraction.",
    )
    # Data sources (same layout as pretrain_critic.py)
    p.add_argument("--noisy_root", default=None,
                   help="Root of noisy dataset (worker*/route_name layout).")
    p.add_argument("--non_noisy_root", default=None,
                   help="Root of non-noisy dataset (database/simlingo/data/simlingo layout).")
    p.add_argument("--noise_sweep_root", default=None,
                   help="Root of noise-sweep dataset (task_*/<route>/measurements+rgb layout).")
    p.add_argument("--max_routes", type=int, default=None,
                   help="Cap the number of routes loaded (applied after shuffling).")
    # Training
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005, help="Target network soft-update rate.")
    p.add_argument("--hidden_dims", default="256,256,256",
                   help="Comma-separated MLP hidden layer sizes, e.g. '256,256,256'.")
    p.add_argument("--device", default="cuda")
    # Feature extraction
    p.add_argument("--routing_command", default="",
                   help="Routing command injected into the InternVL2 prompt for offline data. "
                        "Empty string is a safe default when routing is unknown.")
    p.add_argument("--embedding_cache", default=None,
                   help="Path (.npz) to cache precomputed InternVL2 features across runs. "
                        "Recomputed when checkpoint or dataset changes.")
    # Logging
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--n_log_frames", type=int, default=8)
    # Checkpointing
    p.add_argument("--checkpoint_dir", default="/scratch/current/celinet/critic_pretrain_simlingo")
    p.add_argument("--checkpoint_every", type=int, default=5000)
    # W&B
    p.add_argument("--wandb_project", default="carla_critic_pretrain_simlingo")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    # ── Load SimLingo model ───────────────────────────────────────────────────
    print(f"Loading SimLingo model from {args.simlingo_checkpoint} ...")
    from vlas.simlingo_base import SimLingoBase  # type: ignore

    simlingo_model = SimLingoBase(
        checkpoint_path=args.simlingo_checkpoint,
        device=args.device,
    )
    simlingo_model.model.eval()
    for param in simlingo_model.model.parameters():
        param.requires_grad_(False)
    print("SimLingo loaded and frozen.")

    # ── W&B ──────────────────────────────────────────────────────────────────
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"critic_pretrain_simlingo_{run_ts}",
        mode=args.wandb_mode,
        config=vars(args),
    )

    # ── Scan + load transitions ───────────────────────────────────────────────
    print("Scanning route directories...")
    route_dirs = find_route_dirs(args.noisy_root, args.non_noisy_root, args.noise_sweep_root)
    print(f"Found {len(route_dirs)} routes.")
    if not route_dirs:
        raise RuntimeError("No route directories found — check --noisy_root / --non_noisy_root / --noise_sweep_root.")
    random.shuffle(route_dirs)
    if args.max_routes:
        route_dirs = route_dirs[: args.max_routes]

    print("Loading transitions...")
    all_transitions: List[Transition] = []
    for rd, is_noisy_route in tqdm(route_dirs, desc="load_routes"):
        all_transitions.extend(load_route(rd, discount=args.discount, is_noisy=is_noisy_route))
    if not all_transitions:
        raise RuntimeError("No transitions loaded — check dataset paths and image files.")
    random.shuffle(all_transitions)

    n_noisy = sum(1 for t in all_transitions if t.is_noisy)
    print(
        f"Loaded {len(all_transitions)} transitions "
        f"({n_noisy} noisy, {len(all_transitions) - n_noisy} non-noisy)."
    )

    n_val = max(args.n_log_frames, int(len(all_transitions) * args.val_frac))
    val_transitions = all_transitions[:n_val]
    train_transitions = all_transitions[n_val:]
    print(f"Train: {len(train_transitions)}  Val: {len(val_transitions)}")

    # ── Precompute InternVL2 features ─────────────────────────────────────────
    all_paths = [t.img_path for t in train_transitions] + [t.img_path for t in val_transitions]
    loaded_from_cache = False

    if args.embedding_cache and Path(args.embedding_cache).exists():
        print(f"Loading embeddings from cache: {args.embedding_cache}")
        data = np.load(args.embedding_cache, allow_pickle=True)
        cached_paths = list(data["paths"])
        if (
            set(cached_paths) == set(all_paths)
            and str(data.get("checkpoint", "")) == args.simlingo_checkpoint
        ):
            path_to_idx = {p: i for i, p in enumerate(cached_paths)}
            all_embs = data["embeddings"][[path_to_idx[p] for p in all_paths]]
            train_embeddings = all_embs[: len(train_transitions)]
            val_embeddings = all_embs[len(train_transitions) :]
            print(f"Cache hit: {all_embs.shape[0]} embeddings.")
            loaded_from_cache = True
        else:
            print("Cache mismatch (dataset or checkpoint changed) — recomputing.")

    if not loaded_from_cache:
        print("Precomputing InternVL2 features (this may take a while) ...")
        with torch.no_grad():
            train_embeddings = encode_transitions(
                train_transitions, simlingo_model, args.routing_command
            )
            val_embeddings = encode_transitions(
                val_transitions, simlingo_model, args.routing_command
            )
        print(
            f"Embeddings: train={train_embeddings.shape}  val={val_embeddings.shape}"
        )

        if args.embedding_cache:
            Path(args.embedding_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.embedding_cache,
                embeddings=np.concatenate([train_embeddings, val_embeddings], axis=0),
                paths=np.array(all_paths),
                checkpoint=args.simlingo_checkpoint,
            )
            print(f"Embeddings cached to {args.embedding_cache}")

    # Free SimLingo GPU memory before training
    del simlingo_model
    gc.collect()
    torch.cuda.empty_cache()
    print("SimLingo GPU memory freed.")

    # ── Action caches for Q-value visualization ───────────────────────────────
    noisy_train = [t for t in train_transitions if t.is_noisy] or train_transitions
    clean_train = [t for t in train_transitions if not t.is_noisy] or train_transitions
    cache_n = min(2000, len(train_transitions) // 2)
    noisy_vis_idxs = np.random.choice(len(noisy_train), min(cache_n, len(noisy_train)), replace=False)
    clean_vis_idxs = np.random.choice(len(clean_train), min(cache_n, len(clean_train)), replace=False)
    noisy_cache = np.stack([noisy_train[i].action for i in noisy_vis_idxs])
    clean_cache = np.stack([clean_train[i].action for i in clean_vis_idxs])

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = build_agent(
        vlm_feature_dim=ENCODER_FEATURE_DIM,
        hidden_dims=hidden_dims,
        lr=args.lr,
        device=args.device,
    )
    agent.critic.train()
    agent.actor.eval()

    ckpt_dir = Path(args.checkpoint_dir) / f"run_{run_ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {ckpt_dir}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Starting critic pretraining ...")
    pbar = tqdm(total=args.total_steps, desc="pretrain_critic_simlingo")

    for step in range(1, args.total_steps + 1):
        obs_b, acts_b, rets_b = sample_batch(train_transitions, train_embeddings, args.batch_size)
        info = critic_step(
            agent,
            torch.from_numpy(obs_b).float(),
            torch.from_numpy(acts_b).float(),
            torch.from_numpy(rets_b).float(),
            tau=args.tau,
        )
        pbar.update(1)

        if step % 100 == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=step)

        if step % args.checkpoint_every == 0:
            ckpt_path = str(ckpt_dir / f"step_{step:07d}.pt")
            agent.save(ckpt_path)
            latest = ckpt_dir / "latest.pt"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(f"step_{step:07d}.pt")

        if step % args.log_every == 0:
            log_q_vis(
                agent,
                val_transitions,
                val_embeddings,
                noisy_cache,
                clean_cache,
                step,
                n_frames=args.n_log_frames,
            )

    pbar.close()
    final_path = str(ckpt_dir / "final.pt")
    agent.save(final_path)
    print(f"Final checkpoint -> {final_path}")
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
