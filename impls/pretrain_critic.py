#!/usr/bin/env python3
"""Offline DSRL critic pretraining from noisy + non-noisy CARLA datasets.

Observation encoder: frozen SigLIP2 SO400M (same as the online DSRL setup).
SigLIP encoding happens in PyTorch outside JAX; CarlaObservationEncoder is a
float32 passthrough when image_encoder='siglip'.

Reward mirrors ~/ogbench-carla/ogbench/carla/carla_utils.py defaults:
  progress = 5.0 * route_completion_delta * centering * heading
  speed_limit_pen = 0.1 * max(0, speed/speed_limit - 1)
  collision_pen = -20.0 (per active collision step, noisy data only)
  outside_route_pen = -20.0, traffic_pen = -20.0, crash_stuck_pen = -20.0

Non-noisy data: no violation fields — estimates route_completion_delta from
consecutive ego_matrix positions; centering = heading = 1.0.

Multi-GPU: if multiple JAX devices are visible, critic training uses pmap with
per-device gradient averaging.  SigLIP runs on a separate PyTorch device
(--siglip_device, default 'cuda:0'); JAX uses all visible devices.

Checkpoints saved to --checkpoint_dir every --checkpoint_every steps.

Example
-------
# noise-sweep data only
uv run python impls/pretrain_critic.py \
    --noise_sweep_root /scratch/current/celinet/noise_sweep/data \
    --total_steps 100000 --batch_size 256 --wandb_project carla_critic_pretrain

# legacy noisy + non-noisy datasets
uv run python impls/pretrain_critic.py \
    --noisy_root /scratch/current/celinet/noisy_dataset \
    --non_noisy_root /scratch/current/celinet/non_noisy_dataset \
    --total_steps 100000 --batch_size 256 --wandb_project carla_critic_pretrain
"""

from __future__ import annotations

import argparse
import datetime
import functools
import gzip
import json
import pickle
import random
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import wandb
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from jax_agents.dsrl import DSRLAgent
from utils.flax_utils import TrainState
from utils.siglip_encoder import SigLIPEncoder

# ── Action / reward constants (ogbench-carla defaults) ────────────────────────
ACTION_HORIZON = 10
ACTION_DIM = 4           # DELTA_XY_T_DELTA_XY_SPACE: [dx_t, dy_t, dx_s, dy_s]
ACTION_FLAT_DIM = ACTION_HORIZON * ACTION_DIM   # 40
CONTROL_2D_DIM = 2       # [accel, steer] for control_2d action mode

PROGRESS_WEIGHT = 5.0
SPEED_LIMIT_PEN_WEIGHT = 0.1
COLLISION_PEN = -20.0     # applied per step while contact is active
OUTSIDE_ROUTE_PEN = -20.0  # applied only on rising edge (False→True transition)
TRAFFIC_PEN = -20.0        # applied only on rising edge
SUCCESS_BONUS = 50.0       # from ~/ogbench-carla carla_utils.py
FAILURE_BONUS = -20.0      # from ~/ogbench-carla carla_utils.py

# Default image size (steervla-pi training default)
DEFAULT_IMG_H = 224
DEFAULT_IMG_W = 224

# Camera projection params (simlingo convention)
CAMERA_FOV = 110.0
CAMERA_TVEC = np.array([0.0, 2.0, 1.5], dtype=np.float32)

TRAJ_COLORS = [
    (255, 87, 34), (33, 150, 243), (76, 175, 80), (156, 39, 176),
    (255, 193, 7), (0, 188, 212), (255, 152, 0), (233, 30, 99),
    (96, 125, 139), (205, 220, 57),
]


# ── SigLIP batch encoding ─────────────────────────────────────────────────────

def _infer_chunk(
    enc: SigLIPEncoder,
    paths: list[str],
    include_prompt_subtask: bool,
    sub_batch: int,
    result_holder: list,
    slot: int,
    pbar_update,  # callable(n) to advance shared tqdm bar
) -> None:
    """Inference-only worker: enc is already set up, runs on its assigned device."""
    import torch

    all_embs: list[np.ndarray] = []
    for start in range(0, len(paths), sub_batch):
        batch_paths = paths[start:start + sub_batch]
        pils = []
        for p in batch_paths:
            try:
                pils.append(Image.open(p).convert("RGB"))
            except Exception:
                pils.append(Image.new("RGB", (224, 224), color=0))
        inputs = enc._processor(images=pils, return_tensors="pt")
        inputs = {k: v.to(enc._device) for k, v in inputs.items()}
        with torch.no_grad():
            embs = enc._model.get_image_features(**inputs)
            embs = enc._normalize(embs)
        all_embs.append(embs.detach().cpu().numpy().astype(np.float32))
        pbar_update(len(batch_paths))

    image_embs = np.concatenate(all_embs, axis=0)
    if include_prompt_subtask:
        zero = np.zeros((len(paths), enc.embedding_dim), dtype=np.float32)
        image_embs = np.concatenate([image_embs, zero, zero], axis=-1)
    result_holder[slot] = image_embs


def encode_all_parallel(
    img_paths: list[str],
    model_id: str,
    devices: list[str],
    *,
    include_prompt_subtask: bool = False,
    sub_batch: int = 128,
) -> np.ndarray:
    """Encode all paths in parallel across multiple GPU devices.

    Models are loaded sequentially (HuggingFace from_pretrained is not thread-safe),
    then inference runs in parallel threads — PyTorch releases the GIL during CUDA
    ops so threads run truly in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    n = len(img_paths)
    n_dev = len(devices)
    chunk_size = (n + n_dev - 1) // n_dev
    chunks = [img_paths[i:i + chunk_size] for i in range(0, n, chunk_size)]
    active_devices = devices[:len(chunks)]

    # Load models sequentially — from_pretrained is not thread-safe
    print(f"Loading {len(active_devices)} SigLIP model(s)...")
    encoders = []
    for device in active_devices:
        enc = SigLIPEncoder(model_id=model_id, device=device)
        enc.setup()
        encoders.append(enc)

    results: list[np.ndarray | None] = [None] * len(chunks)
    lock = threading.Lock()

    print(f"Encoding {n} images across {len(chunks)} GPU(s): {active_devices}")
    with tqdm(total=n, desc="SigLIP precompute", unit="img") as pbar:
        def _update(k):
            with lock:
                pbar.update(k)

        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {
                pool.submit(
                    _infer_chunk,
                    encoders[i], chunks[i],
                    include_prompt_subtask, sub_batch, results, i, _update,
                ): i
                for i in range(len(chunks))
            }
            for f in as_completed(futures):
                f.result()

    return np.concatenate(results, axis=0)


def encode_paths_siglip(
    encoder: SigLIPEncoder,
    img_paths: list[str],
    *,
    include_prompt_subtask: bool = False,
    prompt: str = "",
    subtask: str = "",
    sub_batch: int = 128,
) -> np.ndarray:
    """Encode a list of image paths with a single SigLIPEncoder, returning (N, embed_dim) float32."""
    import torch

    encoder.setup()
    all_embs: list[np.ndarray] = []

    for start in range(0, len(img_paths), sub_batch):
        pils = [Image.open(p).convert("RGB") for p in img_paths[start:start + sub_batch]]
        inputs = encoder._processor(images=pils, return_tensors="pt")
        inputs = {k: v.to(encoder._device) for k, v in inputs.items()}
        with encoder._lock:
            with torch.no_grad():
                embs = encoder._model.get_image_features(**inputs)
                embs = encoder._normalize(embs)
        all_embs.append(embs.detach().cpu().numpy().astype(np.float32))

    image_embs = np.concatenate(all_embs, axis=0)

    if not include_prompt_subtask:
        return image_embs

    prompt_emb = encoder.encode_text(prompt)[None]
    subtask_emb = encoder.encode_text(subtask)[None]
    return np.concatenate([
        image_embs,
        np.tile(prompt_emb, (len(img_paths), 1)),
        np.tile(subtask_emb, (len(img_paths), 1)),
    ], axis=-1)


# ── Camera projection ─────────────────────────────────────────────────────────

def _camera_K(w: int, h: int, fov: float = CAMERA_FOV) -> np.ndarray:
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def project_waypoints(wps_xy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Project (N, 2) ego-frame [x_forward, y_right] waypoints to (N, 2) image px."""
    K = _camera_K(img_w, img_h)
    rvec = np.zeros((3, 1), np.float32)
    tvec = CAMERA_TVEC.reshape(3, 1)
    dist = np.zeros((5, 1), np.float32)
    pts_3d = np.zeros((len(wps_xy), 1, 3), dtype=np.float32)
    pts_3d[:, 0, 0] = wps_xy[:, 1].astype(np.float32)  # y_right  → camera x
    pts_3d[:, 0, 2] = wps_xy[:, 0].astype(np.float32)  # x_forward → camera z
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    return pts_2d[:, 0, :]  # (N, 2)


# ── Reward ────────────────────────────────────────────────────────────────────

def _rewards_noisy(measurements: list[dict]) -> np.ndarray:
    """Compute per-step rewards for a noisy-dataset route.

    Uses stored reward_component_* fields for the continuous terms (progress,
    speed-limit, crash-stuck) and rising-edge detection on boolean flags for
    event-based penalties (outside_road, traffic violations) to mirror the
    criteria-delta logic in ~/ogbench-carla carla_utils.py.
    Collision is kept as a per-step contact penalty (matches legacy mode).
    """
    rewards = np.zeros(len(measurements), dtype=np.float32)
    prev_outside = False
    prev_traffic = False
    for t, m in enumerate(measurements):
        # Continuous components stored with exact values in the measurement
        r = (
            float(m.get("reward_component_progress", 0.0))
            + float(m.get("reward_component_speed_limit_pen", 0.0))  # already negative
            + float(m.get("reward_component_crash_stuck_pen", 0.0))  # already negative
        )
        # Collision: per-step contact penalty (legacy mode in carla_utils)
        r += COLLISION_PEN * float(m.get("reward_collision_active", False))
        # Outside-route and traffic: apply only on rising edge (new event)
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


# ── Action from future ego positions ─────────────────────────────────────────

def _waypoints_action(measurements: list[dict], t: int) -> np.ndarray | None:
    if t + ACTION_HORIZON >= len(measurements):
        return None
    origin = np.array(measurements[t]["ego_matrix"], dtype=np.float64)
    R, trans = origin[:3, :3], origin[:3, 3:4]
    wps = []
    for k in range(t, t + ACTION_HORIZON + 1):
        wp = np.array(measurements[k]["ego_matrix"], dtype=np.float64)[:3, 3:4]
        wps.append((R.T @ (wp - trans))[:2, 0])
    deltas = np.diff(np.array(wps), axis=0)      # (H, 2) — physical meters
    # Critic trains in model space to match a_approx_critic at QGF inference time
    # (before steervla_physical_denormalize_actions). dims 0:2 are RLDS-scaled (/7),
    # dims 2:4 are already at physical scale (denorm applies ×1).
    action = np.concatenate([deltas / 7.0, deltas], axis=-1)  # (H, 4)
    return action.reshape(-1).astype(np.float32)


def _clean_control_2d_action(m: dict) -> np.ndarray:
    """Return clean (non-noisy) [accel, steer] from stored controls."""
    throttle = float(m.get("throttle", 0.0))
    steer = float(m.get("steer", 0.0))
    brake = float(m.get("brake", 0.0))
    return np.array([
        float(np.clip(throttle - brake, -1.0, 1.0)),
        float(np.clip(steer, -1.0, 1.0)),
    ], dtype=np.float32)


def _control_2d_action(m: dict, is_noisy: bool) -> np.ndarray:
    """Return [accel, steer] from the stored control for a single step.

    Noisy data: use noisy_throttle/noisy_steer/noisy_brake (the perturbed command
    actually sent to the vehicle — this is where the noise lives).
    Non-noisy data: use the clean expert throttle/steer/brake.
    accel = throttle - brake, both clipped to [-1, 1].
    """
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    img_path: str
    action: np.ndarray        # critic input: (ACTION_FLAT_DIM,) for waypoints, (2,) for control_2d
    mc_return: float
    is_noisy: bool            # True = from noisy dataset, False = from non-noisy
    waypoints: np.ndarray | None = None  # always (ACTION_FLAT_DIM,) from ego_matrix, used for viz
    base_action: np.ndarray | None = None  # (2,) clean expert [accel, steer], zeros if unused


def _load_measurements(meas_dir: Path) -> list[dict] | None:
    files = sorted(meas_dir.glob("*.json.gz"))
    out = []
    for f in files:
        try:
            with gzip.open(f, "rt") as fp:
                out.append(json.load(fp))
        except Exception:
            return None
    return out or None


def load_route(
    route_dir: str,
    discount: float = 0.99,
    is_noisy: bool | None = None,
    action_mode: str = "waypoints",
) -> list[Transition]:
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
        rewards = np.array([
            _step_reward_non_noisy(measurements[t], measurements[t + 1] if t + 1 < n else measurements[t], route_len)
            for t in range(n)
        ], dtype=np.float32)

    results_path = p / "results.json.gz"
    if results_path.exists():
        try:
            with gzip.open(results_path, "rt") as fp:
                res = json.load(fp)
            score = res["scores"].get("score_composed", 0.0)
            rewards[-1] += SUCCESS_BONUS if score >= 95.0 else FAILURE_BONUS
        except Exception:
            pass

    G = _mc_returns(rewards, discount)

    transitions = []
    for t in range(n - ACTION_HORIZON - 1):
        img_path = str(rgb_dir / f"{t:04d}.jpg")
        if not Path(img_path).exists():
            continue
        wp = _waypoints_action(measurements, t)
        if wp is None:
            continue
        if action_mode == "waypoints":
            action, waypoints = wp, None
            base_action = np.zeros(CONTROL_2D_DIM, dtype=np.float32)
        else:
            action = _control_2d_action(measurements[t], bool(is_noisy))
            waypoints = wp
            # Always store the clean expert action; sampling decides whether to use it
            base_action = _clean_control_2d_action(measurements[t])
        transitions.append(Transition(img_path, action, float(G[t]), bool(is_noisy), waypoints, base_action))
    return transitions


def find_route_dirs(
    noisy_root: str | None,
    non_noisy_root: str | None,
    noise_sweep_root: str | None = None,
) -> list[tuple[str, bool]]:
    """Return list of (route_dir, is_noisy) tuples.

    noise_sweep_root expects the layout:
        <noise_sweep_root>/task_*/
            <route_name>/
                measurements/
                rgb/
    All noise_sweep routes are treated as noisy (they carry noisy_throttle/steer/brake).
    """
    dirs: list[tuple[str, bool]] = []
    if noisy_root:
        noisy_path = Path(noisy_root)
        worker_dirs = [d for d in noisy_path.iterdir() if d.is_dir() and d.name.startswith("worker")]
        if worker_dirs:
            # Legacy layout: <noisy_root>/worker*/route_dir/{measurements,rgb}
            for worker in worker_dirs:
                for route in worker.iterdir():
                    if (route / "measurements").exists() and (route / "rgb").exists():
                        dirs.append((str(route), True))
        else:
            # simlingo layout: arbitrary nesting, find all route dirs via rglob
            for meas_path in sorted(noisy_path.rglob("measurements")):
                route = meas_path.parent
                if (route / "rgb").exists():
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
                if route.is_dir() and (route / "measurements").exists() and (route / "rgb").exists():
                    dirs.append((str(route), True))
    return dirs


# ── Batch sampling ────────────────────────────────────────────────────────────

def _transition_base_action(tr: Transition) -> np.ndarray:
    if tr.base_action is not None:
        return tr.base_action
    return np.zeros(CONTROL_2D_DIM, dtype=np.float32)



def sample_batch_siglip(
    transitions: list[Transition],
    batch_size: int,
    encoder: SigLIPEncoder | None,
    include_prompt_subtask: bool,
    sub_batch: int,
    embeddings: np.ndarray | None = None,
    expert_guidance: bool = False,
    lang_dim: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idxs = np.random.randint(len(transitions), size=batch_size)
    acts = np.stack([transitions[i].action for i in idxs])
    if expert_guidance:
        lang = np.stack([_transition_base_action(transitions[i]) for i in idxs])
    else:
        lang = np.zeros((batch_size, lang_dim), dtype=np.float32)
    rets = np.array([transitions[i].mc_return for i in idxs], dtype=np.float32)
    if embeddings is not None:
        obs = embeddings[idxs]
    else:
        assert encoder is not None
        paths = [transitions[i].img_path for i in idxs]
        obs = encode_paths_siglip(encoder, paths,
                                   include_prompt_subtask=include_prompt_subtask,
                                   sub_batch=sub_batch)
    return obs, acts, rets, lang


def sample_batch_impala(
    transitions: list[Transition],
    batch_size: int,
    img_h: int,
    img_w: int,
    expert_guidance: bool = False,
    lang_dim: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idxs = np.random.randint(len(transitions), size=batch_size)
    imgs = [
        np.array(Image.open(transitions[i].img_path).convert("RGB")
                 .resize((img_w, img_h), Image.BILINEAR), dtype=np.uint8)
        for i in idxs
    ]
    acts = np.stack([transitions[i].action for i in idxs])
    if expert_guidance:
        lang = np.stack([_transition_base_action(transitions[i]) for i in idxs])
    else:
        lang = np.zeros((len(idxs), lang_dim), dtype=np.float32)
    rets = np.array([transitions[i].mc_return for i in idxs], dtype=np.float32)
    return np.stack(imgs), acts, rets, lang


# ── JAX critic update ─────────────────────────────────────────────────────────

@jax.jit
def _critic_step(
    network: TrainState,
    obs: jnp.ndarray,           # (B, embed_dim) or (B, H, W, 3)
    actions: jnp.ndarray,       # (B, act_dim)
    targets: jnp.ndarray,       # (B,)
    language_label: jnp.ndarray, # (B, lang_dim); lang_dim=0 for no guidance, 2 for expert
):
    def loss_fn(grad_params):
        obs_e = network.select("obs_encoder")(obs, params=grad_params)
        critic_obs = jnp.concatenate([obs_e, language_label], axis=-1)
        qs = network.select("critic")(critic_obs, actions, params=grad_params)
        loss = jnp.square(qs - targets[None]).mean()
        return loss, {
            "critic_loss": loss,
            "q_mean": qs.mean(),
            "q_std": qs.std(),
            "target_mean": targets.mean(),
            "target_std": targets.std(),
        }
    return network.apply_loss_fn(loss_fn=loss_fn)


@functools.partial(jax.pmap, axis_name="data")
def _critic_step_pmap(
    network: TrainState,
    obs: jnp.ndarray,           # (local_B, embed_dim) or (local_B, H, W, 3)
    actions: jnp.ndarray,       # (local_B, act_dim)
    targets: jnp.ndarray,       # (local_B,)
    language_label: jnp.ndarray, # (local_B, lang_dim)
):
    def loss_fn(grad_params):
        obs_e = network.select("obs_encoder")(obs, params=grad_params)
        critic_obs = jnp.concatenate([obs_e, language_label], axis=-1)
        qs = network.select("critic")(critic_obs, actions, params=grad_params)
        loss = jnp.square(qs - targets[None]).mean()
        return loss, {
            "critic_loss": loss,
            "q_mean": qs.mean(),
            "q_std": qs.std(),
            "target_mean": targets.mean(),
            "target_std": targets.std(),
        }
    grads, info = jax.grad(loss_fn, has_aux=True)(network.params)
    grads = jax.lax.pmean(grads, axis_name="data")
    info = jax.lax.pmean(info, axis_name="data")
    updates, new_opt_state = network.tx.update(grads, network.opt_state, network.params)
    new_params = optax.apply_updates(network.params, updates)
    return network.replace(step=network.step + 1, params=new_params, opt_state=new_opt_state), info


@jax.jit
def _eval_q(
    network: TrainState,
    obs: jnp.ndarray,            # (1, *obs_shape)
    actions: jnp.ndarray,        # (K, act_dim)
    language_label: jnp.ndarray, # (1, lang_dim); lang_dim=0 for no guidance
) -> jnp.ndarray:                # (K,)
    K = actions.shape[0]
    obs_tiled = jnp.repeat(obs, K, axis=0)
    obs_e = network.select("obs_encoder")(obs_tiled)
    lang_tiled = jnp.repeat(language_label, K, axis=0)
    critic_obs = jnp.concatenate([obs_e, lang_tiled], axis=-1)
    qs = network.select("critic")(critic_obs, actions)
    return jnp.min(qs, axis=0)


# ── Visualisation ─────────────────────────────────────────────────────────────

def _draw_trajectory(
    img: np.ndarray,
    action_flat: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dot_radius: int = 4,
) -> np.ndarray:
    h, w = img.shape[:2]
    route_wps = np.vstack([
        np.zeros((1, 2)),
        np.cumsum(action_flat.reshape(ACTION_HORIZON, ACTION_DIM)[:, 2:], axis=0),
    ])
    pts = project_waypoints(route_wps, w, h)
    out = img.copy()
    for i in range(len(pts) - 1):
        u1, v1 = int(round(pts[i, 0])), int(round(pts[i, 1]))
        u2, v2 = int(round(pts[i + 1, 0])), int(round(pts[i + 1, 1]))
        if 0 <= u1 < w and 0 <= v1 < h:
            cv2.circle(out, (u1, v1), dot_radius, color, -1)
            if 0 <= u2 < w and 0 <= v2 < h:
                cv2.line(out, (u1, v1), (u2, v2), color, thickness)
    return out


# Ground-truth trajectory color: bright white, thicker so it stands out
GT_COLOR = (255, 255, 255)

_LEGEND_W = 160      # pixels wide for the legend panel
_LEGEND_ROW_H = 22   # px per legend entry
_LEGEND_PAD = 8      # px padding inside legend


def _make_legend(
    labels: list[str],   # e.g. ["GT", "S1", "S2", ...]
    q_values: list[float],
    colors: list[tuple[int, int, int]],
    img_h: int,
) -> np.ndarray:
    """Return an (img_h, _LEGEND_W, 3) uint8 legend panel."""
    panel = np.zeros((img_h, _LEGEND_W, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # dark background

    # Sort entries by Q-value descending so highest is at top
    order = sorted(range(len(q_values)), key=lambda i: -q_values[i])

    title = "Q-values"
    cv2.putText(panel, title, (_LEGEND_PAD, _LEGEND_PAD + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    for rank, idx in enumerate(order):
        y = _LEGEND_PAD + 26 + rank * _LEGEND_ROW_H
        if y + _LEGEND_ROW_H > img_h:
            break
        color = colors[idx]
        # Color swatch
        cv2.rectangle(panel, (_LEGEND_PAD, y), (_LEGEND_PAD + 14, y + 14), color, -1)
        # Label + Q-value
        text = f"{labels[idx]}  {q_values[idx]:+.2f}"
        cv2.putText(panel, text, (_LEGEND_PAD + 20, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

    return panel


def make_overlay_image(
    img_path: str,
    actions: np.ndarray,                    # (K, act_dim) — actions[0] is always the ground-truth
    q_values: np.ndarray,                   # (K,)
    waypoints: np.ndarray | None = None,    # (K, ACTION_FLAT_DIM) override for viz, e.g. in control_2d mode
    labels: list[str] | None = None,        # if None, auto-generates "GT", "S1", "S2", ...
) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((256, 512, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if labels is None:
        labels = ["GT"] + [f"S{i}" for i in range(1, len(actions))]
    colors = [GT_COLOR] + [TRAJ_COLORS[(i - 1) % len(TRAJ_COLORS)] for i in range(1, len(actions))]

    # Use waypoints override if provided, else fall back to actions (only if 40-d)
    draw_wps = waypoints if waypoints is not None else (actions if actions.shape[1] == ACTION_FLAT_DIM else None)
    if draw_wps is not None:
        for i in range(1, len(draw_wps)):
            img = _draw_trajectory(img, draw_wps[i], colors[i])
        img = _draw_trajectory(img, draw_wps[0], GT_COLOR, thickness=3, dot_radius=5)

    # Append legend panel on the right
    legend = _make_legend(labels, list(q_values), colors, img.shape[0])
    return np.concatenate([img, legend], axis=1)


def log_gt_action_examples(
    network: TrainState,
    val_transitions: list[Transition],
    val_embeddings: np.ndarray | None,
    encoder: SigLIPEncoder | None,
    include_prompt_subtask: bool,
    step: int,
    lang_dim: int = 0,
    n_frames: int = 10,
) -> None:
    """Log GT action overlays from noisy and non-noisy val transitions with mc_return labels."""
    noisy_val = [(i, t) for i, t in enumerate(val_transitions) if t.is_noisy]
    non_noisy_val = [(i, t) for i, t in enumerate(val_transitions) if not t.is_noisy]

    def _make_images(pool: list[tuple[int, Transition]], label: str) -> list[wandb.Image]:
        if not pool:
            return []
        chosen = random.sample(pool, min(n_frames, len(pool)))
        imgs = []
        for val_idx, tr in chosen:
            if val_embeddings is not None:
                obs_np = val_embeddings[val_idx:val_idx + 1]
            elif encoder is not None:
                obs_np = encode_paths_siglip(encoder, [tr.img_path],
                                              include_prompt_subtask=include_prompt_subtask,
                                              sub_batch=1)
            else:
                obs_np = None

            img = cv2.imread(tr.img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            wp_for_draw = tr.waypoints if tr.waypoints is not None else (
                tr.action if len(tr.action) == ACTION_FLAT_DIM else None
            )
            if wp_for_draw is not None:
                img = _draw_trajectory(img, wp_for_draw, GT_COLOR, thickness=3, dot_radius=5)

            q_str = ""
            if obs_np is not None:
                if lang_dim > 0:
                    lang_np = np.asarray(_transition_base_action(tr), dtype=np.float32)[None]
                else:
                    lang_np = np.zeros((1, 0), dtype=np.float32)
                q_gt = float(np.array(_eval_q(
                    network, jnp.asarray(obs_np), jnp.asarray(tr.action[None]),
                    jnp.asarray(lang_np),
                ))[0])
                q_str = f"  Q(GT)={q_gt:.2f}"

            caption = f"[{label}] step={step}  reward={tr.mc_return:.2f}{q_str}"
            imgs.append(wandb.Image(img, caption=caption))
        return imgs

    log_dict: dict = {}
    noisy_imgs = _make_images(noisy_val, "noisy")
    non_noisy_imgs = _make_images(non_noisy_val, "non-noisy")
    if noisy_imgs:
        log_dict["val/gt_noisy"] = noisy_imgs
    if non_noisy_imgs:
        log_dict["val/gt_non_noisy"] = non_noisy_imgs
    if log_dict:
        wandb.log(log_dict, step=step)


def log_visualisations(
    network: TrainState,
    val_transitions: list[Transition],
    noisy_action_cache: np.ndarray,           # (C, act_dim)
    non_noisy_action_cache: np.ndarray,       # (C, act_dim)
    encoder: SigLIPEncoder | None,
    include_prompt_subtask: bool,
    img_h: int,
    img_w: int,
    step: int,
    lang_dim: int = 0,
    n_frames: int = 10,
    n_actions: int = 10,
    n_random_actions: int = 3,
    val_embeddings: np.ndarray | None = None,
    noisy_wp_cache: np.ndarray | None = None,      # (C, 40) waypoints for viz, or None
    non_noisy_wp_cache: np.ndarray | None = None,  # (C, 40) waypoints for viz, or None
) -> None:
    frame_idxs = random.sample(range(len(val_transitions)), min(n_frames, len(val_transitions)))
    images = []
    n_each = max(1, (n_actions - 1) // 2)

    # Estimate action scale from the combined cache for random sampling
    act_dim = noisy_action_cache.shape[1]
    action_std = np.std(
        np.concatenate([noisy_action_cache, non_noisy_action_cache], axis=0), axis=0
    ).clip(1e-3)

    for fi in frame_idxs:
        tr = val_transitions[fi]
        noisy_idxs = np.random.choice(len(noisy_action_cache), size=n_each,
                                      replace=len(noisy_action_cache) < n_each)
        non_noisy_idxs = np.random.choice(len(non_noisy_action_cache), size=n_each,
                                           replace=len(non_noisy_action_cache) < n_each)

        # Random actions sampled from N(0, data_std) — same shape as real actions
        rand_actions = (np.random.randn(n_random_actions, act_dim) * action_std).astype(np.float32)

        sampled_actions = np.concatenate([
            tr.action[None],
            noisy_action_cache[noisy_idxs],
            non_noisy_action_cache[non_noisy_idxs],
            rand_actions,
        ], axis=0)

        n_sampled = 1 + n_each + n_each + n_random_actions
        action_labels = (
            ["GT"]
            + [f"N{i+1}" for i in range(n_each)]
            + [f"C{i+1}" for i in range(n_each)]
            + [f"R{i+1}" for i in range(n_random_actions)]
        )

        # Build parallel waypoints array for viz when actions are not already 40-d
        gt_wp = tr.waypoints if tr.waypoints is not None else (
            tr.action if len(tr.action) == ACTION_FLAT_DIM else None
        )
        if gt_wp is not None and noisy_wp_cache is not None and non_noisy_wp_cache is not None:
            # Random actions get random 40-d waypoints from the same distribution
            rand_wps = (np.random.randn(n_random_actions, ACTION_FLAT_DIM)
                        * np.std(noisy_wp_cache, axis=0).clip(1e-3)).astype(np.float32)
            sampled_wps = np.concatenate([
                gt_wp[None],
                noisy_wp_cache[noisy_idxs],
                non_noisy_wp_cache[non_noisy_idxs],
                rand_wps,
            ], axis=0)
        elif gt_wp is not None and act_dim == ACTION_FLAT_DIM:
            # waypoints mode: random actions are already 40-d, use them directly for viz
            sampled_wps = None  # make_overlay_image falls back to actions
        else:
            sampled_wps = None

        if val_embeddings is not None:
            obs_np = val_embeddings[fi:fi + 1]
        elif encoder is not None:
            obs_np = encode_paths_siglip(
                encoder, [tr.img_path],
                include_prompt_subtask=include_prompt_subtask, sub_batch=1,
            )
        else:
            obs_np = np.array(
                Image.open(tr.img_path).convert("RGB").resize((img_w, img_h), Image.BILINEAR),
                dtype=np.uint8,
            )[None]

        if lang_dim > 0:
            lang_np = np.asarray(_transition_base_action(tr), dtype=np.float32)[None]
        else:
            lang_np = np.zeros((1, 0), dtype=np.float32)
        q_vals = np.array(_eval_q(
            network, jnp.asarray(obs_np), jnp.asarray(sampled_actions), jnp.asarray(lang_np),
        ))
        overlay = make_overlay_image(
            tr.img_path, sampled_actions, q_vals,
            waypoints=sampled_wps, labels=action_labels,
        )
        images.append(wandb.Image(
            overlay,
            caption=(f"step={step}  mc_return={tr.mc_return:.2f}  "
                     f"Q_min={q_vals.min():.2f}  Q_max={q_vals.max():.2f}"),
        ))
    wandb.log({"val/q_visualisations": images}, step=step)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(network: TrainState, ckpt_dir: Path, step: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pkl"
    # Only serialize array data — optax closures (tx/apply_fn) are not picklable
    state = {
        "step": int(network.step),
        "params": jax.tree_util.tree_map(np.asarray, network.params),
        "opt_state": jax.tree_util.tree_map(
            lambda x: np.asarray(x) if hasattr(x, "shape") else x,
            network.opt_state,
        ) if network.opt_state is not None else None,
    }
    with open(path, "wb") as fp:
        pickle.dump(state, fp)
    latest = ckpt_dir / "latest.pkl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)


# ── Config ────────────────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace, siglip_embed_dim: int) -> ml_collections.ConfigDict:
    from jax_agents.dsrl import get_config
    cfg = get_config()
    cfg.lr = args.lr
    cfg.observation_mode = "image"
    cfg.image_encoder = args.image_encoder
    cfg.layer_norm = True
    cfg.critic_ensemble = 2
    cfg.critic_hidden_dims = (256, 256)
    cfg.discount = args.discount

    if args.expert_guidance and args.action_mode == "control_2d":
        # Q(s, a_expert, a): expert [accel, steer] prepended to obs via language_label slot.
        # Compatible with --critic-mode expert in run_carla.sh (critic_feedback_mode=expert_action, lang_dim=2).
        cfg.critic_feedback_mode = "expert_action"
        cfg.language_label_dim = CONTROL_2D_DIM
    else:
        # Q(s, a): pure critic, no extra obs. Compatible with --critic-mode none in run_carla.sh.
        cfg.critic_feedback_mode = "none"
        cfg.language_label_dim = 0

    if args.image_encoder == "siglip":
        cfg.siglip_model_id = args.siglip_model_id
        cfg.siglip_embed_dim = siglip_embed_dim
        cfg.siglip_include_prompt_subtask = args.siglip_include_prompt_subtask
    else:
        cfg.image_impala_width = 1
        cfg.image_impala_stack_sizes = (16, 32, 32)
        cfg.image_impala_num_blocks = 2
        cfg.image_mlp_hidden_dims = (512,)

    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Offline DSRL critic pretraining")
    p.add_argument("--noisy_root", default=None)
    p.add_argument("--non_noisy_root", default=None)
    p.add_argument("--noise_sweep_root", default=None,
                   help="Root of noise-sweep dataset (layout: task_*/<route>/measurements+rgb). "
                        "All routes treated as noisy. Can be combined with noisy_root/non_noisy_root.")
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Total batch size across all JAX devices")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--discount", type=float, default=0.99)
    # Image size (steervla-pi default: 224×224)
    p.add_argument("--img_h", type=int, default=DEFAULT_IMG_H)
    p.add_argument("--img_w", type=int, default=DEFAULT_IMG_W)
    # Encoder
    p.add_argument("--image_encoder", default="siglip", choices=["siglip", "impala"])
    p.add_argument("--siglip_model_id", default="google/siglip2-so400m-patch14-384")
    p.add_argument("--siglip_device", default="cuda:0",
                   help="PyTorch device for SigLIP inference")
    p.add_argument("--siglip_include_prompt_subtask", action="store_true",
                   help="Concatenate empty prompt/subtask embeddings (matches online setup)")
    p.add_argument("--siglip_sub_batch", type=int, default=128,
                   help="Sub-batch size per GPU for SigLIP encoding")
    p.add_argument("--embedding_cache", default=None,
                   help="Path to .npz file for caching precomputed SigLIP embeddings. "
                        "Computed on first run, loaded on subsequent runs.")
    # Logging
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--n_log_frames", type=int, default=10)
    p.add_argument("--n_log_actions", type=int, default=10)
    p.add_argument("--n_random_actions", type=int, default=3,
                   help="Number of random actions to include in Q-value visualization")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--cache_size", type=int, default=5000)
    # Checkpointing
    p.add_argument("--checkpoint_dir", default="/scratch/current/celinet/critic_pretrain",
                   help="Directory for periodic checkpoint saves")
    p.add_argument("--checkpoint_every", type=int, default=1000)
    # Misc
    p.add_argument("--action_mode", default="waypoints", choices=["waypoints", "control_2d"],
                   help="waypoints: 40-d flattened ego-frame deltas (default). "
                        "control_2d: stored [accel, steer] — noisy data uses noisy_throttle/steer/brake "
                        "so the applied noise is directly in the action.")
    p.add_argument("--expert_guidance", action="store_true",
                   help="Condition the critic on the clean expert [accel, steer] as a privileged "
                        "base_action input (prepended to the SigLIP embedding before the critic MLP). "
                        "For noisy data the expert action = clean throttle/steer/brake; for non-noisy "
                        "data it equals the executed action.  Only meaningful with --action_mode control_2d. "
                        "Without this flag the base_action slot is filled with zeros (same architecture, "
                        "no guidance), so both checkpoints are compatible with --pretrained_critic in "
                        "run_carla.sh.")
    p.add_argument("--max_routes", type=int, default=None)
    p.add_argument("--max_transitions", type=int, default=None,
                   help="Cap total transitions after loading (for quick smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default="carla_critic_pretrain")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Multi-GPU setup ───────────────────────────────────────────────────
    devices = jax.devices()
    n_devices = len(devices)
    print(f"JAX devices: {devices}")
    if args.batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={args.batch_size} must be divisible by n_devices={n_devices}"
        )
    local_batch = args.batch_size // n_devices
    use_pmap = n_devices > 1
    print(f"Training mode: {'pmap (multi-GPU)' if use_pmap else 'jit (single device)'}, "
          f"local_batch={local_batch}")

    # ── Encoder setup ─────────────────────────────────────────────────────
    siglip_encoder: SigLIPEncoder | None = None
    siglip_embed_dim = 1152
    if args.image_encoder == "siglip":
        print(f"Loading SigLIP ({args.siglip_model_id}) on {args.siglip_device}...")
        siglip_encoder = SigLIPEncoder(
            model_id=args.siglip_model_id, device=args.siglip_device
        )
        siglip_encoder.setup()
        siglip_embed_dim = siglip_encoder.embedding_dim
        if args.siglip_include_prompt_subtask:
            siglip_embed_dim *= 3
        print(f"SigLIP embedding dim: {siglip_embed_dim}")

    # ── W&B ───────────────────────────────────────────────────────────────
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"critic_pretrain_{run_ts}",
        mode=args.wandb_mode,
        config=vars(args),
    )

    # ── Scan + load transitions ───────────────────────────────────────────
    print("Scanning route directories...")
    route_dirs = find_route_dirs(args.noisy_root, args.non_noisy_root, args.noise_sweep_root)
    print(f"Found {len(route_dirs)} routes.")
    random.shuffle(route_dirs)
    if args.max_routes:
        route_dirs = route_dirs[:args.max_routes]

    print("Loading transitions...")
    all_transitions: list[Transition] = []
    for rd, is_noisy_route in tqdm(route_dirs, desc="load_routes"):
        all_transitions.extend(load_route(rd, discount=args.discount,
                                          is_noisy=is_noisy_route, action_mode=args.action_mode))
    if not all_transitions:
        raise RuntimeError("No transitions loaded — check dataset paths.")
    random.shuffle(all_transitions)
    if args.max_transitions:
        all_transitions = all_transitions[:args.max_transitions]
    n_noisy = sum(1 for t in all_transitions if t.is_noisy)
    n_non_noisy = len(all_transitions) - n_noisy
    print(f"Loaded {len(all_transitions)} transitions ({n_noisy} noisy, {n_non_noisy} non-noisy).")
    n_val = max(args.n_log_frames, int(len(all_transitions) * args.val_frac))
    val_transitions = all_transitions[:n_val]
    train_transitions = all_transitions[n_val:]
    print(f"Train: {len(train_transitions)}  Val: {len(val_transitions)}")

    # ── Action caches (one per dataset type for even-split logging) ───────
    cache_n = min(args.cache_size, len(train_transitions) // 2)
    noisy_train = [t for t in train_transitions if t.is_noisy]
    non_noisy_train = [t for t in train_transitions if not t.is_noisy]
    if not noisy_train:
        noisy_train = train_transitions
    if not non_noisy_train:
        non_noisy_train = train_transitions

    def _viz_waypoints(tr: Transition) -> np.ndarray | None:
        """Return 40-d waypoints for visualization; None if unavailable."""
        if tr.waypoints is not None:
            return tr.waypoints
        if len(tr.action) == ACTION_FLAT_DIM:
            return tr.action
        return None

    def _build_cache(pool: list[Transition], n: int) -> tuple[np.ndarray, np.ndarray | None]:
        idxs = np.random.choice(len(pool), size=min(n, len(pool)), replace=False)
        actions = np.stack([pool[i].action for i in idxs])
        wps = [_viz_waypoints(pool[i]) for i in idxs]
        waypoints = np.stack(wps) if all(w is not None for w in wps) else None
        return actions, waypoints

    noisy_action_cache, noisy_wp_cache = _build_cache(noisy_train, cache_n)
    non_noisy_action_cache, non_noisy_wp_cache = _build_cache(non_noisy_train, cache_n)
    print(f"Action caches: {len(noisy_action_cache)} noisy, {len(non_noisy_action_cache)} non-noisy")

    # ── Pre-compute SigLIP embeddings (frozen encoder → cache everything) ─
    train_embeddings: np.ndarray | None = None
    val_embeddings: np.ndarray | None = None
    if args.image_encoder == "siglip":
        all_paths = [t.img_path for t in train_transitions] + [t.img_path for t in val_transitions]

        # Cache stores {path: embedding} so it's valid across any train/val split or seed.
        loaded_from_cache = False
        if args.embedding_cache and Path(args.embedding_cache).exists():
            print(f"Loading SigLIP embeddings from cache: {args.embedding_cache}")
            data = np.load(args.embedding_cache, allow_pickle=True)
            cached_paths = list(data["paths"])
            if (set(cached_paths) == set(all_paths)
                    and str(data["model_id"]) == args.siglip_model_id):
                path_to_idx = {p: i for i, p in enumerate(cached_paths)}
                all_embs = data["embeddings"][[path_to_idx[p] for p in all_paths]]
                train_embeddings = all_embs[:len(train_transitions)]
                val_embeddings = all_embs[len(train_transitions):]
                print(f"Cache hit: {all_embs.shape[0]} embeddings loaded.")
                loaded_from_cache = True
            else:
                print("Cache mismatch (dataset or model changed) — recomputing.")

        if not loaded_from_cache:
            import torch
            n_torch_gpus = torch.cuda.device_count()
            precompute_devices = [f"cuda:{i}" for i in range(n_torch_gpus)] if n_torch_gpus > 1 else [args.siglip_device]
            print(f"Pre-computing SigLIP embeddings across {len(precompute_devices)} GPU(s)...")
            all_embs = encode_all_parallel(
                all_paths, args.siglip_model_id, precompute_devices,
                include_prompt_subtask=args.siglip_include_prompt_subtask,
                sub_batch=args.siglip_sub_batch,
            )
            train_embeddings = all_embs[:len(train_transitions)]
            val_embeddings = all_embs[len(train_transitions):]
            print(f"Embeddings computed: {all_embs.shape}")

            if args.embedding_cache:
                Path(args.embedding_cache).parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    args.embedding_cache,
                    embeddings=all_embs,
                    paths=np.array(all_paths),
                    model_id=args.siglip_model_id,
                )
                print(f"Embeddings saved to cache: {args.embedding_cache}")

        # Free all PyTorch GPU memory before JAX pmap initializes.
        # encode_all_parallel loads one model per GPU; without explicit cleanup those
        # allocations linger and OOM the JAX pmap init on multi-GPU runs.
        import gc, torch
        del siglip_encoder
        siglip_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        print("PyTorch GPU memory freed.")

    # ── Build agent ───────────────────────────────────────────────────────
    act_dim = CONTROL_2D_DIM if args.action_mode == "control_2d" else ACTION_FLAT_DIM
    config = build_config(args, siglip_embed_dim)
    lang_dim = int(config.get("language_label_dim", 0))
    if args.image_encoder == "siglip":
        ex_obs = jnp.zeros((1, siglip_embed_dim), dtype=jnp.float32)
    else:
        ex_obs = jnp.zeros((1, args.img_h, args.img_w, 3), dtype=jnp.uint8)
    ex_actions = jnp.zeros((1, act_dim), dtype=jnp.float32)
    agent = DSRLAgent.create(
        seed=args.seed, ex_observations=ex_obs, ex_actions=ex_actions, config=config,
    )
    network = agent.network

    # ── Checkpoint directory ──────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir) / f"run_{run_ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")
    print(f"lang_dim={lang_dim} ({'expert guidance' if lang_dim > 0 else 'no guidance'})")

    # ── pmap replication ──────────────────────────────────────────────────
    if use_pmap:
        network_rep = jax.device_put_replicated(network, devices)
    else:
        network_rep = None

    # ── JIT warm-up ───────────────────────────────────────────────────────
    print("Warming up JIT...")
    if args.image_encoder == "siglip":
        dummy_obs = np.zeros((args.batch_size, siglip_embed_dim), dtype=np.float32)
    else:
        dummy_obs = np.zeros((args.batch_size, args.img_h, args.img_w, 3), dtype=np.uint8)
    dummy_acts = np.zeros((args.batch_size, act_dim), dtype=np.float32)
    dummy_rets = np.zeros(args.batch_size, dtype=np.float32)
    dummy_lang = np.zeros((args.batch_size, lang_dim), dtype=np.float32)

    if use_pmap:
        obs_sh = dummy_obs.reshape(n_devices, local_batch, *dummy_obs.shape[1:])
        acts_sh = dummy_acts.reshape(n_devices, local_batch, -1)
        rets_sh = dummy_rets.reshape(n_devices, local_batch)
        lang_sh = dummy_lang.reshape(n_devices, local_batch, -1)
        network_rep, _ = _critic_step_pmap(
            network_rep, jnp.asarray(obs_sh), jnp.asarray(acts_sh), jnp.asarray(rets_sh),
            jnp.asarray(lang_sh),
        )
        network = jax.tree_util.tree_map(lambda x: x[0], network_rep)
    else:
        network, _ = _critic_step(
            network, jnp.asarray(dummy_obs), jnp.asarray(dummy_acts), jnp.asarray(dummy_rets),
            jnp.asarray(dummy_lang),
        )
    print("JIT warm-up done.")

    # ── Initial GT action sanity check ───────────────────────────────────
    print("Logging GT action examples (step 0 sanity check)...")
    log_gt_action_examples(
        network, val_transitions, val_embeddings, siglip_encoder,
        args.siglip_include_prompt_subtask, step=0,
        lang_dim=lang_dim, n_frames=args.n_log_frames,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print("Starting pretraining...")
    pbar = tqdm(total=args.total_steps, desc="pretrain_critic")

    for step in range(1, args.total_steps + 1):
        # Sample batch
        if args.image_encoder == "siglip":
            obs_b, acts_b, rets_b, lang_b = sample_batch_siglip(
                train_transitions, args.batch_size, siglip_encoder,
                args.siglip_include_prompt_subtask, args.siglip_sub_batch,
                embeddings=train_embeddings,
                expert_guidance=args.expert_guidance,
                lang_dim=lang_dim,
            )
        else:
            obs_b, acts_b, rets_b, lang_b = sample_batch_impala(
                train_transitions, args.batch_size, args.img_h, args.img_w,
                expert_guidance=args.expert_guidance,
                lang_dim=lang_dim,
            )

        # Gradient step
        if use_pmap:
            obs_sh = obs_b.reshape(n_devices, local_batch, *obs_b.shape[1:])
            acts_sh = acts_b.reshape(n_devices, local_batch, -1)
            rets_sh = rets_b.reshape(n_devices, local_batch)
            lang_sh = lang_b.reshape(n_devices, local_batch, -1)
            network_rep, info_rep = _critic_step_pmap(
                network_rep,
                jnp.asarray(obs_sh), jnp.asarray(acts_sh), jnp.asarray(rets_sh),
                jnp.asarray(lang_sh),
            )
            network = jax.tree_util.tree_map(lambda x: x[0], network_rep)
            info = jax.tree_util.tree_map(lambda x: float(x[0]), info_rep)
        else:
            network, info_jax = _critic_step(
                network, jnp.asarray(obs_b), jnp.asarray(acts_b), jnp.asarray(rets_b),
                jnp.asarray(lang_b),
            )
            info = {k: float(v) for k, v in info_jax.items()}
            if use_pmap:
                network_rep = jax.device_put_replicated(network, devices)

        pbar.update(1)

        if step % 100 == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=step)

        if step % args.checkpoint_every == 0:
            save_checkpoint(network, ckpt_dir, step)
            # Re-replicate after unreplicate
            if use_pmap:
                network_rep = jax.device_put_replicated(network, devices)

        if step % args.log_every == 0:
            log_visualisations(
                network, val_transitions,
                noisy_action_cache, non_noisy_action_cache,
                siglip_encoder, args.siglip_include_prompt_subtask,
                args.img_h, args.img_w,
                step=step, lang_dim=lang_dim,
                n_frames=args.n_log_frames, n_actions=args.n_log_actions,
                n_random_actions=args.n_random_actions,
                val_embeddings=val_embeddings,
                noisy_wp_cache=noisy_wp_cache, non_noisy_wp_cache=non_noisy_wp_cache,
            )
            log_gt_action_examples(
                network, val_transitions, val_embeddings, siglip_encoder,
                args.siglip_include_prompt_subtask, step=step,
                lang_dim=lang_dim, n_frames=args.n_log_frames,
            )

    pbar.close()
    save_checkpoint(network, ckpt_dir, args.total_steps)
    print(f"Final checkpoint saved → {ckpt_dir}")
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
