"""Frozen SimLingo base policy wrapper for residual SAC.

Loads the SimLingo DrivingModel from a DeepSpeed checkpoint, runs inference
to produce a base action (accel, steer) via PID, and simultaneously exposes
the VLM driving features (mean-pooled last-layer hidden states) that the
residual SAC actor/critic use as their observation.

The same 1024×512 front-camera image is passed to both the VLM backbone
(to produce features for the residual) and the waypoint heads (to produce
the base action). No output normalization is applied anywhere.

Usage::

    base = SimLingoBase("/path/to/epoch=013.ckpt", device="cuda")
    base_action, vlm_features = base.get_action_and_features(
        simlingo_image=obs["simlingo_image"],  # (H, W, 3) uint8
        ego_state=obs["state"],                # (25,) float32
        target_point=obs["target_point"],      # (2,) float32  (unused in command mode)
        routing_command=obs["routing_command"], # str
    )
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


# ── Exact copies of PID controllers from the SimLingo repo ───────────────────
# Embedded here to avoid importing team_code.{transfuser_utils,nav_planner}
# which have top-level `import carla` that fails outside the leaderboard env.
# Values and logic are identical to agent_simlingo.py.

class _SpeedPIDController:
    """Exact copy of team_code.transfuser_utils.PIDController.
    agent_simlingo.py: self.speed_controller = t_u.PIDController(k_p=..., k_i=..., k_d=...)
    """
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self.window.append(error)
        if len(self.window) >= 2:
            integral = np.mean(self.window)
            derivative = self.window[-1] - self.window[-2]
        else:
            integral = 0.0
            derivative = 0.0
        return self.k_p * error + self.k_i * integral + self.k_d * derivative


class _LateralPIDController:
    """Exact copy of team_code.nav_planner.LateralPIDController.
    agent_simlingo.py: self.turn_controller = LateralPIDController(inference_mode=False)
    """
    def __init__(
        self,
        k_p=3.118357247806046,
        k_d=1.3782508892109167,
        k_i=0.6406067986034124,
        speed_scale=0.9755321901954155,
        speed_offset=1.9152884533402488,
        default_lookahead=24,
        speed_threshold=23.150102938235136,
        n=6,
        inference_mode=False,
    ):
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        self.speed_scale = speed_scale
        self.speed_offset = speed_offset
        self.default_lookahead = default_lookahead
        self.speed_threshold = speed_threshold
        self.n = n
        self.inference_mode = inference_mode
        self._saved_window = []
        self._window = []

    def step(self, route_np, current_speed):
        current_speed = current_speed * 3.6  # m/s → km/h
        if self.inference_mode:
            n_lookahead = np.clip(
                self.speed_scale * current_speed + self.speed_offset, 24, 105
            ) / 10
            n_lookahead = n_lookahead - 2
            n_lookahead = int(min(n_lookahead, route_np.shape[0] - 1))
        else:
            n_lookahead = int(
                min(
                    np.clip(
                        self.speed_scale * current_speed + self.speed_offset, 24, 105
                    ),
                    route_np.shape[0] - 1,
                )
            )
        n_lookahead = min(n_lookahead, len(route_np) - 1)
        desired_heading_vec = route_np[n_lookahead]
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        heading_error = heading_error * 180.0 / np.pi / 90.0
        self._window.append(heading_error)
        self._window = self._window[-self.n:]
        derivative = 0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        integral = np.mean(self._window)
        steering = np.clip(
            self.k_p * heading_error + self.k_d * derivative + self.k_i * integral,
            -1.0, 1.0,
        ).item()
        return steering

    def save_state(self):
        self._saved_window = self._window.copy()

    def load_state(self):
        self._window = self._saved_window.copy()


# Speed PID parameters from GlobalConfig (config_simlingo.py)
_SPEED_KP = 1.75
_SPEED_KI = 1.0
_SPEED_KD = 2.0
_SPEED_N = 20

# ── Repo paths ────────────────────────────────────────────────────────────────
_REBUTTAL_ROOT = Path(__file__).resolve().parents[2] / "simlingo-rebuttal"
for _p in [str(_REBUTTAL_ROOT), str(_REBUTTAL_ROOT / "team_code")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── SimLingo model config ────────────────────────────────────────────────────
# DrivingAdaptor creates: 20 route-query tokens + 10 speed-wps tokens = 30 total
_DRIVING_TOKEN_LEN = 30
# InternVL2-1B Qwen2 backbone hidden size
_VLM_FEATURE_DIM = 896
VLM_FEATURE_DIM = _VLM_FEATURE_DIM  # public export
# EGO_STATE_IDX_SPEED is index 15 in the 25-dim state vector
_EGO_STATE_IDX_SPEED = 15


def _build_model_from_hydra_config(hydra_cfg_path: Path, cache_dir: str):
    """Instantiate DrivingModel using the training Hydra config."""
    import hydra
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    cfg = OmegaConf.load(str(hydra_cfg_path))
    model_cfg = cfg.model
    data_cfg = cfg.data_module

    variant = model_cfg.vision_model.variant  # e.g. "OpenGVLab/InternVL2-1B"
    processor = AutoProcessor.from_pretrained(
        variant,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    model = hydra.utils.instantiate(
        model_cfg,
        cfg_data_module=data_cfg,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False,
    )
    return model




class SimLingoBase:
    """Frozen SimLingo inference engine exposing base_action and VLM features."""

    # PID parameters from GlobalConfig
    _BRAKE_SPEED = 0.4
    _BRAKE_RATIO = 1.1
    _CLIP_DELTA = 1.0
    _CLIP_THROTTLE = 1.0
    _CARLA_FPS = 20
    _WP_DILATION = 1
    _DATA_SAVE_FREQ = 5

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        ckpt_dir = Path(checkpoint_path)

        if cache_dir is None:
            # Use simlingo's pretrained dir if it exists, else HF default cache
            candidate = Path("/home/celinet/simlingo/pretrained")
            cache_dir = str(candidate) if candidate.exists() else None

        # Search for Hydra config by walking up from the checkpoint dir
        hydra_cfg_path = None
        for ancestor in [ckpt_dir.parent, ckpt_dir.parent.parent, ckpt_dir.parent.parent.parent]:
            candidate = ancestor / ".hydra" / "config.yaml"
            if candidate.exists():
                hydra_cfg_path = candidate
                break
        if hydra_cfg_path is None:
            raise FileNotFoundError(
                f"Hydra config.yaml not found in any ancestor of {ckpt_dir}"
            )

        print(f"[SimLingoBase] Loading model from {hydra_cfg_path} ...", flush=True)
        model = _build_model_from_hydra_config(hydra_cfg_path, cache_dir)

        # Load consolidated checkpoint weights
        pt_path = ckpt_dir / "pytorch_model.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"pytorch_model.pt not found at {pt_path}")
        print(f"[SimLingoBase] Loading weights from {pt_path} ...", flush=True)
        state_dict = torch.load(str(pt_path), map_location="cpu")
        # pytorch_model.pt from zero_to_fp32 may have keys prefixed with 'module.'
        if all(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[SimLingoBase] Missing keys ({len(missing)}): {missing[:5]}...", flush=True)
        if unexpected:
            print(f"[SimLingoBase] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...", flush=True)

        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        # Match training dtype (bfloat16)
        model = model.to(torch.bfloat16)
        self.model = model

        # Register the same custom special tokens that training used.
        # This ensures smallest_added_id in replace_placeholder_tokens points past
        # <IMG_CONTEXT> so image context tokens are never treated as placeholders.
        _CUSTOM_SPECIAL_TOKENS = [
            '<WAYPOINTS>', '<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>',
            '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>',
            '<TARGET_POINT>',
        ]
        model.tokenizer.add_special_tokens(
            {'additional_special_tokens': _CUSTOM_SPECIAL_TOKENS}
        )

        # Load InternVL2 conversation module for prompt construction
        variant = OmegaConf.load(str(hydra_cfg_path)).model.vision_model.variant
        self._variant = variant
        self._conv_module = self._load_conv_module(variant, cache_dir)
        self._eos_token_id = self._get_eos_token_id(variant, model)

        # Image preprocessing
        from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
        from transformers import AutoConfig
        self._dynamic_preprocess = dynamic_preprocess
        self._build_transform = build_transform
        tmp_cfg = AutoConfig.from_pretrained(variant, trust_remote_code=True)
        image_size = tmp_cfg.force_image_size or tmp_cfg.vision_config.image_size
        patch_size = tmp_cfg.vision_config.patch_size
        self._num_image_token = int((image_size // patch_size) ** 2 * (tmp_cfg.downsample_ratio ** 2))
        self._img_transform = build_transform(input_size=image_size)

        # Speed PID: exact copy of t_u.PIDController (see _SpeedPIDController above)
        # agent_simlingo.py: self.speed_controller = t_u.PIDController(k_p=1.75, k_i=1.0, k_d=2.0)
        self._speed_controller = _SpeedPIDController(
            k_p=_SPEED_KP, k_i=_SPEED_KI, k_d=_SPEED_KD, n=_SPEED_N
        )
        # Lateral PID: exact copy of nav_planner.LateralPIDController (see _LateralPIDController above)
        # agent_simlingo.py: self.turn_controller = LateralPIDController(inference_mode=False)
        self._turn_controller = _LateralPIDController(inference_mode=False)
        self._last_route_interp: Optional[np.ndarray] = None  # stored for per-tick steer calls

        # Feature capture hook
        self._driving_features: Optional[torch.Tensor] = None
        self._register_feature_hook()

        # Cached last-inference outputs for video overlay
        self._last_speed_wps: Optional[np.ndarray] = None   # (10, 2) ego-frame
        self._last_route: Optional[np.ndarray] = None        # (20, 2) ego-frame

        print("[SimLingoBase] Ready.", flush=True)

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _load_conv_module(self, variant: str, cache_dir: Optional[str]):
        """Load the InternVL2 conversation.py module used for prompt construction."""
        model_name = variant.split("/")[1]
        # Check candidate paths
        candidates = [
            Path(cache_dir or "") / model_name / "conversation.py",
            Path("/home/celinet/simlingo/pretrained") / model_name / "conversation.py",
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--OpenGVLab--{model_name}" /
            "snapshots",
        ]
        conv_path = None
        for c in candidates:
            if c.is_file():
                conv_path = c
                break
            if c.is_dir():
                # Find the snapshot subdir
                snaps = sorted(c.iterdir())
                for snap in snaps:
                    cand = snap / "conversation.py"
                    if cand.exists():
                        conv_path = cand
                        break
                if conv_path:
                    break

        if conv_path is None:
            raise FileNotFoundError(
                f"Could not find conversation.py for {variant}. "
                "Run a snapshot download or set cache_dir."
            )
        spec = importlib.util.spec_from_file_location("conv_module", str(conv_path))
        conv_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conv_module)
        return conv_module

    def _get_eos_token_id(self, variant: str, model) -> int:
        tokenizer = model.tokenizer
        if "InternVL2-4B" in variant:
            return tokenizer.added_tokens_encoder.get("<|end|>", tokenizer.eos_token_id)
        elif "InternVL2-2B" in variant:
            return tokenizer.added_tokens_encoder.get("<|im_end|>", tokenizer.eos_token_id)
        else:
            return tokenizer.eos_token_id

    def _register_feature_hook(self) -> None:
        """Hook the inner transformer to capture last-layer hidden states.

        Architecture: LLM → PeftModel → LoraModel → Qwen2ForCausalLM → Qwen2Model
        We hook Qwen2Model which returns BaseModelOutputWithPast with last_hidden_state.
        """
        parent = self

        def _hook(module, input, output):
            # output is BaseModelOutputWithPast; last_hidden_state is (batch, seq, hidden)
            hs = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            parent._last_lm_features = hs.detach()

        self._last_lm_features: Optional[torch.Tensor] = None
        # Walk: language_model.model → PeftModel/LoraModel → CausalLM → inner Qwen2Model
        inner = self.model.language_model.model  # PeftModel
        if hasattr(inner, "base_model"):
            inner = inner.base_model.model       # Qwen2ForCausalLM (via LoraModel.model)
        if hasattr(inner, "model"):
            inner = inner.model                  # Qwen2Model
        inner.register_forward_hook(_hook)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _preprocess_image(self, image_hwc: np.ndarray) -> torch.Tensor:
        """Convert (H, W, 3) uint8 numpy to the tiled pixel_values tensor.

        Matches training: cut bottom 4.8/16 of the image (removes bonnet),
        then tile with max_num=2, use_thumbnail=False (use_global_img=False).
        """
        from PIL import Image as _PIL_Image
        # Crop bottom 4.8/16 to match training preprocessing
        h = image_hwc.shape[0]
        crop_h = int(h - (h * 4.8) // 16)
        image_hwc = image_hwc[:crop_h, :, :]
        pil_img = _PIL_Image.fromarray(image_hwc)
        patches = self._dynamic_preprocess(
            pil_img,
            min_num=1,
            max_num=2,
            image_size=448,
            use_thumbnail=False,
        )
        pixel_values = torch.stack([self._img_transform(p) for p in patches])  # (num_patches, 3, H, W)
        return pixel_values.unsqueeze(0)  # (1, num_patches, 3, H, W)

    def _build_language_label(
        self,
        speed_ms: float,
        target_points: np.ndarray,
    ):
        """Build the LanguageLabel for target_point_command mode (matches checkpoint training).

        target_points: (2, 2) float32 ego-frame waypoints [current_tp, next_tp].
        Prompt: "Current speed: X m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"
        """
        from simlingo_training.utils.custom_types import LanguageLabel

        prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        prompt = f"Current speed: {round(speed_ms, 1)} m/s. {prompt_tp} What should the ego do next?"

        IMG_START = "<img>"
        IMG_END = "</img>"
        IMG_CONTEXT = "<IMG_CONTEXT>"
        # max_num=2, use_thumbnail=False → always 2 patches per image
        _NUM_PATCHES_IN_PROMPT = 2
        image_tokens = IMG_START + IMG_CONTEXT * self._num_image_token * _NUM_PATCHES_IN_PROMPT + IMG_END

        template = self._conv_module.get_conv_template("internlm2-chat")
        template.append_message(template.roles[0], f"<image>\n{prompt}")
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        # Remove system prompt
        system_prompt = template.system_template.replace("{system_message}", template.system_message) + template.sep
        query = query.replace(system_prompt, "")
        query = query.replace("<image>", image_tokens, 1)

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(
            [query],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        phrase_ids = tokenized["input_ids"]
        phrase_valid = phrase_ids != tokenizer.pad_token_id
        phrase_mask = phrase_valid

        # Build placeholder_values: maps <TARGET_POINT> token id → (2, 2) coords array
        # Matches agent_simlingo.py lines 479-484
        token_id = tokenizer.convert_tokens_to_ids('<TARGET_POINT>')
        placeholder_values = [{token_id: target_points}]

        return LanguageLabel(
            phrase_ids=phrase_ids.to(self.device),
            phrase_valid=phrase_valid.to(self.device),
            phrase_mask=phrase_mask.to(self.device),
            placeholder_values=placeholder_values,
            language_string=[query],
            loss_masking=None,
        )

    def _compute_desired_speeds(self, speed_wps_np: np.ndarray) -> np.ndarray:
        """Extract the desired speed for each of the 10 predicted waypoints.

        Returns (10,) float32 array of m/s targets, one per waypoint interval.
        Uses the same distance-over-0.5s formula as agent_simlingo.control_pid().
        Last two entries copy k=7 since boundary waypoints span only 0.25s.
        """
        desired_speeds = np.zeros(10, dtype=np.float32)
        for k in range(8):
            desired_speeds[k] = np.linalg.norm(speed_wps_np[k] - speed_wps_np[k + 2]) * 2.0
        desired_speeds[8] = desired_speeds[7]
        desired_speeds[9] = desired_speeds[7]
        return desired_speeds

    def accel_for_desired_speed(self, desired_speed: float, current_speed: float) -> float:
        """PID throttle/brake for one CARLA tick.  Call once per tick, not per chunk.

        Returns accel in [-1, 1]: positive = throttle, -1 = full brake.
        """
        brake = (desired_speed < self._BRAKE_SPEED) or (
            current_speed / max(desired_speed, 1e-6) > self._BRAKE_RATIO
        )
        delta = float(np.clip(desired_speed - current_speed, 0.0, self._CLIP_DELTA))
        throttle = float(np.clip(self._speed_controller.step(delta), 0.0, self._CLIP_THROTTLE))
        return throttle if not brake else -1.0

    def _lateral_control(self, route_np: np.ndarray, speed: float) -> float:
        """Compute steer and cache the interpolated route for per-tick reuse.

        Exactly mirrors agent_simlingo.control_pid lines 824-829:
            route_interp = self.interpolate_waypoints(route_waypoints.squeeze())
            steer = self.turn_controller.step(route_interp, speed)
            steer = np.clip(steer, -1.0, 1.0)
            steer = round(steer, 3)
        """
        route_interp = _interpolate_waypoints(route_np)
        self._last_route_interp = route_interp  # store for steer_for_speed() calls
        steer = self._turn_controller.step(route_interp, speed)
        steer = float(np.clip(steer, -1.0, 1.0))
        return round(steer, 3)

    def steer_for_speed(self, current_speed: float) -> float:
        """Run the lateral PID for one tick using the route from the last VLM call.

        Call this every CARLA tick within a chunk to let the PID integrate
        error history properly (rather than holding a constant steer).  Uses
        speed-dependent lookahead but the same interpolated route as the VLM call.
        Matches the per-tick turn_controller.step() call in agent_simlingo.py.
        """
        if self._last_route_interp is None:
            return 0.0
        steer = self._turn_controller.step(self._last_route_interp, current_speed)
        return round(float(np.clip(steer, -1.0, 1.0)), 3)

    def reset_pid(self) -> None:
        """Call at the start of each new episode to clear PID integrator history."""
        # Recreate speed controller (no reset() method — same as agent recreation between routes)
        self._speed_controller = _SpeedPIDController(
            k_p=_SPEED_KP, k_i=_SPEED_KI, k_d=_SPEED_KD, n=_SPEED_N
        )
        # Clear lateral PID error history
        self._turn_controller._window = []
        self._turn_controller._saved_window = []
        self._last_route_interp = None

    @torch.no_grad()
    def get_chunk_and_features(
        self,
        simlingo_image: np.ndarray,
        ego_state: np.ndarray,
        target_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """VLM inference → (desired_speeds (10,), route_interp (N,2), vlm_features (896,)).

        target_points: (2, 2) float32 ego-frame waypoints [[tp_x, tp_y], [next_tp_x, next_tp_y]].
        desired_speeds[k] is the target speed (m/s) for waypoint k of the 10-step
        predicted trajectory.

        route_interp is the 0.1m-spaced interpolated predicted route in the current
        ego frame — pass it to steer_for_speed(current_speed) every CARLA tick to let
        the lateral PID integrate feedback across the chunk (speed-dependent lookahead
        is recomputed each tick even though the route is from this VLM call).
        """
        from simlingo_training.utils.custom_types import DrivingInput
        try:
            from team_code.simlingo_utils import get_camera_intrinsics, get_camera_extrinsics  # type: ignore
        except ImportError:
            get_camera_intrinsics = get_camera_extrinsics = None

        speed_ms = float(ego_state[_EGO_STATE_IDX_SPEED])
        # target_points: (2, 2) ego-frame [current, next] target waypoints
        target_points = np.asarray(target_points, dtype=np.float32).reshape(2, 2)

        # ── Image preprocessing ───────────────────────────────────────────────
        pixel_values = self._preprocess_image(simlingo_image)  # (1, num_patches, 3, H, W)
        num_patches = pixel_values.shape[1]
        C, H, W = pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
        # DrivingInput expects (B=1, T=1, num_patches, C, H, W)
        camera_images = pixel_values.view(1, 1, num_patches, C, H, W)
        camera_images = camera_images.to(self.device).to(torch.bfloat16)

        image_sizes = torch.tensor(
            [[simlingo_image.shape[0], simlingo_image.shape[1]]], dtype=torch.long
        )

        # ── Camera intrinsics / extrinsics ────────────────────────────────────
        try:
            cam_intrinsics = get_camera_intrinsics(
                simlingo_image.shape[1], simlingo_image.shape[0], 110
            ).unsqueeze(0).float().to(self.device)
            cam_extrinsics = get_camera_extrinsics().unsqueeze(0).float().to(self.device)
        except Exception:
            cam_intrinsics = torch.eye(3, device=self.device).unsqueeze(0).float()
            cam_extrinsics = torch.eye(4, device=self.device).unsqueeze(0).float()

        # ── Language label (target_point_command format) ──────────────────────
        lang_label = self._build_language_label(speed_ms, target_points)

        # ── Target point tensor (first waypoint, shape (1, 2)) ────────────────
        tp_tensor = torch.from_numpy(target_points[0]).unsqueeze(0).float().to(self.device)

        # ── Assemble DrivingInput ─────────────────────────────────────────────
        driving_input = DrivingInput(
            camera_images=camera_images,
            image_sizes=image_sizes,
            camera_intrinsics=cam_intrinsics,
            camera_extrinsics=cam_extrinsics,
            vehicle_speed=torch.tensor([[speed_ms]], dtype=torch.float32, device=self.device),
            target_point=tp_tensor,
            prompt=lang_label,
            prompt_inference=lang_label,
        )

        # ── Forward pass ──────────────────────────────────────────────────────
        self._last_lm_features = None
        speed_wps, route, language = self.model(driving_input)

        # ── Extract VLM driving features ──────────────────────────────────────
        if self._last_lm_features is not None:
            # The last _DRIVING_TOKEN_LEN tokens correspond to driving queries
            features = self._last_lm_features  # (1, seq_len, hidden_size)
            driving_feats = features[:, -_DRIVING_TOKEN_LEN:, :]  # (1, 30, 896)
            vlm_features = driving_feats.mean(dim=1).squeeze(0).float().cpu().numpy()  # (896,)
        else:
            vlm_features = np.zeros(_VLM_FEATURE_DIM, dtype=np.float32)

        # ── Waypoints → chunk ─────────────────────────────────────────────────
        if speed_wps is not None and route is not None:
            speed_wps_np = speed_wps[0].float().cpu().numpy()   # (10, 2)
            route_np = route[0].float().cpu().numpy()            # (20, 2)
            self._last_speed_wps = speed_wps_np
            self._last_route = route_np
            desired_speeds = self._compute_desired_speeds(speed_wps_np)
            # _lateral_control stores route_interp in self._last_route_interp
            self._lateral_control(route_np, speed_ms)
        else:
            desired_speeds = np.zeros(10, dtype=np.float32)
            self._last_route_interp = np.zeros((1, 2), dtype=np.float32)

        route_interp = self._last_route_interp

        return desired_speeds, route_interp, vlm_features

    @torch.no_grad()
    def get_action_and_features(
        self,
        simlingo_image: np.ndarray,
        ego_state: np.ndarray,
        target_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-step wrapper: VLM inference → (base_action (2,), vlm_features (896,)).

        target_points: (2, 2) float32 ego-frame waypoints.
        Calls get_chunk_and_features() then computes one action for the current tick.
        For chunk execution use get_chunk_and_features() + accel_for_desired_speed()
        + steer_for_speed() in the outer loop.
        """
        desired_speeds, _route_interp, vlm_features = self.get_chunk_and_features(
            simlingo_image, ego_state, target_points
        )
        current_speed = float(ego_state[_EGO_STATE_IDX_SPEED])
        accel = self.accel_for_desired_speed(desired_speeds[0], current_speed)
        steer = self.steer_for_speed(current_speed)
        base_action = np.array([accel, steer], dtype=np.float32)
        return base_action, vlm_features



def _interpolate_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Evenly interpolate waypoints to 0.1 m spacing (mirrors SimLingo's interpolate_waypoints)."""
    from scipy.interpolate import PchipInterpolator
    wps = np.concatenate([np.zeros((1, waypoints.shape[1])), waypoints.copy()], axis=0)
    shift = np.roll(wps, 1, axis=0)
    shift[0] = shift[1]
    dists = np.linalg.norm(wps - shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(len(dists)) * 1e-4  # strictly increasing
    interp = PchipInterpolator(dists, wps, axis=0)
    x = np.arange(0.1, dists[-1], 0.1)
    pts = interp(x)
    return pts if pts.shape[0] > 0 else wps[None, -1]
