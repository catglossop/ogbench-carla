"""SimLingo SteerVLA: the hierarchical SimLingo policy as a drop-in for :class:`vlas.steervla.SteerVLAActor`.

Two torch SimLingo ``DrivingModel`` checkpoints from the simlingo-steervla repo run in this process:

* **HL planner** (``model_type: hl``) -- image + ego-state history + routing command -> text
  ``"<reasoning>\\n\\nDriving Behavior: <meta action>"``. The meta action is the subtask.
* **LL policy** (``model_type: ll``) -- image + ``"Command: <meta action>"`` -> speed waypoints and route.

The LL output is re-expressed as the OpenPI chunk layout the CARLA env already decodes
(``DELTA_XY_T_DELTA_XY_SPACE``, RLDS units), so the rest of ``main_carla`` -- DSRL rollout, CAST
relabel, video overlays, checkpoint cadence -- is unchanged. Select it with ``steervla.vla =
"simlingo_steervla"``; see ``configs/simlingo_steervla_cast_relabel_train_config.py``.

Online HL training mirrors ``SteerVLAActor.update_hl``: the same pool sampling (``HLPoolSamplingMixin``)
over the cast_relabel HL dataset plus offline replay pools (``vlas/extract_simlingo_hl_replay.py``),
the same throttle / checkpoint contract, and SimLingo's own tokenization and language loss. The LL is
frozen. Online samples supervise only the text after ``Driving Behavior:`` (the rollout reasoning is
kept as context); replay samples supervise the full answer, as in SimLingo training.

Needs the ``simlingo`` extra (peft, hydra-core, pytorch-lightning, timm) and the simlingo-steervla
checkout on disk (``steervla.simlingo_source_root``).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, MutableMapping

import numpy as np
import torch

from vlas.hl_pool_sampling import HLPoolSamplingMixin
from vlas.steervla import SteerVLAActor

DRIVING_BEHAVIOR_MARKER = "Driving Behavior:"
# Registered at SimLingo training time; must be added before any tokenization so the
# placeholder-token id range matches the checkpoint.
SIMLINGO_SPECIAL_TOKENS = [
    "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>",
]
# ``ogbench.carla.carla_utils`` ego-state vector indices.
_EGO_IDX_SPEED = 15
_EGO_IDX_YAW_DEG = 5
# ``DELTA_XY_T_DELTA_XY_SPACE`` RLDS scaling of the speed-waypoint columns; the env multiplies by it
# (``steervla_simlingo_control.denormalize_actions``). Route columns are meters, unscaled.
_SPEED_XY_SCALE = 7.0
_CARLA_FPS = 20
# SimLingo HL ego history: ``ego_state_history_count`` entries 2 s apart (0.5 Hz), oldest first.
_EGO_HISTORY_INTERVAL_TICKS = int(_CARLA_FPS / 0.5)
_TRAINING_HL_FALLBACK = "Continue driving safely."


# --------------------------------------------------------------------------------------------- #
# Checkpoint + model loading                                                                     #
# --------------------------------------------------------------------------------------------- #


def resolve_simlingo_weights(path: str | Path) -> Path:
    """Weights file for a SimLingo run dir, ``epoch=XXX.ckpt`` dir, exported step dir, or file."""
    p = Path(path).expanduser()
    if p.is_file():
        return p
    if (p / "checkpoints").is_dir():
        epochs = sorted(p.glob("checkpoints/epoch=*.ckpt"), key=lambda d: int(d.name.split("=")[1].split(".")[0]))
        if epochs:
            p = epochs[-1]
    for candidate in (
        p / "converted" / "pytorch_model.bin",
        p / "pytorch_model.bin",
        p / "pytorch_model.pt",
        p / "checkpoint" / "mp_rank_00_model_states.pt",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No SimLingo weights under {p} (converted/pytorch_model.bin or DeepSpeed model states).")


def find_simlingo_hydra_config(weights: Path) -> Path:
    for parent in weights.parents:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .hydra/config.yaml above {weights}")


def _load_state_dict(weights: Path) -> dict[str, torch.Tensor]:
    # Own training checkpoints; DeepSpeed model-state pickles need weights_only=False on torch>=2.6.
    state = torch.load(str(weights), map_location="cpu", weights_only=False)
    if "module" in state and isinstance(state["module"], dict):
        state = state["module"]
    state = state.get("state_dict", state)
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def _import_simlingo(source_root: Path) -> None:
    root = str(source_root)
    if root not in sys.path:
        sys.path.insert(0, root)


class _SimLingoModel:
    """One loaded SimLingo checkpoint plus the prompt / image plumbing its training used."""

    def __init__(self, checkpoint: str | Path, *, source_root: Path, device: torch.device, expect_type: str):
        import hydra
        from omegaconf import OmegaConf
        from transformers import AutoConfig, AutoProcessor

        from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess

        self.weights = resolve_simlingo_weights(checkpoint)
        self.config_path = find_simlingo_hydra_config(self.weights)
        cfg = OmegaConf.load(str(self.config_path))
        model_type = str(cfg.model.get("model_type", "ll"))
        if model_type != expect_type:
            raise ValueError(f"{self.weights} is a model_type={model_type!r} checkpoint, expected {expect_type!r}")
        self.model_type = model_type
        # The source DrivingModel constructs a waypoint adaptor even for an HL-only checkpoint.
        # Give that unused adaptor a valid shape; ``generate`` bypasses it for the HL path below.
        if model_type == "hl" and not cfg.model.predict_route_as_wps and not cfg.model.speed_wps_mode:
            cfg.model.speed_wps_mode = "2d"
            cfg.model.predict_route_as_wps = True
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        self.cfg = cfg
        self.dataset_cfg = cfg.data_module.base_dataset
        self.variant = str(cfg.model.vision_model.variant)
        self.device = device
        self.cache_root = source_root / "pretrained"

        processor = AutoProcessor.from_pretrained(self.variant, trust_remote_code=True)
        self.tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
        self.tokenizer.add_special_tokens({"additional_special_tokens": SIMLINGO_SPECIAL_TOKENS})
        self.tokenizer.padding_side = "left"

        # Same construction as simlingo's agent_steervla.py: instantiate under bf16 so new modules
        # (adaptors, waypoint encoder) match the checkpoint dtype, then load the trained weights.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=str(self.cache_root / self.variant.split("/")[1]),
                _recursive_=False,
            )
        finally:
            torch.set_default_dtype(default_dtype)
        missing, unexpected = model.load_state_dict(_load_state_dict(self.weights), strict=False)
        if missing or unexpected:
            print(
                f"[simlingo_steervla] {expect_type.upper()} load: {len(missing)} missing / {len(unexpected)} "
                f"unexpected keys (first: {missing[:3]} {unexpected[:3]})",
                flush=True,
            )
        # Captured before anything is frozen: exactly the parameter set SimLingo training updated
        # (LoRA on the LLM, plus the vision tower unless ``vision_model.freeze``).
        self.trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.model = model.to(device=device, dtype=torch.bfloat16).eval()
        print(f"[simlingo_steervla] loaded {expect_type.upper()} {self.weights} on {device}", flush=True)

        tmp = AutoConfig.from_pretrained(self.variant, trust_remote_code=True)
        image_size = tmp.force_image_size or tmp.vision_config.image_size
        self.image_size = int(image_size)
        self.num_image_token = int((image_size // tmp.vision_config.patch_size) ** 2 * (tmp.downsample_ratio**2))
        self._transform = build_transform(input_size=self.image_size)
        self._dynamic_preprocess = dynamic_preprocess
        self._conv_module = self._load_conv_module()

    def _load_conv_module(self):
        path = self.cache_root / self.variant.split("/")[1] / "conversation.py"
        if not path.is_file():
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=self.variant, local_dir=str(path.parent))
        spec = importlib.util.spec_from_file_location("get_conv_template", str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["get_conv_template"] = module
        spec.loader.exec_module(module)
        return module

    def pixel_tiles(self, rgb_hwc: np.ndarray) -> torch.Tensor:
        """``(num_patches, 3, S, S)`` tiles, after the dataset's ``cut_bottom_quarter`` crop."""
        from PIL import Image

        rgb = np.asarray(rgb_hwc, dtype=np.uint8)
        if bool(self.dataset_cfg.get("cut_bottom_quarter", True)):
            h = rgb.shape[0]
            rgb = rgb[: int(h - (h * 4.8) // 16)]
        tiles = self._dynamic_preprocess(
            Image.fromarray(rgb),
            image_size=self.image_size,
            use_thumbnail=bool(self.cfg.model.vision_model.use_global_img),
            max_num=2,
        )
        return torch.stack([self._transform(t) for t in tiles])

    def question_label(self, prompts: list[str], num_patches: int):
        """Inference prompt (user turn + open assistant turn), exactly as agent_steervla.format_input."""
        from simlingo_training.utils.custom_types import LanguageLabel

        queries = []
        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.num_image_token * num_patches + "</img>"
        for prompt in prompts:
            template = self._conv_module.get_conv_template("internlm2-chat")
            template.append_message(template.roles[0], "<image>\n" + prompt)
            template.append_message(template.roles[1], None)
            system = template.system_template.replace("{system_message}", template.system_message) + template.sep
            queries.append(template.get_prompt().replace(system, "").replace("<image>", image_tokens, 1))
        enc = self.tokenizer(queries, padding=True, return_tensors="pt", add_special_tokens=False)
        valid = enc["input_ids"] != self.tokenizer.pad_token_id
        return LanguageLabel(
            phrase_ids=enc["input_ids"].to(self.device),
            phrase_valid=valid.to(self.device),
            phrase_mask=valid.to(self.device),
            placeholder_values=[{} for _ in prompts],
            language_string=queries,
            loss_masking=None,
        )

    def driving_input(self, pixel_tiles: torch.Tensor, prompt_label, question_label=None):
        from simlingo_training.utils.custom_types import DrivingInput

        b = int(pixel_tiles.shape[0])
        return DrivingInput(
            camera_images=pixel_tiles.to(self.device, dtype=torch.bfloat16),  # (B, T=1, patches, 3, S, S)
            image_sizes=None,
            camera_intrinsics=torch.eye(3, device=self.device).expand(b, 3, 3),
            camera_extrinsics=torch.eye(4, device=self.device).expand(b, 4, 4),
            vehicle_speed=torch.zeros(b, 1, device=self.device),
            target_point=torch.zeros(b, 2, device=self.device),
            prompt=prompt_label,
            prompt_inference=question_label if question_label is not None else prompt_label,
        )

    @torch.no_grad()
    def generate(self, rgb_hwc: np.ndarray, prompt: str):
        tiles = self.pixel_tiles(rgb_hwc)
        label = self.question_label([prompt], int(tiles.shape[0]))
        driving_input = self.driving_input(tiles[None, None], label)
        if self.model_type != "hl":
            return self.model(driving_input)

        adaptor_dict = self.model.adaptors(driving_input, inference=True)
        adaptor_dict = self.model.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=adaptor_dict,
            pixel_values=driving_input.camera_images,
            placeholder_values=driving_input.prompt_inference.placeholder_values,
            wp_encoder=self.model.wp_encoder,
        )
        if self.variant == "OpenGVLab/InternVL2-4B":
            eos = self.tokenizer.added_tokens_encoder["<|end|>"]
        elif self.variant == "OpenGVLab/InternVL2-2B":
            eos = self.tokenizer.added_tokens_encoder["<|im_end|>"]
        else:
            eos = self.tokenizer.eos_token_id
        tokens, _ = self.model.language_model.greedy_sample(
            adaptor_dict["language_inputs"],
            eos_token_id=eos,
            max_new_tokens=100,
            input_embed_matrix=self.model.adaptors.language.embed_tokens.weight,
            logit_matrix=self.model.adaptors.language.lm_head.weight,
            attention_mask=adaptor_dict["language_inputs_mask"],
        )
        return None, None, self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, rgb_hwc: np.ndarray, prompts: list[str]):
        """Batch one scene's candidate queries without the upstream HL row loop."""
        tiles = self.pixel_tiles(rgb_hwc)
        label = self.question_label(prompts, int(tiles.shape[0]))
        inputs = self.driving_input(tiles[None, None].expand(len(prompts), -1, -1, -1, -1, -1), label)
        model = self.model
        if model.model_type != "hl":
            # Match single-query positional indexing. Padding different-length LL
            # prompts shifts the upstream model's positions and changes waypoints.
            lengths = label.phrase_valid.sum(dim=1).cpu().tolist()
            groups = {}
            for i, length in enumerate(lengths):
                groups.setdefault(length, []).append(i)
            outputs = [None] * len(prompts)
            for indices in groups.values():
                group_label = self.question_label([prompts[i] for i in indices], int(tiles.shape[0]))
                group_input = self.driving_input(
                    tiles[None, None].expand(len(indices), -1, -1, -1, -1, -1), group_label)
                speed, route, _ = model(group_input)
                if speed is None or route is None:
                    raise RuntimeError('Batched LL returned no waypoints.')
                for j, i in enumerate(indices):
                    outputs[i] = (speed[j], route[j])
            return (torch.stack([x[0] for x in outputs]), torch.stack([x[1] for x in outputs]), []), None
        if not model.predict_language or model.adaptors.driving is not None:
            raise RuntimeError("Batched HL requires the language-only planner.")
        adaptor = model.adaptors(inputs, inference=True)
        adaptor = model.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=adaptor, pixel_values=inputs.camera_images,
            placeholder_values=inputs.prompt_inference.placeholder_values,
            camera_images_history=inputs.camera_images_history,
        )
        variant = model.language_model.variant
        eos = (model.tokenizer.added_tokens_encoder['<|end|>'] if variant == 'OpenGVLab/InternVL2-4B'
               else model.tokenizer.added_tokens_encoder['<|im_end|>'] if variant == 'OpenGVLab/InternVL2-2B'
               else model.tokenizer.eos_token_id)
        tokens, _ = model.language_model.greedy_sample(
            adaptor['language_inputs'], eos_token_id=eos, max_new_tokens=512,
            input_embed_matrix=model.adaptors.language.embed_tokens.weight,
            logit_matrix=model.adaptors.language.lm_head.weight,
            attention_mask=adaptor['language_inputs_mask'],
        )
        overflow = ~(tokens == eos).any(dim=1)
        return (None, None, model.tokenizer.batch_decode(tokens, skip_special_tokens=True)), overflow.cpu().numpy()


# --------------------------------------------------------------------------------------------- #
# Actor                                                                                          #
# --------------------------------------------------------------------------------------------- #


def split_hl_output(text: str) -> tuple[str, str]:
    """``(reasoning, subtask)`` from ``"<reasoning>\\n\\nDriving Behavior: <subtask>"``."""
    text = str(text or "").strip()
    idx = text.find(DRIVING_BEHAVIOR_MARKER)
    if idx < 0:
        return "", " ".join(text.split())
    return text[:idx].strip(), " ".join(text[idx + len(DRIVING_BEHAVIOR_MARKER):].split())


class SimLingoSteerVLAActor(HLPoolSamplingMixin):
    """SimLingo HL -> LL policy with SteerVLAActor's rollout, CAST-HL-update and checkpoint contract."""

    # Chunk-cache / re-anchor helpers are model-agnostic but pinned to SteerVLAActor's class body by
    # ``test_reanchor_cached_chunk.py``, so they are borrowed rather than moved.
    _EGO_STATE_IDX_X = SteerVLAActor._EGO_STATE_IDX_X
    _EGO_STATE_IDX_Y = SteerVLAActor._EGO_STATE_IDX_Y
    _EGO_STATE_IDX_YAW_DEG = SteerVLAActor._EGO_STATE_IDX_YAW_DEG
    _ROUTE_XY_FORMATS = SteerVLAActor._ROUTE_XY_FORMATS
    _shift_cached_action_chunk = SteerVLAActor._shift_cached_action_chunk
    _ego_pose_from_state = SteerVLAActor._ego_pose_from_state
    _current_ego_pose = SteerVLAActor._current_ego_pose
    _reanchor_disabled = SteerVLAActor._reanchor_disabled
    _reanchor_route_to_current_pose = SteerVLAActor._reanchor_route_to_current_pose
    replay_action_chunk_from_pose = SteerVLAActor.replay_action_chunk_from_pose
    _prune_checkpoints = staticmethod(SteerVLAActor._prune_checkpoints)

    def __init__(
        self,
        *,
        hl_checkpoint: str,
        ll_checkpoint: str,
        simlingo_source_root: str,
        raw_obs_holder: MutableMapping[str, Any] | None,
        image_key: str = "image_viz",
        routing_command: str = "follow the road.",
        output_action_format: str = "DELTA_XY_T_DELTA_XY_SPACE",
        action_horizon: int = 10,
        action_dim: int = 4,
        actions_per_model_query: int = 1,
        actions_per_cot: int = 1,
        env_steps_per_chunk_row: int = 5,
        reanchor_cached_chunk: bool = True,
        training_gpu_rank: int = -1,
        hl_training_gpu_rank: int = -1,
        load_trainable_params: bool = False,
        hl_dataset_dir: str | Path | None = None,
        hl_update_every: int = 1,
        hl_update_batch_size: int = 16,
        hl_update_num_steps: int = 1,
        hl_micro_batch_size: int = 8,
        hl_lr: float | None = 1e-5,
        hl_weight_decay: float = 0.1,
        hl_grad_clip: float = 1.0,
        hl_replay_root: str | Path | None = None,
        hl_replay_pools: list[dict] | None = None,
        hl_online_weight: float = 1.0,
        hl_online_bad_fraction: float = -1.0,
        hl_online_precursor_fraction: float = -1.0,
        hl_online_backfill_from_replay: bool = True,
        use_adaptive_sampling: bool = False,
        adaptive_sampling_weights: dict | None = None,
        hl_min_online_samples: int = 1,
        hl_keep_last_rounds: int = 0,
        noise_scale: float = 1.0,
        cot_temperature: float = 0.0,
        hl_kl_coef: float = 0.0,
        hl_log_kl: bool = True,
    ) -> None:
        if str(output_action_format).strip().lower() != "delta_xy_t_delta_xy_space":
            raise ValueError("simlingo_steervla emits DELTA_XY_T_DELTA_XY_SPACE chunks only.")
        if int(action_dim) != 4:
            raise ValueError("simlingo_steervla emits action_dim=4 chunks (speed xy + route xy).")

        self.raw_obs_holder = raw_obs_holder
        self.image_key = str(image_key)
        self.routing_command = str(routing_command)
        self.output_action_format = str(output_action_format)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.actions_per_model_query = max(1, int(actions_per_model_query))
        self.actions_per_cot = max(1, int(actions_per_cot))
        self.env_steps_per_chunk_row = max(1, int(env_steps_per_chunk_row))
        self.reanchor_cached_chunk = bool(reanchor_cached_chunk)

        # Attributes main_carla / DSRLAgent read off any VLA actor (see steervla.py for meaning).
        self.load_trainable_params = bool(load_trainable_params)
        self.model_cfg = None  # No OpenPI observation / replay-token fields for this actor.
        self.train_cfg = None
        self._qgf_config = None
        self.include_ego_history = False
        self.proprio_norm = False
        self.sampling_seed = 0
        # HL CoT sampling temperature (0 = greedy, SimLingo's default); the LL always decodes greedily.
        self.cot_temperature = float(cot_temperature)
        # KL tether of the HL update to the starting checkpoint, as SteerVLAActor's hl_kl_coef / hl_log_kl.
        self.hl_kl_coef = float(hl_kl_coef)
        self.hl_log_kl = bool(hl_log_kl)
        self._hl_ref_model = None
        self.noise_scale = float(noise_scale)
        self.debug_noise = False
        self.debug_noise_samples = 0
        self.use_best_noise = False
        self.debug_noise_log_every_n_steps = 0
        self.last_action_was_cached = False
        self.last_reanchor: dict[str, float] | None = None
        self._reanchor_disabled_reason: str | None = None

        self.hl_dataset_dir: Path | None = Path(hl_dataset_dir) if hl_dataset_dir is not None else None
        self.hl_update_every = max(1, int(hl_update_every))
        self.hl_update_batch_size = max(1, int(hl_update_batch_size))
        self.hl_update_num_steps = max(1, int(hl_update_num_steps))
        self.hl_micro_batch_size = max(1, int(hl_micro_batch_size))
        self.hl_lr = float(hl_lr) if hl_lr else 1e-5
        self.hl_weight_decay = float(hl_weight_decay)
        self.hl_grad_clip = float(hl_grad_clip)
        self._init_hl_pool_sampling(
            hl_replay_root=hl_replay_root,
            hl_replay_pools=hl_replay_pools,
            hl_online_weight=hl_online_weight,
            hl_online_bad_fraction=hl_online_bad_fraction,
            hl_online_precursor_fraction=hl_online_precursor_fraction,
            hl_online_backfill_from_replay=hl_online_backfill_from_replay,
            use_adaptive_sampling=use_adaptive_sampling,
            adaptive_sampling_weights=adaptive_sampling_weights,
            hl_min_online_samples=hl_min_online_samples,
            hl_keep_last_rounds=hl_keep_last_rounds,
        )

        # JAX GPU ranks index the visible CUDA devices in the same order torch does.
        self.device = torch.device(f"cuda:{int(training_gpu_rank)}" if int(training_gpu_rank) >= 0 else "cuda")
        self.hl_device = (
            torch.device(f"cuda:{int(hl_training_gpu_rank)}") if int(hl_training_gpu_rank) >= 0 else self.device
        )
        source_root = Path(simlingo_source_root).expanduser().resolve()
        _import_simlingo(source_root)
        self.hl = _SimLingoModel(hl_checkpoint, source_root=source_root, device=self.hl_device, expect_type="hl")
        self.ll = _SimLingoModel(ll_checkpoint, source_root=source_root, device=self.device, expect_type="ll")
        self.ll.model.requires_grad_(False)
        # DrivingModel.forward calls greedy_sample without a temperature; inject cot_temperature for the HL.
        hl_lm = self.hl.model.language_model
        greedy_sample = hl_lm.greedy_sample

        def sample_with_temperature(*args, **kwargs):
            kwargs.setdefault("temperature", self.cot_temperature)
            return greedy_sample(*args, **kwargs)

        hl_lm.greedy_sample = sample_with_temperature
        if bool(self.hl.dataset_cfg.get("use_history_image", False)):
            raise NotImplementedError(
                f"{self.hl.config_path} sets use_history_image=True; the history-frame input is not wired "
                "into this actor (rollout image buffer + HL-sample storage). The bellman HL trains without it."
            )
        self._use_ego_history = bool(self.hl.dataset_cfg.get("use_ego_state_history", False))
        self._ego_history_count = int(self.hl.dataset_cfg.get("ego_state_history_count", 3))
        self._ego_hist: deque[tuple[float, float]] = deque(
            maxlen=_EGO_HISTORY_INTERVAL_TICKS * self._ego_history_count + 2
        )

        self._cached_action_chunk: np.ndarray | None = None
        self._cached_action_pose: np.ndarray | None = None
        self._cached_action_step = 0
        self._cot: dict[str, str] | None = None
        self._cot_age = 0

        self._hl_params: list[torch.nn.Parameter] = []
        self._hl_master: list[torch.Tensor] = []
        self._hl_optimizer: torch.optim.Optimizer | None = None
        if self.load_trainable_params:
            self._setup_hl_training()

    @property
    def sampling_seed(self) -> int:
        return self._sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, value: int) -> None:
        # main_carla sets this per run and per eval episode; HL sampling at cot_temperature > 0 uses torch's RNG.
        self._sampling_seed = int(value)
        torch.manual_seed(self._sampling_seed)

    # ---- rollout ------------------------------------------------------------------------------ #

    def _raw(self) -> dict[str, Any] | None:
        raw = None if self.raw_obs_holder is None else self.raw_obs_holder.get("obs")
        return raw if isinstance(raw, dict) else None

    def _push_ego_history(self, raw: dict[str, Any]) -> None:
        # BoN queries several candidates from the same observation. Record it once.
        if getattr(self, "_last_history_raw", None) is raw:
            return
        self._last_history_raw = raw
        state = np.asarray(raw.get("state"), dtype=np.float32).reshape(-1)
        if state.size > _EGO_IDX_SPEED:
            self._ego_hist.append((float(state[_EGO_IDX_SPEED]), float(state[_EGO_IDX_YAW_DEG])))

    def _ego_history_line(self) -> str:
        """SimLingo's training-format history line; oldest-available entry when the run is younger."""
        if not self._use_ego_history or not self._ego_hist:
            return ""
        picked = []
        for i in range(self._ego_history_count, 0, -1):
            offset = 1 + i * _EGO_HISTORY_INTERVAL_TICKS
            picked.append(self._ego_hist[-offset] if len(self._ego_hist) >= offset else self._ego_hist[0])
        speeds = " ".join(f"{round(s, 1)} m/s" for s, _ in picked)
        headings = " ".join(f"{round(h, 1)} degrees" for _, h in picked)
        return f"Speed history: {speeds} Heading history: {headings}\n"

    def hl_prompt(self, raw: dict[str, Any]) -> str:
        state = np.asarray(raw.get("state"), dtype=np.float32).reshape(-1)
        speed = round(float(state[_EGO_IDX_SPEED]), 1) if state.size > _EGO_IDX_SPEED else 0.0
        routing = raw.get("routing_command")
        routing = str(routing).strip() if isinstance(routing, str) and routing.strip() else self.routing_command
        return f"{self._ego_history_line()}Current speed: {speed} m/s\nCommand: {routing}"

    def _image(self, raw: dict[str, Any]) -> np.ndarray:
        image = raw.get(self.image_key)
        if image is None:
            raise KeyError(f"raw obs has no {self.image_key!r} (set steervla.image_key)")
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"raw obs {self.image_key!r} has shape {image.shape}, expected (H, W, 3)")
        return image

    def _sample_cot(self, raw: dict[str, Any]) -> dict[str, str]:
        prompt = self.hl_prompt(raw)
        _, _, language = self.hl.generate(self._image(raw), prompt)
        text = str(language[0] if language else "")
        reasoning, subtask = split_hl_output(text)
        return {
            "prompt": prompt,
            "output": text,
            "reasoning": reasoning,
            "subtask": subtask or _TRAINING_HL_FALLBACK,
        }

    def _stash_cot_in_raw(self, raw: dict[str, Any] | None) -> None:
        if raw is None or self._cot is None:
            return
        # String fields only: main_carla keys the OpenPI token backfill on raw["reasoning"].
        raw["subtask_text"] = self._cot["subtask"]
        raw["reasoning_text"] = self._cot["reasoning"]
        # Full HL prompt: what cast_relabel stores and update_hl re-tokenizes.
        raw["openpi_prompt_raw_text"] = self._cot["prompt"]
        # Short display/critic prompt, like SteerVLA's: main_carla SigLIP-encodes this field, and the
        # ego-history line pushes the full prompt past SigLIP's 64-token text limit.
        raw["openpi_prompt_text"] = self._cot["prompt"].rsplit("\n", 1)[-1]
        raw["simlingo_hl_output"] = self._cot["output"]

    @staticmethod
    def waypoints_to_chunk(speed_wps: np.ndarray, route: np.ndarray, horizon: int) -> np.ndarray:
        """SimLingo cumulative waypoints -> ``(horizon, 4)`` DELTA_XY_T_DELTA_XY_SPACE chunk in RLDS units.

        The env rebuilds ``cumsum(chunk[:, :2] * 7)`` as the 0.25 s speed waypoints and
        ``cumsum([0; chunk[:, 2:]])`` as the route, so the speed columns are the per-step displacements
        / 7 and the route columns the spatial deltas after SimLingo's leading origin point.
        """
        speed_wps = np.asarray(speed_wps, dtype=np.float64).reshape(-1, 2)
        route = np.asarray(route, dtype=np.float64).reshape(-1, 2)
        if route.shape[0] > 1 and float(np.linalg.norm(route[0])) < 0.25:
            route = route[1:]

        def deltas(points: np.ndarray) -> np.ndarray:
            d = np.diff(np.concatenate([np.zeros((1, 2)), points], axis=0), axis=0)[:horizon]
            if d.shape[0] < horizon:
                d = np.concatenate([d, np.repeat(d[-1:], horizon - d.shape[0], axis=0)], axis=0)
            return d

        chunk = np.concatenate([deltas(speed_wps) / _SPEED_XY_SCALE, deltas(route)], axis=1)
        return chunk.astype(np.float32)

    def _query_chunk(self, raw: dict[str, Any]) -> np.ndarray:
        if self._cot is None or self._cot_age >= self.actions_per_cot:
            self._cot = self._sample_cot(raw)
            self._cot_age = 0
        state = np.asarray(raw.get("state"), dtype=np.float32).reshape(-1)
        speed = round(float(state[_EGO_IDX_SPEED]), 1) if state.size > _EGO_IDX_SPEED else 0.0
        ll_prompt = f"Current speed: {speed} m/s. Command: {self._cot['subtask']} Predict the waypoints."
        speed_wps, route, _ = self.ll.generate(self._image(raw), ll_prompt)
        if speed_wps is None or route is None:
            raise RuntimeError("SimLingo LL returned no waypoints (check predict_route_as_wps / speed_wps_mode).")
        return self.waypoints_to_chunk(
            speed_wps[0].float().cpu().numpy(), route[0].float().cpu().numpy(), self.action_horizon
        ).reshape(1, -1)

    @torch.no_grad()
    def sample_candidates(self, n: int, *, temperature: float, raw: dict[str, Any], **kwargs):
        """Fresh independent HL samples followed by a batched LL waypoint pass."""
        self._push_ego_history(raw)
        prompt = self.hl_prompt(raw)
        previous_temperature = self.cot_temperature
        self.cot_temperature = float(temperature)
        started = time.monotonic()
        try:
            (_, _, texts), overflow = self.hl.generate_batch(self._image(raw), [prompt] * n)
        finally:
            self.cot_temperature = previous_temperature
        hl_seconds = time.monotonic() - started
        parsed = [split_hl_output(text) for text in texts]
        subtasks = [subtask or _TRAINING_HL_FALLBACK for _, subtask in parsed]
        state = np.asarray(raw.get('state'), dtype=np.float32).reshape(-1)
        speed = round(float(state[_EGO_IDX_SPEED]), 1) if state.size > _EGO_IDX_SPEED else 0.0
        prompts = [f'Current speed: {speed} m/s. Command: {subtask} Predict the waypoints.' for subtask in subtasks]
        started = time.monotonic()
        (speed_wps, routes, _), _ = self.ll.generate_batch(self._image(raw), prompts)
        if speed_wps is None or routes is None:
            raise RuntimeError('Batched SimLingo LL returned no waypoints.')
        speed_wps = speed_wps.float().cpu().numpy()
        routes = routes.float().cpu().numpy()
        chunks = np.stack([self.waypoints_to_chunk(w, r, self.action_horizon).reshape(-1)
                           for w, r in zip(speed_wps, routes)])
        self._cot = dict(prompt=prompt, output=texts[0], reasoning=parsed[0][0], subtask=subtasks[0])
        self._stash_cot_in_raw(raw)
        self.last_candidate_timings = dict(hl_s=hl_seconds, ll_s=time.monotonic() - started, candidates=n)
        print(f'[hierarchical-batch] {self.last_candidate_timings}', flush=True)
        return dict(actions_normalized=chunks, subtask_texts=subtasks,
                    reasoning_texts=[x[0] for x in parsed], reasoning_overflowed=overflow)

    def _next_cached_action(self, batch_size: int) -> np.ndarray | None:
        """Serve a held chunk, shifted one row per ``env_steps_per_chunk_row`` and re-anchored (see steervla)."""
        if self.actions_per_model_query <= 1 or batch_size != 1 or self._cached_action_chunk is None:
            return None
        max_cached = min(self.actions_per_model_query, self.action_horizon * self.env_steps_per_chunk_row)
        if self._cached_action_step >= max_cached:
            self._cached_action_chunk = None
            self._cached_action_step = 0
            return None
        row = self._cached_action_step // self.env_steps_per_chunk_row
        out = self._shift_cached_action_chunk(self._cached_action_chunk, row)
        out = self._reanchor_route_to_current_pose(out, self._cached_action_chunk)
        self._cached_action_step += 1
        return out

    def __call__(self, observations: Any, noise: Any):
        """DSRL ``vla_sample_fn``: image/state come from ``raw_obs_holder``; ``noise`` only sets the batch size."""
        import jax.numpy as jnp

        batch_size = int(np.asarray(noise).shape[0])
        raw = self._raw()
        if raw is None:
            raise RuntimeError('SimLingoSteerVLAActor needs raw_obs_holder["obs"] (the CARLA gym dict).')
        self._push_ego_history(raw)
        self._cot_age += 1
        cached = self._next_cached_action(batch_size)
        self.last_action_was_cached = cached is not None
        if cached is None:
            out = np.repeat(self._query_chunk(raw), batch_size, axis=0)
            if self.actions_per_model_query > 1 and batch_size == 1:
                self._cached_action_chunk = out.copy()
                self._cached_action_pose = self._current_ego_pose()
                self._cached_action_step = 1
        else:
            out = cached
        self._stash_cot_in_raw(raw)
        return jnp.asarray(out, dtype=jnp.float32)

    def decode_last_batch_subtasks(self) -> list[str]:
        """Expose the sampled meta-action through main_carla's BoN label contract."""
        return [self._cot["subtask"]] if self._cot is not None else []

    def decode_last_batch_reasoning(self) -> list[str]:
        """Expose the text before ``Driving Behavior:`` for diagnostics."""
        return [self._cot["reasoning"]] if self._cot is not None else []

    def reset_candidate_cache(self) -> None:
        """Draw a fresh HL/LL candidate while preserving observed ego history."""
        history = self._ego_hist.copy()
        last_raw = getattr(self, "_last_history_raw", None)
        self.reset_action_cache()
        self._ego_hist.extend(history)
        self._last_history_raw = last_raw

    def reset_action_cache(self) -> None:
        """Episode reset: drop the held chunk, the held CoT and the ego history."""
        self._cached_action_chunk = None
        self._cached_action_pose = None
        self._cached_action_step = 0
        self._cot = None
        self._cot_age = 0
        self._ego_hist.clear()
        self._last_history_raw = None

    # ---- online HL update --------------------------------------------------------------------- #

    def _setup_hl_training(self) -> None:
        named = dict(self.hl.model.named_parameters())
        self._hl_params = [named[n] for n in self.hl.trainable_names]
        for p in self.hl.model.parameters():
            p.requires_grad_(False)
        for p in self._hl_params:
            p.requires_grad_(True)
        # bf16 compute, fp32 master weights + AdamW state (SimLingo trained 16-mixed).
        self._hl_master = [p.detach().float().clone().requires_grad_(True) for p in self._hl_params]
        self._hl_optimizer = torch.optim.AdamW(
            self._hl_master, lr=self.hl_lr, weight_decay=self.hl_weight_decay, betas=(0.9, 0.999)
        )
        if self.hl_kl_coef > 0.0 or self.hl_log_kl:
            # Frozen snapshot of the starting HL: the reference policy for KL(pi_theta || pi_ref).
            self._hl_ref_model = copy.deepcopy(self.hl.model).eval().requires_grad_(False)
        n = sum(p.numel() for p in self._hl_params)
        print(
            f"[simlingo_steervla] HL trainable: {len(self._hl_params)} tensors / {n / 1e6:.1f}M params "
            f"on {self.hl_device} (lr={self.hl_lr}, wd={self.hl_weight_decay})",
            flush=True,
        )

    def _read_hl_record(self, e: dict[str, Any]) -> dict[str, Any] | None:
        from vlas.steervla import strip_cot_sentinels

        path = Path(e["dir"]) / str(e["file"])
        try:
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                import cv2

                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    return None
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                with np.load(path) as z:
                    image = np.asarray(z["image"], dtype=np.uint8)
        except Exception:
            return None  # Half-written .npz (cast_relabel writes concurrently) or unreadable frame.
        pool = str(e.get("pool", "online"))
        prompt = str(e.get("prompt") or "").strip()
        subtask = strip_cot_sentinels(e.get("subtask", ""))
        if not prompt or not subtask:
            return None
        online = pool == "online"
        # Online relabels keep the rollout reasoning as context and supervise only the subtask;
        # cast_relabel's ``reasoning`` is the VLM's SteerVLA-style replacement, not SimLingo's format.
        reasoning = str(e.get("original_reasoning") or "").strip() if online else ""
        if not reasoning:
            reasoning = strip_cot_sentinels(e.get("reasoning", ""))
        return {
            "image": image,
            "prompt": prompt,
            "subtask": subtask,
            "reasoning": reasoning,
            "loss_scope": "subtask" if online else "full",
            "pool": pool,
            "label": e.get("label"),
            "credit_source": e.get("credit_source", ""),
            "policy_version": int(e.get("policy_version", -1)),
            "sample_id": str(path),
        }

    @staticmethod
    def hl_answer(record: dict[str, Any]) -> str:
        reasoning = str(record.get("reasoning") or "").strip()
        subtask = str(record["subtask"]).strip()
        return f"{reasoning}\n\n{DRIVING_BEHAVIOR_MARKER} {subtask}" if reasoning else f"{DRIVING_BEHAVIOR_MARKER} {subtask}"

    def _build_hl_example(self, records: list[dict[str, Any]]):
        """SimLingo DrivingExample with the training chat template and per-record loss scope."""
        from simlingo_training.utils.custom_types import DrivingExample, DrivingLabel, LanguageLabel
        from simlingo_training.utils.internvl2_utils import get_custom_chat_template

        m = self.hl
        tiles = [m.pixel_tiles(r["image"]) for r in records]
        num_patches = int(tiles[0].shape[0])
        if any(int(t.shape[0]) != num_patches for t in tiles):
            raise ValueError("HL batch mixes image aspect ratios (different tile counts); frames must share a size.")
        pixels = torch.stack(tiles)[:, None]  # (B, 1, patches, 3, S, S)
        conversations = [
            [
                {"role": "user", "content": [{"type": "text", "text": r["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": self.hl_answer(r)}]},
            ]
            for r in records
        ]
        conv, question = get_custom_chat_template(
            conversations, m.tokenizer, m.variant, m.num_image_token * num_patches, cache_root_dir=str(m.cache_root)
        )
        loss_mask = conv["loss_masking"].clone()
        if any(r["loss_scope"] == "subtask" for r in records):
            enc = m.tokenizer(
                conv["language_string"], padding=True, return_tensors="pt",
                add_special_tokens=False, return_offsets_mapping=True,
            )
            if not torch.equal(enc["input_ids"], conv["phrase_ids"]):
                raise RuntimeError("re-tokenization with offsets disagrees with the chat-template tokenization")
            for i, r in enumerate(records):
                if r["loss_scope"] != "subtask":
                    continue
                text = conv["language_string"][i]
                start = text.rfind(DRIVING_BEHAVIOR_MARKER)
                if start < 0:
                    loss_mask[i] = False
                    continue
                start += len(DRIVING_BEHAVIOR_MARKER)
                loss_mask[i] &= enc["offset_mapping"][i, :, 0] >= start

        def label(d, mask):
            return LanguageLabel(
                phrase_ids=d["phrase_ids"].to(m.device),
                phrase_valid=d["phrase_valid"].to(m.device),
                phrase_mask=d["phrase_mask"].to(m.device),
                placeholder_values=[{} for _ in records],
                language_string=d["language_string"],
                loss_masking=None if mask is None else mask.to(m.device),
            )

        b = len(records)
        example = DrivingExample(
            driving_input=m.driving_input(pixels, label(conv, loss_mask), label(question, None)),
            driving_label=DrivingLabel(
                waypoints=torch.zeros(b, 11, 2, device=m.device),
                path=torch.zeros(b, 20, 2, device=m.device),
                answer=LanguageLabel(None, None, None, None, [self.hl_answer(r) for r in records], None),
                image_ff_org=torch.zeros(b, 1, device=m.device),
                eval_infos=None,
                refined_commentary=None,
            ),
            run_id=None,
        )
        # Next-token targets: token t+1 is supervised where the mask is set.
        return example, int(loss_mask[:, 1:].sum())

    def update_hl(
        self,
        *,
        batch_size: int | None = None,
        num_steps: int | None = None,
        rng: Any = None,
        global_step: int | None = None,
    ) -> dict[str, float]:
        """Throttled HL gradient steps on the cast_relabel HL dataset (+ replay pools); same contract as SteerVLA."""
        del rng
        if self._hl_optimizer is None:
            return self._hl_skip("actor was not loaded trainable (needs steervla.load_trainable_params=True)")
        if self.hl_dataset_dir is None:
            return self._hl_skip(
                "hl_dataset_dir is unset — main_carla only wires it when cast_relabel.enabled "
                "and load_trainable_params are both true"
            )
        self._hl_update_calls += 1
        if self.hl_update_every > 1 and (self._hl_update_calls % self.hl_update_every != 0):
            return {}
        bs = int(batch_size or self.hl_update_batch_size)
        ns = int(num_steps or self.hl_update_num_steps)
        records = self._load_hl_batch(bs)
        if records is None:
            return self._hl_skip(
                f"waiting for the online HL pool to start: {self._hl_pool_size}/"
                f"{self.hl_min_online_samples} samples under {self.hl_dataset_dir}"
            )
        self._last_hl_skip_reason = None
        reuse_info = self._record_hl_sample_uses(records, ns)
        self._dump_hl_batch(records, global_step=global_step)

        micro = [records[i : i + self.hl_micro_batch_size] for i in range(0, len(records), self.hl_micro_batch_size)]
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats(self.hl_device)
        model = self.hl.model
        model.train()
        losses, grad_norm, total_tokens = [], 0.0, 0
        kl_info: dict[str, float] = {}
        try:
            for _ in range(max(1, ns)):
                built = [self._build_hl_example(chunk) for chunk in micro]
                total_tokens = sum(n for _, n in built)
                if total_tokens == 0:
                    return self._hl_skip("HL batch has no supervised tokens (empty subtasks after masking)")
                step_loss = 0.0
                n_rows = sum(int(ex.driving_input.camera_images.shape[0]) for ex, _ in built)
                kl_sums = {"kl_to_ref": 0.0, "ce_theta": 0.0, "ce_ref": 0.0, "log_ratio": 0.0}
                for example, _ in built:
                    loss_dict, _ = model.forward_loss(example, per_sample=True)
                    token_loss, token_mask = loss_dict["language_loss"]
                    token_loss, token_mask = token_loss.float(), token_mask.float()
                    micro_sum = (token_loss * token_mask).sum()
                    objective = micro_sum / total_tokens
                    if self._hl_ref_model is not None:
                        # k3 KL(pi_theta || pi_ref) on the per-example mean CE of the supervised tokens --
                        # the estimator and reference of steervla._openpi_hl_train_step. The reference is
                        # forward-only, so gradient flows through ce_theta alone.
                        n_tok = token_mask.sum(-1).clamp(min=1.0)
                        ce_theta = (token_loss * token_mask).sum(-1) / n_tok
                        with torch.no_grad():
                            ref_dict, _ = self._hl_ref_model.forward_loss(example, per_sample=True)
                            ce_ref = (ref_dict["language_loss"][0].float() * token_mask).sum(-1) / n_tok
                        log_ratio = ce_ref - ce_theta
                        kl = torch.exp(-log_ratio) + log_ratio - 1.0
                        weight = float(ce_theta.shape[0]) / n_rows
                        if self.hl_kl_coef > 0.0:
                            objective = objective + self.hl_kl_coef * kl.mean() * weight
                        for key, value in (("kl_to_ref", kl), ("ce_theta", ce_theta), ("ce_ref", ce_ref), ("log_ratio", log_ratio)):
                            kl_sums[key] += float(value.detach().mean()) * weight
                    objective.backward()
                    step_loss += float(micro_sum.detach()) / total_tokens
                if self._hl_ref_model is not None:
                    kl_info = dict(kl_sums, kl_penalty=self.hl_kl_coef * kl_sums["kl_to_ref"])
                for p, master in zip(self._hl_params, self._hl_master):
                    master.grad = None if p.grad is None else p.grad.detach().float()
                    p.grad = None
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self._hl_master, self.hl_grad_clip))
                self._hl_optimizer.step()
                self._hl_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for p, master in zip(self._hl_params, self._hl_master):
                        p.copy_(master.to(p.dtype))
                losses.append(step_loss)
                self._hl_grad_steps += 1
        finally:
            model.eval()
        self._hl_updates_applied += 1

        out: dict[str, float] = {
            "loss": float(np.mean(losses)),
            "grad_norm": grad_norm,
            "supervised_tokens": float(total_tokens),
            "lr": float(self.hl_lr),
            "n_samples": float(len(records)),
            "n_distinct_samples": float(len({r.get("sample_id") for r in records if r.get("sample_id")})),
            "n_online": float(sum(1 for r in records if r["pool"] == "online")),
            "policy_updates": float(self._hl_grad_steps),
            "updates_applied": float(self._hl_updates_applied),
            "update_calls": float(self._hl_update_calls),
            "update_seconds": float(time.time() - t0),
            "hl_mem_peak_gb": float(torch.cuda.max_memory_allocated(self.hl_device)) / 1e9,
        }
        out.update(reuse_info)
        out.update(kl_info)
        print(
            f"[simlingo_steervla.update_hl] loss={out['loss']:.4f} grad_norm={grad_norm:.3f} "
            f"policy_updates={self._hl_grad_steps} bs={len(records)} (online {int(out['n_online'])}) "
            f"tokens={total_tokens} {out['update_seconds']:.1f}s peak={out['hl_mem_peak_gb']:.1f}GB",
            flush=True,
        )
        return out

    def _dump_hl_batch(self, records: list[dict[str, Any]], *, global_step: int | None) -> None:
        """Text view of the exact batch each update trained on (images omitted), next to the HL dataset."""
        if self.hl_dataset_dir is None:
            return
        try:
            out_dir = Path(self.hl_dataset_dir).parent / "simlingo_hl_batches"
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {k: r.get(k) for k in ("sample_id", "pool", "label", "credit_source", "loss_scope", "prompt")}
                | {"answer": self.hl_answer(r)}
                for r in records
            ]
            name = f"update_{self._hl_updates_applied + 1:06d}_step_{int(global_step or -1)}.json"
            (out_dir / name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - logging must never break the update.
            print(f"[simlingo_steervla.update_hl] batch dump failed (non-fatal): {exc}", flush=True)

    def save_checkpoint(self, out_root: str | Path, step: int, *, keep_last: int = 0) -> Path | None:
        """Export the HL as ``<out_root>/<step>/pytorch_model.bin`` + ``.hydra/config.yaml``.

        Loadable as ``steervla.hl_checkpoint=<out_root>/<step>`` here, and by simlingo's
        ``agent_steervla.py`` (``resolve_checkpoint`` / ``find_hydra_config``).
        """
        if self._hl_optimizer is None:
            print("[simlingo_steervla.save_checkpoint] skipped: HL not loaded trainable.", flush=True)
            return None
        step_dir = Path(out_root) / str(int(step))
        (step_dir / ".hydra").mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.hl.model.state_dict().items()}
        torch.save(state, step_dir / "pytorch_model.bin")
        shutil.copyfile(self.hl.config_path, step_dir / ".hydra" / "config.yaml")
        print(f"[simlingo_steervla.save_checkpoint] wrote HL checkpoint -> {step_dir}", flush=True)
        if int(keep_last) > 0:
            self._prune_checkpoints(out_root, keep_last=int(keep_last))
        return step_dir


def create_simlingo_steervla_sample_fn(
    steervla_cfg: MutableMapping[str, Any],
    raw_obs_holder: MutableMapping[str, Any],
    *,
    training_gpu_rank: int = -1,
    noise_scale: float = 1.0,
) -> tuple[Callable[[Any, Any], Any], SimLingoSteerVLAActor]:
    """Same contract as :func:`vlas.steervla.create_steervla_pi0_cot_sample_fn`, for ``steervla.vla='simlingo_steervla'``."""
    srank = steervla_cfg.get("training_gpu_rank", None)
    if srank is None:
        srank = training_gpu_rank
    for key in ("hl_checkpoint", "ll_checkpoint", "simlingo_source_root"):
        if not steervla_cfg.get(key):
            raise ValueError(f"steervla.vla='simlingo_steervla' requires steervla.{key}")
    generic_ckpt = str(steervla_cfg.get("checkpoint") or "")
    if generic_ckpt and generic_ckpt != str(steervla_cfg["hl_checkpoint"]):
        print(
            f"[simlingo_steervla] steervla.checkpoint={generic_ckpt} is not used; the HL loads from "
            f"steervla.hl_checkpoint={steervla_cfg['hl_checkpoint']}",
            flush=True,
        )
    actor = SimLingoSteerVLAActor(
        hl_checkpoint=str(steervla_cfg["hl_checkpoint"]),
        ll_checkpoint=str(steervla_cfg["ll_checkpoint"]),
        simlingo_source_root=str(steervla_cfg["simlingo_source_root"]),
        raw_obs_holder=raw_obs_holder,
        image_key=str(steervla_cfg.get("image_key", "image_viz")),
        routing_command=str(steervla_cfg.get("routing_command", "follow the road.")),
        output_action_format=steervla_cfg.get("output_action_format") or "DELTA_XY_T_DELTA_XY_SPACE",
        action_horizon=int(steervla_cfg.get("action_horizon", 10)),
        action_dim=int(steervla_cfg.get("action_dim", 4)),
        actions_per_model_query=int(steervla_cfg.get("actions_per_model_query", 1)),
        actions_per_cot=int(steervla_cfg.get("actions_per_cot", 1)),
        env_steps_per_chunk_row=int(steervla_cfg.get("env_steps_per_chunk_row", 5)),
        reanchor_cached_chunk=bool(steervla_cfg.get("reanchor_cached_chunk", True)),
        training_gpu_rank=int(srank),
        hl_training_gpu_rank=int(steervla_cfg.get("hl_training_gpu_rank", -1)),
        load_trainable_params=bool(steervla_cfg.get("load_trainable_params", False)),
        hl_dataset_dir=steervla_cfg.get("hl_dataset_dir"),
        hl_update_every=int(steervla_cfg.get("hl_update_every", 1)),
        hl_update_batch_size=int(steervla_cfg.get("hl_update_batch_size", 16)),
        hl_update_num_steps=int(steervla_cfg.get("hl_update_num_steps", 1)),
        hl_micro_batch_size=int(steervla_cfg.get("hl_micro_batch_size", 8)),
        hl_lr=steervla_cfg.get("hl_lr"),
        hl_weight_decay=float(steervla_cfg.get("hl_weight_decay", 0.1)),
        hl_grad_clip=float(steervla_cfg.get("hl_grad_clip", 1.0)),
        hl_replay_root=steervla_cfg.get("hl_replay_root"),
        hl_replay_pools=steervla_cfg.get("hl_replay_pools"),
        hl_online_weight=float(steervla_cfg.get("hl_online_weight", 1.0)),
        hl_online_bad_fraction=float(steervla_cfg.get("hl_online_bad_fraction", -1.0)),
        hl_online_precursor_fraction=float(steervla_cfg.get("hl_online_precursor_fraction", -1.0)),
        hl_online_backfill_from_replay=bool(steervla_cfg.get("hl_online_backfill_from_replay", True)),
        use_adaptive_sampling=bool(steervla_cfg.get("use_adaptive_sampling", False)),
        adaptive_sampling_weights=(
            dict(steervla_cfg["adaptive_sampling_weights"]) if steervla_cfg.get("adaptive_sampling_weights") else None
        ),
        hl_min_online_samples=int(steervla_cfg.get("hl_min_online_samples", 1)),
        hl_keep_last_rounds=int(steervla_cfg.get("hl_keep_last_rounds", 0)),
        noise_scale=float(steervla_cfg.get("noise_scale", noise_scale)),
        cot_temperature=float(steervla_cfg.get("cot_temperature", 0.0)),
        hl_kl_coef=float(steervla_cfg.get("hl_kl_coef", 0.0)),
        hl_log_kl=bool(steervla_cfg.get("hl_log_kl", True)),
    )

    def sample_fn(observations: Any, noise: Any):
        return actor(observations, noise)

    sample_fn.reset_action_cache = actor.reset_action_cache  # type: ignore[attr-defined]
    return sample_fn, actor
