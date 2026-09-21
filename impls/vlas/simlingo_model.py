"""Torch-only SimLingo checkpoint loading, shared with the isolated HL worker."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch

DRIVING_BEHAVIOR_MARKER = "Driving Behavior:"
SIMLINGO_SPECIAL_TOKENS = [
    "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>",
]

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
        return self.model(self.driving_input(tiles[None, None], label))

    @torch.no_grad()
    def generate_batch(self, rgb_hwc: np.ndarray, prompt: str, num_samples: int):
        """``num_samples`` independent continuations of one scene, in a single forward pass.

        Every Best-of-N candidate at an env step is sampled from the same image and the same
        prompt -- only the drawn tokens differ -- so the tiling, the vision tower and the prompt
        encoding are shared here instead of being repeated once per candidate. greedy_sample
        already samples row-wise and tracks per-row completion, so the rows stay independent.
        """
        tiles = self.pixel_tiles(rgb_hwc)
        label = self.question_label([prompt] * int(num_samples), int(tiles.shape[0]))
        batched = tiles[None, None].expand(int(num_samples), -1, -1, -1, -1, -1).contiguous()
        return self.model(self.driving_input(batched, label))


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
