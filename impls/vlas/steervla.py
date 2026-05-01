"""Minimal SteerVLA / OpenPI loader used by ``main_carla`` checkpoint smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import config as _config

from .utils import LocalActor, RemoteActor


class SteerVLALocalActor(LocalActor):
    """Loads JAX params via ``TrainConfig.model.load`` after ``maybe_download``."""

    def __init__(self, actor_config: str, checkpoint_path: str) -> None:
        super().__init__(actor_config, checkpoint_path)
        self.checkpoint_dir: Optional[Path] = None
        self.policy = None

    def setup(self) -> None:
        cfg = _config.get_config(self.actor_config)
        path = download.maybe_download(self.checkpoint_path)
        self.checkpoint_dir = Path(path).resolve()
        params_dir = self.checkpoint_dir / "params"
        if not params_dir.exists():
            raise FileNotFoundError(f"Expected OpenPI checkpoint params at {params_dir}")
        params = _model.restore_params(params_dir)
        self.policy = cfg.model.load(params)

    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError("Wire observation dict + policy inference for offline eval.")

    def get_cot(self, state: Dict[str, Any]) -> dict:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError("Update is not supported for SteerVLA actor at this time.")


class SteerVLAActor:
    """Remote HTTP server or local OpenPI checkpoint."""

    def __init__(
        self,
        actor_url: Optional[str] = None,
        actor_config: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.actor_url = actor_url
        self.actor_config = actor_config
        self.checkpoint_path = checkpoint_path

        if actor_url is not None:
            self.actor = RemoteActor(actor_url)
            self.actor.get_info()
        else:
            assert actor_config is not None and checkpoint_path is not None
            self.actor = SteerVLALocalActor(actor_config, checkpoint_path)
            self.actor.setup()
