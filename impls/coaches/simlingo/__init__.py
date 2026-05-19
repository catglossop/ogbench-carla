from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_carla_python_api_on_path() -> None:
    root = os.environ.get("CARLA_PYTHON_API_ROOT")
    if not root and os.environ.get("CARLA_ROOT"):
        root = str(Path(os.environ["CARLA_ROOT"]).resolve() / "PythonAPI" / "carla")
    if not root:
        return
    p = str(Path(root).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_carla_python_api_on_path()

from .autopilot import AutoPilot
