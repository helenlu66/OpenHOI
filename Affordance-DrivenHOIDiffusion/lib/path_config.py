"""Affordance-Driven HOI Diffusion entry point for shared OpenHOI paths."""

from __future__ import annotations

import sys
from pathlib import Path

_openhoi_mirror = Path(__file__).resolve().parents[1]
if str(_openhoi_mirror) not in sys.path:
    sys.path.insert(0, str(_openhoi_mirror))

from openhoi_paths import afford_grab_pkl, data_path, data_root, grab_data_root  # noqa: E402

__all__ = ["afford_grab_pkl", "data_path", "data_root", "grab_data_root"]
