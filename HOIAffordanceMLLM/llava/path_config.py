"""HOIAffordanceMLLM entry point for shared OpenHOI path configuration."""

from __future__ import annotations

import sys
from pathlib import Path

_openhoi_mirror = Path(__file__).resolve().parents[2].parent
if str(_openhoi_mirror) not in sys.path:
    sys.path.insert(0, str(_openhoi_mirror))

from openhoi_paths import (  # noqa: E402
    affdata_test_json,
    affdata_test_points,
    affdata_train_json,
    affdata_train_points,
    afford_grab_pkl,
    data_path,
    data_root,
    log_dir,
    mm_projector,
    uni3d_checkpoint,
    vision_ckpt,
)

__all__ = [
    "affdata_test_json",
    "affdata_test_points",
    "affdata_train_json",
    "affdata_train_points",
    "afford_grab_pkl",
    "data_path",
    "data_root",
    "log_dir",
    "mm_projector",
    "uni3d_checkpoint",
    "vision_ckpt",
]
