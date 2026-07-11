"""Single source of truth for OpenHOI filesystem paths (Python).

Environment variable names match OpenHOI/scripts/openhoi_env.sh.
"""

from __future__ import annotations

import os
from pathlib import Path

_OPENHOI_ROOT = Path(__file__).resolve().parent


def _path_from_env(env_name: str, default_relative: tuple[str, ...]) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    return data_path(*default_relative)


def openhoi_root() -> Path:
    return Path(os.environ.get("OPENHOI_ROOT", str(_OPENHOI_ROOT))).expanduser().resolve()


def data_root() -> Path:
    return Path(
        os.environ.get("OPENHOI_DATA_ROOT", str(Path.home() / "openhoi-data"))
    ).expanduser().resolve()


def data_path(*parts: str) -> Path:
    return data_root().joinpath(*parts)


def log_dir() -> Path:
    return Path(os.environ.get("OPENHOI_LOG_DIR", str(data_root() / "log_dir"))).expanduser().resolve()


def llm_model() -> Path:
    return _path_from_env("OPENHOI_LLM_VERSION", ("ShapeLLM_7B_gapartnet_v1.0",))


def gapartnet_meta() -> Path:
    return _path_from_env("OPENHOI_META_PATH", ("shapellm", "gapartnet_sft_27k_openai.json"))


def gapartnet_pcs() -> Path:
    return _path_from_env("OPENHOI_PCS_PATH", ("shapellm", "gapartnet_pcs"))


def vision_ckpt() -> Path:
    return _path_from_env("OPENHOI_VISION_CKPT", ("zeroshot", "large", "best_lvis.pth"))


def mm_projector() -> Path:
    return _path_from_env("OPENHOI_MM_PROJECTOR", ("shapellm", "7b", "mm_projector.bin"))


def uni3d_checkpoint() -> Path:
    return _path_from_env("OPENHOI_UNI3D_CHECKPOINT", ("uni3d", "modelzoo", "uni3d-b", "model.pt"))


def affdata_train_points() -> Path:
    return _path_from_env("OPENHOI_AFFDATA_TRAIN_POINTS", ("affdata", "point_train_all.txt"))


def affdata_train_json() -> Path:
    return _path_from_env("OPENHOI_AFFDATA_TRAIN_JSON", ("affdata", "json_train_all.txt"))


def affdata_test_points() -> Path:
    return _path_from_env("OPENHOI_AFFDATA_TEST_POINTS", ("affdata", "point_test_all.txt"))


def affdata_test_json() -> Path:
    return _path_from_env("OPENHOI_AFFDATA_TEST_JSON", ("affdata", "json_test_all.txt"))


def afford_grab_pkl() -> Path:
    return _path_from_env("OPENHOI_AFFORD_GRAB_PKL", ("afford", "data_grab.pkl"))


def grab_data_root() -> Path:
    return _path_from_env("OPENHOI_GRAB_DATA_ROOT", ("grab",))


def mllm_checkpoint() -> Path:
    return _path_from_env(
        "OPENHOI_MLLM_CHECKPOINT",
        ("checkpoints", "hoi-affordance-mllm"),
    )

