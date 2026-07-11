#!/usr/bin/env bash
# OpenHOI path defaults. Source from PhysAwareHOI scripts/paths.env.sh or directly.
# Override any value by exporting it before sourcing.

if [ -n "${BASH_VERSION:-}" ]; then
  _openhoi_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _openhoi_env_dir="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
  _openhoi_env_dir="$(cd "$(dirname "$0")" && pwd)"
fi
export OPENHOI_ROOT="${OPENHOI_ROOT:-$(cd "${_openhoi_env_dir}/.." && pwd)}"
export OPENHOI_DATA_ROOT="${OPENHOI_DATA_ROOT:-${HOME}/openhoi-data}"
export OPENHOI_LOG_DIR="${OPENHOI_LOG_DIR:-${OPENHOI_DATA_ROOT}/log_dir}"
export OPENHOI_VENV="${OPENHOI_VENV:-${OPENHOI_ROOT}/../.venv-openhoi}"

export OPENHOI_LLM_VERSION="${OPENHOI_LLM_VERSION:-${OPENHOI_DATA_ROOT}/ShapeLLM_7B_gapartnet_v1.0}"
export OPENHOI_META_PATH="${OPENHOI_META_PATH:-${OPENHOI_DATA_ROOT}/shapellm/gapartnet_sft_27k_openai.json}"
export OPENHOI_PCS_PATH="${OPENHOI_PCS_PATH:-${OPENHOI_DATA_ROOT}/shapellm/gapartnet_pcs}"
export OPENHOI_VISION_CKPT="${OPENHOI_VISION_CKPT:-${OPENHOI_DATA_ROOT}/zeroshot/large/best_lvis.pth}"
export OPENHOI_MM_PROJECTOR="${OPENHOI_MM_PROJECTOR:-${OPENHOI_DATA_ROOT}/shapellm/7b/mm_projector.bin}"
# Fix uni3d path: HF stores under modelzoo/uni3d-b/model.pt
export OPENHOI_UNI3D_CHECKPOINT="${OPENHOI_UNI3D_CHECKPOINT:-${OPENHOI_DATA_ROOT}/uni3d/modelzoo/uni3d-b/model.pt}"
export OPENHOI_AFFDATA_TRAIN_POINTS="${OPENHOI_AFFDATA_TRAIN_POINTS:-${OPENHOI_DATA_ROOT}/affdata/point_train_all.txt}"
export OPENHOI_AFFDATA_TRAIN_JSON="${OPENHOI_AFFDATA_TRAIN_JSON:-${OPENHOI_DATA_ROOT}/affdata/json_train_all.txt}"
export OPENHOI_AFFDATA_TEST_POINTS="${OPENHOI_AFFDATA_TEST_POINTS:-${OPENHOI_DATA_ROOT}/affdata/point_test_all.txt}"
export OPENHOI_AFFDATA_TEST_JSON="${OPENHOI_AFFDATA_TEST_JSON:-${OPENHOI_DATA_ROOT}/affdata/json_test_all.txt}"
export OPENHOI_AFFORD_GRAB_PKL="${OPENHOI_AFFORD_GRAB_PKL:-${OPENHOI_DATA_ROOT}/afford/data_grab.pkl}"
export OPENHOI_GRAB_DATA_ROOT="${OPENHOI_GRAB_DATA_ROOT:-${OPENHOI_DATA_ROOT}/grab}"
export OPENHOI_CHECKPOINTS_ROOT="${OPENHOI_CHECKPOINTS_ROOT:-${OPENHOI_DATA_ROOT}/checkpoints}"
export OPENHOI_HOI_DATA_ROOT="${OPENHOI_HOI_DATA_ROOT:-${OPENHOI_DATA_ROOT}/data}"
export OPENHOI_LORA_CHECKPOINT="${OPENHOI_LORA_CHECKPOINT:-${OPENHOI_ROOT}/HOIAffordanceMLLM/checkpoints/shapellm-7b-gapartnet-v1.0-lora}"
export OPENHOI_MLLM_CHECKPOINT="${OPENHOI_MLLM_CHECKPOINT:-${OPENHOI_CHECKPOINTS_ROOT}/hoi-affordance-mllm}"

export PYTHONPATH="${OPENHOI_ROOT}:${PYTHONPATH:-}"
