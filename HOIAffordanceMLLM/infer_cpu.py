"""Minimal CPU/macOS inference entry point for HOIAffordanceMLLM.

This is a thin driver that runs a single affordance-prediction forward pass on
CPU. All CPU/CUDA compatibility lives in ``openhoi_cpu`` (imported first, before
any ``llava`` import); this file only builds the model and feeds it one sample,
so the upstream package is used unmodified.

Examples
--------
Cheap self-test of the CPU geometry ops (no 7B LLM, runs in seconds)::

    python infer_cpu.py --check-ops

Full single-sample affordance inference (loads the 7B backbone, needs lots of
RAM -- see the note at the bottom of this file)::

    python infer_cpu.py \
        --points ~/openhoi-data/affdata/test/point_Microwave_1.txt \
        --question "Which part should be manipulated to open it?" \
        --output /tmp/affordance_Microwave_1.txt

Weight locations default to ``$OPENHOI_DATA_ROOT`` (``~/openhoi-data``) and can
be overridden with the ``OPENHOI_*`` environment variables or the CLI flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# MUST come before importing torch/llava so the CUDA-only imports resolve and
# the .cuda()/torch.load patches are in place.
import openhoi_cpu

import numpy as np
import torch


def data_root() -> Path:
    return openhoi_cpu.data_root()


def parse_args() -> argparse.Namespace:
    root = data_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-ops", action="store_true",
                   help="Run only the CPU geometry-op self-test (no LLM) and exit.")
    p.add_argument("--llm", default=os.environ.get("OPENHOI_LLM_VERSION",
                   str(root / "ShapeLLM_7B_gapartnet_v1.0")),
                   help="Path to the ShapeLLM-7B backbone directory.")
    p.add_argument("--vision-ckpt", default=os.environ.get("OPENHOI_VISION_CKPT",
                   str(root / "zeroshot" / "large" / "best_lvis.pth")))
    p.add_argument("--mm-projector", default=os.environ.get("OPENHOI_MM_PROJECTOR",
                   str(root / "shapellm" / "7b" / "mm_projector.bin")))
    p.add_argument("--aff-ckpt", default=os.environ.get("OPENHOI_AFF_CKPT", ""),
                   help="Optional trained HOIAffordanceMLLM checkpoint (state_dict "
                        "or DeepSpeed 'mp_rank_00_model_states.pt'). Without it the "
                        "affordance heads are randomly initialised.")
    p.add_argument("--points", default=str(root / "affdata" / "test" / "point_Microwave_1.txt"),
                   help="Point-cloud .txt file (space separated; cols 2:5 are xyz).")
    p.add_argument("--raw-points", default="",
                   help="Nx3 xyz file used AS-IS (no resample/renormalize), so the "
                        "output affordance aligns index-for-index with these points. "
                        "Use this to consume Stage B's dumped 1024-pt object cloud.")
    p.add_argument("--question", default="Which part of this object should be manipulated?")
    p.add_argument("--answer", default="The affordance region is [AFF].",
                   help="Assistant turn; must contain the [AFF] token.")
    p.add_argument("--sample-points", type=int, default=2048)
    p.add_argument("--dtype", choices=["mixed", "float32", "bfloat16"], default="mixed",
                   help="'mixed' (default): LLaMA transformer in bf16 (~14GB), point/"
                        "vision/affordance heads in fp32 -- fits in ~18GB and keeps CPU "
                        "convs valid. 'float32' needs ~50GB. 'bfloat16' fails on CPU convs.")
    p.add_argument("--output", default="", help="Where to write per-point affordance scores.")
    p.add_argument("--force", action="store_true",
                   help="Skip the RAM safety check and load the 7B backbone anyway.")
    return p.parse_args()


def _total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError):
        return 0.0


def check_memory(dtype: str, force: bool) -> None:
    """Refuse to load the 7B backbone if it is likely to exhaust RAM.

    Loading the ShapeLLM-7B backbone in float32 peaks at ~50 GB (model + the
    checkpoint shard being copied in). On a smaller machine this hard-crashes
    the OS, so we bail out early unless --force is given.
    """
    total = _total_ram_gb()
    if total <= 0:
        return
    peak_gb = {"float32": 50.0, "mixed": 24.0, "bfloat16": 24.0}[dtype]
    if dtype == "bfloat16":
        print("[warn] pure bfloat16 fails on CPU convolutions (NNPack). Use --dtype mixed "
              "for a CPU-friendly bf16-LLM + fp32-heads run.")
    if not force and peak_gb > total - 4:
        sys.exit(
            f"[abort] loading the 7B backbone in {dtype} needs ~{peak_gb:.0f} GB peak, "
            f"but this machine has only {total:.0f} GB RAM.\n"
            f"        Running it anyway will exhaust memory and can crash the OS.\n"
            f"        Use --check-ops to validate the CPU port, run on a larger host,\n"
            f"        or pass --force to override this guard.")


# --------------------------------------------------------------------------- #
# CPU geometry-op self-test (validates the knn_cuda / pointnet2_ops fallbacks)
# --------------------------------------------------------------------------- #

def check_ops() -> None:
    torch.manual_seed(0)
    from llava.model.language_model.affordancellm import DGCNN_Propagation
    from llava.model.Uni3D.models import uni3d

    dg = DGCNN_Propagation(k=4).eval()
    with torch.no_grad():
        out = dg(torch.randn(1, 3, 32), torch.randn(1, 768, 32),
                 torch.randn(1, 3, 64), torch.randn(1, 768, 64))
    print(f"[ok] DGCNN_Propagation (KNN fallback) -> {tuple(out.shape)}")

    model = uni3d.create_uni3d().eval()
    with torch.no_grad():
        feats = model.encode_pc(torch.randn(1, 2048, 6))
    print(f"[ok] Uni3D.encode_pc (pointnet2 FPS fallback) -> {len(feats)} tensors")
    print("CPU op self-test PASSED")


# --------------------------------------------------------------------------- #
# Point-cloud loading (mirrors ReasonSegDataset.extract_point_file + pc_normalize)
# --------------------------------------------------------------------------- #

def load_raw_point_cloud(path: str) -> torch.Tensor:
    """Load an Nx3 cloud as-is (order preserved, no resample/renormalize) so the
    resulting affordance aligns index-for-index with the given points."""
    arr = np.loadtxt(path, dtype=np.float32)
    xyz = arr[:, :3]
    return torch.from_numpy(xyz.T).float().unsqueeze(0)  # (1, 3, N)


def load_point_cloud(path: str, num_points: int) -> torch.Tensor:
    """Return xyz points as a (1, 3, num_points) float32 tensor."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append([float(x) for x in parts[2:]])  # drop the 2 index columns
    arr = np.asarray(rows, dtype=np.float32)
    xyz = arr[:, 0:3]

    # resample to a fixed count (replicates the fixed 2048/N sampling upstream uses)
    n = xyz.shape[0]
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.concatenate([np.arange(n), np.random.choice(n, num_points - n, replace=True)])
    xyz = xyz[idx]

    # pc_normalize: center + unit sphere
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if scale > 0:
        xyz = xyz / scale

    pts = torch.from_numpy(xyz.T).float().unsqueeze(0)  # (1, 3, N)
    return pts


# --------------------------------------------------------------------------- #
# Model construction (the CPU-safe subset of inference.py::train())
# --------------------------------------------------------------------------- #

# Submodules kept in fp32 in 'mixed' mode (everything except the LLaMA
# transformer + lm_head). CPU convs live in point_model/vision_tower and require
# fp32; the affordance heads want fp32 precision.
_MIXED_FP32_MODULES = [
    "model.vision_tower", "model.mm_projector", "model.point_model", "model.text_hidden_fcs",
    "projection", "Geometry_Correlation", "propagation_2", "propagation_1", "propagation_0",
    "dgcnn_pro_1", "dgcnn_pro_2", "decoder",
]


def _get_submodule(model, dotted):
    obj = model
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def apply_mixed_precision(model) -> None:
    """LLaMA transformer + lm_head stay bf16; listed submodules go fp32, with
    dtype casts at the two LLM<->head boundaries."""
    for name in _MIXED_FP32_MODULES:
        _get_submodule(model, name).float()

    # Point tokens leave the fp32 mm_projector -> must be bf16 to enter the LLM.
    _get_submodule(model, "model.mm_projector").register_forward_hook(
        lambda _m, _i, out: out.to(torch.bfloat16))

    # LLM hidden states (bf16) enter the fp32 text_hidden_fcs head -> cast to fp32.
    def _to_fp32_pre(_module, args):
        return tuple(a.float() if torch.is_tensor(a) else a for a in args)

    _get_submodule(model, "model.text_hidden_fcs")[0].register_forward_pre_hook(_to_fp32_pre)


def build_model(args, device):
    import transformers
    from llava import conversation as conversation_lib

    load_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.llm, model_max_length=2048, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[AFF]")
    seg_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    from llava.model import LISAForCausalLM

    print(f"[..] loading ShapeLLM-7B backbone in {args.dtype} (this is the slow / memory-heavy step)")
    # NOTE: low_cpu_mem_usage is intentionally OFF. Large parts of this model
    # (the Uni3D point backbone and the affordance heads) are not in the LLM
    # checkpoint; with low_cpu_mem_usage they stay as meta tensors and cannot
    # later be moved to a device.
    model = LISAForCausalLM.from_pretrained(
        args.llm,
        seg_token_idx=seg_token_idx,
        use_mm_start_end=True,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.dtype == "mixed":
        apply_mixed_precision(model)
        # point_model's Uni3D weights were loaded (bf16-truncated) during __init__;
        # reload them at full fp32 now that the module is fp32.
        uni3d_ck = torch.load("/root/tmp/uni3d/model.pt", map_location="cpu")  # remapped by openhoi_cpu
        sd = {k.replace("module.", ""): v for k, v in uni3d_ck["module"].items()}
        model.model.point_model.load_state_dict(sd)

    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]

    model_args = SimpleNamespace(
        vision_tower="ReConV2/cfgs/pretrain/large/openshape.yaml",
        vision_tower_path=args.vision_ckpt,
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        pretrain_mm_mlp_adapter=args.mm_projector,
        mm_projector_type="mlp2x_gelu",
        mm_use_pt_start_end=False,
        mm_use_pt_patch_token=False,
        prompt_token_num=32,
        with_ape=True, with_local=True, with_global=True, with_color=True,
    )

    model.get_model().initialize_vision_modules(model_args, fsdp=None)
    model.get_vision_tower().load_model()
    model.config.mm_use_pt_start_end = False
    model.config.mm_use_pt_patch_token = False
    model.config.with_color = True
    model.config.sample_points_num = args.sample_points
    model.initialize_vision_tokenizer(model_args, tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    if args.aff_ckpt:
        load_affordance_checkpoint(model, args.aff_ckpt)

    # Do NOT force a global dtype in mixed mode (it would collapse the bf16/fp32 split).
    model = model.to(device)
    if args.dtype != "mixed":
        model = model.to(load_dtype)
    model.eval()
    return model, tokenizer, seg_token_idx


def load_affordance_checkpoint(model, ckpt_path: str, lora_alpha: int = 16, lora_r: int = 8) -> None:
    """Load the trained affordance heads + [AFF] embeddings and merge LoRA.

    Expects the small file produced by extract_aff_heads.py (keys already have
    the PEFT `base_model.model.` prefix stripped)."""
    print(f"[..] loading affordance checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "module" in sd:  # raw DeepSpeed file, not the extract
        sys.exit("[error] --aff-ckpt points at the raw 59GB DeepSpeed file. Run "
                 "extract_aff_heads.py first and pass the small aff_extracted.pt.")

    lora = {k: v for k, v in sd.items() if ".lora_A." in k or ".lora_B." in k}
    base = {k: v for k, v in sd.items() if k not in lora}

    filled = sum(1 for k in base if k in dict(model.named_parameters())
                 or k in dict(model.named_buffers()))
    model.load_state_dict(base, strict=False)
    print(f"[ok] loaded {filled}/{len(base)} head/embedding tensors")

    # Merge LoRA deltas (W += (alpha/r) * B @ A) into the base q/v projections.
    scaling = lora_alpha / lora_r
    params = dict(model.named_parameters())
    merged = 0
    with torch.no_grad():
        for ak in (k for k in lora if ".lora_A." in k):
            bk = ak.replace(".lora_A.", ".lora_B.")
            target = ak.split(".lora_A.")[0] + ".weight"
            if bk in lora and target in params:
                delta = (lora[bk].float() @ lora[ak].float()) * scaling
                params[target].add_(delta.to(params[target].dtype))
                merged += 1
    print(f"[ok] merged {merged} LoRA adapters (scale={scaling})")


# --------------------------------------------------------------------------- #
# Single-sample forward
# --------------------------------------------------------------------------- #

def run_inference(args, device) -> None:
    from llava.constants import DEFAULT_POINT_TOKEN
    from llava.mm_utils import tokenizer_point_token
    from llava import conversation as conversation_lib

    model, tokenizer, _ = build_model(args, device)

    if args.raw_points:
        points = load_raw_point_cloud(args.raw_points).to(device)
    else:
        points = load_point_cloud(args.points, args.sample_points).to(device)
    n_pts = points.shape[-1]

    conv = conversation_lib.conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_POINT_TOKEN + "\n" + args.question)
    conv.append_message(conv.roles[1], args.answer)
    prompt = conv.get_prompt()

    input_ids = tokenizer_point_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    labels = input_ids.clone()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).to(device)
    offset = torch.tensor([0, 1], dtype=torch.long, device=device)
    aff_label = torch.zeros((1, n_pts), dtype=torch.float32, device=device)
    logist_label = [torch.zeros(1, device=device)]

    print("[..] running forward pass ...")
    with torch.no_grad():
        _loss, pred_affordance, _ = model(
            points=points, input_ids=input_ids, labels=labels,
            attention_masks=attention_masks, offset=offset,
            aff_label=aff_label, logist_label=logist_label,
        )
    scores = pred_affordance[0].float().reshape(-1).cpu().numpy()
    print(f"[ok] affordance scores: n={scores.size} "
          f"min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}")

    if args.output:
        np.savetxt(args.output, scores, fmt="%.6f")
        print(f"[ok] wrote per-point scores to {args.output}")


def main() -> None:
    args = parse_args()
    device = openhoi_cpu.install()
    print(f"[..] device: {device}")

    if args.check_ops:
        check_ops()
        return

    for label, path in [("LLM", args.llm), ("vision", args.vision_ckpt), ("mm_projector", args.mm_projector)]:
        if not Path(path).exists():
            sys.exit(f"[error] {label} path not found: {path}\n"
                     f"        Set OPENHOI_DATA_ROOT or pass the matching --flag.")
    check_memory(args.dtype, args.force)
    run_inference(args, device)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Memory note
# -----------------------------------------------------------------------------
# The 7B backbone is ~28 GB in float32 (~14 GB in bfloat16). The published
# trained checkpoint (checkpoints/hoi-affordance-mllm/mp_rank_00_model_states.pt)
# is a ~59 GB DeepSpeed ZeRO state file and cannot be fully materialised on a
# 36 GB machine. Use `--check-ops` to validate the CPU port itself, and run the
# full forward on a host with enough RAM (or with `--dtype bfloat16`, no
# `--aff-ckpt`) to smoke-test the pipeline.
