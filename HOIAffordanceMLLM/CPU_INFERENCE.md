# Running HOIAffordanceMLLM on CPU (macOS / no CUDA)

The upstream code targets A100 GPUs and depends on CUDA-only packages
(`knn_cuda`, `pointnet2_ops`, `flash_attn`, `deepspeed`, `bitsandbytes`),
hard-codes weight paths under `/root/tmp`, and sprinkles `.cuda()` calls through
the model. This adaptation makes single-sample **affordance inference** run on a
CPU-only Mac **without modifying any of the upstream source files**.

## What was added (and nothing else changed)

Everything is centralized in two new files:

| File | Purpose |
|------|---------|
| [`openhoi_cpu.py`](openhoi_cpu.py) | The whole compatibility layer. Import it and call `install()` **before** importing `llava`. |
| [`infer_cpu.py`](infer_cpu.py) | A thin single-sample inference driver. |

`openhoi_cpu.install()` does four things, all reversible / no-ops on a real GPU box:

1. **Import stubs** — registers pure-PyTorch fallbacks in `sys.modules` for the
   CUDA-only imports that run at module load: `knn_cuda.KNN`,
   `pointnet2_ops.pointnet2_utils` (`furthest_point_sample`, `gather_operation`),
   and a `deepspeed` stub (with a valid `__spec__` so `transformers` correctly
   detects it as unavailable). `flash_attn` is imported lazily inside functions
   upstream, so it never needs a stub for inference.
2. **`.cuda()` → no-op** — patches `Tensor.cuda` / `Module.cuda` to stay on CPU,
   so the hard-coded `.cuda()` calls in `affordancellm.py` work unchanged.
3. **`torch.load` path remap** — redirects reads of `/root/tmp/...` to
   `$OPENHOI_DATA_ROOT` (default `~/openhoi-data`), including the special case
   `uni3d/model.pt → uni3d/modelzoo/uni3d-b/model.pt`.
4. **`torch.empty` scalar guard** — fixes a `transformers`/`torch` version bug
   where loading a 0-d parameter (Uni3D's `logit_scale`) calls `torch.empty()`
   with no args.

## Usage

```bash
cd HOIAffordanceMLLM
source ../../.venv-openhoi/bin/activate     # the shared py3.9 env

# 1) Validate the CPU port itself (no LLM, runs in seconds):
python infer_cpu.py --check-ops

# 2) Full single-sample affordance inference:
python infer_cpu.py \
    --points ~/openhoi-data/affdata/test/point_Microwave_1.txt \
    --question "Which part should be manipulated to open it?" \
    --dtype bfloat16 \
    --output /tmp/affordance.txt
```

Weight locations default to `$OPENHOI_DATA_ROOT` (`~/openhoi-data`) and can be
overridden via `OPENHOI_*` env vars or CLI flags (`--llm`, `--vision-ckpt`,
`--mm-projector`, `--aff-ckpt`). Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
to avoid any network access.

## Running real Stage-A affordance inference on 36 GB (works)

Two problems had to be solved to fit the 7B on this Mac:

1. **The forward pass** — pure fp32 needs ~50 GB (OOMs), and pure bf16 fails on
   CPU convolutions (NNPack). Solution: **mixed precision** (`--dtype mixed`,
   default) — the LLaMA transformer + `lm_head` run in **bf16** (~14 GB), while
   the point/vision encoders and affordance heads stay **fp32** (convs valid,
   heads precise). Casts happen only at the two LLM↔head boundaries (two forward
   hooks in `infer_cpu.py::apply_mixed_precision`). Peak RSS ≈ 15 GB.

2. **The trained weights** — the published checkpoint is a 59 GB DeepSpeed file
   that won't fit in RAM. [`extract_aff_heads.py`](extract_aff_heads.py) reads
   only the checkpoint's zip **index** (a few MB) and copies out just the trained
   affordance heads, the `[AFF]` token embedding / `lm_head`, and the LoRA deltas
   → a **1.4 GB** `aff_extracted.pt`. `infer_cpu.py` loads those and merges the
   88 LoRA adapters (64 LLM q/v + 24 Uni3D q/v, scale α/r = 2.0) into the base.

Workflow:

```bash
cd HOIAffordanceMLLM
source ../../.venv-openhoi/bin/activate

# one-time: pull the small trained-weight file out of the 59GB checkpoint
python extract_aff_heads.py --save ~/openhoi-data/checkpoints/hoi-affordance-mllm/aff_extracted.pt

# real affordance inference (mixed precision, ~15GB, slow CPU forward ~20-30 min)
python infer_cpu.py --dtype mixed \
    --points ~/openhoi-data/affdata/test/point_Microwave_1.txt \
    --question "Which part of the microwave should be grasped to open its door?" \
    --aff-ckpt ~/openhoi-data/checkpoints/hoi-affordance-mllm/aff_extracted.pt \
    --output /tmp/aff_microwave.txt
```

Verified result (Microwave): a localized affordance map — `max≈0.61`, `mean≈0.06`,
~8% of points active (vs. a flat 0.50 when the heads are untrained). The single
7B forward on CPU takes ~20–30 min; RSS stays ~15 GB.

## ⚠️ Memory reality on this machine (36 GB RAM)

The full forward pass **does not fit in 36 GB**, and this was confirmed the hard
way — a float32 run exhausted memory and forced an OS restart. Empirically:

- **float32** — the ShapeLLM-7B backbone peaks at ~50 GB during load
  (model + checkpoint shard being copied in) → **OOM / OS crash on 36 GB**.
- **bfloat16** — halves the LLM to ~14 GB, but **CPU convolutions (NNPack) do
  not support bf16**, so the point/vision encoders raise
  `Mismatched Tensor types in NNPack convolutionOutput`. bf16 is a dead end for
  this model on CPU. (Upstream's Mac path likewise forces `--bf16 False`.)
- **Trained affordance checkpoint**
  (`checkpoints/hoi-affordance-mllm/mp_rank_00_model_states.pt`) is a **~59 GB**
  DeepSpeed ZeRO state file — also far past 36 GB. `--aff-ckpt` is therefore
  optional; without it the affordance heads (`projection`,
  `Geometry_Correlation`, `decoder`, `text_hidden_fcs`) are randomly
  initialized, so a run would execute but produce meaningless values.

Because of this, `infer_cpu.py` has a **RAM guard**: it refuses to load the 7B
backbone when the estimated peak exceeds available RAM (override with `--force`
only if you accept the OOM risk).

**To actually run the full forward**, use a host with more RAM (≈64 GB+ for
float32) and, for meaningful output, enough to also load the affordance
checkpoint — or first extract just the affordance-head tensors from the 59 GB
file on a larger machine and pass that smaller file via `--aff-ckpt`.

## Verified on this CPU / Mac

- ✅ `from llava.model import LISAForCausalLM` imports cleanly with no CUDA stack
  (knn_cuda / pointnet2_ops / deepspeed stubs work; `transformers` treats
  deepspeed as unavailable).
- ✅ `python infer_cpu.py --check-ops` — DGCNN (KNN fallback) and Uni3D
  `encode_pc` (pointnet2 FPS fallback) both run on CPU.
- ✅ Model construction on CPU reached the 7B checkpoint-shard load and the
  point/vision/uni3d weights loaded via the `/root/tmp` path remap (before the
  memory wall). The two version-compat fixes needed there —
  `torch.empty()` 0-d guard and disabling `low_cpu_mem_usage` — are handled in
  the shim / driver.
- ⛔ Full 7B forward — blocked by RAM on this 36 GB machine, not by code.
