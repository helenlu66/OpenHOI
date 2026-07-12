# Running Stage B (Affordance-Driven HOI Diffusion) on CPU / macOS

Stage B turns a text instruction + object + **affordance map** (from Stage A)
into a hand+object interaction motion sequence. This runs comfortably on CPU
(all models are ≤115 MB); the compatibility work is centralized so upstream
source is untouched.

## New files (nothing upstream changed)

| File | Purpose |
|------|---------|
| [`openhoi_cpu_b.py`](openhoi_cpu_b.py) | The whole compatibility layer. `install()` before importing `lib.*`. |
| [`infer_stageb_cpu.py`](infer_stageb_cpu.py) | Single-sample driver (replicates `create_hoi`'s plain-sampling path on CPU). |
| [`setup_stageb_cpu.sh`](setup_stageb_cpu.sh) | Symlinks the config's relative asset paths to real locations. |

`openhoi_cpu_b.install()` (all CPU-only bits gated behind `not torch.cuda.is_available()`, so it's inert on a GPU box):

1. **pytorch3d stub** — the plain inference path only imports `knn_points` +
   `Meshes` (the functions that use them aren't on this path), so a pure-torch
   stub avoids building pytorch3d on Apple Silicon. Installed only if the real
   pytorch3d is missing.
2. **`.cuda()` / `.to("cuda")` → CPU**, **MPS disabled** (so sentence-transformers
   doesn't put tensors on `mps` and clash), **`torch.load` → `map_location=cpu`**.
3. **NumPy alias restore** (`np.bool`/`np.int`/… removed in NumPy ≥1.24; this
   repo predates that).

The driver also applies two fixes that are **not** CPU-specific (they'd be needed
on GPU too, so they live in the driver, not a source edit):

- `contact.use_scale=False`, `cond_dim=1601` — the shipped config (1602 /
  use_scale=True) does not match the released `contact_estimator.pth` (1665).
- A corrected `proc_cond_contact_estimator_cov_map` — the upstream `use_scale=False`
  branch references `enc_text_expand` before assignment (only the `True` branch
  was ever run). Patched in via monkeypatch.

## Usage

```bash
cd Affordance-DrivenHOIDiffusion
./setup_stageb_cpu.sh                       # one-time: link data/, mano, checkpoints
source ../../.venv-openhoi/bin/activate

python infer_stageb_cpu.py \
    --text "Lift the elephant with the right hand." \
    --aff-map /tmp/aff_elephant_1024.txt \   # Stage A output aligned to the object cloud
    --output /tmp/hoi_elephant.npz
```

Output `.npz`: `refined_x_lhand (T,99)`, `refined_x_rhand (T,99)`,
`refined_x_obj (T,9)` — MANO hand params + object 6-DoF over ~T frames. Without
`--aff-map` a uniform placeholder is used (pipeline test only). Verified on CPU:
"Lift the elephant" → 150-frame motion, diffusion ~1000 steps at ~18 it/s.

## True end-to-end (aligned affordance)

The affordance map must align **index-for-index** with the 1024-point object
cloud Stage B uses, so Stage A is run on *that exact cloud* (no resampling):

```bash
# 1) Stage B: export the object's 1024x3 cloud
python infer_stageb_cpu.py --text "Lift the elephant with the right hand." \
    --dump-obj-pc /tmp/elephant_1024.txt

# 2) Stage A: affordance on those exact points (see ../HOIAffordanceMLLM)
python ../HOIAffordanceMLLM/infer_cpu.py --dtype mixed \
    --raw-points /tmp/elephant_1024.txt \
    --aff-ckpt ~/openhoi-data/checkpoints/hoi-affordance-mllm/aff_extracted.pt \
    --question "Which part of the elephant should be grasped to lift it up?" \
    --output /tmp/aff_elephant_1024.txt

# 3) Stage B: generate motion conditioned on the real affordance
python infer_stageb_cpu.py --text "Lift the elephant with the right hand." \
    --aff-map /tmp/aff_elephant_1024.txt --output /tmp/hoi_elephant.npz
```
