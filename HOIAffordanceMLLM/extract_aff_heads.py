"""Extract just the trained affordance-head tensors from OpenHOI's 59 GB
DeepSpeed checkpoint, without loading the whole file into RAM.

A torch checkpoint is a zip archive: a small `data.pkl` index describes every
tensor and points at raw storage blobs in `.../data/<n>`. We unpickle ONLY the
index (a few MB) with stubbed `persistent_load` / `_rebuild_tensor` so no tensor
bytes are read, then copy out just the storages whose parameter name matches the
affordance heads. Peak RAM stays tiny.

Usage:
    python extract_aff_heads.py --list          # print matching keys + shapes
    python extract_aff_heads.py --save aff_heads.pt
"""

from __future__ import annotations

import argparse
import pickle
import zipfile
from functools import reduce

import torch

# Everything trained on top of the frozen backbones plus the pieces the [AFF]
# token needs (its trained embedding row / lm_head) and the LoRA deltas on the
# LLM attention projections. All are small except embed_tokens/lm_head (~0.5 GB).
HEAD_KEYS = (
    "projection", "Geometry_Correlation", "decoder",
    "text_hidden_fcs", "propagation_", "dgcnn_pro_",
    "embed_tokens", "lm_head", ".lora_A.", ".lora_B.",
)

# PEFT wraps the model as base_model.model.<...>; strip that to match a plain
# LISAForCausalLM state_dict.
_PEFT_PREFIX = "base_model.model."

_STORAGE_DTYPE = {
    "FloatStorage": torch.float32, "HalfStorage": torch.float16,
    "BFloat16Storage": torch.bfloat16, "DoubleStorage": torch.float64,
    "LongStorage": torch.int64, "IntStorage": torch.int32,
    "ShortStorage": torch.int16, "CharStorage": torch.int8,
    "ByteStorage": torch.uint8, "BoolStorage": torch.bool,
}


class _StorageStub:
    """Placeholder returned by persistent_load; carries no data."""
    def __init__(self, key, dtype):
        self.key = key
        self.dtype = dtype


class _TensorDesc:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = stride


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_a, **_k):
    return _TensorDesc(storage, storage_offset, size, stride)


class _IndexUnpickler(pickle.Unpickler):
    """Unpickles the checkpoint index, stubbing tensor reconstruction."""

    def persistent_load(self, pid):
        # pid == ('storage', storage_type, key, location, numel)
        _tag, storage_type, key, _location, _numel = pid
        name = getattr(storage_type, "__name__", str(storage_type))
        dtype = _STORAGE_DTYPE.get(name, torch.float32)
        return _StorageStub(str(key), dtype)

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if name == "_rebuild_parameter":
            return lambda desc, *_a, **_k: desc
        if name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        try:
            return super().find_class(module, name)
        except Exception:
            # Any exotic class in the index we don't need -> harmless stand-in.
            return lambda *a, **k: None


def _find_module_dict(obj):
    """DeepSpeed model_states root is a dict; the weights live under 'module'."""
    if isinstance(obj, dict):
        if "module" in obj and isinstance(obj["module"], dict):
            return obj["module"]
        return obj
    return {}


def load_index(pt_path: str):
    zf = zipfile.ZipFile(pt_path)
    prefix = zf.namelist()[0].split("/")[0]
    with zf.open(f"{prefix}/data.pkl") as f:
        root = _IndexUnpickler(f).load()
    return zf, prefix, _find_module_dict(root)


def read_tensor(zf, prefix, desc: "_TensorDesc") -> torch.Tensor:
    raw = zf.read(f"{prefix}/data/{desc.storage.key}")
    flat = torch.frombuffer(bytearray(raw), dtype=desc.storage.dtype)
    n = reduce(lambda a, b: a * b, desc.size, 1)
    return flat[desc.offset: desc.offset + n].reshape(desc.size).clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/Users/luh1/openhoi-data/checkpoints/"
                    "hoi-affordance-mllm/mp_rank_00_model_states.pt")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    zf, prefix, module = load_index(args.ckpt)
    matches = {k: v for k, v in module.items()
               if isinstance(v, _TensorDesc) and any(s in k for s in HEAD_KEYS)}
    print(f"[ok] index parsed: {len(module)} params total, {len(matches)} match head keys")

    if args.list:
        for k in sorted(matches):
            d = matches[k]
            print(f"  {k:60s} {str(d.size):20s} {d.storage.dtype}")
        return

    if args.save:
        out = {}
        for k, d in matches.items():
            key = k[len(_PEFT_PREFIX):] if k.startswith(_PEFT_PREFIX) else k
            out[key] = read_tensor(zf, prefix, d)
        torch.save(out, args.save)
        nbytes = sum(t.numel() * t.element_size() for t in out.values())
        print(f"[ok] saved {len(out)} tensors ({nbytes/1e6:.1f} MB) to {args.save}")


if __name__ == "__main__":
    main()
