"""Stage B (Affordance-Driven HOI Diffusion) CPU/macOS inference driver.

Replicates the plain-sampling path of start/create_hoi.py on CPU: text +
object + affordance map -> coarse diffusion motion -> refined hand+object
motion. All CPU/CUDA/pytorch3d compatibility lives in openhoi_cpu_b (imported
first); this driver only orchestrates, so lib/* stays unmodified.

Run from the Affordance-DrivenHOIDiffusion directory (relative config/asset
paths + the data/, checkpoints/ symlinks created by setup_stageb_cpu.sh):

    python infer_stageb_cpu.py --text "Lift the elephant with the right hand." \
        --aff-map /tmp/aff_elephant_1024.txt --output /tmp/hoi_elephant.npz

--aff-map is a text file of 1024 floats aligned to the object's obj.pkl point
cloud (produce it with Stage A run on the SAME cloud; see --dump-obj-pc). If
omitted, a uniform placeholder map is used just to exercise the pipeline.
"""

from __future__ import annotations

import argparse

import openhoi_cpu_b  # must precede any lib.* import

import numpy as np
import torch
from easydict import EasyDict as edict
from hydra import compose, initialize
from omegaconf import OmegaConf


def load_cfg():
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")
    return edict(OmegaConf.to_object(cfg))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", default="Lift the elephant with the right hand.")
    p.add_argument("--aff-map", default="", help="1024 affordance values aligned to the object cloud.")
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--output", default="/tmp/hoi_stageb.npz")
    p.add_argument("--video", default="", help="Also render an mp4 of the interaction here.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dump-obj-pc", default="",
                   help="Write the object's 1024x3 normalized cloud here (feed it to Stage A) and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    device = openhoi_cpu_b.install()
    print(f"[..] device: {device}")

    from lib.models.mano import build_mano_aa
    from lib.networks.clip import load_and_freeze_clip, encoded_text
    from lib.models.object import build_object_model
    from lib.utils.model_utils import (
        build_refiner, build_model_and_diffusion, build_seq_cvae,
        build_mpnet, build_pointnetfeat, build_contact_estimator,
    )
    from lib.utils.demo_utils import get_object_hand_info, get_valid_mask_bunch
    from lib.utils.proc import proc_obj_feat_final, proc_cond_contact_estimator, proc_refiner_input
    import lib.utils.proc as _procmod

    # Upstream bug: the use_scale=False branch of proc_cond_contact_estimator_cov_map
    # references enc_text_expand before it is defined (only the use_scale=True path
    # was ever exercised). The released contact checkpoint needs use_scale=False, so
    # patch in a corrected version centrally rather than editing lib/utils/proc.py.
    def _cond_cov_map_fixed(obj_scale, obj_feat, enc_text, npts, use_scale, affordance_map):
        enc_text_expand = enc_text.unsqueeze(1).expand(-1, npts, -1)
        if use_scale:
            obj_scale_expand2 = obj_scale.unsqueeze(1).unsqueeze(2).expand(-1, npts, -1)
            return torch.cat([obj_scale_expand2, obj_feat, enc_text_expand, affordance_map], dim=2)
        return torch.cat([obj_feat, enc_text_expand, affordance_map], dim=2)

    _procmod.proc_cond_contact_estimator_cov_map = _cond_cov_map_fixed

    cfg = load_cfg()
    # The published contact-estimator checkpoint was trained without the scale
    # term: condition = obj_feat(1088) + text(512) + affordance(1) = 1601. The
    # shipped config defaults (use_scale=True / cond_dim=1602) don't match it.
    cfg.contact.use_scale = False
    cfg.contact.cond_dim = 1601
    data_cfg = cfg.dataset
    text = [args.text]

    print("[..] building models (CPU) ...")
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_cfg.flat_hand)
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_cfg.flat_hand)
    clip_model = load_and_freeze_clip(cfg.clip.clip_version)
    if device.type == "cpu":
        clip_model = clip_model.float()  # CLIP ships fp16; fp16 addmm is unsupported on CPU
    texthom, diffusion = build_model_and_diffusion(cfg, lhand_layer, rhand_layer, test=True)
    seq_cvae = build_seq_cvae(cfg, test=True)
    pointnet = build_pointnetfeat(cfg, test=True)
    pointnet(torch.randn([1, 1024, 3]))  # warm up lazy layers
    contact_estimator = build_contact_estimator(cfg, test=True)
    refiner = build_refiner(cfg, test=True)
    object_model = build_object_model(data_cfg.data_obj_pc_path)
    try:
        mpnet = build_mpnet(cfg)
    except Exception as e:  # sentence-transformers model may be unavailable offline
        print(f"[warn] mpnet unavailable ({e}); using CLIP-only object/hand search")
        mpnet = None

    print(f"[..] resolving object + hand for: {text[0]!r}")
    (is_lhand, is_rhand, obj_pc_org, obj_pc_normal_org, normalized_obj_pc, point_sets,
     obj_cent, obj_scale, obj_verts, obj_faces, obj_top_idx, obj_pc_top_idx) = \
        get_object_hand_info(object_model, clip_model, text, data_cfg.obj_root, data_cfg, mpnet)

    # NOTE: we deliberately do NOT correct the upstream CLIP-similarity hand
    # guess, even though it is unreliable (CLIP text embeddings barely separate
    # "right" vs "left" hand). Faithfully surfacing that failure mode is the point
    # of the evaluation harness in model_eval/.
    bs, npts = normalized_obj_pc.shape[:2]
    print(f"[ok] object cloud: {tuple(normalized_obj_pc.shape)}  lhand={is_lhand} rhand={is_rhand}")

    if args.dump_obj_pc:
        np.savetxt(args.dump_obj_pc, normalized_obj_pc[0].cpu().numpy(), fmt="%.6f")
        print(f"[ok] wrote object cloud ({npts}x3) to {args.dump_obj_pc}; run Stage A on it and pass --aff-map")
        return

    # Affordance map (1, npts, 1) aligned to the object cloud.
    if args.aff_map:
        vals = np.loadtxt(args.aff_map).reshape(-1)
        assert vals.shape[0] == npts, f"aff-map has {vals.shape[0]} values, expected {npts}"
        affordance_map = torch.from_numpy(vals).float().view(1, npts, 1)
        print(f"[ok] loaded affordance map from {args.aff_map} (mean={vals.mean():.3f})")
    else:
        affordance_map = torch.ones(1, npts, 1)
        print("[warn] no --aff-map given; using a uniform placeholder (pipeline test only)")

    enc_text = encoded_text(clip_model, text)
    enc_none_text = encoded_text(clip_model, [""] * bs)
    obj_feat = pointnet(normalized_obj_pc)

    duration = (seq_cvae.decode(enc_text) * 150).long()
    valid_mask_lhand, valid_mask_rhand, valid_mask_obj = get_valid_mask_bunch(
        is_lhand, is_rhand, data_cfg.max_nframes, duration)

    print("[..] contact estimator (affordance -> contact) ...")
    obj_feat_final, est_contact_map = proc_obj_feat_final(
        contact_estimator, obj_scale, obj_cent, obj_feat, enc_text, npts,
        cfg.texthom.use_obj_scale_centroid, cfg.contact.use_scale,
        cfg.texthom.use_contact_feat, affordance_map)

    print("[..] TextHOM diffusion sampling (slow on CPU) ...")
    coarse_x_lhand, coarse_x_rhand, coarse_x_obj = diffusion.sampling(
        texthom, obj_feat_final, enc_text, enc_none_text, data_cfg.max_nframes,
        cfg.texthom.hand_nfeats, cfg.texthom.obj_nfeats,
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
        device=device, guidance_rate=args.guidance)

    if est_contact_map is None:
        cond = proc_cond_contact_estimator(obj_scale, obj_feat, enc_text, npts, cfg.contact.use_scale)
        est_contact_map = (contact_estimator.decode(cond)[..., 0] > 0.5).long()

    print("[..] refiner ...")
    input_lhand, input_rhand, refined_x_obj = proc_refiner_input(
        coarse_x_lhand, coarse_x_rhand, coarse_x_obj, lhand_layer, rhand_layer,
        obj_pc_org, obj_pc_normal_org, valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
        est_contact_map, data_cfg.name, obj_pc_top_idx=obj_pc_top_idx)
    refined_x_lhand, refined_x_rhand = refiner(
        input_lhand, input_rhand, valid_mask_lhand=valid_mask_lhand, valid_mask_rhand=valid_mask_rhand)

    # Convert params -> meshes (for saving + rendering) via the upstream helpers.
    from lib.utils.data import process_hand_result, process_obj_result

    def _lh(): return bool(is_lhand[0]) if hasattr(is_lhand, "__len__") else bool(is_lhand)
    def _rh(): return bool(is_rhand[0]) if hasattr(is_rhand, "__len__") else bool(is_rhand)

    obj_v = obj_verts[0] if isinstance(obj_verts, (list, tuple)) else obj_verts
    obj_verts_tf = process_obj_result(obj_v, refined_x_obj[0], data_cfg.name).detach().cpu().numpy()
    obj_faces_np = np.asarray(obj_faces[0] if isinstance(obj_faces, (list, tuple)) else obj_faces)

    save = dict(
        text=text[0],
        refined_x_lhand=refined_x_lhand[0].detach().cpu().numpy(),
        refined_x_rhand=refined_x_rhand[0].detach().cpu().numpy(),
        refined_x_obj=refined_x_obj[0].detach().cpu().numpy(),
        nframes=int(duration[0].item()),
        obj_verts_tf=obj_verts_tf,
        obj_faces=obj_faces_np,
    )
    if _lh():
        lv, lf = process_hand_result(lhand_layer, refined_x_lhand[0])
        save["lhand_verts"] = lv.detach().cpu().numpy(); save["lhand_faces"] = lf.cpu().numpy()
    if _rh():
        rv, rf = process_hand_result(rhand_layer, refined_x_rhand[0])
        save["rhand_verts"] = rv.detach().cpu().numpy(); save["rhand_faces"] = rf.cpu().numpy()

    np.savez(args.output, **save)
    print(f"[ok] wrote motion to {args.output}  "
          f"(lhand {tuple(refined_x_lhand[0].shape)}, rhand {tuple(refined_x_rhand[0].shape)}, "
          f"obj {tuple(refined_x_obj[0].shape)}, ~{int(duration[0].item())} frames)")

    if args.video:
        import render_hoi
        print(f"[..] rendering video -> {args.video}")
        render_hoi.render_from_npz(args.output, args.video, fps=args.fps)
        print(f"[ok] wrote video to {args.video}")


if __name__ == "__main__":
    main()
