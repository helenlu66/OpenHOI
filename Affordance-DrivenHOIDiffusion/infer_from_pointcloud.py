"""OpenHOI Affordance-Driven HOI Diffusion inference from an externally supplied
(e.g. RaiSim-settled) object point cloud, dumping raw hand/object trajectories
for playback elsewhere instead of rendering a video.

Companion to demo/demo.py: reuses the same model-building and diffusion/refiner
pipeline, but the object geometry is loaded from a JSON pose file + the
dataset's own object mesh (transformed into that pose) rather than picked
automatically by CLIP text search. This lets a caller pass "the object as it
actually sits in a physics sim" instead of the dataset's canonical pose.

The model itself still predicts object/hand trajectories in its own learned
(GRAB capture-space) coordinate convention regardless of the input point
cloud's world pose -- verified empirically, frame-0 output is nowhere near
the input pose. So after generation we rigidly align frame 0 of the
predicted object pose onto the caller's settled pose and apply that same
rigid transform to every frame of the object *and* both hands (they share
one generation-space frame), producing a trajectory anchored to the caller's
world frame.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

import hydra
import numpy as np
import torch
import trimesh
from easydict import EasyDict as edict
from omegaconf import OmegaConf

from lib.device_utils import get_inference_device
from lib.models.mano import build_mano_aa
from lib.networks.clip import encoded_text, encoded_text_normalized, load_and_freeze_clip
from lib.utils.demo_utils import get_valid_mask_bunch, search_hand
from lib.utils.proc import pc_normalize, proc_obj_feat_final_old, proc_refiner_input
from lib.utils.proc_output import get_hand_joints_w_tip
from lib.utils.rot import rot6d_to_rotmat
from lib.utils.model_utils import (
    build_contact_estimator,
    build_model_and_diffusion,
    build_mpnet,
    build_pointnetfeat,
    build_refiner,
    build_seq_cvae,
)


def _load_settled_pose(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    position = np.array(data["position"], dtype=np.float64)
    quat_wxyz = np.array(data["quaternion"], dtype=np.float64)
    w, x, y, z = quat_wxyz
    rotmat = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotmat, position


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config) -> None:
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)

    device = get_inference_device()
    print(f"Using device: {device}")
    if device.type == "cpu":
        os.environ.setdefault("OPENHOI_FORCE_CPU", "1")

    data_config = config.dataset
    dataset_name = data_config.name
    max_nframes = data_config.max_nframes
    hand_nfeats = config.texthom.hand_nfeats
    obj_nfeats = config.texthom.obj_nfeats

    text = config.text if isinstance(config.text, str) else config.text[0]
    object_name = config.object_name
    settled_pose_json = config.settled_pose_json
    out_npz = config.out_npz
    force_both_hands = bool(config.get("force_both_hands", True))
    # TextHOM hardcodes obj_dim=2048=1024(global feat)+1024(per-point contact
    # map) whenever use_contact_feat is set, so the sampled point cloud must
    # have exactly 1024 points regardless of the README's general "2048"
    # point-cloud-preprocessing guidance (that applies elsewhere).
    npoints = int(config.get("npoints", 1024))

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).to(device)
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).to(device)

    refiner = build_refiner(config, test=True)
    texthom, diffusion = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    clip_model = load_and_freeze_clip(config.clip.clip_version).to(device)
    mpnet = build_mpnet(config)
    seq_cvae = build_seq_cvae(config, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    contact_estimator = build_contact_estimator(config, test=True)

    dump_data = torch.randn([1, 1024, 3], device=device)
    pointnet(dump_data)

    settled_rotmat, settled_position = _load_settled_pose(settled_pose_json)

    obj_file = osp.join(data_config.obj_root, f"{object_name}.ply")
    obj_mesh_canonical = trimesh.load(obj_file, maintain_order=True)
    verts_canonical = np.asarray(obj_mesh_canonical.vertices.copy())
    verts_settled = verts_canonical @ settled_rotmat.T + settled_position
    obj_mesh_settled = trimesh.Trimesh(
        vertices=verts_settled, faces=obj_mesh_canonical.faces, process=False
    )
    print(
        f"object={object_name} settled_position={settled_position} "
        f"mesh_bounds={obj_mesh_settled.bounds.tolist()}"
    )

    sampled_points, face_index = trimesh.sample.sample_surface(obj_mesh_settled, npoints)
    sampled_normals = obj_mesh_settled.face_normals[face_index]

    normalized_pc, obj_cent, obj_scale = pc_normalize(
        np.asarray(sampled_points), return_params=True
    )

    obj_pc = torch.from_numpy(np.asarray(sampled_points)).float().to(device).unsqueeze(0)
    obj_pc_normal = torch.from_numpy(np.asarray(sampled_normals)).float().to(device).unsqueeze(0)
    normalized_obj_pc = torch.from_numpy(normalized_pc).float().to(device).unsqueeze(0)
    obj_cent_t = torch.from_numpy(obj_cent).float().to(device).unsqueeze(0)
    obj_scale_t = torch.tensor([obj_scale], dtype=torch.float32, device=device)

    if force_both_hands:
        is_lhand, is_rhand = 1, 1
        print("hand type: both hands (forced)")
    else:
        text_feat_clip = encoded_text_normalized(clip_model, [text])[0]
        text_feat_mpnet = mpnet.encode([text], convert_to_tensor=True)[0]
        is_lhand, is_rhand = search_hand(clip_model, mpnet, text_feat_clip, text_feat_mpnet)

    bs, npts = normalized_obj_pc.shape[:2]
    enc_text = encoded_text(clip_model, [text])
    enc_none_text = encoded_text(clip_model, [""] * bs)
    obj_feat = pointnet(normalized_obj_pc)

    obj_feat_final, est_contact_map = proc_obj_feat_final_old(
        contact_estimator,
        obj_scale_t,
        obj_cent_t,
        obj_feat,
        enc_text,
        npts,
        config.texthom.use_obj_scale_centroid,
        config.contact.use_scale,
        config.texthom.use_contact_feat,
    )

    duration = seq_cvae.decode(enc_text)
    duration *= 150
    duration = duration.long()
    valid_mask_lhand, valid_mask_rhand, valid_mask_obj = get_valid_mask_bunch(
        [is_lhand], [is_rhand], max_nframes, duration
    )

    with torch.no_grad():
        coarse_x_lhand, coarse_x_rhand, coarse_x_obj = diffusion.sampling(
            texthom,
            obj_feat_final,
            enc_text,
            enc_none_text,
            max_nframes,
            hand_nfeats,
            obj_nfeats,
            valid_mask_lhand,
            valid_mask_rhand,
            valid_mask_obj,
            device=device,
        )

        input_lhand, input_rhand, refined_x_obj = proc_refiner_input(
            coarse_x_lhand,
            coarse_x_rhand,
            coarse_x_obj,
            lhand_layer,
            rhand_layer,
            obj_pc,
            obj_pc_normal,
            valid_mask_lhand,
            valid_mask_rhand,
            valid_mask_obj,
            est_contact_map,
            dataset_name,
            obj_pc_top_idx=None,
        )

        refined_x_lhand, refined_x_rhand = refiner(
            input_lhand,
            input_rhand,
            valid_mask_lhand=valid_mask_lhand,
            valid_mask_rhand=valid_mask_rhand,
        )

        text_duration = duration[0].item()
        refined_x_obj_sampled = refined_x_obj[0][:text_duration]
        obj_trans = refined_x_obj_sampled[:, :3].detach().cpu().numpy()
        obj_rotmat = (
            rot6d_to_rotmat(refined_x_obj_sampled[:, 3:9]).reshape(-1, 3, 3).detach().cpu().numpy()
        )

        lhand_joints = None
        rhand_joints = None
        if is_lhand:
            lhand_joints = (
                get_hand_joints_w_tip(refined_x_lhand[:, :text_duration], lhand_layer)[0]
                .detach()
                .cpu()
                .numpy()
            )
        if is_rhand:
            rhand_joints = (
                get_hand_joints_w_tip(refined_x_rhand[:, :text_duration], rhand_layer)[0]
                .detach()
                .cpu()
                .numpy()
            )

    print(
        f"generated {text_duration} frames; frame-0 obj_trans={obj_trans[0]} "
        f"(compare to settled_position={settled_position})"
    )

    # The model predicts trans/rot in its own learned (GRAB capture-space)
    # convention -- feeding it a world-frame point cloud does NOT make its
    # output world-anchored (verified empirically: frame-0 obj_trans is
    # nowhere near settled_position). Rigidly align frame 0 of the generated
    # object pose to our RaiSim settled pose, and apply that same rigid
    # transform to every frame of both the object and the hand joints (they
    # share one generation-space frame), so the whole clip lands correctly on
    # the table instead of floating at the model's native coordinates.
    align_R = settled_rotmat @ obj_rotmat[0].T
    align_t = settled_position - align_R @ obj_trans[0]

    obj_trans = obj_trans @ align_R.T + align_t
    obj_rotmat = np.einsum("ij,tjk->tik", align_R, obj_rotmat)
    if lhand_joints is not None:
        lhand_joints = lhand_joints @ align_R.T + align_t
    if rhand_joints is not None:
        rhand_joints = rhand_joints @ align_R.T + align_t

    print(f"  aligned frame-0 obj_trans={obj_trans[0]} (should match settled_position)")

    os.makedirs(osp.dirname(osp.abspath(out_npz)), exist_ok=True)
    np.savez(
        out_npz,
        obj_trans=obj_trans,
        obj_rotmat=obj_rotmat,
        lhand_joints=lhand_joints if lhand_joints is not None else np.zeros((0, 21, 3)),
        rhand_joints=rhand_joints if rhand_joints is not None else np.zeros((0, 21, 3)),
        is_lhand=bool(is_lhand),
        is_rhand=bool(is_rhand),
        fps=config.fps,
        text=text,
        object_name=object_name,
    )
    print(f"Saved trajectory: {out_npz}")


if __name__ == "__main__":
    main()
