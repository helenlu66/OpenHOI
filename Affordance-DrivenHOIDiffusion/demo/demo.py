"""OpenHOI Affordance-Driven HOI Diffusion demo (README demo.sh equivalent)."""

from __future__ import annotations

import os
import os.path as osp
import re
import sys

sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import hydra
import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf

from lib.device_utils import get_inference_device
from lib.models.mano import build_mano_aa
from lib.models.object import build_object_model
from lib.networks.clip import encoded_text, load_and_freeze_clip
from lib.utils.demo_utils import get_object_hand_info, get_valid_mask_bunch, proc_results
from lib.utils.file import save_video
from lib.utils.model_utils import (
    build_contact_estimator,
    build_model_and_diffusion,
    build_mpnet,
    build_pointnetfeat,
    build_refiner,
    build_seq_cvae,
)
from lib.utils.proc import proc_obj_feat_final_old, proc_refiner_input

try:
    import pytorch3d  # noqa: F401
    from lib.utils.renderer import Renderer
    from lib.utils.visualize import render_videos
    _HAS_PYTORCH3D = True
except ImportError:
    _HAS_PYTORCH3D = False
    Renderer = None  # type: ignore[misc, assignment]
    render_videos = None  # type: ignore[misc, assignment]
    from lib.utils.renderer_pyrender import render_mesh_video


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip("[] "))
    return slug.strip("_") or "sample"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
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

    test_text = getattr(config, "test_text", None) or config.text
    if isinstance(test_text, str):
        test_text = [test_text]
    nsamples = int(getattr(config, "nsamples", 1) or 1)

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).to(device)
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).to(device)

    refiner = build_refiner(config, test=True)
    texthom, diffusion = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    clip_model = load_and_freeze_clip(config.clip.clip_version).to(device)
    mpnet = build_mpnet(config)
    seq_cvae = build_seq_cvae(config, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    contact_estimator = build_contact_estimator(config, test=True)
    object_model = build_object_model(data_config.data_obj_pc_path)

    dump_data = torch.randn([1, 1024, 3], device=device)
    pointnet(dump_data)

    if _HAS_PYTORCH3D:
        renderer = Renderer(device=str(device), camera=f"{dataset_name}_front", fps=config.fps)
    else:
        renderer = None
        print("pytorch3d not found; using pyrender fallback renderer.")
    output_dir = osp.join(osp.dirname(__file__), "..", "outputs", "demo")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for text in test_text:
            for sample_idx in range(nsamples):
                print(f"Generating: {text!r} (sample {sample_idx + 1}/{nsamples})")
                (
                    is_lhand,
                    is_rhand,
                    obj_pc_org,
                    obj_pc_normal_org,
                    normalized_obj_pc,
                    point_sets,
                    obj_cent,
                    obj_scale,
                    obj_verts,
                    obj_faces,
                    obj_top_idx,
                    obj_pc_top_idx,
                ) = get_object_hand_info(
                    object_model,
                    clip_model,
                    [text],
                    data_config.obj_root,
                    data_config,
                    mpnet,
                )

                bs, npts = normalized_obj_pc.shape[:2]
                enc_text = encoded_text(clip_model, [text])
                enc_none_text = encoded_text(clip_model, [""] * bs)
                obj_feat = pointnet(normalized_obj_pc)

                obj_feat_final, est_contact_map = proc_obj_feat_final_old(
                    contact_estimator,
                    obj_scale,
                    obj_cent,
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
                    is_lhand, is_rhand, max_nframes, duration
                )

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
                    obj_pc_org,
                    obj_pc_normal_org,
                    valid_mask_lhand,
                    valid_mask_rhand,
                    valid_mask_obj,
                    est_contact_map,
                    dataset_name,
                    obj_pc_top_idx=obj_pc_top_idx,
                )

                refined_x_lhand, refined_x_rhand = refiner(
                    input_lhand,
                    input_rhand,
                    valid_mask_lhand=valid_mask_lhand,
                    valid_mask_rhand=valid_mask_rhand,
                )

                text_duration = duration[0].item()
                obj_verts_text = obj_verts[0]
                obj_faces_text = obj_faces[0]
                is_lhand_text = is_lhand[0]
                is_rhand_text = is_rhand[0]
                obj_top_idx_text = obj_top_idx[0] if dataset_name == "arctic" else None

                refined_x_lhand_sampled = refined_x_lhand[0][:text_duration]
                refined_x_rhand_sampled = refined_x_rhand[0][:text_duration]
                refined_x_obj_sampled = refined_x_obj[0][:text_duration]

                refined_obj_verts_tf, refined_lhand_verts, lhand_faces, refined_rhand_verts, rhand_faces = proc_results(
                    refined_x_lhand_sampled,
                    refined_x_rhand_sampled,
                    refined_x_obj_sampled,
                    obj_verts_text,
                    lhand_layer,
                    rhand_layer,
                    is_lhand_text,
                    is_rhand_text,
                    dataset_name,
                    obj_top_idx_text,
                )

                if _HAS_PYTORCH3D:
                    merged_video = render_videos(
                        renderer,
                        refined_lhand_verts,
                        lhand_faces,
                        refined_rhand_verts,
                        rhand_faces,
                        refined_obj_verts_tf,
                        obj_faces_text,
                        is_lhand_text,
                        is_rhand_text,
                    )
                else:
                    verts_list = [refined_obj_verts_tf]
                    faces_list = [obj_faces_text]
                    mesh_kinds = ["obj"]
                    if is_lhand_text:
                        verts_list.append(refined_lhand_verts)
                        faces_list.append(lhand_faces)
                        mesh_kinds.append("lhand")
                    if is_rhand_text:
                        verts_list.append(refined_rhand_verts)
                        faces_list.append(rhand_faces)
                        mesh_kinds.append("rhand")
                    merged_video = render_mesh_video(
                        verts_list,
                        faces_list,
                        mesh_kinds=mesh_kinds,
                        fps=config.fps,
                        camera=f"{dataset_name}_front",
                    )

                save_name = f"{_slug(text)}_{sample_idx + 1}.mp4"
                save_path = osp.join(output_dir, save_name)
                save_video(merged_video, fps=config.fps, save_path=save_path)
                print(f"Saved video: {save_path}")


if __name__ == "__main__":
    main()
