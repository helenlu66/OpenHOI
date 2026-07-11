"""Lightweight mesh video renderer (pyrender fallback when pytorch3d is unavailable)."""

from __future__ import annotations

import numpy as np
import pyrender
import trimesh

from lib.utils.rot import get_rotmat_x, get_rotmat_y

MESH_COLORS = {
    "obj": np.array([235, 206, 135, 255], dtype=np.uint8),
    "lhand": np.array([180, 105, 255, 255], dtype=np.uint8),
    "rhand": np.array([47, 180, 37, 255], dtype=np.uint8),
    "default": np.array([160, 160, 160, 255], dtype=np.uint8),
}


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up_cam = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up_cam
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _grab_front_rotation() -> np.ndarray:
    """Match pytorch3d get_grab_front_camera() orientation."""
    return get_rotmat_y(np.pi) @ get_rotmat_x(-np.pi / 2)


def _grab_front_eye(target: np.ndarray, extent: float) -> np.ndarray:
    """Place the GRAB front camera relative to the scene center and size."""
    rot = _grab_front_rotation()
    # pytorch3d uses T = [0, -1, 0.8]; camera center C = -R^T @ T
    base_offset = -rot.T @ np.array([0.0, -1.0, 0.8])
    base_distance = np.linalg.norm(base_offset)
    # Pull the camera closer for small scenes so meshes fill the frame.
    desired_distance = max(extent * 1.35, 0.35)
    return target + base_offset / base_distance * desired_distance


def _frame_bounds(vertices_list: list, frame_idx: int) -> tuple[np.ndarray, float]:
    points = []
    for verts in vertices_list:
        if verts is None:
            continue
        frame = _to_numpy(verts[frame_idx])
        if frame.size == 0:
            continue
        if not np.isfinite(frame).all():
            continue
        points.append(frame)
    if not points:
        return np.zeros(3, dtype=np.float64), 0.25
    all_points = np.concatenate(points, axis=0)
    center = all_points.mean(axis=0)
    extent = float(np.max(all_points.max(axis=0) - all_points.min(axis=0)))
    return center, max(extent, 0.12)


def _camera_pose_for_frame(
    vertices_list: list,
    frame_idx: int,
    camera: str,
) -> np.ndarray:
    center, extent = _frame_bounds(vertices_list, frame_idx)
    if camera.endswith("_front"):
        eye = _grab_front_eye(center, extent)
        return _look_at_pose(eye, center, np.array([0.0, 1.0, 0.0]))

    distance = max(extent * 1.6, 0.5)
    eye = center + np.array([0.0, -distance, distance * 0.85])
    return _look_at_pose(eye, center, np.array([0.0, 1.0, 0.0]))


def render_mesh_video(
    vertices_list: list,
    faces_list: list,
    mesh_kinds: list | None = None,
    img_size: int = 512,
    fps: int = 30,
    camera: str = "grab_front",
) -> np.ndarray:
    """Render a batch of mesh frames to an RGB numpy video [T, H, W, 3]."""
    if mesh_kinds is None:
        mesh_kinds = ["default"] * len(vertices_list)

    duration = _to_numpy(vertices_list[0]).shape[0]
    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    frames = []

    for frame_idx in range(duration):
        scene = pyrender.Scene(
            bg_color=[0.12, 0.12, 0.12, 1.0],
            ambient_light=[0.45, 0.45, 0.45],
        )
        camera_pose = _camera_pose_for_frame(vertices_list, frame_idx, camera)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.5, znear=0.01, zfar=20.0)
        scene.add(cam, pose=camera_pose)
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.5),
            pose=camera_pose,
        )

        for verts, faces, kind in zip(vertices_list, faces_list, mesh_kinds):
            if verts is None or faces is None:
                continue
            vertices = _to_numpy(verts[frame_idx])
            if not np.isfinite(vertices).all():
                continue
            face_array = _to_numpy(faces).astype(np.int64)
            mesh = trimesh.Trimesh(vertices=vertices, faces=face_array, process=False)
            mesh.visual.vertex_colors = MESH_COLORS.get(kind, MESH_COLORS["default"])
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        color, _ = renderer.render(scene)
        frames.append(color.copy())

    renderer.delete()
    return np.stack(frames, axis=0)
