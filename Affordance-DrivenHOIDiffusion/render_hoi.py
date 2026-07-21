"""Headless renderer for Stage B hand-object motion -> mp4.

Uses matplotlib's Agg backend (no OpenGL / display needed, so it works over SSH
and on headless macOS, unlike pyrender/open3d offscreen). Reads the mesh arrays
that infer_stageb_cpu.py stores in its output .npz.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

_LHAND_COLOR = (0.20, 0.45, 0.90)
_RHAND_COLOR = (0.90, 0.35, 0.25)
_OBJ_COLOR = (0.60, 0.60, 0.62)
_LIGHT_DIR = np.array([0.35, -0.45, 0.82], dtype=np.float32)
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
_OBJ_TRAIL_COLOR = (0.28, 0.28, 0.30)


def _global_bounds(vert_arrays):
    v = np.concatenate([a.reshape(-1, 3) for a in vert_arrays], axis=0)
    center = (v.max(0) + v.min(0)) / 2.0
    radius = float((v.max(0) - v.min(0)).max()) / 2.0 * 1.15 + 1e-6
    return center, radius


def _trajectory_points(obj_verts, hands):
    pts = [obj_verts.mean(axis=1)]
    pts.extend(verts.mean(axis=1) for verts, _, _ in hands)
    return pts


def _camera_bounds(obj_verts, hands, mode, pad):
    if mode == "trajectory":
        center, radius = _global_bounds([obj_verts] + [h[0] for h in hands])
        radius *= pad
        return center - radius, center + radius

    centers = _trajectory_points(obj_verts, hands)
    path = np.concatenate(centers, axis=0)
    center = (path.max(0) + path.min(0)) / 2.0
    path_extent = (path.max(0) - path.min(0)).max()
    body_radius = max(
        float(np.linalg.norm(obj_verts - obj_verts.mean(axis=1, keepdims=True), axis=2).max()),
        *[float(np.linalg.norm(h[0] - h[0].mean(axis=1, keepdims=True), axis=2).max()) for h in hands],
    )
    radius = (path_extent / 2.0 + body_radius) * pad + 1e-6
    return center - radius, center + radius


def _face_colors(verts, faces, color=_OBJ_COLOR, alpha=0.65, shade=True):
    base = np.asarray(color, dtype=np.float32)
    if not shade:
        return np.repeat([[base[0], base[1], base[2], alpha]], len(faces), axis=0)

    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    diffuse = np.clip(normals @ _LIGHT_DIR, 0.0, 1.0)
    intensity = 0.42 + 0.58 * diffuse
    rgb = np.clip(base[None, :] * intensity[:, None], 0.0, 1.0)
    return np.concatenate([rgb, np.full((len(faces), 1), alpha)], axis=1)


def render_motion(out_path, obj_verts, obj_faces, hands, fps=30,
                  elev=18.0, azim=60.0, max_obj_faces=2500, dpi=100,
                  obj_alpha=0.65, shade_obj=True, obj_edges=False,
                  camera="world", camera_pad=1.35, trails=True):
    """obj_verts:(T,No,3) obj_faces:(Fo,3) hands:list of (verts(T,V,3),faces,color)."""
    T = obj_verts.shape[0]
    obj_faces = np.asarray(obj_faces)
    if obj_faces.shape[0] > max_obj_faces:  # keep matplotlib fast on dense meshes
        sel = np.linspace(0, obj_faces.shape[0] - 1, max_obj_faces).astype(int)
        obj_faces = obj_faces[sel]

    lo, hi = _camera_bounds(obj_verts, hands, camera, camera_pad)
    obj_centers, *hand_centers = _trajectory_points(obj_verts, hands)

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(5, 5), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([-0.15, -0.15, 1.3, 1.3])  # fill the figure
        obj_edgecolor = (0.38, 0.38, 0.40, 0.35) if obj_edges else "none"
        ax.add_collection3d(Poly3DCollection(
            obj_verts[t][obj_faces],
            facecolors=_face_colors(obj_verts[t], obj_faces, alpha=obj_alpha, shade=shade_obj),
            edgecolor=obj_edgecolor,
            linewidth=0.25 if obj_edges else 0.0))
        for verts, faces, color in hands:
            ax.add_collection3d(Poly3DCollection(
                verts[t][faces], facecolor=color, edgecolor="none", alpha=0.95))
        if trails:
            ax.plot(obj_centers[:t + 1, 0], obj_centers[:t + 1, 1], obj_centers[:t + 1, 2],
                    color=_OBJ_TRAIL_COLOR, linewidth=1.6, alpha=0.85)
            for centers, (_, _, color) in zip(hand_centers, hands):
                ax.plot(centers[:t + 1, 0], centers[:t + 1, 1], centers[:t + 1, 2],
                        color=color, linewidth=1.4, alpha=0.75)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(img.copy())
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path


def render_from_npz(npz_path, out_path, fps=30, **kwargs):
    d = np.load(npz_path, allow_pickle=True)
    n = int(d["nframes"]) if "nframes" in d else d["obj_verts_tf"].shape[0]
    n = max(1, min(n, d["obj_verts_tf"].shape[0]))
    hands = []
    if "lhand_verts" in d.files:
        hands.append((d["lhand_verts"][:n], d["lhand_faces"], _LHAND_COLOR))
    if "rhand_verts" in d.files:
        hands.append((d["rhand_verts"][:n], d["rhand_faces"], _RHAND_COLOR))
    return render_motion(out_path, d["obj_verts_tf"][:n], d["obj_faces"], hands, fps=fps, **kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=60.0)
    ap.add_argument("--obj-alpha", type=float, default=0.65)
    ap.add_argument("--flat-obj", action="store_true", help="Disable normal-based object shading.")
    ap.add_argument("--obj-edges", action="store_true", help="Draw faint object triangle edges.")
    ap.add_argument("--camera", choices=["world", "trajectory"], default="world",
                    help="'world' keeps a wide fixed frame around the motion path; "
                         "'trajectory' fits all vertices across all frames.")
    ap.add_argument("--camera-pad", type=float, default=1.35,
                    help="Scale factor for the fixed camera bounds.")
    ap.add_argument("--no-trails", action="store_true", help="Hide object/hand center trajectory traces.")
    args = ap.parse_args()
    out = args.out or args.npz.rsplit(".", 1)[0] + ".mp4"
    print("[ok] wrote", render_from_npz(
        args.npz, out, fps=args.fps, elev=args.elev, azim=args.azim,
        obj_alpha=args.obj_alpha, shade_obj=not args.flat_obj, obj_edges=args.obj_edges,
        camera=args.camera, camera_pad=args.camera_pad, trails=not args.no_trails))
