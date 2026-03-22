"""
Fit a 3D hair mesh to a 3D head mesh: uniform scale, optional root-to-tip taper, translation.

Loads .obj / .glb / .gltf via trimesh (and open3d fallback). FBX may require assimp;
install `trimesh[easy]` or system assimp for best format support.

Usage:
  python fit_hair_to_head.py --hair hair.obj --head head.obj --out hair_fitted.obj
  from fit_hair_to_head import fit_hair_to_head
  fit_hair_to_head("hair.glb", "head.obj", "out.obj")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import trimesh
except ImportError as e:
    raise SystemExit("Install trimesh: pip install 'trimesh[easy]'") from e


def _scene_to_trimesh(loaded: Any) -> "trimesh.Trimesh":
    """Turn a trimesh.Scene or Trimesh into a single Trimesh."""
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("Scene contains no triangle meshes.")
        return trimesh.util.concatenate(geoms)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise TypeError(f"Unsupported loaded type: {type(loaded)}")


def load_mesh_trimesh(path: Path) -> "trimesh.Trimesh":
    """Load mesh with trimesh (obj, glb, gltf, stl, ply, …)."""
    loaded = trimesh.load(str(path), skip_materials=True)
    if loaded is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return _scene_to_trimesh(loaded)


def load_mesh_open3d(path: Path) -> "trimesh.Trimesh":
    """Fallback: Open3D → trimesh."""
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(path))
    if m.is_empty():
        raise ValueError(f"Open3D: empty mesh {path}")
    m.compute_vertex_normals()
    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def load_mesh(path: str | Path) -> "trimesh.Trimesh":
    """Try trimesh first, then Open3D."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return load_mesh_trimesh(path)
    except Exception as e:
        LOGGER.info("trimesh load failed (%s), trying Open3D…", e)
    try:
        return load_mesh_open3d(path)
    except Exception as e2:
        raise RuntimeError(
            f"Could not load {path} with trimesh or Open3D. "
            f"For FBX, install assimp bindings or convert to .obj/.glb. "
            f"Errors: {e!r}; {e2!r}"
        ) from e2


def _pick_up_axis(bounds: np.ndarray) -> int:
    """Axis with largest extent (typical head height)."""
    extents = bounds[1] - bounds[0]
    return int(np.argmax(extents))


def _axis_unit(up: int) -> np.ndarray:
    u = np.zeros(3, dtype=np.float64)
    u[up] = 1.0
    return u


def _apply_root_tip_taper(
    vertices: np.ndarray,
    up: int,
    root_scale: float,
    tip_scale: float,
) -> np.ndarray:
    """Widen toward tips: scale horizontal offset from centroid (roots narrower, tips wider)."""
    v = vertices.astype(np.float64).copy()
    coord = v[:, up]
    cmin, cmax = float(coord.min()), float(coord.max())
    span = cmax - cmin
    if span < 1e-12:
        return v
    t = (coord - cmin) / span
    factor = root_scale + (tip_scale - root_scale) * t
    c = v.mean(axis=0)
    hvec = v - c
    hvec[:, up] = 0.0
    hvec *= factor[:, None]
    out = c + hvec
    out[:, up] = v[:, up]
    return out


def fit_hair_to_head(
    hair_file: str | Path,
    head_file: str | Path,
    output_file: str | Path,
    *,
    head_size_ratio: float = 0.92,
    vertical_offset: float = 0.0,
    apply_taper: bool = False,
    taper_root_scale: float = 0.98,
    taper_tip_scale: float = 1.04,
    up_axis: Optional[int] = None,
) -> "trimesh.Trimesh":
    """
    Scale and translate hair mesh to sit on the head mesh.

    Parameters
    ----------
    hair_file, head_file
        Paths to hair and head meshes (.obj, .glb, .gltf, .stl, .ply; FBX if assimp available).
    output_file
        Where to save fitted hair (.obj, .glb, .ply, .stl supported by trimesh).
    head_size_ratio
        Target: max extent of (scaled) hair bbox / max extent of head bbox along up axis.
    vertical_offset
        Extra shift along up axis after alignment (scene units; positive = toward tips).
    apply_taper
        If True, apply mild root-to-tip scale along the vertical column for a natural silhouette.
    taper_root_scale, taper_tip_scale
        In-plane taper at root vs tip (only if apply_taper).
    up_axis
        0, 1, or 2 for X, Y, Z as “up”. None = auto (longest head bbox edge).

    Returns
    -------
    trimesh.Trimesh
        The fitted hair mesh (also written to output_file).
    """
    hair = load_mesh(hair_file)
    head = load_mesh(head_file)

    hair.remove_unreferenced_vertices()
    head.remove_unreferenced_vertices()

    hb = head.bounds.astype(np.float64)
    hair_b = hair.bounds.astype(np.float64)

    if up_axis is None:
        up = _pick_up_axis(hb)
    else:
        up = int(up_axis) % 3

    e_head = hb[1] - hb[0]
    e_hair = hair_b[1] - hair_b[0]
    head_extent_up = float(e_head[up])
    hair_extent_up = float(e_hair[up])
    if hair_extent_up < 1e-12:
        raise ValueError("Hair mesh has near-zero extent along up axis.")

    # Uniform scale so hair “height” matches a fraction of head size (character proportion).
    scale = (head_extent_up * head_size_ratio) / hair_extent_up

    head_center = head.vertices.mean(axis=0)
    hair_center = hair.vertices.mean(axis=0)

    v = hair.vertices.astype(np.float64).copy()
    v -= hair_center
    v *= scale
    v += hair_center

    # Recompute bounds after scale
    hair_b = np.stack([v.min(axis=0), v.max(axis=0)], axis=0)
    hair_extent_up = float(hair_b[1, up] - hair_b[0, up])

    if apply_taper:
        v = _apply_root_tip_taper(v, up, taper_root_scale, taper_tip_scale)
        hair_b = np.stack([v.min(axis=0), v.max(axis=0)], axis=0)

    # Align: hair root (min along up) to head top (max along up), center on head in other axes.
    head_top = float(hb[1, up])
    hair_bottom = float(hair_b[0, up])
    hair_centroid = v.mean(axis=0)

    trans = np.zeros(3, dtype=np.float64)
    trans[up] = head_top - hair_bottom + vertical_offset
    for ax in range(3):
        if ax != up:
            trans[ax] = head_center[ax] - hair_centroid[ax]

    v += trans

    out_mesh = trimesh.Trimesh(vertices=v, faces=hair.faces, process=False)
    if hair.visual is not None:
        try:
            out_mesh.visual = hair.visual.copy()
        except Exception:
            pass

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(str(out_path))
    LOGGER.info("Exported fitted hair to %s (%d vertices, %d faces)", out_path, len(v), len(hair.faces))
    return out_mesh


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Fit a hair mesh to a head mesh (scale + place).")
    p.add_argument("--hair", required=True, help="Hair mesh (.obj, .glb, …)")
    p.add_argument("--head", required=True, help="Head mesh")
    p.add_argument("--out", "-o", required=True, help="Output mesh path")
    p.add_argument("--ratio", type=float, default=0.92, help="Hair height / head height target (default 0.92)")
    p.add_argument("--v-offset", type=float, default=0.0, help="Extra translation along up axis")
    p.add_argument("--taper", action="store_true", help="Apply root-to-tip proportional scale")
    p.add_argument("--up", type=int, choices=[0, 1, 2], default=None, help="Up axis: 0=X 1=Y 2=Z (default: auto)")
    args = p.parse_args()

    fit_hair_to_head(
        args.hair,
        args.head,
        args.out,
        head_size_ratio=args.ratio,
        vertical_offset=args.v_offset,
        apply_taper=args.taper,
        up_axis=args.up,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
