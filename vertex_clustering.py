#!/usr/bin/env python3
"""
Vertex Clustering Mesh Simplifier for NIfTI volumes

Usage:
    python vertex_clustering.py volume.nii.gz output_mesh.obj --iso-level 0.5 --target-verts 50000

Workflow:
1. Load a medical volume stored as .nii or .nii.gz.
2. Extract a surface mesh with marching cubes at the requested iso-level.
3. Simplify the mesh via vertex clustering down to ~target vertices.

Requirements:
    pip install numpy nibabel scikit-image trimesh
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import trimesh
from nibabel import load as load_nifti
from nibabel.nifti1 import Nifti1Image
from skimage.measure import marching_cubes

import trimesh.repair as repair
from trimesh import smoothing, util

try:
    from pymeshfix import MeshFix
except ImportError:  # pragma: no cover - optional dependency
    MeshFix = None


def compute_voxel_indices(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Map 3D points to integer voxel indices given a cell size and origin.
    """
    # Numerical stability: protect against divide-by-zero voxel size
    vs = max(float(voxel_size), 1e-18)
    idx = np.floor((points - origin) / vs).astype(np.int64)
    return idx


def unique_voxel_count(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> int:
    """
    Count unique occupied voxel cells for given voxel size.
    """
    idx = compute_voxel_indices(points, origin, voxel_size)
    # Use rows as tuples for uniqueness
    # np.unique on axis=0 is efficient enough for large arrays
    uniq = np.unique(idx, axis=0)
    return uniq.shape[0]


def choose_voxel_size_for_target(
    mesh_factory: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    original_vertex_count: int,
    target_verts: int,
    extent_max: float,
    tolerance_ratio: float = 0.02,
    max_iter: int = 25,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Binary search for a voxel size that yields ~ target vertices after clustering.
    Returns the chosen voxel size and the corresponding mesh (vertices, faces).
    """
    target_verts = int(target_verts)
    tol_abs = max(1, int(math.ceil(target_verts * tolerance_ratio)))

    # Start with the unmodified mesh (voxel size -> 0)
    best_vs = 0.0
    best_vertices, best_faces = mesh_factory(0.0)
    best_count = best_vertices.shape[0]
    best_err = abs(best_count - target_verts)
    if best_err <= tol_abs or target_verts >= original_vertex_count:
        return best_vs, best_vertices, best_faces

    # Establish an upper bound with sufficiently coarse voxels
    high_vs = extent_max if extent_max > 0 else 1.0
    vertices_high, faces_high = mesh_factory(high_vs)
    count_high = vertices_high.shape[0]
    err_high = abs(count_high - target_verts)
    if err_high < best_err:
        best_vs, best_vertices, best_faces = high_vs, vertices_high, faces_high
        best_count, best_err = count_high, err_high

    expand_iter = 0
    while count_high > target_verts and expand_iter < max_iter * 4:
        high_vs *= 2.0
        vertices_high, faces_high = mesh_factory(high_vs)
        count_high = vertices_high.shape[0]
        err_high = abs(count_high - target_verts)
        if err_high < best_err:
            best_vs, best_vertices, best_faces = high_vs, vertices_high, faces_high
            best_count, best_err = count_high, err_high
        expand_iter += 1

    low_vs = 0.0

    for _ in range(max_iter):
        mid_vs = (low_vs + high_vs) / 2.0
        if mid_vs <= 0.0:
            mid_vs = np.nextafter(0.0, high_vs)
        vertices_mid, faces_mid = mesh_factory(mid_vs)
        count_mid = vertices_mid.shape[0]
        err_mid = abs(count_mid - target_verts)

        if err_mid < best_err:
            best_vs, best_vertices, best_faces = mid_vs, vertices_mid, faces_mid
            best_count, best_err = count_mid, err_mid

        if err_mid <= tol_abs:
            return mid_vs, vertices_mid, faces_mid

        if count_mid > target_verts:
            low_vs = mid_vs
        else:
            high_vs = mid_vs

    return best_vs, best_vertices, best_faces


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_triangle_2d(p: np.ndarray, tri: np.ndarray, eps: float) -> bool:
    signs = [_orient2d(tri[i], tri[(i + 1) % 3], p) for i in range(3)]
    pos = any(s > eps for s in signs)
    neg = any(s < -eps for s in signs)
    return not (pos and neg)


def _on_segment_2d(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect_2d(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, eps: float) -> bool:
    o1 = _orient2d(a1, a2, b1)
    o2 = _orient2d(a1, a2, b2)
    o3 = _orient2d(b1, b2, a1)
    o4 = _orient2d(b1, b2, a2)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps):
        return True

    if abs(o1) <= eps and _on_segment_2d(a1, a2, b1, eps):
        return True
    if abs(o2) <= eps and _on_segment_2d(a1, a2, b2, eps):
        return True
    if abs(o3) <= eps and _on_segment_2d(b1, b2, a1, eps):
        return True
    if abs(o4) <= eps and _on_segment_2d(b1, b2, a2, eps):
        return True

    return False


def _triangles_intersect_2d(tri_a: np.ndarray, tri_b: np.ndarray, eps: float) -> bool:
    for point in tri_a:
        if _point_in_triangle_2d(point, tri_b, eps):
            return True
    for point in tri_b:
        if _point_in_triangle_2d(point, tri_a, eps):
            return True

    for i in range(3):
        a1 = tri_a[i]
        a2 = tri_a[(i + 1) % 3]
        for j in range(3):
            b1 = tri_b[j]
            b2 = tri_b[(j + 1) % 3]
            if _segments_intersect_2d(a1, a2, b1, b2, eps):
                return True
    return False


def triangles_intersect(tri_a: np.ndarray, tri_b: np.ndarray, eps: float = 1e-9) -> bool:
    tri_a = np.asarray(tri_a, dtype=np.float64)
    tri_b = np.asarray(tri_b, dtype=np.float64)

    min_a = tri_a.min(axis=0)
    max_a = tri_a.max(axis=0)
    min_b = tri_b.min(axis=0)
    max_b = tri_b.max(axis=0)
    if np.any(max_a < min_b - eps) or np.any(max_b < min_a - eps):
        return False

    n1 = np.cross(tri_a[1] - tri_a[0], tri_a[2] - tri_a[0])
    n2 = np.cross(tri_b[1] - tri_b[0], tri_b[2] - tri_b[0])
    len1 = np.linalg.norm(n1)
    len2 = np.linalg.norm(n2)
    if len1 < eps or len2 < eps:
        return False

    n1 /= len1
    n2 /= len2

    dist2 = np.dot(tri_b - tri_a[0], n1)
    if np.all(dist2 > eps) or np.all(dist2 < -eps):
        return False

    dist1 = np.dot(tri_a - tri_b[0], n2)
    if np.all(dist1 > eps) or np.all(dist1 < -eps):
        return False

    cross_norms = np.cross(n1, n2)
    if np.linalg.norm(cross_norms) < eps:
        axis = np.abs(n1)
        drop = int(np.argmax(axis))
        tri_a_2d = np.delete(tri_a, drop, axis=1)
        tri_b_2d = np.delete(tri_b, drop, axis=1)
        return _triangles_intersect_2d(tri_a_2d, tri_b_2d, eps)

    edges_a = [tri_a[1] - tri_a[0], tri_a[2] - tri_a[1], tri_a[0] - tri_a[2]]
    edges_b = [tri_b[1] - tri_b[0], tri_b[2] - tri_b[1], tri_b[0] - tri_b[2]]
    for ea in edges_a:
        for eb in edges_b:
            axis = np.cross(ea, eb)
            length = np.linalg.norm(axis)
            if length < eps:
                continue
            axis /= length
            proj_a = np.dot(tri_a, axis)
            proj_b = np.dot(tri_b, axis)
            if proj_a.max() < proj_b.min() - eps or proj_b.max() < proj_a.min() - eps:
                return False
    return True


def count_self_intersections(mesh: trimesh.Trimesh, eps: float = 1e-9) -> Optional[int]:
    if mesh.faces.shape[0] == 0:
        return 0
    try:
        tree = mesh.triangles_tree
    except BaseException:
        return None

    triangles = np.asarray(mesh.triangles)
    faces = mesh.faces
    bounds = np.hstack((triangles.min(axis=1), triangles.max(axis=1)))

    adjacency = set(tuple(sorted(pair)) for pair in mesh.face_adjacency)
    vertex_sets = [set(face.tolist()) for face in faces]

    self_int = 0
    for idx, box in enumerate(bounds):
        # RTree expects min bounds followed by max bounds
        hits = tree.intersection(tuple(box), objects=False)
        for j in hits:
            if j <= idx:
                continue
            if (idx, j) in adjacency or (j, idx) in adjacency:
                continue
            if vertex_sets[idx].intersection(vertex_sets[j]):
                continue
            if triangles_intersect(triangles[idx], triangles[j], eps=eps):
                self_int += 1
    return self_int


def cluster_vertices(
    vertices: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    representative: str = "centroid"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster vertices into voxels and produce:
    - new_vertices: (K, 3) array of representative points (K = occupied cells)
    - old_to_new: (N,) array mapping each original vertex index to new vertex index
    """
    idx = compute_voxel_indices(vertices, origin, voxel_size)
    # Map voxel index tuple -> list of vertex indices
    # Pack as a single integer key using a structured array for speed
    # But simpler: use lex sort and groupby

    # Get unique voxel cells and inverse map
    uniq_cells, inverse = np.unique(idx, axis=0, return_inverse=True)
    K = uniq_cells.shape[0]

    # Aggregate points per cluster
    new_vertices = np.zeros((K, 3), dtype=vertices.dtype)

    if representative == "voxel_center":
        # Center of each voxel cell: origin + (cell_idx + 0.5) * voxel_size
        new_vertices = origin + (uniq_cells.astype(np.float64) + 0.5) * float(voxel_size)
    else:
        # Centroid of points in each cluster
        # Sum per cluster and divide by counts
        counts = np.bincount(inverse, minlength=K).astype(np.float64)
        sums = np.zeros((K, 3), dtype=np.float64)
        np.add.at(sums, inverse, vertices)
        new_vertices = (sums / counts[:, None]).astype(vertices.dtype)

    old_to_new = inverse  # each original vertex maps to its cluster index
    return new_vertices, old_to_new


def rebuild_faces(old_faces: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:
    """
    Remap faces from old vertex indices to new cluster indices, remove
    degenerate and duplicate faces, and return compacted face array.
    """
    if old_faces.size == 0:
        return old_faces

    f = old_to_new[old_faces]
    # Remove degenerate faces (any repeated vertex in a triangle)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    if f.size == 0:
        return f

    # Remove duplicate faces irrespective of winding
    f_sorted = np.sort(f, axis=1)
    uniq, idx = np.unique(f_sorted, axis=0, return_index=True)
    f = f[idx]

    return f


def compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove unreferenced vertices and reindex faces accordingly.
    """
    if faces.size == 0:
        # No faces: return unique vertices only
        uniq_v, inv = np.unique(vertices, axis=0, return_inverse=True)
        return uniq_v, np.empty((0, 3), dtype=np.int64)

    used = np.unique(faces.reshape(-1))
    map_old_to_new = -np.ones(vertices.shape[0], dtype=np.int64)
    map_old_to_new[used] = np.arange(used.shape[0], dtype=np.int64)

    new_vertices = vertices[used]
    new_faces = map_old_to_new[faces]
    return new_vertices, new_faces


def simplify_mesh_vertex_clustering(
    mesh: trimesh.Trimesh,
    target_vertices: int,
    tolerance_ratio: float = 0.02,
    max_iter: int = 25,
    representative: str = "centroid"
) -> trimesh.Trimesh:
    """
    Simplify a mesh using vertex clustering towards target vertex count.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be trimesh.Trimesh")

    V = mesh.vertices.view(np.ndarray)
    F = mesh.faces.view(np.ndarray) if mesh.faces is not None else np.empty((0, 3), dtype=np.int64)

    n = V.shape[0]
    target_vertices = int(target_vertices)

    if target_vertices >= n:
        # Nothing to do
        return mesh.copy()

    # Bounding box (axis-aligned)
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    extent = np.maximum(vmax - vmin, 1e-12)  # avoid zeros
    extent_max = float(np.max(extent))
    if not np.isfinite(extent_max) or extent_max <= 0.0:
        return mesh.copy()

    def build_mesh(voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
        if voxel_size <= np.finfo(float).eps:
            return V.copy(), F.copy()
        clustered_vertices, old_to_new = cluster_vertices(
            V, vmin, voxel_size, representative=representative
        )
        clustered_faces = rebuild_faces(F, old_to_new)
        return compact_mesh(clustered_vertices, clustered_faces)

    _, new_vertices, new_faces = choose_voxel_size_for_target(
        build_mesh,
        original_vertex_count=n,
        target_verts=target_vertices,
        extent_max=extent_max,
        tolerance_ratio=tolerance_ratio,
        max_iter=max_iter,
    )

    # Build result mesh
    simplified = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    # Optional: recompute normals for better shading
    try:
        simplified.rezero()
        simplified.remove_duplicate_faces()
        simplified.remove_degenerate_faces()
        simplified.remove_unreferenced_vertices()
        simplified.fix_normals()
    except Exception:
        pass

    return simplified


def marching_cubes_from_nifti(
    volume_path: Path,
    iso_level: float,
    step_size: int = 1,
    allow_degenerate: bool = False,
) -> trimesh.Trimesh:
    """
    Convert a NIfTI volume into a surface mesh using marching cubes.
    """
    img: Nifti1Image = load_nifti(str(volume_path))
    data = img.get_fdata()

    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape}")

    verts, faces, normals, values = marching_cubes(
        data,
        level=iso_level,
        step_size=max(1, step_size),
        allow_degenerate=allow_degenerate,
    )

    affine = img.affine
    verts_h = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
    verts_world = (affine @ verts_h.T).T[:, :3]

    mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


def clean_mesh(
    mesh: trimesh.Trimesh,
    component_area_ratio: float,
    smooth_iters: int,
) -> trimesh.Trimesh:
    mesh = mesh.copy()

    mesh.remove_infinite_values()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()

    repair.fix_winding(mesh)
    repair.fix_inversion(mesh, multibody=True)
    repair.fix_normals(mesh, multibody=True)
    repair.fill_holes(mesh)

    if MeshFix is not None:
        try:
            fixer = MeshFix(mesh.vertices, mesh.faces)
            fixer.repair(verbose=False, joincomp=True, remove_smallest_components=True)
            v_fixed, f_fixed = fixer.return_arrays()
            if v_fixed.size and f_fixed.size:
                mesh = trimesh.Trimesh(
                    vertices=np.asarray(v_fixed),
                    faces=np.asarray(f_fixed, dtype=np.int64),
                    process=False,
                )
        except Exception:
            pass

    if component_area_ratio > 0.0:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            total_area = sum(c.area for c in components)
            threshold = total_area * component_area_ratio
            keep = [c for c in components if c.area >= threshold]
            if not keep:
                keep = [max(components, key=lambda c: c.area)]
            mesh = util.concatenate(keep)

    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    if smooth_iters > 0:
        smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iters)
        mesh.rezero()

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    repair.fix_normals(mesh, multibody=False)

    mesh = mesh.process(validate=True)
    repair.fix_normals(mesh, multibody=False)

    return mesh


def mesh_diagnostics(mesh: trimesh.Trimesh) -> dict:
    if mesh.faces.shape[0] == 0:
        return {
            "self_intersections": 0,
            "non_manifold_edges": 0,
            "degenerate_triangles": 0,
            "flipped_normals": 0,
            "disconnected_components": 0,
            "overlapping_faces": 0,
        }

    area_tol = 1e-12
    areas = np.asarray(mesh.area_faces)
    degenerate = int(np.sum(areas <= area_tol))

    edges_unique = np.asarray(mesh.edges_unique)
    edge_inverse = np.asarray(mesh.edges_unique_inverse)
    if edges_unique.size == 0:
        non_manifold = 0
    else:
        edge_counts = np.bincount(edge_inverse, minlength=len(edges_unique))
        non_manifold = int(np.sum(edge_counts > 2))

    normals = mesh.face_normals
    adjacency = np.asarray(mesh.face_adjacency)
    if adjacency.size:
        n0 = normals[adjacency[:, 0]]
        n1 = normals[adjacency[:, 1]]
        cosang = np.einsum("ij,ij->i", n0, n1)
        cosang = np.clip(cosang, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosang))
        flips = int(np.sum(angles > 150.0))
    else:
        flips = 0

    components = mesh.split(only_watertight=False)
    disconnected = len(components)

    sorted_faces = np.sort(mesh.faces, axis=1)
    _, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    overlapping = int(np.sum(counts > 1))

    self_intersections = count_self_intersections(mesh)

    return {
        "self_intersections": self_intersections,
        "non_manifold_edges": non_manifold,
        "degenerate_triangles": degenerate,
        "flipped_normals": flips,
        "disconnected_components": disconnected,
        "overlapping_faces": overlapping,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract a mesh from a NIfTI volume via marching cubes and simplify with vertex clustering."
    )
    p.add_argument("--input", required=True, help="Input NIfTI volume (.nii or .nii.gz).")
    p.add_argument("--output", required=True, help="Output mesh file (.obj, .ply, .stl, etc.)")
    p.add_argument("--iso-level", type=float, required=True, help="Iso-surface level for marching cubes.")
    p.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Marching cubes step size (larger skips voxels for faster but rougher meshes).",
    )
    p.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Allow marching cubes to return degenerate triangles.",
    )
    p.add_argument("--target-verts", type=int, required=True, help="Desired number of vertices (approximate).")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Relative tolerance on target (default 0.02 = 2%%).",
    )
    p.add_argument("--max-iter", type=int, default=25, help="Max iterations for binary search (default 25).")
    p.add_argument("--representative", choices=["centroid", "voxel_center"], default="centroid",
                   help="Representative point inside each voxel (default: centroid).")
    p.add_argument(
        "--component-threshold",
        type=float,
        default=0.01,
        help="Minimum surface-area ratio for connected components to keep (0 disables pruning).",
    )
    p.add_argument(
        "--smooth-iters",
        type=int,
        default=5,
        help="Number of Taubin smoothing iterations applied after simplification (0 disables).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input volume not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        mesh = marching_cubes_from_nifti(
            input_path,
            iso_level=args.iso_level,
            step_size=args.step_size,
            allow_degenerate=args.allow_degenerate,
        )
    except Exception as exc:
        print(f"Failed to extract mesh from volume: {exc}", file=sys.stderr)
        sys.exit(1)

    if mesh.is_empty:
        print("Marching cubes produced an empty mesh.", file=sys.stderr)
        sys.exit(1)

    simplified = simplify_mesh_vertex_clustering(
        mesh,
        target_vertices=args.target_verts,
        tolerance_ratio=args.tolerance,
        max_iter=args.max_iter,
        representative=args.representative
    )

    simplified = clean_mesh(
        simplified,
        component_area_ratio=max(0.0, float(args.component_threshold)),
        smooth_iters=max(0, int(args.smooth_iters)),
    )

    try:
        simplified.export(args.output)
        v_in = len(mesh.vertices)
        v_out = len(simplified.vertices)
        f_in = len(mesh.faces)
        f_out = len(simplified.faces)
        print(f"Simplified: {v_in}→{v_out} vertices, {f_in}→{f_out} faces")
        print(f"Saved: {args.output}")
    except Exception as e:
        print(f"Failed to save mesh: {e}", file=sys.stderr)
        sys.exit(1)

    diagnostics = mesh_diagnostics(simplified)
    print("Diagnostics:")
    si = diagnostics["self_intersections"]
    if si is None:
        print("  self-intersections: unavailable (requires rtree or python-fcl).")
    else:
        print(f"  self-intersections: {si}")
    print(f"  non-manifold edges: {diagnostics['non_manifold_edges']}")
    print(f"  degenerate triangles: {diagnostics['degenerate_triangles']}")
    print(f"  flipped normals: {diagnostics['flipped_normals']}")
    print(f"  disconnected components: {diagnostics['disconnected_components']}")
    print(f"  overlapping faces: {diagnostics['overlapping_faces']}")


if __name__ == "__main__":
    main()