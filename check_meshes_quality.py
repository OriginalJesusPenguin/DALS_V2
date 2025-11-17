#!/usr/bin/env python3
import argparse
from pathlib import Path

import trimesh

from vertex_clustering import mesh_diagnostics


def analyze_mesh(path: Path):
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    return mesh, mesh_diagnostics(mesh)


def format_diag(diag, watertight):
    parts = [
        f"self_intersections={diag['self_intersections'] if diag['self_intersections'] is not None else 'n/a'}",
        f"non_manifold_edges={diag['non_manifold_edges']}",
        f"degenerate_triangles={diag['degenerate_triangles']}",
        f"flipped_normals={diag['flipped_normals']}",
        f"disconnected_components={diag['disconnected_components']}",
        f"overlapping_faces={diag['overlapping_faces']}",
        f"watertight={watertight}",
    ]
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/ralbe/DALS/mesh_autodecoder/data/vertex_clustering",
        help="Root directory containing meshes.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    meshes = sorted(root.rglob("*.obj"))
    if not meshes:
        print(f"No .obj meshes found under {root}.")
        return

    print(f"Found {len(meshes)} meshes under {root}")
    totals = {
        "non_manifold_edges": 0,
        "degenerate_triangles": 0,
        "flipped_normals": 0,
        "disconnected_components": 0,
        "overlapping_faces": 0,
        "self_intersections": 0,
        "watertight_fail": 0,
    }
    counted_self = 0

    for mesh_path in meshes:
        try:
            mesh, diag = analyze_mesh(mesh_path)
        except Exception as exc:
            print(f"[ERROR] {mesh_path}: {exc}")
            continue

        rel = mesh_path.relative_to(root)
        print(
            f"{rel}: verts={len(mesh.vertices)}, faces={len(mesh.faces)}, "
            f"{format_diag(diag, mesh.is_watertight)}"
        )

        for key in ("non_manifold_edges", "degenerate_triangles",
                    "flipped_normals", "disconnected_components",
                    "overlapping_faces"):
            totals[key] += int(diag[key])
        if diag["self_intersections"] is not None:
            totals["self_intersections"] += int(diag["self_intersections"])
            counted_self += 1
        if not mesh.is_watertight:
            totals["watertight_fail"] += 1

    print("\nSummary:")
    for key in ("non_manifold_edges", "degenerate_triangles",
                "flipped_normals", "disconnected_components",
                "overlapping_faces"):
        print(f"  total {key}: {totals[key]}")
    if counted_self:
        print(f"  total self_intersections: {totals['self_intersections']}")
    else:
        print("  self_intersections: n/a")
    print(f"  meshes not watertight: {totals['watertight_fail']}")


if __name__ == "__main__":
    main()

