#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shape_similarity.py

Compute:
1) Earth Mover's Distance (EMD) between two point clouds sampled from .obj meshes.
2) Laplace–Beltrami eigenvalues from a triangular mesh (cotangent Laplacian).

Author: You + Copilot
"""

import argparse
import time
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import eigsh

# ------------------------------------------------------------
#                       OBJ LOADING
# ------------------------------------------------------------

def _parse_face_index(token):
    """
    Parse an OBJ face token like '3', '3/4', or '3/4/5' and return the vertex index (0-based).
    """
    # OBJ is 1-based indices; we convert to 0-based
    return int(token.split('/')[0]) - 1

def load_obj_vertices_faces(path):
    """
    Minimal OBJ loader that returns (V, F)
      V: (n, 3) float
      F: (m, 3) int, triangulated
    Ignores materials, normals, and texture coordinates.
    Performs simple fan triangulation for polygons with >3 vertices.
    """
    print(f"[DEBUG] Loading OBJ from: {path}")
    verts = []
    faces = []
    with open(path, 'r') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f' and len(parts) >= 4:
                # Fan triangulate if polygon
                idxs = [_parse_face_index(tok) for tok in parts[1:]]
                v0 = idxs[0]
                for k in range(1, len(idxs) - 1):
                    faces.append([v0, idxs[k], idxs[k+1]])

    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)
    print(f"[DEBUG] Loaded mesh: {V.shape[0]} vertices, {F.shape[0]} triangles")
    return V, F

# ------------------------------------------------------------
#                SURFACE SAMPLING (UNIFORM BY AREA)
# ------------------------------------------------------------

def triangle_areas(V, F):
    """
    Compute area for each triangle face.
    """
    v0 = V[F[:, 0], :]
    v1 = V[F[:, 1], :]
    v2 = V[F[:, 2], :]
    areas = 0.5 * norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return areas

def sample_points_on_mesh(V, F, n_samples, seed=42):
    """
    Uniformly sample points on the surface using area-weighted face sampling
    + uniform barycentric sampling per selected face.

    Returns:
      P: (n_samples, 3)
    """
    rng = np.random.default_rng(seed)
    areas = triangle_areas(V, F)
    total_area = np.sum(areas)
    if total_area <= 0:
        raise ValueError("Mesh has zero total area; cannot sample.")

    probs = areas / total_area
    # Choose faces according to area
    face_idx = rng.choice(len(F), size=n_samples, p=probs, replace=True)

    # For each chosen face, sample barycentric coordinates
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    # Warp to uniform on triangle
    sqrt_u = np.sqrt(u)
    b0 = 1.0 - sqrt_u
    b1 = sqrt_u * (1.0 - v)
    b2 = sqrt_u * v

    tri = F[face_idx]
    p0 = V[tri[:, 0], :]
    p1 = V[tri[:, 1], :]
    p2 = V[tri[:, 2], :]
    P = b0[:, None] * p0 + b1[:, None] * p1 + b2[:, None] * p2

    print(f"[DEBUG] Sampled {n_samples} points (area-weighted).")
    return P

# ------------------------------------------------------------
#                  EMD (OPTIMAL TRANSPORT)
# ------------------------------------------------------------

def pairwise_distances(P, Q):
    """
    Compute pairwise Euclidean distances between two point sets:
      P: (n, d), Q: (m, d)
    Returns:
      C: (n, m)
    """
    # Efficient broadcast: ||P - Q|| = sqrt(||P||^2 + ||Q||^2 - 2 P Q^T)
    P2 = np.sum(P * P, axis=1)[:, None]
    Q2 = np.sum(Q * Q, axis=1)[None, :]
    C2 = P2 + Q2 - 2.0 * (P @ Q.T)
    C2 = np.maximum(C2, 0.0)  # numerical safety
    C = np.sqrt(C2)
    return C

def emd_via_linprog(P, Q, verbose=True):
    """
    Compute EMD using a FROM-SCRATCH linear programming formulation.
    This is exact but scales as O((n+m)*nm) memory—only feasible for small n,m (<= ~300).

    P, Q: (n, d), (m, d)
    Treat as uniform distributions: a_i = 1/n, b_j = 1/m.

    Returns:
      emd_value (float)
    """
    t0 = time.time()
    n, m = P.shape[0], Q.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)

    if verbose:
        print(f"[DEBUG][EMD-LP] Building cost matrix for {n}x{m} points...")
    C = pairwise_distances(P, Q)
    c = C.flatten()  # objective coefficients (length n*m)

    # Constraints:
    # Sum_j gamma_{ij} = a_i  (for all i)
    # Sum_i gamma_{ij} = b_j  (for all j)
    # gamma_{ij} >= 0
    if verbose:
        print(f"[DEBUG][EMD-LP] Building equality constraints...")
    Aeq_rows = n + m
    Aeq_cols = n * m
    A_eq = sparse.lil_matrix((Aeq_rows, Aeq_cols), dtype=np.float64)
    beq = np.zeros(Aeq_rows, dtype=np.float64)

    # Row constraints: for each i, sum over j
    for i in range(n):
        row = i
        start = i * m
        A_eq[row, start:start + m] = 1.0
        beq[row] = a[i]

    # Column constraints: for each j, sum over i
    for j in range(m):
        row = n + j
        # All positions with column j are at indices i*m + j for i in [0..n-1]
        A_eq[row, j:Aeq_cols:m] = 1.0
        beq[row] = b[j]

    A_eq = A_eq.tocsr()

    # Bounds: gamma_{ij} >= 0
    bounds = [(0, None)] * (n * m)

    if verbose:
        print(f"[DEBUG][EMD-LP] Solving LP with scipy.optimize.linprog (HiGHS)...")
    res = linprog(c, A_eq=A_eq, b_eq=beq, bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    emd_value = res.fun
    if verbose:
        print(f"[DEBUG][EMD-LP] Done. EMD = {emd_value:.6f}. Time: {time.time() - t0:.2f}s")
    return emd_value

def emd_via_pot(P, Q, reg=None, verbose=True):
    """
    Compute EMD using POT (Python Optimal Transport).
    If reg is None -> exact EMD (network simplex) via ot.emd2.
    If reg > 0    -> entropic regularized OT via ot.sinkhorn2 (faster, approximate).

    Returns:
      emd_value (float)
    """
    try:
        import ot  # POT
    except Exception as e:
        raise ImportError("POT (package 'POT') not installed. Try: pip install POT") from e

    n, m = P.shape[0], Q.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = pairwise_distances(P, Q)

    if reg is None:
        if verbose:
            print("[DEBUG][EMD-POT] Calling ot.emd2 (exact EMD)...")
        val = ot.emd2(a, b, C)
        if verbose:
            print(f"[DEBUG][EMD-POT] EMD = {val:.6f}")
        return float(val)
    else:
        if verbose:
            print(f"[DEBUG][EMD-POT] Calling ot.sinkhorn2 with reg={reg}...")
        val, _ = ot.sinkhorn2(a, b, C, reg)
        # sinkhorn2 returns transport cost; for Euclidean metric, this is comparable to EMD
        if verbose:
            print(f"[DEBUG][EMD-POT] Sinkhorn cost ≈ {float(val):.6f}")
        return float(val)

# ------------------------------------------------------------
#        COTANGENT LAPLACIAN + LUMPED MASS (LB EIGS)
# ------------------------------------------------------------

def cotangent(a, b, c):
    """
    Compute cotangent of angle at vertex 'a' in triangle (a, b, c).
    cot(alpha) = (ab · ac) / ||ab × ac||
    """
    u = b - a
    v = c - a
    cross_n = np.cross(u, v)
    denom = norm(cross_n)
    if denom < 1e-16:
        return 0.0
    return float(np.dot(u, v) / denom)

def build_cotangent_laplacian(V, F):
    """
    Build the symmetric cotangent Laplacian L and lumped mass matrix M.

    L = (1/2) * sum_over_faces [cot(alpha_ij) + cot(beta_ij)] for edge (i, j), with L_ij = -w_ij, and L_ii = sum_j w_ij.
    M = diagonal with barycentric (lumped) vertex areas: A_i = sum_{faces incident to i} (area(face)/3)

    Returns:
      L: (n, n) scipy.sparse.csr_matrix
      M: (n, n) scipy.sparse.csr_matrix
    """
    n_verts = V.shape[0]
    I = []
    J = []
    W = []  # weights for off-diagonals; we'll accumulate and later build L

    print("[DEBUG][LB] Building cotangent weights and lumped mass...")
    areas = triangle_areas(V, F)
    vertex_area = np.zeros(n_verts, dtype=np.float64)

    # Accumulate per-triangle contributions
    for t, (i, j, k) in enumerate(F):
        vi, vj, vk = V[i], V[j], V[k]

        # Update lumped mass (1/3 of triangle area to each vertex)
        a = areas[t]
        vertex_area[i] += a / 3.0
        vertex_area[j] += a / 3.0
        vertex_area[k] += a / 3.0

        # Cotangents at each angle
        cot_alpha = cotangent(vj, vi, vk)  # angle at j with vectors (j->i, j->k)
        cot_beta  = cotangent(vi, vj, vk)  # angle at i
        cot_gamma = cotangent(vk, vi, vj)  # angle at k (careful: function signature is cot at first arg)

        # Edges opposite to each angle contribute to weights:
        # w_ij += 0.5 * cot(gamma) ; because gamma is angle at k opposite edge (i,j)
        w_ij = 0.5 * cot_gamma
        w_jk = 0.5 * cot_alpha
        w_ki = 0.5 * cot_beta

        # Store symmetric pairs (i,j), (j,k), (k,i)
        # We'll accumulate duplicates in a sparse matrix
        I += [i, j, j, k, k, i]
        J += [j, i, k, j, i, k]
        W += [w_ij, w_ij, w_jk, w_jk, w_ki, w_ki]

    # Build sparse off-diagonal weight matrix Wmat (sum duplicates)
    Wmat = sparse.coo_matrix((W, (I, J)), shape=(n_verts, n_verts), dtype=np.float64).tocsr()

    # Laplacian: L = D - W, where D_ii = sum_j W_ij, and L_ij = -W_ij for i != j
    diag = np.array(Wmat.sum(axis=1)).ravel()
    L = sparse.diags(diag, 0) - Wmat

    # Lumped mass matrix M (diagonal)
    M = sparse.diags(vertex_area, 0)

    print("[DEBUG][LB] L and M built. "
          f"nnz(L)={L.nnz}, min(area)={vertex_area.min():.3e}, max(area)={vertex_area.max():.3e}")
    return L, M

def laplace_beltrami_eigs(V, F, k=10, which='SM'):
    """
    Compute k smallest-magnitude eigenpairs of the generalized problem:
      L x = lambda M x
    using the cotangent Laplacian (L) and lumped mass (M).

    NOTE:
      - The smallest eigenvalue is ~0 (one per connected component).
      - We do NOT normalize for scale (as requested).
      - For better numerical stability we use shift-invert around sigma=0.

    Returns:
      evals: (k,) eigenvalues (ascending)
      evecs: (n, k) eigenvectors
    """
    L, M = build_cotangent_laplacian(V, F)

    # Ensure SPD-ish: tiny regularization on L to avoid exact singularity if needed
    reg = 1e-12
    L_reg = L + reg * M

    print(f"[DEBUG][LB] Solving generalized eigensystem for k={k} (this may take a moment)...")
    # Use shift-invert near sigma=0 to retrieve smallest eigenvalues reliably
    evals, evecs = eigsh(A=L_reg, k=k, M=M, sigma=0.0, which='LM')

    # Sort ascending
    order = np.argsort(evals)
    evals = np.real(evals[order])
    evecs = np.real(evecs[:, order])

    print("[DEBUG][LB] Eigenvalues (first 10 shown if available):")
    for i, lam in enumerate(evals[:10]):
        print(f"  lambda[{i}] = {lam:.8e}")

    return evals, evecs

# ------------------------------------------------------------
#                END-TO-END PIPELINE (EXAMPLE)
# ------------------------------------------------------------

def compute_pointcloud_from_obj(path, n_samples=5000, seed=42):
    V, F = load_obj_vertices_faces(path)
    P = sample_points_on_mesh(V, F, n_samples=n_samples, seed=seed)
    return P, V, F

def main():
    parser = argparse.ArgumentParser(description="Compute EMD and Laplace-Beltrami eigenvalues from .obj meshes.")
    parser.add_argument("--mesh_a", type=str, required=True, help="Path to first OBJ file")
    parser.add_argument("--mesh_b", type=str, required=True, help="Path to second OBJ file")
    parser.add_argument("--samples", type=int, default=2000, help="Number of surface samples per mesh for EMD")
    parser.add_argument("--emd_method", type=str, choices=["lp", "pot", "sinkhorn"], default="pot",
                        help="EMD solver: 'lp' (exact, small), 'pot' (exact via POT), 'sinkhorn' (fast approx via POT)")
    parser.add_argument("--sinkhorn_reg", type=float, default=0.01, help="Entropic regularization (if emd_method=sinkhorn)")
    parser.add_argument("--lb_k", type=int, default=20, help="Number of Laplace-Beltrami eigenvalues to compute")
    args = parser.parse_args()

    print("[DEBUG] ----- Sampling point clouds for EMD -----")
    P, V_A, F_A = compute_pointcloud_from_obj(args.mesh_a, n_samples=args.samples, seed=123)
    Q, V_B, F_B = compute_pointcloud_from_obj(args.mesh_b, n_samples=args.samples, seed=456)

    # Optional: translate to centroid for numerical stability (does NOT affect EMD if both are translated differently, 
    # EMD is translation-sensitive; but centering can help numerics in distance computation).
    # Commented out to keep true as-is distances:
    # P = P - P.mean(axis=0, keepdims=True)
    # Q = Q - Q.mean(axis=0, keepdims=True)

    print("[DEBUG] ----- Computing EMD between point clouds -----")
    if args.emd_method == "lp":
        if args.samples > 300:
            print("[WARN] LP EMD scales poorly. Consider --samples<=300 or use --emd_method pot/sinkhorn.")
        emd_val = emd_via_linprog(P, Q, verbose=True)
    elif args.emd_method == "pot":
        emd_val = emd_via_pot(P, Q, reg=None, verbose=True)
    else:
        emd_val = emd_via_pot(P, Q, reg=args.sinkhorn_reg, verbose=True)

    print(f"[RESULT] EMD between sampled point clouds: {emd_val:.6f}")

    print("[DEBUG] ----- Laplace–Beltrami eigenvalues (as-is, no scale normalization) -----")
    t0 = time.time()
    evals_A, _ = laplace_beltrami_eigs(V_A, F_A, k=args.lb_k)
    evals_B, _ = laplace_beltrami_eigs(V_B, F_B, k=args.lb_k)
    print(f"[RESULT] First {args.lb_k} LB eigenvalues (Mesh A):")
    print("  " + ", ".join(f"{x:.6e}" for x in evals_A))
    print(f"[RESULT] First {args.lb_k} LB eigenvalues (Mesh B):")
    print("  " + ", ".join(f"{x:.6e}" for x in evals_B))
    print(f"[DEBUG] LB total time: {time.time() - t0:.2f}s")

main()