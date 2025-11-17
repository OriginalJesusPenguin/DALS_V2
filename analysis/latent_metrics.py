#!/usr/bin/env python3
"""
Utilities for analysing MeshDecoder latent representations.

The helpers in this module cover three families of metrics:

1. Local extrinsic deformation measures (per-vertex displacement, per-face
   stretch/strain derived from deformation gradients, and area-weighted
   vertex aggregates).
2. Intrinsic spectral descriptors (Laplace–Beltrami eigenspectra and
   Heat Kernel Signatures) with distances to a baseline mesh.
3. Latent-space sensitivity diagnostics, both via finite differences and
   automatic differentiation, including latent pullback metrics.

All computations assume the existence of a baseline mesh obtained by
decoding the mean latent vector stored in the training checkpoint.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding
try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - Torch < 2.6 fallback
    def add_safe_globals(_items):
        return None

from pytorch3d.structures import Meshes
from scipy import sparse
from scipy.sparse.linalg import eigsh

EPS = 1e-8


@dataclasses.dataclass
class DeformationResults:
    displacement: np.ndarray  # (N, 3)
    displacement_norm: np.ndarray  # (N,)
    face_metrics: Mapping[str, np.ndarray]
    vertex_metrics: Mapping[str, np.ndarray]


@dataclasses.dataclass
class SpectralResults:
    eigenvalues: np.ndarray
    baseline_eigenvalues: np.ndarray
    spectrum_distance: float
    hks: np.ndarray
    baseline_hks: np.ndarray
    hks_distance_per_vertex: np.ndarray


@dataclasses.dataclass
class LatentSensitivity:
    per_vertex_sensitivity: np.ndarray  # (N, d) or (N,) for aggregated norms
    global_scores: np.ndarray  # (d,)
    settings: Mapping[str, float]
    jacobian_vectors: Optional[np.ndarray] = None  # (d, N, 3)


def _ensure_safe_globals() -> None:
    try:
        add_safe_globals([Embedding])
    except Exception:  # pragma: no cover - PyTorch < 2.6
        pass


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict:
    _ensure_safe_globals()
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,  # PyTorch >= 2.6
        )
    except TypeError:
        ckpt = torch.load(
            checkpoint_path,
            map_location=device,
        )
    if "latent_vectors" not in ckpt:
        raise KeyError("Checkpoint is missing `latent_vectors` embedding.")
    return ckpt


def decoder_from_checkpoint(
    checkpoint: Mapping, device: torch.device
) -> Tuple[torch.nn.Module, Meshes]:
    from model.mesh_decoder import MeshDecoder  # Local import to avoid cycles
    from pytorch3d.ops import subdivide_meshes

    hparams = checkpoint["hparams"]
    decoder = MeshDecoder(
        latent_features=hparams["latent_features"],
        steps=hparams["steps"],
        hidden_features=hparams["hidden_features"],
        subdivide=hparams["subdivide"],
        mode=hparams["decoder_mode"],
        norm=hparams["normalization"][0],
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.to(device)
    decoder.eval()

    template: Meshes = checkpoint["template"].to(device)
    template = subdivide_meshes.SubdivideMeshes()(template).to(device)
    return decoder, template


def baseline_latent_vector(checkpoint: Mapping) -> Tensor:
    latent_vectors: Embedding = checkpoint["latent_vectors"]
    weights = latent_vectors.weight.detach()
    return weights.mean(dim=0)


def load_latent_tensor(path: Path) -> Tensor:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        for key in ("latent", "latent_vector", "latent_vectors"):
            if key in tensor:
                tensor = tensor[key]
                break
        else:
            raise KeyError(f"No latent vector found in {path}")
    tensor = torch.as_tensor(tensor).float()
    if tensor.ndim == 2:
        tensor = tensor.squeeze(0)
    return tensor


def resolve_latent_path(
    latent_path: Optional[Path],
    latent_root: Path,
    status: str,
    index: int,
) -> Path:
    if latent_path is not None:
        return latent_path

    if not latent_root.exists():
        raise FileNotFoundError(f"Latent root directory {latent_root} does not exist.")

    candidates: List[Path] = []
    for pattern in (f"{status}_*.pt", f"{status}_*.pth"):
        candidates.extend(sorted(latent_root.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No latent codes found for status '{status}' in {latent_root}"
        )
    if index < 0 or index >= len(candidates):
        raise IndexError(
            f"Latent index {index} is out of bounds for status '{status}'. "
            f"Found {len(candidates)} candidates."
        )
    return candidates[index]


def decode_latent(
    decoder: torch.nn.Module, template: Meshes, latent: Tensor, device: torch.device
) -> Meshes:
    latent_batch = latent.detach().to(device).unsqueeze(0)
    with torch.no_grad():
        decoded = decoder(template.clone(), latent_batch)[-1]
    return decoded.cpu()


def _verts_and_faces(mesh: Meshes) -> Tuple[np.ndarray, np.ndarray]:
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy().astype(np.int32)
    return verts, faces


def triangle_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    v0 = V[F[:, 0], :]
    v1 = V[F[:, 1], :]
    v2 = V[F[:, 2], :]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def cotangent(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = b - a
    v = c - a
    cross = np.cross(u, v)
    denom = np.linalg.norm(cross)
    if denom < 1e-16:
        return 0.0
    return float(np.dot(u, v) / denom)


def build_cotangent_laplacian(
    V: np.ndarray, F: np.ndarray
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    n_verts = V.shape[0]
    I: List[int] = []
    J: List[int] = []
    W: List[float] = []

    areas = triangle_areas(V, F)
    vertex_area = np.zeros(n_verts, dtype=np.float64)

    for f_idx, (i, j, k) in enumerate(F):
        vi, vj, vk = V[i], V[j], V[k]
        area = areas[f_idx]
        share = area / 3.0
        vertex_area[i] += share
        vertex_area[j] += share
        vertex_area[k] += share

        cot_gamma = cotangent(vk, vi, vj)
        cot_alpha = cotangent(vj, vk, vi)
        cot_beta = cotangent(vi, vj, vk)

        w_ij = 0.5 * cot_beta
        w_jk = 0.5 * cot_gamma
        w_ki = 0.5 * cot_alpha

        I.extend([i, j, j, k, k, i])
        J.extend([j, i, k, j, i, k])
        W.extend([w_ij, w_ij, w_jk, w_jk, w_ki, w_ki])

    Wmat = sparse.coo_matrix((W, (I, J)), shape=(n_verts, n_verts)).tocsr()
    diag = np.array(Wmat.sum(axis=1)).ravel()
    L = sparse.diags(diag, 0) - Wmat
    M = sparse.diags(vertex_area, 0)
    return L, M


def laplace_beltrami_eigs(
    V: np.ndarray, F: np.ndarray, k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    L, M = build_cotangent_laplacian(V, F)
    reg = 1e-12
    L_reg = L + reg * M
    evals, evecs = eigsh(A=L_reg, k=k, M=M, sigma=0.0, which="LM")
    order = np.argsort(evals)
    evals = np.real(evals[order])
    evecs = np.real(evecs[:, order])
    return evals, evecs


def per_vertex_displacement(
    baseline_mesh: Meshes, target_mesh: Meshes
) -> Tuple[np.ndarray, np.ndarray]:
    base_verts, _ = _verts_and_faces(baseline_mesh)
    target_verts, _ = _verts_and_faces(target_mesh)
    if base_verts.shape != target_verts.shape:
        raise ValueError("Baseline and target meshes must share vertex topology.")
    displacement = target_verts - base_verts
    norms = np.linalg.norm(displacement, axis=1)
    return displacement, norms


def _face_deformation_gradients(
    baseline_mesh: Meshes, target_mesh: Meshes
) -> Dict[str, np.ndarray]:
    base_verts, faces = _verts_and_faces(baseline_mesh)
    target_verts, _ = _verts_and_faces(target_mesh)
    areas = triangle_areas(base_verts, faces)

    sigma1 = np.empty(len(faces))
    sigma2 = np.empty(len(faces))
    kappa = np.empty(len(faces))
    area_change = np.empty(len(faces))
    log_sigma1 = np.empty(len(faces))
    log_sigma2 = np.empty(len(faces))

    for i, (i0, i1, i2) in enumerate(faces):
        p1, p2, p3 = base_verts[[i0, i1, i2]]
        q1, q2, q3 = target_verts[[i0, i1, i2]]

        Ds = np.stack((p2 - p1, p3 - p1), axis=1)  # (3, 2)
        Dt = np.stack((q2 - q1, q3 - q1), axis=1)

        if np.linalg.norm(np.cross(Ds[:, 0], Ds[:, 1])) < EPS:
            sigma1[i] = sigma2[i] = 1.0
            kappa[i] = 1.0
            area_change[i] = 1.0
            log_sigma1[i] = log_sigma2[i] = 0.0
            continue

        Ds_pinv = np.linalg.pinv(Ds, rcond=1e-8)
        F = Dt @ Ds_pinv  # (3, 3)
        S = np.linalg.svd(F, compute_uv=False)
        sig1, sig2 = S[:2]
        sig1 = float(max(sig1, EPS))
        sig2 = float(max(sig2, EPS))
        sigma1[i] = sig1
        sigma2[i] = sig2
        kappa[i] = sig1 / sig2
        area_change[i] = sig1 * sig2
        log_sigma1[i] = math.log(sig1)
        log_sigma2[i] = math.log(sig2)

    return {
        "area": areas,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "kappa": kappa,
        "area_change": area_change,
        "log_sigma1": log_sigma1,
        "log_sigma2": log_sigma2,
    }


def _vertex_aggregate_from_faces(
    faces: np.ndarray, face_metrics: Mapping[str, np.ndarray], areas: np.ndarray
) -> Dict[str, np.ndarray]:
    n_verts = faces.max() + 1
    weights = np.zeros(n_verts, dtype=np.float64)
    accumulators: Dict[str, np.ndarray] = {
        key: np.zeros(n_verts, dtype=np.float64) for key in face_metrics
    }

    for face_idx, (v0, v1, v2) in enumerate(faces):
        w = areas[face_idx] / 3.0
        weights[v0] += w
        weights[v1] += w
        weights[v2] += w
        for key, values in face_metrics.items():
            accumulators[key][v0] += w * values[face_idx]
            accumulators[key][v1] += w * values[face_idx]
            accumulators[key][v2] += w * values[face_idx]

    weights = np.maximum(weights, EPS)
    return {key: val / weights for key, val in accumulators.items()}


def compute_deformation_metrics(
    baseline_mesh: Meshes, target_mesh: Meshes
) -> DeformationResults:
    displacement, disp_norm = per_vertex_displacement(baseline_mesh, target_mesh)
    face_data = _face_deformation_gradients(baseline_mesh, target_mesh)
    faces = baseline_mesh.faces_packed().cpu().numpy()
    vertex_data = _vertex_aggregate_from_faces(faces, face_data, face_data["area"])
    return DeformationResults(
        displacement=displacement,
        displacement_norm=disp_norm,
        face_metrics=face_data,
        vertex_metrics=vertex_data,
    )


def laplace_beltrami_spectrum(
    verts: np.ndarray, faces: np.ndarray, k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = laplace_beltrami_eigs(verts, faces, k=k)
    return np.asarray(evals), np.asarray(evecs)


def heat_kernel_signature(
    evals: np.ndarray, evecs: np.ndarray, times: Sequence[float]
) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    phi_squared = evecs**2  # (N, k)
    decay = np.exp(-np.outer(evals, times))  # (k, T)
    return (phi_squared[:, :, None] * decay[None, :, :]).sum(axis=1)


def spectral_metrics(
    verts: np.ndarray,
    faces: np.ndarray,
    baseline_verts: np.ndarray,
    baseline_faces: np.ndarray,
    k: int = 20,
    weights: Optional[Sequence[float]] = None,
    times: Optional[Sequence[float]] = None,
) -> SpectralResults:
    evals, evecs = laplace_beltrami_spectrum(verts, faces, k=k)
    base_evals, base_evecs = laplace_beltrami_spectrum(baseline_verts, baseline_faces, k=k)

    if weights is None:
        weights = np.ones(k, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != k:
        raise ValueError("weights length must match k eigenvalues.")

    spec_dist = math.sqrt(np.sum(weights * (evals - base_evals) ** 2))

    if times is None:
        times = np.geomspace(1e-4, 1e-1, num=16)
    hks = heat_kernel_signature(evals, evecs, times)
    baseline_hks = heat_kernel_signature(base_evals, base_evecs, times)
    hks_diff = np.linalg.norm(hks - baseline_hks, axis=1)

    return SpectralResults(
        eigenvalues=evals,
        baseline_eigenvalues=base_evals,
        spectrum_distance=spec_dist,
        hks=hks,
        baseline_hks=baseline_hks,
        hks_distance_per_vertex=hks_diff,
    )


def finite_difference_latent_sensitivity(
    decoder: torch.nn.Module,
    template: Meshes,
    baseline_latent: Tensor,
    latent_indices: Optional[Iterable[int]] = None,
    epsilon: float = 1e-2,
    device: Optional[torch.device] = None,
    return_jacobians: bool = False,
) -> LatentSensitivity:
    device = device or baseline_latent.device
    base_mesh = decode_latent(decoder, template, baseline_latent, device)
    base_verts, faces = _verts_and_faces(base_mesh)
    areas = triangle_areas(base_verts, faces)
    area_weights = _vertex_aggregate_from_faces(
        faces, {"area": np.ones(len(faces))}, areas
    )["area"]

    latent_dim = baseline_latent.numel()
    if latent_indices is None:
        latent_indices = range(latent_dim)
    latent_indices = list(latent_indices)

    per_vertex = []
    jacobians: List[np.ndarray] = []
    global_scores = []

    for idx in latent_indices:
        unit = torch.zeros_like(baseline_latent)
        unit[idx] = epsilon

        plus_latent = baseline_latent + unit
        minus_latent = baseline_latent - unit
        mesh_plus = decode_latent(decoder, template, plus_latent, device)
        mesh_minus = decode_latent(decoder, template, minus_latent, device)

        v_plus, _ = _verts_and_faces(mesh_plus)
        v_minus, _ = _verts_and_faces(mesh_minus)

        jac = (v_plus - v_minus) / (2.0 * epsilon)
        sens = np.linalg.norm(jac, axis=1)
        per_vertex.append(sens)
        global_scores.append(math.sqrt(float(np.sum(area_weights * sens**2))))
        if return_jacobians:
            jacobians.append(jac)

    per_vertex_array = np.stack(per_vertex, axis=1)  # (N, len(indices))
    global_scores_array = np.asarray(global_scores, dtype=np.float64)
    jac_stack = np.stack(jacobians, axis=0) if return_jacobians else None
    return LatentSensitivity(
        per_vertex_sensitivity=per_vertex_array,
        global_scores=global_scores_array,
        settings={"epsilon": epsilon},
        jacobian_vectors=jac_stack,
    )


def autodiff_vertex_jacobian_norms(
    decoder: torch.nn.Module,
    template: Meshes,
    latent: Tensor,
    num_probes: int = 16,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    device = device or latent.device
    latent = latent.detach().to(device)
    template = template.to(device)

    def decode(latent_vec: Tensor) -> Tensor:
        mesh = decoder(template.clone(), latent_vec.unsqueeze(0))[-1]
        return mesh.verts_packed()

    base_output = decode(latent)
    n_verts = base_output.shape[0]
    accum = torch.zeros(n_verts, device=device)

    for _ in range(num_probes):
        direction = torch.randn_like(latent)
        _, jvp = torch.autograd.functional.jvp(
            decode,
            latent,
            direction,
            create_graph=False,
        )
        accum += jvp.pow(2).sum(dim=1)

    return torch.sqrt(accum / num_probes).cpu().numpy()


def pullback_metric(
    decoder: torch.nn.Module,
    template: Meshes,
    latent: Tensor,
    area_weights: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    latent_indices: Optional[Iterable[int]] = None,
    epsilon: float = 1e-2,
) -> Tuple[np.ndarray, List[int]]:
    device = device or latent.device
    latent = latent.detach().to(device)

    template = template.to(device)
    baseline_mesh = decode_latent(decoder, template, latent, device)

    if area_weights is None:
        verts_np, faces = _verts_and_faces(baseline_mesh)
        areas = triangle_areas(verts_np, faces)
        area_weights = _vertex_aggregate_from_faces(
            faces, {"area": np.ones(len(faces))}, areas
        )["area"]

    latent_dim = latent.numel()
    if latent_indices is None:
        latent_indices = range(latent_dim)
    latent_indices = list(latent_indices)

    fd = finite_difference_latent_sensitivity(
        decoder,
        template,
        latent,
        latent_indices=latent_indices,
        epsilon=epsilon,
        device=device,
        return_jacobians=True,
    )
    if fd.jacobian_vectors is None:
        raise RuntimeError("Jacobian vectors were not computed.")

    jac = fd.jacobian_vectors  # (len(indices), N, 3)
    n_dims = len(latent_indices)
    metric = np.zeros((n_dims, n_dims), dtype=np.float64)

    for a, jac_a in enumerate(jac):
        for b, jac_b in enumerate(jac):
            metric[a, b] = np.sum(
                area_weights * np.sum(jac_a * jac_b, axis=1)
            )

    return metric, latent_indices


