"""
Mesh Segmentation Refinement Pipeline

This script implements a complete pipeline for refining mesh segmentations using:
1. A trained segmentation CNN to generate initial predictions
2. A mesh decoder for generating refined 3D meshes
3. Optimization-based refinement using volume gradients and mesh deformation

Author: Generated for mesh autodecoder project
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Add the project root to Python path
import sys
project_root = '/home/ralbe/DALS/mesh_autodecoder'
sys.path.append(project_root)

# Standard library imports
import os
import time
import random
import datetime
from collections import defaultdict
from glob import glob
from os.path import join

# Third-party scientific computing
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, sobel
from skimage.measure import marching_cubes

# PyTorch ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

# PyTorch3D for 3D mesh operations
import pytorch3d
import pytorch3d.io
import pytorch3d.utils
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_batch
from pytorch3d.ops import (
    sample_points_from_meshes,
    knn_points,
    GraphConv,
    SubdivideMeshes,
    laplacian,
)
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing

# MONAI medical imaging library
from monai.networks.nets import UNet, VNet, DynUNet, UNETR, SegResNet
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric, compute_meandice, compute_hausdorff_distance
from monai.data import CacheDataset, ThreadDataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.transforms import (
    AddChanneld, AsDiscrete, Compose, ToDeviced, 
    EnsureTyped, EnsureType, KeepLargestConnectedComponent,
)

# Visualization
import plotly
import plotly.subplots
import plotly.graph_objects as go
import plotly.express as px

# Project-specific imports
from util.mesh_plot import plot_wireframe, plot_meshes_wireframe, plot_mesh, plot_meshes
from util.data import load_meshes_in_dir
from util.rasterize import rasterize_vol
from model.mesh_decoder import MeshDecoder, seed_everything
from model.loss import mesh_bl_quality_loss, mesh_edge_loss_highdim, mesh_laplacian_loss_highdim
from model.dals_segmenter import SynVNet_8h2s, _get_kernels_strides

# Try importing metrics with error handling for C++ extension
try:
    from util.metrics import point_metrics, self_intersections
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import metrics module: {e}")
    METRICS_AVAILABLE = False
    
    def point_metrics(*args, **kwargs):
        return {}
    
    def self_intersections(meshes):
        return torch.zeros(len(meshes)), torch.zeros(1)

# Initialize subdivide once for later use
subdivide = SubdivideMeshes()




# =============================================================================
# CONFIGURATION
# =============================================================================

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimization parameters
MAX_ITER_INIT = 0      # Initial rough registration iterations
LR_INIT = 1e-2         # Learning rate for initial registration
MAX_ITER_ALIGN = 20    # Alignment optimization iterations (offset/scale)
LR_ALIGN = 1e-2        # Learning rate for alignment
MAX_ITER_DEFORM = 50   # Deformation optimization iterations
LR_DEFORM = 1e-4       # Learning rate for deformation

# Visualization parameters
ENABLE_SLICE_VISUALIZATION = False
ENABLE_3D_MESH_VISUALIZATION = False
ENABLE_ALIGNMENT_VISUALIZATION = False

# Metrics parameters
NUM_POINTS_FOR_METRICS = 100
NUM_POINTS_FOR_OPTIMIZATION = 5000

# Data paths
TRAIN_DATA_PATH = '/scratch/ralbe/dals_data/train_data_mixed.pt'
VAL_DATA_PATH = '/scratch/ralbe/dals_data/val_data_mixed.pt'
# Alternative paths for small subset:
# TRAIN_DATA_PATH = '/home/ralbe/DALS/mesh_autodecoder/data/train_data.pt'
# VAL_DATA_PATH = '/home/ralbe/DALS/mesh_autodecoder/data/val_data.pt'

# Model paths
VNET_MODEL_PATH = '/home/ralbe/DALS/mesh_autodecoder/models/ConvNetTrainer_VNet_2025-10-15_15-36.ckpt'
MESH_DECODER_PATH = '/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-10-27_14-04-44.ckpt'

# Initialize keep_largest_cc for processing prediction masks
keep_largest_cc = KeepLargestConnectedComponent(None)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def dict_unzip(d):
    """
    Convert a dictionary of arrays into a list of dictionaries.
    
    Args:
        d: Dictionary where values are arrays of the same length
        
    Returns:
        List of dictionaries, one for each element in the arrays
    """
    num_elems = len(d[list(d.keys())[0]])
    out = [dict() for _ in range(num_elems)]
    for key, array in d.items():
        assert len(array) == num_elems, "All arrays must have same length"
        for i, v in enumerate(array):
            out[i][key] = v
    return out


def compute_face_area_vert_weights(mesh):
    """
    Compute vertex weights based on face areas for mesh operations.
    
    Args:
        mesh: PyTorch3D Meshes object
        
    Returns:
        Tensor of shape (N, 1) containing vertex weights
    """
    weights = torch.zeros(len(mesh.verts_packed()), device=mesh.device)
    faces = mesh.faces_packed()
    face_areas = mesh.faces_areas_packed()
    weights[faces[:, 0]] += face_areas / 3
    weights[faces[:, 1]] += face_areas / 3
    weights[faces[:, 2]] += face_areas / 3
    return weights.unsqueeze(1)


def sample_vol(verts, vol):
    """
    Trilinearly sample values from a volumetric tensor at given vertex locations.

    Args:
        verts: (N, 3) tensor of 3D coordinates, normalized to [-1, 1] (grid_sample coordinates)
        vol: (1, C, D, H, W) volumetric tensor (batch=1, C=channels/classes, D/H/W=depth/height/width)
        
    Returns:
        (N, C) tensor of sampled values at each input vertex (or (N,) if C=1)
    """
    # Reshape verts for 5D grid_sample input: Single batch, no time, all verts as a flat "spatial" axis
    grid = verts.view(1, 1, 1, *verts.shape)            # Shape: (1, 1, 1, N, 3)
    nc = vol.shape[1]                                   # Number of channels (C)
    # Trilinearly interpolate volumetric values at each point
    samples = F.grid_sample(vol, grid, align_corners=False, padding_mode='border')
    # Reshape output as (N, C) and remove unnecessary singleton dimensions
    return samples.view(nc, len(verts)).T.squeeze()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_slice_comparison(image, label, pred, central_slice=None):
    """
    Visualize image slice with ground truth and prediction overlays.
    
    Args:
        image: Input image tensor
        label: Ground truth label tensor
        pred: Prediction tensor
        central_slice: Slice index to visualize (default: middle slice)
    """
    if not ENABLE_SLICE_VISUALIZATION:
        return
        
    if central_slice is None:
        central_slice = image.shape[0] // 2
    
    img_slice = image[..., central_slice].cpu().numpy()
    label_slice = label[..., central_slice].cpu().numpy()
    pred_slice = pred[1, ..., central_slice].cpu().numpy() if pred.dim() > 3 else pred[..., central_slice].cpu().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(img_slice, cmap='gray')
    axs[0].set_title('Image (slice)')
    axs[0].axis('off')
    
    axs[1].imshow(img_slice, cmap='gray')
    axs[1].imshow(label_slice, alpha=0.4, cmap='Reds')
    axs[1].set_title('Mask (GT)')
    axs[1].axis('off')
    
    axs[2].imshow(img_slice, cmap='gray')
    axs[2].imshow(pred_slice > 0.5, alpha=0.4, cmap='Blues')
    axs[2].set_title('Prediction (thr=0.5)')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_mesh_comparison(prediction_mesh, label_mesh, title="Mesh Comparison"):
    """
    Visualize comparison between prediction and ground truth meshes.
    
    Args:
        prediction_mesh: Predicted mesh object
        label_mesh: Ground truth mesh object
        title: Title for the plot
    """
    if not ENABLE_3D_MESH_VISUALIZATION:
        return
        
    fig = go.Figure()
    plot_meshes_wireframe(prediction_mesh, fig=fig, opacity=0.5)
    plot_meshes_wireframe(label_mesh, fig=fig, opacity=0.5)
    fig.update_layout(height=800, title=title)
    fig.show()


# =============================================================================
# MESH PROCESSING FUNCTIONS
# =============================================================================

def create_ground_truth_mesh(label, size0):
    """
    Create ground truth mesh from label using marching cubes.
    
    Args:
        label: Ground truth label tensor
        size0: Size parameter for normalization
        
    Returns:
        PyTorch3D Meshes object containing the ground truth mesh
    """
    verts, faces, _, _ = marching_cubes(label.numpy())
    print(f'GT mask converted to mesh of size {verts.shape}')
    print(f'GT mesh COM (image space): {verts.mean(axis=0)}')
    
    label_mesh = Meshes(
        [torch.from_numpy((verts.copy()) / size0 * 2 - 1).float()],
        [torch.from_numpy(faces.copy()).long()]
    )
    print(f'GT mesh COM (-1,1 space): {label_mesh.verts_packed().mean(axis=0)}')
    
    return label_mesh


def process_prediction_mask(pred, size0):
    """
    Process prediction mask and compute gradients.
    
    Args:
        pred: Prediction tensor from the segmentation model
        size0: Size parameter for normalization
        
    Returns:
        Tuple of (thresholded_mask, processed_mask, gradient_tensor)
    """
    blabel = pred[1]
    blabel_thr = keep_largest_cc(blabel > 0.5)
    # Ensure blabel_thr is on CPU for further downstream use as well as numpy/scipy
    blabel_thr_cpu = blabel_thr.cpu()
    blabel_thr_np = blabel_thr_cpu.numpy()
    blabel_udf = torch.from_numpy(
        distance_transform_edt(~blabel_thr_np) + distance_transform_edt(blabel_thr_np)
    )
    blabel = blabel.unsqueeze(0).unsqueeze(0).float()
    blabel_udf = blabel_udf.unsqueeze(0).unsqueeze(0).float()

    # Compute gradients using Sobel filter; move data to CPU and numpy if needed
    udf_np = blabel_udf[0, 0].cpu().numpy()
    blabel_dx = torch.from_numpy(sobel(udf_np, axis=2)) / size0
    blabel_dy = torch.from_numpy(sobel(udf_np, axis=1)) / size0
    blabel_dz = torch.from_numpy(sobel(udf_np, axis=0)) / size0

    blabel_grad = torch.stack([blabel_dx, blabel_dy, blabel_dz], dim=0).unsqueeze(0)

    return blabel_thr_cpu, blabel, blabel_grad


def create_prediction_mesh(blabel_thr, size0):
    """
    Create prediction mesh from thresholded binary label.
    
    Args:
        blabel_thr: Thresholded binary label tensor
        size0: Size parameter for normalization
        
    Returns:
        Tuple of (mesh_in_image_space, mesh_in_normalized_space)
    """
    # Ensure input is on CPU before converting to numpy
    blabel_thr_cpu = blabel_thr.cpu()
    mc_verts, mc_faces, _, _ = marching_cubes(blabel_thr_cpu.numpy())

    prediction_mesh = Meshes(
        [torch.from_numpy(mc_verts.copy())],
        [torch.from_numpy(mc_faces.copy())]
    )
    prediction_mesh_img_space = Meshes(
        [torch.from_numpy((mc_verts.copy()) / size0 * 2 - 1)],
        [torch.from_numpy(mc_faces.copy())]
    )

    print(f'Predicted mask converted to mesh of size {mc_verts.shape}')
    print(f'Predicted Mesh COM (image space): {prediction_mesh.verts_packed().mean(axis=0)}')
    print(f'Predicted Mesh COM (-1,1 space): {prediction_mesh_img_space.verts_packed().mean(axis=0)}')

    return prediction_mesh, prediction_mesh_img_space


def visualize_mesh_comparison(prediction_mesh, label_mesh, title="Mesh Comparison"):
    """Visualize comparison between prediction and ground truth meshes."""
    if not ENABLE_3D_MESH_VISUALIZATION:
        return
        
    fig = go.Figure()
    plot_meshes_wireframe(prediction_mesh, fig=fig, opacity=0.5)
    plot_meshes_wireframe(label_mesh, fig=fig, opacity=0.5)
    fig.update_layout(height=800, title=title)
    fig.show()


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def compute_initial_alignment(mc_verts, size0):
    """
    Compute initial offset and scale for mesh alignment.
    
    Args:
        mc_verts: Mesh vertices from marching cubes
        size0: Size parameter for normalization
        
    Returns:
        Tuple of (offset, scale) tensors normalized to [-1, 1] space
    """
    mc_verts = torch.from_numpy(mc_verts.copy())
    offset = mc_verts.mean(dim=0, keepdims=True)
    scale = torch.max(torch.norm(mc_verts - offset, dim=1))
    
    print(f'Initial offset: {offset}')
    print(f'Initial scale: {scale}')
    
    # Normalize to [-1, 1] space
    offset = offset / size0 * 2 - 1
    scale = scale.unsqueeze(0) / size0 * 2
    
    print(f'Normalized offset: {offset}')
    print(f'Normalized scale: {scale}')
    
    return offset, scale


def perform_initial_registration(decoder, search_template, lv, offset, scale, blabel_grad, 
                               max_iter_init, lr_init, verts_mean):
    """
    Perform initial rough registration using volume gradients.
    
    Args:
        decoder: Mesh decoder model
        search_template: Template mesh for optimization
        lv: Latent vector
        offset: Initial offset tensor
        scale: Initial scale tensor
        blabel_grad: Gradient tensor from prediction mask
        max_iter_init: Maximum iterations for initial registration
        lr_init: Learning rate for initial registration
        verts_mean: Mean of vertices
        
    Returns:
        Tuple of (predicted_mesh, updated_offset, updated_scale)
    """
    with torch.no_grad():
        pred_mesh = decoder(search_template, lv)[-1]
        verts = pred_mesh.verts_packed()
        normals = pred_mesh.verts_normals_packed()
        weights = compute_face_area_vert_weights(pred_mesh)
        weights_sum = weights.sum()
        
        print(f"Ran decoder for initial alignment mesh. verts shape: {verts.shape}")
        
        for i in range(max_iter_init):
            f_ext = sample_vol((verts - verts_mean) * scale + offset, blabel_grad)
            offset -= lr_init * torch.sum(weights * f_ext, dim=0) / weights_sum
            scale -= lr_init * torch.sum(weights * torch.sum(f_ext * normals, dim=1, keepdim=True)) / weights_sum
            print(f"Initial registration iter {i+1}/{max_iter_init}: offset={offset.cpu().numpy().squeeze()}, scale={scale.cpu().numpy().squeeze()}")
    
    return pred_mesh, offset, scale


def optimize_alignment_offset_scale(
    decoder,
    search_template,
    lv,
    offset,
    scale,
    prediction_mesh_img_space,
    max_iters,
    lr,
):
    """
    Refine offset and scale to align the average decoded mesh to the initial
    prediction by minimizing Chamfer distance (no latent optimization here).
    """
    offset = offset.clone().detach().requires_grad_(True)
    scale = scale.clone().detach().requires_grad_(True)

    optim = torch.optim.Adam([offset, scale], lr=lr)

    # Fixed target points from the initial prediction (in normalized space)
    target_points = sample_points_from_meshes(
        prediction_mesh_img_space, NUM_POINTS_FOR_OPTIMIZATION
    ).to(offset.device)

    # Expand latent vector to match template vertices
    lv_expanded = lv.repeat(len(search_template.verts_packed()), 1)

    for _ in range(max_iters):
        optim.zero_grad()
        with torch.no_grad():
            pred_mesh = decoder(search_template, lv_expanded, expand_lv=False)[-1]
        pred_points = sample_points_from_meshes(
            pred_mesh, NUM_POINTS_FOR_OPTIMIZATION
        )
        verts_mean_iter = pred_mesh.verts_packed().mean()
        pred_points_t = (pred_points - verts_mean_iter) * scale + offset
        chamfer_val = chamfer_distance(pred_points_t, target_points)
        loss = chamfer_val[0] if isinstance(chamfer_val, tuple) else chamfer_val
        loss.backward()
        optim.step()

    return offset.detach(), scale.detach()

def perform_mesh_deformation_optimization(decoder, search_template, lv, offset, scale, 
                                        target_points,
                                        max_iter_deform, lr_deform):
    """
    Perform mesh deformation optimization using latent vector updates.
    Uses Chamfer loss against the scaled prediction and Laplacian regularization.
    
    Args:
        decoder: Mesh decoder model
        search_template: Template mesh for optimization
        lv: Latent vector
        offset: Offset tensor
        scale: Scale tensor
        target_points: Target points to match (from scaled prediction)
        max_iter_deform: Maximum iterations for deformation
        lr_deform: Learning rate for deformation
        
    Returns:
        Tuple of (predicted_mesh, updated_offset, updated_scale, energies, chamfers)
    """
    print(f"DEBUG perform_mesh_deformation_optimization:")
    print(f"  target_points shape: {target_points.shape}")
    print(f"  target_points device: {target_points.device}")
    print(f"  offset shape: {offset.shape}, scale shape: {scale.shape}")
    print(f"  offset device: {offset.device}, scale device: {scale.device}")
    print(f"  lv shape before repeat: {lv.shape}")
    
    lv = lv.repeat(len(search_template.verts_packed()), 1)
    lv.requires_grad_(True)
    print(f"Latent vector repeated for vertices, lv shape: {lv.shape}")
    
    optim = torch.optim.Adam([lv], lr=lr_deform)
    
    decoder.eval()
    decoder.requires_grad_(False)
    energies = []
    chamfers = []
    
    for it in range(max_iter_deform):
        optim.zero_grad()
        pred_mesh = decoder(search_template, lv, expand_lv=False)[-1]
        
        points, _ = sample_points_from_meshes(pred_mesh, NUM_POINTS_FOR_OPTIMIZATION, return_normals=True)
        verts_mean_iter = pred_mesh.verts_packed().mean()
        transformed_points = (points[0] - verts_mean_iter) * scale + offset
        
        # Debug print statements
        if it == 0:  # Only print on first iteration
            print(f"DEBUG: points shape: {points.shape}")
            print(f"DEBUG: transformed_points shape: {transformed_points.shape}")
            print(f"DEBUG: target_points shape: {target_points.shape}")
            print(f"DEBUG: scale shape: {scale.shape}, offset shape: {offset.shape}")
            print(f"DEBUG: verts_mean_iter shape: {verts_mean_iter.shape}")
        
        chamfer_val = chamfer_distance(transformed_points, target_points)
        chamfer_loss = chamfer_val[0] if isinstance(chamfer_val, tuple) else chamfer_val
        energies.append(0.0)
        chamfers.append(chamfer_loss.cpu().item())
        loss = 0
        loss += chamfer_loss
        loss += 1e8 * mesh_laplacian_loss_highdim(pred_mesh, lv, p=2)
        loss.backward()
        optim.step()
        
        if it % 10 == 0:  # Print progress every 10 iterations
            print(f"Deformation iter {it+1}/{max_iter_deform}: Chamfer={chamfer_loss.cpu().item():.4f}")
    
    return pred_mesh, offset, scale, energies, chamfers


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def compute_evaluation_metrics(pred_mesh, label_mesh, blabel_thr, label, verts_mean, 
                              offset, scale, size0, num_points=NUM_POINTS_FOR_METRICS):
    """
    Compute Dice, Hausdorff, and Chamfer distance metrics.
    
    Args:
        pred_mesh: Predicted mesh object
        label_mesh: Ground truth mesh object
        blabel_thr: Thresholded binary label
        label: Ground truth label tensor
        verts_mean: Mean of vertices
        offset: Offset tensor
        scale: Scale tensor
        size0: Size parameter for normalization
        num_points: Number of points for sampling metrics
        
    Returns:
        Dictionary containing all computed metrics
    """
    with torch.no_grad():
        # Rasterize predicted mesh to volume
        rasterize_verts = (pred_mesh.verts_packed() - verts_mean) * scale + offset
        rasterize_verts = rasterize_verts[:, :]
        rasterize_mesh = Meshes([rasterize_verts], [pred_mesh.faces_packed()])
        pred_mesh_vol = rasterize_vol(rasterize_mesh, label.shape)
        print(f"Rasterized predicted mesh to volume. pred_mesh_vol shape: {pred_mesh_vol.shape}")
        
        # Sample points for Chamfer distance
        label_points = sample_points_from_meshes(label_mesh, num_points).to(device)
        
        # Create baseline mesh from thresholded prediction
        verts, faces, _, _ = marching_cubes(blabel_thr.cpu().numpy())
        verts_tensor = torch.from_numpy(verts.copy()).to(device)
        faces_tensor = torch.from_numpy(faces.copy()).to(device)
        blabel_mesh = Meshes([verts_tensor / size0 * 2 - 1], [faces_tensor])
        print(f"Recomputed blabel_mesh for baseline metrics. Verts: {verts.shape}, Faces: {faces.shape}")
        
        # Move all arrays/tensors involved in metric computations to same device for MONAI
        pred_mesh_vol_device = pred_mesh_vol.to(device)
        label_device = label.to(device)
        blabel_thr_device = blabel_thr.to(device)
        
        # Compute metrics for our method
        dice_mesh = compute_meandice(
            one_hot(pred_mesh_vol_device.unsqueeze(0).unsqueeze(0), 2),
            one_hot(label_device.unsqueeze(0).unsqueeze(0), 2),
        )[0][1].cpu().item()
        
        haus_mesh = compute_hausdorff_distance(
            one_hot(pred_mesh_vol_device.unsqueeze(0).unsqueeze(0), 2),
            one_hot(label_device.unsqueeze(0).unsqueeze(0), 2),
        ).cpu().item()
        
        chamfer_mesh = chamfer_distance(
            sample_points_from_meshes(rasterize_mesh, num_points).to(device),
            label_points
        )[0].cpu().item() * 1e4
        
        # Compute metrics for baseline (convnet)
        dice_conv = compute_meandice(
            one_hot(blabel_thr_device.unsqueeze(0).unsqueeze(0), 2),
            one_hot(label_device.unsqueeze(0).unsqueeze(0), 2),
        )[0][1].cpu().item()
        
        haus_conv = compute_hausdorff_distance(
            one_hot(blabel_thr_device.unsqueeze(0).unsqueeze(0), 2),
            one_hot(label_device.unsqueeze(0).unsqueeze(0), 2),
        ).cpu().item()
        
        chamfer_conv = chamfer_distance(
            sample_points_from_meshes(blabel_mesh, num_points).to(device),
            label_points
        )[0].cpu().item() * 1e4
        
        print(f"Metrics - Dice: {dice_mesh:.4f} (ours), {dice_conv:.4f} (convnet) | "
              f"Hausdorff: {haus_mesh:.4f} (ours), {haus_conv:.4f} (convnet) | "
              f"Chamfer: {chamfer_mesh:.2f} (ours), {chamfer_conv:.2f} (convnet)")
        
        return {
            'dice_mesh': dice_mesh,
            'haus_mesh': haus_mesh,
            'chamfer_mesh': chamfer_mesh,
            'dice_conv': dice_conv,
            'haus_conv': haus_conv,
            'chamfer_conv': chamfer_conv,
            'pred_mesh_vol': pred_mesh_vol,
            'rasterize_mesh': rasterize_mesh
        }


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_single_sample(idx, image, label, model, decoder, latent_vectors, template, 
                         search_template, size0):
    """
    Process a single validation sample through the complete pipeline.
    
    Args:
        idx: Sample index
        image: Input image tensor
        label: Ground truth label tensor
        model: Segmentation CNN model
        decoder: Mesh decoder model
        latent_vectors: Latent vector embeddings
        template: Template mesh
        search_template: Subdivided template mesh
        size0: Size parameter for normalization
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n======= Processing validation sample {idx+1} =======")
    
    # Get prediction from model
    with torch.no_grad():
        pred = torch.sigmoid(model(image.unsqueeze(0).unsqueeze(0).to(device)).squeeze()).to(device)
    
    # Visualize slice comparison
    visualize_slice_comparison(image, label, pred)
    
    # Create ground truth mesh
    label_mesh = create_ground_truth_mesh(label, size0).to(device)
    
    # Process prediction mask
    blabel_thr, blabel, blabel_grad = process_prediction_mask(pred, size0)
    blabel_grad = blabel_grad.to(device)
    
    # Create prediction mesh
    prediction_mesh, prediction_mesh_img_space = create_prediction_mesh(blabel_thr, size0)
    
    # Visualize mesh comparison
    visualize_mesh_comparison(prediction_mesh_img_space, label_mesh, 'Blue: Predicted, Red: GT')
    
    # Compute initial alignment
    mc_verts, mc_faces, _, _ = marching_cubes(blabel_thr.numpy())
    offset, scale = compute_initial_alignment(mc_verts, size0)
    offset = offset.to(device)
    scale = scale.to(device)
    
    # Get initial latent vector
    lv = latent_vectors.weight.data.mean(dim=0).clone().unsqueeze(0)

    # Optional: force-based rough registration (kept but can be 0 iters)
    pred_mesh, offset, scale = perform_initial_registration(
        decoder, search_template, lv, offset, scale, blabel_grad,
        MAX_ITER_INIT, LR_INIT, prediction_mesh.verts_packed().mean()
    )

    # New: refine offset/scale by minimizing Chamfer to initial prediction
    print("Stage 1: Optimizing offset and scale alignment...")
    offset, scale = optimize_alignment_offset_scale(
        decoder, search_template, lv, offset, scale,
        prediction_mesh_img_space, max_iters=MAX_ITER_ALIGN, lr=LR_ALIGN
    )
    print(f"Alignment complete. Final offset: {offset.cpu().numpy().squeeze()}, scale: {scale.cpu().numpy().squeeze()}")
    
    # Visualize alignment result
    if ENABLE_ALIGNMENT_VISUALIZATION:
        fig = go.Figure()
        plot_meshes_wireframe(pred_mesh, fig=fig)
        plot_meshes_wireframe(label_mesh, fig=fig)
        fig.update_layout(height=800)
        fig.show()
    
    # Store alignment-only results
    all_meshes_only_align.append(pred_mesh.clone())
    all_offsets_only_align.append(offset.clone())
    all_scales_only_align.append(scale.clone())
    
    # Prepare target points from the scaled prediction (fixed during deformation)
    print(f"DEBUG: Preparing target points from prediction_mesh_img_space")
    print(f"DEBUG: prediction_mesh_img_space device: {prediction_mesh_img_space.device}")
    target_points_result = sample_points_from_meshes(
        prediction_mesh_img_space, NUM_POINTS_FOR_OPTIMIZATION
    )
    print(f"DEBUG: sample_points_from_meshes returned type: {type(target_points_result)}")
    if isinstance(target_points_result, tuple):
        print(f"DEBUG: It's a tuple! Len={len(target_points_result)}")
        target_points = target_points_result[0].to(device)
    else:
        target_points = target_points_result.to(device)
    print(f"DEBUG: target_points final shape: {target_points.shape}, device: {target_points.device}")

    # Perform mesh deformation optimization against scaled prediction
    print("Stage 2: Optimizing latent code with Chamfer + Laplacian...")
    pred_mesh, offset, scale, energies, chamfers = perform_mesh_deformation_optimization(
        decoder, search_template, lv, offset, scale, target_points,
        MAX_ITER_DEFORM, LR_DEFORM
    )

    # Compute evaluation metrics
    metrics = compute_evaluation_metrics(
        pred_mesh, label_mesh, blabel_thr, label, pred_mesh.verts_packed().mean(),
        offset, scale, size0
    )
    
    # Store results
    all_meshes.append(pred_mesh.detach())
    all_offsets.append(offset)
    all_scales.append(scale)
    all_verts_means.append(pred_mesh.verts_packed().mean())
    all_rasterized_vols.append(metrics['pred_mesh_vol'].cpu().clone())
    all_label_meshes.append(label_mesh)
    all_blabel_thr.append(blabel_thr)
    
    return metrics


def load_data():
    """
    Load training and validation data from files.
    
    Returns:
        Tuple of (train_data, val_data, size_info)
    """
    print('Loading data...')
    train_data = torch.load(TRAIN_DATA_PATH, map_location='cpu')
    val_data = torch.load(VAL_DATA_PATH, map_location='cpu')

    # Sanity checks
    for split, data in {'train': train_data, 'val': val_data}.items():
        n_img, n_lab, n_msk = data['images'].shape[0], data['labels'].shape[0], data['masks'].shape[0]
        assert n_img == n_lab == n_msk, f"Mismatch in {split} data entries"
        s_img, s_lab, s_msk = data['images'].shape[1:], data['labels'].shape[1:], data['masks'].shape[1:]
        assert s_img == s_lab == s_msk, f"Mismatch in {split} data sizes"

    train_n = train_data['images'].shape[0]
    val_n = val_data['images'].shape[0]
    size = train_data['images'].shape[1:]
    print(f"Train: {train_n} images, Val: {val_n} images, Image size: {size}")

    train_data = dict_unzip(train_data)
    val_data = dict_unzip(val_data)
    
    return train_data, val_data, size


def load_segmentation_model():
    """
    Load the trained segmentation CNN model.
    
    Returns:
        Loaded segmentation model
    """
    print('Loading segmentation ConvNet model...')
    checkpoint = torch.load(VNET_MODEL_PATH, map_location='cpu')
    hparams = checkpoint['hparams']
    model_type = hparams['model'].upper()

    if model_type == 'RESUNET':
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif model_type == 'VNET':
        model = VNet(spatial_dims=3, in_channels=1, out_channels=2)
    elif model_type == 'DYNUNET':
        kernels, strides = _get_kernels_strides(hparams['data_size'], hparams['data_spacing'])
        model = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            kernel_size=kernels, strides=strides,
            upsample_kernel_size=strides[1:], norm_name='instance',
        )
    elif model_type == 'UNETR':
        model = UNETR(
            in_channels=1, out_channels=2, img_size=hparams['data_size'],
            feature_size=16, hidden_size=384, mlp_dim=1536, num_heads=6,
            pos_embed="conv", norm_name="instance",
            res_block=True, dropout_rate=0.0,
        )
    elif model_type == 'UNET':
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        )
    elif model_type == 'SYNVNET3D':
        model = SynVNet_8h2s(
            n_channels=1, n_classes=2, n_filters=16,
            normalization='batchnorm', has_dropout=True,
        )
    elif model_type == 'SEGRESNET':
        model = SegResNet(spatial_dims=3, in_channels=1, out_channels=2)
    else:
        raise ValueError(f'Unknown model: {model_type}')

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    model.requires_grad = False
    return model


def load_mesh_decoder():
    """
    Load the trained mesh decoder model and associated components.
    
    Returns:
        Tuple of (decoder, latent_vectors, template)
    """
    print('Loading mesh decoder model...')
    checkpoint = torch.load(MESH_DECODER_PATH, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    template = checkpoint.get('template')
    print("\nTEMPLATE info:")
    if isinstance(template, dict):
        print(f"  Keys: {list(template.keys())}")
        for k, v in template.items():
            if hasattr(v, 'shape'):
                print(f"    - {k}: shape {tuple(v.shape)}")
            else:
                print(f"    - {k}: type {type(v)}")
    elif hasattr(template, 'shape'):
        print(f"  Shape: {tuple(template.shape)}")
    else:
        print(f"  Type: {type(template)}")

    print("\nSCHEDULER STATE DICT keys:", list(checkpoint.get('scheduler_state_dict', {}).keys()))
    for k, v in checkpoint.get('scheduler_state_dict', {}).items():
        print(f"    - {k}: {type(v)}")

    print("\nOPTIMIZER STATE DICT keys:", list(checkpoint.get('optimizer_state_dict', {}).keys()))

    print(f"\nEPOCH: {checkpoint.get('epoch', 'N/A')}")
    print(f"BEST EPOCH: {checkpoint.get('best_epoch', 'N/A')}")
    print(f"BEST LOSS: {checkpoint.get('best_loss', 'N/A')}")
    print(f"BEST EPOCH LOSSES: {checkpoint.get('best_epoch_losses', 'N/A')}")

    hparams = checkpoint['hparams']

    decoder = MeshDecoder(
        hparams['latent_features'],
        hparams['steps'],
        hparams['hidden_features'],
        hparams['subdivide'],
        mode=hparams['decoder_mode'],
        norm=hparams['normalization'][0],
    )
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    decoder = decoder.to(device)

    latent_vectors = checkpoint['latent_vectors'].to(device)
    latent_vectors.eval()
    template = checkpoint['template'].to(device)

    num_vertices = template.verts_packed().shape[0]
    print(f"Number of mesh vertices in template: {num_vertices}")
    
    return decoder, latent_vectors, template


if __name__ == "__main__":
    # Load data and models
    train_data, val_data, size = load_data()
    model = load_segmentation_model()
    decoder, latent_vectors, template = load_mesh_decoder()
    
    # Initialize storage lists for results
    dice_scores = []
    dice_scores_convnet = []
    hausdorff_dists = []
    hausdorff_dists_convnet = []
    chamfer_dists = []
    chamfer_dists_convnet = []

    all_meshes = []
    all_offsets = []
    all_scales = []
    all_meshes_only_align = []
    all_offsets_only_align = []
    all_scales_only_align = []
    all_label_meshes = []
    all_verts_means = []
    all_rasterized_vols = []
    all_blabel_thr = []

    print("Starting validation loop over", len(val_data), "samples.")
    print(f"Configuration: MAX_ITER_INIT={MAX_ITER_INIT}, LR_INIT={LR_INIT}")
    print(f"Alignment: MAX_ITER_ALIGN={MAX_ITER_ALIGN}, LR_ALIGN={LR_ALIGN}")
    print(f"Deformation: MAX_ITER_DEFORM={MAX_ITER_DEFORM}, LR_DEFORM={LR_DEFORM}")
    print(f"Visualization: slice={ENABLE_SLICE_VISUALIZATION}, 3D={ENABLE_3D_MESH_VISUALIZATION}, alignment={ENABLE_ALIGNMENT_VISUALIZATION}")

    # Create search template once  
    search_template = subdivide(template.cpu()).to(device)
    size0 = size[0]

    # Process each validation sample
    for idx in trange(len(val_data)):
        image = val_data[idx]['images']
        label = val_data[idx]['labels']
        size0 = image.shape[0]
        
        metrics = process_single_sample(
            idx, image, label, model, decoder, latent_vectors, template, search_template, size0
        )
        
        # Store metrics
        dice_scores.append(metrics['dice_mesh'])
        dice_scores_convnet.append(metrics['dice_conv'])
        hausdorff_dists.append(metrics['haus_mesh'])
        hausdorff_dists_convnet.append(metrics['haus_conv'])
        chamfer_dists.append(metrics['chamfer_mesh'])
        chamfer_dists_convnet.append(metrics['chamfer_conv'])

    # Convert metric lists to numpy arrays for final computation
    dice_scores = np.array(dice_scores)
    hausdorff_dists = np.array(hausdorff_dists)
    chamfer_dists = np.array(chamfer_dists)
    dice_scores_convnet = np.array(dice_scores_convnet)
    hausdorff_dists_convnet = np.array(hausdorff_dists_convnet)
    chamfer_dists_convnet = np.array(chamfer_dists_convnet)

    # Print final results
    print("\n======= Evaluation Complete =======")
    print(f"Final mean/std dice (ours): {dice_scores.mean():.4f} ± {dice_scores.std():.4f}")
    print(f"Final mean/std dice (convnet): {dice_scores_convnet.mean():.4f} ± {dice_scores_convnet.std():.4f}")
    print(f"Final mean/std hausdorff (ours): {hausdorff_dists.mean():.4f} ± {hausdorff_dists.std():.4f}")
    print(f"Final mean/std hausdorff (convnet): {hausdorff_dists_convnet.mean():.4f} ± {hausdorff_dists_convnet.std():.4f}")
    print(f"Final mean/std chamfer (ours): {chamfer_dists.mean():.2f} ± {chamfer_dists.std():.2f}")
    print(f"Final mean/std chamfer (convnet): {chamfer_dists_convnet.mean():.2f} ± {chamfer_dists_convnet.std():.2f}")
    print('Done')

