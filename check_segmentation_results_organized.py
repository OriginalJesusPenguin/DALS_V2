# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import os
import sys
# Add the project root to Python path
project_root = '/home/ralbe/DALS/mesh_autodecoder'
sys.path.append(project_root)
import random
import datetime
import time
from collections import defaultdict
from glob import glob
from os.path import join

# Third-party scientific computing
import numpy as np
from scipy.ndimage import distance_transform_edt, sobel
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch3D
import pytorch3d
import pytorch3d.io
import pytorch3d.utils
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import (
    Meshes,
    Pointclouds,
    join_meshes_as_batch,
)
from pytorch3d.ops import (
    sample_points_from_meshes,
    knn_points,
    GraphConv,
    SubdivideMeshes,
    laplacian,
)
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
)
from pytorch3d._C import point_face_dist_forward

# MONAI medical imaging library
from monai.networks.nets import (
    UNet,
    VNet,
    DynUNet,
    UNETR,
)
from monai.data import (
    CacheDataset,
    ThreadDataLoader,
    decollate_batch,
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, compute_meandice, compute_hausdorff_distance
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    Compose,
    ToDeviced,
    EnsureTyped,
    EnsureType,
    KeepLargestConnectedComponent,
)
from monai.networks.utils import one_hot

# Visualization
import matplotlib.pyplot as plt
import plotly
import plotly.subplots
import plotly.graph_objects as go
import plotly.express as px

# Progress bars
from tqdm import tqdm, trange

# Project-specific imports
from util.mesh_plot import (
    plot_wireframe,
    plot_meshes_wireframe,
    plot_mesh,
    plot_meshes,
)
from util.data import load_meshes_in_dir
from util.metrics import point_metrics, self_intersections
from util.rasterize import rasterize_vol

from model.mesh_decoder import (
    MeshDecoder,
    seed_everything,
)
from model.loss import (
    mesh_bl_quality_loss,
    mesh_edge_loss_highdim,
    mesh_laplacian_loss_highdim,
)


# Initialize subdivide once for later use
subdivide = SubdivideMeshes()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def dict_unzip(d):
    """Turn dict of arrays into list of dicts."""
    num_elems = len(d[list(d.keys())[0]])
    out = [dict() for _ in range(num_elems)]
    for key, array in d.items():
        assert len(array) == num_elems, "All arrays must have same length"
        for i, v in enumerate(array):
            out[i][key] = v
    return out

# =============================================================================
# MESH PROCESSING FUNCTIONS
# =============================================================================

def meshify_label_gt(label, size0, device):
    """
    Given a volumetric mask `label`, extract a surface mesh and normalize vertices to [-1,1],
    then recenter so the center of mass (COM) is 0.
    Returns a Meshes object.
    """
    verts, faces, _, _ = marching_cubes(label.numpy())
    verts = torch.from_numpy(verts.copy()).float().to(device)
    faces = torch.from_numpy(faces.copy()).long().to(device)
    # Normalize to [-1, 1]
    verts_normed = verts / size0 * 2 - 1
    # Center the COM at 0
    com = verts_normed.mean(dim=0, keepdim=True)
    verts_normed_centered = verts_normed - com
    return Meshes([verts_normed_centered], [faces])

def compute_mesh_geometric_stats(verts):
    """
    Compute center of mass (mean of verts) and bounding radius (furthest from mean).
    """
    com = verts.mean(dim=0, keepdims=True)
    dists = torch.norm(verts - com, dim=1)
    r = torch.max(dists)
    return com, r

def normalize_mesh_verts(verts, com, r, size0):
    """
    Center, scale, translate, and normalize mesh vertices to [-1, 1] cube.
    """
    centered = verts - verts.mean(dim=0)
    scaled = centered * r
    translated = scaled + com
    return (translated / size0) * 2 - 1

def meshify_preds(decoder, latent_vector, template, size0, com, r):
    """
    Given model, mean latent code, template mesh, output a Meshes object (normalized to [-1,1]).
    This function is mesh-agnostic: you pass in any template (coarse, subdivided, etc).
    """
    pred = decoder(template, latent_vector, expand_lv=True)[-1]
    verts = pred.verts_packed().float()
    faces = pred.faces_packed().long()
    verts_normed = normalize_mesh_verts(verts, com, r, size0)
    return Meshes([verts_normed], [faces])

def sample_vol(verts, vol):
    """
    Trilinearly sample values from a volumetric tensor at given vertex locations.

    Args:
        verts: (N, 3) tensor of 3D coordinates, normalized to [-1, 1] (i.e., grid_sample coordinates)
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

def compute_face_area_vert_weights(mesh):
    """Compute vertex weights based on face areas."""
    weights = torch.zeros(len(mesh.verts_packed()), device=mesh.device)
    faces = mesh.faces_packed()
    face_areas = mesh.faces_areas_packed()
    weights[faces[:, 0]] += face_areas / 3
    weights[faces[:, 1]] += face_areas / 3
    weights[faces[:, 2]] += face_areas / 3
    return weights.unsqueeze(1)

def split_and_collapse_sphere(mesh, template, split_thr=4/3, collapse_thr=4/5):
    """Split and collapse edges of a sphere mesh based on length thresholds."""
    edges = mesh.edges_packed()
    mverts = mesh.verts_packed()
    tverts = template.verts_packed()
    edge_lengths = torch.norm(mverts[edges[:, 1]] - mverts[edges[:, 1]], dim=1)
    
    mean_length = edge_lengths.mean()
    v0, v1 = tverts[edges[:, 0]], tverts[edges[:, 1]]
    
    split_idx = edge_lengths > split_thr * mean_length
    new_verts_split = 0.5 * (v0[split_idx] + v1[split_idx])
    new_verts_split /= new_verts_split.norm(dim=1, keepdim=True)
    
    collapse_idx = edge_lengths < collapse_thr * mean_length
    keep_vert_idx = torch.full((len(tverts),), True)
    keep_vert_idx[edges[collapse_idx].flatten()] = False
    new_verts_collapse = 0.5 * (v0[collapse_idx] + v1[collapse_idx])
    new_verts_collapse /= new_verts_collapse.norm(dim=1, keepdim=True)
    
    ch = ConvexHull(torch.cat([
        tverts[keep_vert_idx], 
        new_verts_split, 
        new_verts_collapse,
    ]).cpu())
    
    return Meshes(
        [torch.from_numpy(ch.points).float()],
        [torch.from_numpy(ch.simplices)],
    )

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Optimization parameters
MAX_ITER_INIT = 0      # Initial rough registration iterations
LR_INIT = 1e-2         # Learning rate for initial registration
MAX_ITER_DEFORM = 50   # Deformation optimization iterations
LR_DEFORM = 1e-4       # Learning rate for deformation

# Visualization parameters
ENABLE_SLICE_VISUALIZATION = False
ENABLE_3D_MESH_VISUALIZATION = False
ENABLE_ALIGNMENT_VISUALIZATION = False

# Metrics parameters
NUM_POINTS_FOR_METRICS = 100
NUM_POINTS_FOR_OPTIMIZATION = 5000

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_slice_comparison(image, label, pred, central_slice=None):
    """Visualize image slice with ground truth and prediction overlays."""
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
    """Visualize comparison between prediction and ground truth meshes."""
    if not ENABLE_3D_MESH_VISUALIZATION:
        return
        
    fig = go.Figure()
    plot_meshes_wireframe(prediction_mesh, fig=fig, opacity=0.5)
    plot_meshes_wireframe(label_mesh, fig=fig, opacity=0.5)
    fig.update_layout(height=800, title=title)
    fig.show()

# =============================================================================
# MESH CREATION AND PROCESSING FUNCTIONS
# =============================================================================

def create_ground_truth_mesh(label, size0, device):
    """Create ground truth mesh from label using marching cubes."""
    verts, faces, _, _ = marching_cubes(label.numpy())
    label_mesh = Meshes(
        [torch.from_numpy((verts.copy()) / size0 * 2 - 1).float().to(device)],
        [torch.from_numpy(faces.copy()).long().to(device)]
    )
    
    return label_mesh

def process_prediction_mask(pred, size0):
    """Process prediction mask and compute gradients."""
    keep_largest_cc = KeepLargestConnectedComponent(None)
    blabel = pred[1]
    blabel_thr = keep_largest_cc(blabel > 0.5)
    blabel_udf = torch.from_numpy(distance_transform_edt(~blabel_thr) + distance_transform_edt(blabel_thr))
    blabel = blabel.unsqueeze(0).unsqueeze(0).float()
    blabel_udf = blabel_udf.unsqueeze(0).unsqueeze(0).float()
    
    # Compute gradients using Sobel filter
    blabel_dx = torch.from_numpy(sobel(blabel_udf[0,0], axis=2)) / size0
    blabel_dy = torch.from_numpy(sobel(blabel_udf[0,0], axis=1)) / size0
    blabel_dz = torch.from_numpy(sobel(blabel_udf[0,0], axis=0)) / size0
    
    blabel_grad = torch.stack([blabel_dx, blabel_dy, blabel_dz], dim=0).unsqueeze(0)
    
    return blabel_thr, blabel, blabel_grad

def create_prediction_mesh(blabel_thr, size0, device):
    """Create prediction mesh from thresholded binary label."""
    mc_verts, mc_faces, _, _ = marching_cubes(blabel_thr.numpy())
    
    prediction_mesh = Meshes(
        [torch.from_numpy(mc_verts.copy()).to(device)],
        [torch.from_numpy(mc_faces.copy()).to(device)]
    )
    prediction_mesh_img_space = Meshes(
        [torch.from_numpy((mc_verts.copy()) / size0 * 2 - 1).to(device)],
        [torch.from_numpy(mc_faces.copy()).to(device)]
    )
    
    
    return prediction_mesh, prediction_mesh_img_space

# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def compute_initial_alignment(mc_verts, size0):
    """Compute initial offset and scale for mesh alignment."""
    mc_verts = torch.from_numpy(mc_verts.copy())
    offset = mc_verts.mean(dim=0, keepdims=True)
    scale = torch.max(torch.norm(mc_verts - offset, dim=1))
    
    # Normalize to [-1, 1] space
    offset = offset / size0 * 2 - 1
    scale = scale.unsqueeze(0) / size0 * 2
    
    return offset, scale

def perform_initial_registration(decoder, search_template, lv, offset, scale, blabel_grad, 
                               max_iter_init, lr_init, verts_mean, device):
    """Perform initial rough registration using volume gradients."""
    with torch.no_grad():
        pred_mesh = decoder(search_template, lv)[-1]
        verts = pred_mesh.verts_packed()
        normals = pred_mesh.verts_normals_packed()
        weights = compute_face_area_vert_weights(pred_mesh)
        weights_sum = weights.sum()
        
        for i in range(max_iter_init):
            f_ext = sample_vol((verts - verts_mean) * scale + offset, blabel_grad.to(device))
            offset -= lr_init * torch.sum(weights * f_ext, dim=0) / weights_sum
            scale -= lr_init * torch.sum(weights * torch.sum(f_ext * normals, dim=1, keepdim=True)) / weights_sum
    
    return pred_mesh, offset, scale

def perform_mesh_deformation_optimization(decoder, search_template, lv, offset, scale, 
                                        blabel_grad, label_points, verts_mean,
                                        max_iter_deform, lr_deform, lr_init, device):
    """Perform mesh deformation optimization using latent vector updates."""
    lv = lv.repeat(len(search_template.verts_packed()), 1)
    lv.requires_grad_(True)
    
    optim = torch.optim.Adam([lv], lr=lr_deform)
    
    decoder.eval()
    decoder.requires_grad_(False)
    # Keep decoder on the same device as the data
    decoder.to(device)
    
    offset = offset.to(device).float()
    scale = scale.to(device).float()
    verts_mean = verts_mean.to(device).float()
    blabel_grad = blabel_grad.to(device)
    label_points = label_points.to(device)

    energies = []
    chamfers = []
    
    for it in range(max_iter_deform):
        optim.zero_grad()
        pred_mesh = decoder(search_template, lv, expand_lv=False)[-1]
        
        points, normals = sample_points_from_meshes(pred_mesh, NUM_POINTS_FOR_OPTIMIZATION, return_normals=True)
        with torch.no_grad():
            f_ext = 0.1 * sample_vol((points[0] - verts_mean) * scale + offset, blabel_grad.to(device)).unsqueeze(0)
            offset -= lr_init * f_ext.mean(dim=1)
            scale -= lr_init * torch.mean(torch.sum(f_ext * normals, dim=-1), dim=-1)
            
            energies.append(f_ext.mean().cpu().item())
            chamfers.append(chamfer_distance(points - verts_mean + offset, label_points.to(device))[0].cpu().item())
        
        points.backward(f_ext, retain_graph=True)
        
        loss = 0
        loss += 1e8 * mesh_laplacian_loss_highdim(pred_mesh, lv, p=8)
        loss.backward()
        optim.step()
    
    return pred_mesh, offset, scale, energies, chamfers

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def compute_evaluation_metrics(pred_mesh, label_mesh, blabel_thr, label, verts_mean, 
                              offset, scale, size0, device, num_points=NUM_POINTS_FOR_METRICS):
    """Compute Dice, Hausdorff, and Chamfer distance metrics."""
    with torch.no_grad():
        # Rasterize predicted mesh to volume
        rasterize_verts = (pred_mesh.verts_packed() - verts_mean) * scale + offset
        rasterize_verts = rasterize_verts[:, :]
        rasterize_mesh = Meshes([rasterize_verts], [pred_mesh.faces_packed()])
        pred_mesh_vol = rasterize_vol(rasterize_mesh, label.shape)
        
        # Sample points for Chamfer distance
        label_points = sample_points_from_meshes(label_mesh, num_points).cpu()
        
        # Create baseline mesh from thresholded prediction
        verts, faces, _, _ = marching_cubes(blabel_thr.numpy())
        blabel_mesh = Meshes([torch.from_numpy(verts.copy()) / size0 * 2 - 1], [torch.from_numpy(faces.copy())])
        
        # Compute metrics for our method
        pred_one_hot = one_hot(pred_mesh_vol.unsqueeze(0).unsqueeze(0).to(device), 2)
        label_one_hot = one_hot(label.unsqueeze(0).unsqueeze(0).to(device), 2)
        dice_mesh = compute_meandice(
            pred_one_hot,
            label_one_hot,
        )[0][1].cpu().item()
        
        haus_mesh = compute_hausdorff_distance(
            one_hot(pred_mesh_vol.unsqueeze(0).unsqueeze(0), 2),
            one_hot(label.unsqueeze(0).unsqueeze(0), 2),
        ).cpu().item()
        
        chamfer_mesh = chamfer_distance(
            sample_points_from_meshes(rasterize_mesh, num_points).cpu(),
            label_points
        )[0].cpu().item() * 1e4
        
        # Compute metrics for baseline (convnet)
        dice_conv = compute_meandice(
            one_hot(blabel_thr.unsqueeze(0).unsqueeze(0), 2),
            one_hot(label.unsqueeze(0).unsqueeze(0), 2),
        )[0][1].cpu().item()
        
        haus_conv = compute_hausdorff_distance(
            one_hot(blabel_thr.unsqueeze(0).unsqueeze(0), 2),
            one_hot(label.unsqueeze(0).unsqueeze(0), 2),
        ).cpu().item()
        
        chamfer_conv = chamfer_distance(
            sample_points_from_meshes(blabel_mesh, num_points).cpu(),
            label_points
        )[0].cpu().item() * 1e4
        
        
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
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_single_sample(idx, image, label, model, decoder, latent_vectors, template, 
                         search_template, size0, storage_lists, device):
    """Process a single validation sample through the complete pipeline."""
    
    # Unpack storage lists
    (all_meshes, all_offsets, all_scales, all_meshes_only_align, 
     all_offsets_only_align, all_scales_only_align, all_label_meshes, 
     all_verts_means, all_rasterized_vols, all_blabel_thr) = storage_lists
    
    # Get prediction from model
    with torch.no_grad():
        pred = torch.sigmoid(model(image.unsqueeze(0).unsqueeze(0).to(device)).squeeze()).cpu()
    
    # Visualize slice comparison
    visualize_slice_comparison(image, label, pred)
    
    # Create ground truth mesh
    label_mesh = create_ground_truth_mesh(label, size0, device)
    
    # Process prediction mask
    blabel_thr, blabel, blabel_grad = process_prediction_mask(pred, size0)
    
    # Create prediction mesh
    prediction_mesh, prediction_mesh_img_space = create_prediction_mesh(blabel_thr, size0, device)
    
    # Visualize mesh comparison
    visualize_mesh_comparison(prediction_mesh_img_space, label_mesh, 'Blue: Predicted, Red: GT')
    
    # Compute initial alignment
    mc_verts, mc_faces, _, _ = marching_cubes(blabel_thr.numpy())
    offset, scale = compute_initial_alignment(mc_verts, size0)
    
    # Get initial latent vector
    lv = latent_vectors.weight.data.mean(dim=0).clone().unsqueeze(0).to(device)

    # Perform initial registration
    pred_mesh, offset, scale = perform_initial_registration(
        decoder, search_template, lv, offset, scale, blabel_grad,
        MAX_ITER_INIT, LR_INIT, prediction_mesh.verts_packed().mean(), device
    )
    
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
    
    # Perform mesh deformation optimization
    label_points = sample_points_from_meshes(label_mesh, NUM_POINTS_FOR_OPTIMIZATION).cpu()
    pred_mesh, offset, scale, energies, chamfers = perform_mesh_deformation_optimization(
        decoder, search_template, lv, offset, scale, blabel_grad, label_points,
        pred_mesh.verts_packed().mean(), MAX_ITER_DEFORM, LR_DEFORM, LR_INIT, device
    )
    
    # Compute evaluation metrics
    metrics = compute_evaluation_metrics(
        pred_mesh, label_mesh, blabel_thr, label, pred_mesh.verts_packed().mean(),
        offset, scale, size0, device
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

# =============================================================================
# MAIN EXECUTION CODE
# =============================================================================

def main():
    """Main execution function for the segmentation evaluation pipeline."""
    
    # =============================================================================
    # DATA LOADING AND PREPARATION
    # =============================================================================
    
    print("Loading data...")
    
    # CirrMRI600+ 64 64 64 size, all images (prepare_full_dataset.py).
    train_data = torch.load('/scratch/ralbe/dals_data/train_data_mixed.pt', map_location='cpu')
    val_data = torch.load('/scratch/ralbe/dals_data/test_data_mixed.pt', map_location='cpu') # This is inference, so we checking on test data
    
    # CirrMRI600+ 64 64 64 size, small subset (prepare_small_dataset.py).
    # train_data = torch.load('/home/ralbe/DALS/mesh_autodecoder/data/train_data.pt', map_location='cpu')
    # val_data = torch.load('/home/ralbe/DALS/mesh_autodecoder/data/val_data.pt', map_location='cpu')
    
    # Assert that all shapes have the same first dimension (number of images)
    assert train_data['images'].shape[0] == train_data['labels'].shape[0] == train_data['masks'].shape[0], "Mismatch in train data entries"
    assert val_data['images'].shape[0] == val_data['labels'].shape[0] == val_data['masks'].shape[0], "Mismatch in val data entries"
    
    # Assert that spatial dimensions match
    assert train_data['images'].shape[1:] == train_data['labels'].shape[1:] == train_data['masks'].shape[1:], "Mismatch in train data sizes"
    assert val_data['images'].shape[1:] == val_data['labels'].shape[1:] == val_data['masks'].shape[1:], "Mismatch in val data sizes"
    
    # Extract number of images and spatial size
    train_n = train_data['images'].shape[0]
    val_n = val_data['images'].shape[0]
    size = train_data['images'].shape[1:]  # Tuple with spatial dims
    
    print(f"Train: {train_n} images, Val: {val_n} images, Image size: {size}")
    
    # Convert to list of dictionaries
    train_data = dict_unzip(train_data)
    val_data = dict_unzip(val_data)
    
    # =============================================================================
    # MODEL LOADING
    # =============================================================================
    
    print("Loading models...")
    
    # Load segmentation model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path  = '/home/ralbe/DALS/mesh_autodecoder/models/ConvNetTrainer_DynUNet_2025-10-15_15-36.ckpt'
    checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/models/ConvNetTrainer_ResUNet_2025-10-15_15-36.ckpt'
    checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/models/ConvNetTrainer_SEGRESNET_2025-10-15_18-37.ckpt'
    checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/models/ConvNetTrainer_SYNVNET3D_2025-10-15_17-41.ckpt'
    checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/models/ConvNetTrainer_UNET_2025-10-15_16-23.ckpt'
    checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/models/ConvNetTrainer_UNETR_2025-10-15_15-36.ckpt'
    checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/models/ConvNetTrainer_VNet_2025-10-15_15-36.ckpt'
        # checkpoint = torch.load('/home/ralbe/DALS/mesh_autodecoder/ConvNetTrainer_ResUNet_2025-10-15_15-36.ckpt', map_location='cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Initialize segmentation model based on checkpoint
    if checkpoint['hparams']['model'] == 'ResUNet':
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif checkpoint['hparams']['model'] == 'VNet':
        model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
        )
    elif checkpoint['hparams']['model'] == 'DynUNet':
        from model.dals_segmenter import _get_kernels_strides
        kernels, strides = _get_kernels_strides(
            checkpoint['hparams']['data_size'],
            checkpoint['hparams']['data_spacing'],
        )
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name='instance',
        )
    elif checkpoint['hparams']['model'] == 'UNETR':
        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=checkpoint['hparams']['data_size'],
            feature_size=16,
            hidden_size=768//2,
            mlp_dim=3072//2,
            num_heads=12//2,
            pos_embed="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    else:
        raise ValueError('Unknown model')
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    model.requires_grad = False
    
    # Load DALS model checkpoint
    # checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/scripts/rami_scripts/MeshDecoderTrainer_2025-10-15_15-31.ckpt'
    checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/models/old_meshdecoders/MeshDecoderTrainer_2025-10-17_13-46-26.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    
    # Initialize DALS decoder
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
    decoder = decoder.to(device).eval()
    
    # Load latent vectors and template
    latent_vectors = checkpoint['latent_vectors']
    latent_vectors = latent_vectors.to(device).eval()
    template = checkpoint['template'].to(device)
    
    # Initialize post-processing
    keep_largest_cc = KeepLargestConnectedComponent(None)
    print("Models loaded successfully")
    
    # =============================================================================
    # EVALUATION SETUP
    # =============================================================================
    
    # Initialize storage lists
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
    
    # Create search template once
    # Work around PyTorch3D device issue by doing subdivision on CPU then moving to device
    if device.type == 'cuda':
        # Move template to CPU for subdivision, then back to CUDA
        template_cpu = template.cpu()
        search_template_cpu = subdivide(template_cpu)
        search_template = search_template_cpu.to(device)
    else:
        # On CPU, just do subdivision directly
        search_template = subdivide(template)
    
    # =============================================================================
    # MAIN EVALUATION LOOP
    # =============================================================================
    
    # Create storage lists tuple
    storage_lists = (all_meshes, all_offsets, all_scales, all_meshes_only_align, 
                    all_offsets_only_align, all_scales_only_align, all_label_meshes, 
                    all_verts_means, all_rasterized_vols, all_blabel_thr)
    
    # Process each validation sample
    for idx in trange(len(val_data)):
        image = val_data[idx]['images']
        label = val_data[idx]['labels']
        size0 = image.shape[0]
        
        metrics = process_single_sample(
            idx, image, label, model, decoder, latent_vectors, template, search_template, size0, storage_lists, device
        )
        
        # Store metrics
        dice_scores.append(metrics['dice_mesh'])
        dice_scores_convnet.append(metrics['dice_conv'])
        hausdorff_dists.append(metrics['haus_mesh'])
        hausdorff_dists_convnet.append(metrics['haus_conv'])
        chamfer_dists.append(metrics['chamfer_mesh'])
        chamfer_dists_convnet.append(metrics['chamfer_conv'])
    
    # =============================================================================
    # RESULTS COMPUTATION AND DISPLAY
    # =============================================================================
    
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
    
    return {
        'dice_scores': dice_scores,
        'dice_scores_convnet': dice_scores_convnet,
        'hausdorff_dists': hausdorff_dists,
        'hausdorff_dists_convnet': hausdorff_dists_convnet,
        'chamfer_dists': chamfer_dists,
        'chamfer_dists_convnet': chamfer_dists_convnet,
        'all_meshes': all_meshes,
        'all_label_meshes': all_label_meshes,
    }

if __name__ == "__main__":
    try:
        results = main()
        print("Script completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
