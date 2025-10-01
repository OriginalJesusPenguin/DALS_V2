#!/usr/bin/env python3
"""
Script to visualize latent vectors using UMAP projection.
Reads latent_vectors.pt and creates a 2D scatter plot with:
- Cirrhotic samples in red
- Healthy samples in blue
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import argparse
from pathlib import Path
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import re

def load_latent_vectors(filepath):
    """Load latent vectors and filenames from the .pt file."""
    print(f"Loading latent vectors from: {filepath}")
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
    except:
        data = torch.load(filepath, map_location='cpu')
    filenames = data['filenames']
    latent_vectors = data['latent_vectors']
    
    print(f"Loaded {len(filenames)} samples")
    print(f"Latent vector shape: {latent_vectors[0].shape}")
    
    return filenames, latent_vectors

def prepare_latent_data(latent_vectors):
    """Convert list of latent vectors to a 2D numpy array for UMAP."""
    # Stack all latent vectors into a single array
    # For global mode: each lv is (1, latent_dim) -> stack to (n_samples, latent_dim)
    # For local mode: each lv is (n_vertices, latent_dim) -> we'll take the mean
    
    print("Preparing latent data for UMAP...")
    
    # Check if we're in global or local mode
    if len(latent_vectors[0].shape) == 2 and latent_vectors[0].shape[0] == 1:
        # Global mode: (1, latent_dim)
        print("Detected global mode - using latent vectors directly")
        latent_array = torch.stack(latent_vectors).squeeze(1).detach().numpy()  # (n_samples, latent_dim)
    else:
        # Local mode: (n_vertices, latent_dim) - take mean across vertices
        print("Detected local mode - taking mean across vertices")
        latent_array = torch.stack([lv.mean(dim=0) for lv in latent_vectors]).detach().numpy()  # (n_samples, latent_dim)
    
    print(f"Final latent array shape: {latent_array.shape}")
    return latent_array

def load_clinical_data(csv_path):
    """Load clinical data from CSV file."""
    print(f"Loading clinical data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create a mapping from Patient ID to Radiological Evaluation
    clinical_map = {}
    for _, row in df.iterrows():
        patient_id = str(row['Patient ID'])
        rad_eval = int(row['Radiological Evaluation'])
        clinical_map[patient_id] = rad_eval
    
    print(f"Loaded clinical data for {len(clinical_map)} patients")
    print(f"Radiological Evaluation distribution:")
    for severity in [1, 2, 3]:
        count = sum(1 for v in clinical_map.values() if v == severity)
        print(f"  Severity {severity}: {count} patients")
    
    return clinical_map

def get_sample_labels(filenames, clinical_map=None):
    """Extract labels from filenames and clinical data."""
    labels = []
    severity_labels = []
    
    for filename in filenames:
        # Extract patient ID from filename (e.g., "cirrhotic_56.obj" -> "56")
        if filename.startswith('cirrhotic_'):
            patient_id = filename.replace('cirrhotic_', '').replace('.obj', '')
            labels.append('cirrhotic')
            
            # Get severity from clinical data
            if clinical_map and patient_id in clinical_map:
                severity = clinical_map[patient_id]
                severity_labels.append(severity)
            else:
                severity_labels.append(0)  # Unknown severity
                
        elif filename.startswith('healthy_'):
            labels.append('healthy')
            severity_labels.append(0)  # Healthy = 0
        else:
            labels.append('unknown')
            severity_labels.append(0)
    
    return labels, severity_labels

def perform_umap(latent_array, n_neighbors=15, min_dist=0.1, random_state=42):
    """Perform UMAP dimensionality reduction."""
    print("Performing UMAP dimensionality reduction...")
    print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=True
    )
    
    embedding = reducer.fit_transform(latent_array)
    print(f"UMAP embedding shape: {embedding.shape}")
    
    return embedding

def create_scatter_plot(embedding, labels, severity_labels, filenames, output_path):
    """Create and save the scatter plot with probability density backgrounds."""
    print("Creating scatter plot with probability density backgrounds...")
    
    # Create figure
    plt.figure(figsize=(8, 7))
    
    # Separate data by severity (0: Healthy, 1: Mild, 2: Moderate, 3: Severe)
    healthy_mask = np.array(severity_labels) == 0
    mild_mask = np.array(severity_labels) == 1
    moderate_mask = np.array(severity_labels) == 2
    severe_mask = np.array(severity_labels) == 3
    unknown_mask = np.array(labels) == 'unknown'
    
    # Create a grid for density estimation
    x_min, x_max = embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5
    y_min, y_max = embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Plot probability density backgrounds for each severity level
    if np.any(healthy_mask):
        print("Computing healthy density...")
        healthy_points = embedding[healthy_mask]
        healthy_kde = gaussian_kde(healthy_points.T)
        healthy_density = np.reshape(healthy_kde(positions).T, xx.shape)
        
        # Create blue colormap for healthy
        blue_cmap = LinearSegmentedColormap.from_list('blue_density', 
                                                    ['white', 'lightblue', 'blue', 'darkblue'], N=256)
        plt.contourf(xx, yy, healthy_density, levels=20, cmap=blue_cmap, alpha=0.3)
        plt.contour(xx, yy, healthy_density, levels=10, colors='blue', alpha=0.6, linewidths=0.5)
    
    if np.any(mild_mask):
        print("Computing mild cirrhosis density...")
        mild_points = embedding[mild_mask]
        mild_kde = gaussian_kde(mild_points.T)
        mild_density = np.reshape(mild_kde(positions).T, xx.shape)
        
        # Create yellow colormap for mild
        yellow_cmap = LinearSegmentedColormap.from_list('yellow_density', 
                                                      ['white', 'lightyellow', 'yellow', 'orange'], N=256)
        plt.contourf(xx, yy, mild_density, levels=20, cmap=yellow_cmap, alpha=0.3)
        plt.contour(xx, yy, mild_density, levels=10, colors='orange', alpha=0.6, linewidths=0.5)
    
    if np.any(moderate_mask):
        print("Computing moderate cirrhosis density...")
        moderate_points = embedding[moderate_mask]
        moderate_kde = gaussian_kde(moderate_points.T)
        moderate_density = np.reshape(moderate_kde(positions).T, xx.shape)
        
        # Create orange colormap for moderate
        orange_cmap = LinearSegmentedColormap.from_list('orange_density', 
                                                       ['white', 'moccasin', 'orange', 'darkorange'], N=256)
        plt.contourf(xx, yy, moderate_density, levels=20, cmap=orange_cmap, alpha=0.3)
        plt.contour(xx, yy, moderate_density, levels=10, colors='darkorange', alpha=0.6, linewidths=0.5)
    
    if np.any(severe_mask):
        print("Computing severe cirrhosis density...")
        severe_points = embedding[severe_mask]
        severe_kde = gaussian_kde(severe_points.T)
        severe_density = np.reshape(severe_kde(positions).T, xx.shape)
        
        # Create red colormap for severe
        red_cmap = LinearSegmentedColormap.from_list('red_density', 
                                                   ['white', 'lightcoral', 'red', 'darkred'], N=256)
        plt.contourf(xx, yy, severe_density, levels=20, cmap=red_cmap, alpha=0.3)
        plt.contour(xx, yy, severe_density, levels=10, colors='red', alpha=0.6, linewidths=0.5)
    
    # Plot individual points on top
    if np.any(healthy_mask):
        plt.scatter(embedding[healthy_mask, 0], embedding[healthy_mask, 1], 
                   c='blue', label=f'Healthy (n={np.sum(healthy_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkblue', linewidth=1.0, zorder=5)
    
    if np.any(mild_mask):
        plt.scatter(embedding[mild_mask, 0], embedding[mild_mask, 1], 
                   c='yellow', label=f'Mild Cirrhosis (n={np.sum(mild_mask)})', 
                   alpha=0.8, s=60, edgecolors='orange', linewidth=1.0, zorder=5)
    
    if np.any(moderate_mask):
        plt.scatter(embedding[moderate_mask, 0], embedding[moderate_mask, 1], 
                   c='orange', label=f'Moderate Cirrhosis (n={np.sum(moderate_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkorange', linewidth=1.0, zorder=5)
    
    if np.any(severe_mask):
        plt.scatter(embedding[severe_mask, 0], embedding[severe_mask, 1], 
                   c='red', label=f'Severe Cirrhosis (n={np.sum(severe_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkred', linewidth=1.0, zorder=5)
    
    if np.any(unknown_mask):
        plt.scatter(embedding[unknown_mask, 0], embedding[unknown_mask, 1], 
                   c='gray', label=f'Unknown (n={np.sum(unknown_mask)})', 
                   alpha=0.8, s=60, edgecolors='black', linewidth=1.0, zorder=5)
    
    # Customize plot
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('Latent Vector UMAP Projection with Probability Density', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Legend is already positioned by matplotlib automatically
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show plot
    plt.show()

def create_binary_plot(embedding, labels, filenames, output_path):
    """Create a binary plot showing cirrhotic vs healthy with probability density backgrounds."""
    print("Creating binary plot (cirrhotic vs healthy)...")
    
    # Create figure
    plt.figure(figsize=(8, 7))
    
    # Separate data by binary labels
    cirrhotic_mask = np.array(labels) == 'cirrhotic'
    healthy_mask = np.array(labels) == 'healthy'
    unknown_mask = np.array(labels) == 'unknown'
    
    # Create a grid for density estimation
    x_min, x_max = embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5
    y_min, y_max = embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Plot probability density backgrounds for each class
    if np.any(healthy_mask):
        print("Computing healthy density...")
        healthy_points = embedding[healthy_mask]
        healthy_kde = gaussian_kde(healthy_points.T)
        healthy_density = np.reshape(healthy_kde(positions).T, xx.shape)
        
        # Create blue colormap for healthy
        blue_cmap = LinearSegmentedColormap.from_list('blue_density', 
                                                    ['white', 'lightblue', 'blue', 'darkblue'], N=256)
        plt.contourf(xx, yy, healthy_density, levels=20, cmap=blue_cmap, alpha=0.3)
        plt.contour(xx, yy, healthy_density, levels=10, colors='blue', alpha=0.6, linewidths=0.5)
    
    if np.any(cirrhotic_mask):
        print("Computing cirrhotic density...")
        cirrhotic_points = embedding[cirrhotic_mask]
        cirrhotic_kde = gaussian_kde(cirrhotic_points.T)
        cirrhotic_density = np.reshape(cirrhotic_kde(positions).T, xx.shape)
        
        # Create red colormap for cirrhotic
        red_cmap = LinearSegmentedColormap.from_list('red_density', 
                                                   ['white', 'lightcoral', 'red', 'darkred'], N=256)
        plt.contourf(xx, yy, cirrhotic_density, levels=20, cmap=red_cmap, alpha=0.3)
        plt.contour(xx, yy, cirrhotic_density, levels=10, colors='red', alpha=0.6, linewidths=0.5)
    
    # Plot individual points on top
    if np.any(healthy_mask):
        plt.scatter(embedding[healthy_mask, 0], embedding[healthy_mask, 1], 
                   c='blue', label=f'Healthy (n={np.sum(healthy_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkblue', linewidth=1.0, zorder=5)
    
    if np.any(cirrhotic_mask):
        plt.scatter(embedding[cirrhotic_mask, 0], embedding[cirrhotic_mask, 1], 
                   c='red', label=f'Cirrhotic (n={np.sum(cirrhotic_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkred', linewidth=1.0, zorder=5)
    
    if np.any(unknown_mask):
        plt.scatter(embedding[unknown_mask, 0], embedding[unknown_mask, 1], 
                   c='gray', label=f'Unknown (n={np.sum(unknown_mask)})', 
                   alpha=0.8, s=60, edgecolors='black', linewidth=1.0, zorder=5)
    
    # Customize plot
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('Latent Vector UMAP Projection: Cirrhotic vs Healthy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Legend is already positioned by matplotlib automatically
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Binary plot saved to: {output_path}")
    
    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize latent vectors using UMAP with clinical data')
    parser.add_argument('--latent_file', type=str, default='latent_vectors.pt',
                       help='Path to latent_vectors.pt file')
    parser.add_argument('--clinical_file', type=str, 
                       default='/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/cirrhotic_dataset.csv',
                       help='Path to clinical data CSV file')
    parser.add_argument('--output', type=str, default='latent_umap_plot.png',
                       help='Output plot filename')
    parser.add_argument('--plot_mode', type=str, choices=['severity', 'binary'], default='severity',
                       help='Plot mode: "severity" for cirrhosis severity levels (0:Healthy, 1:Mild, 2:Moderate, 3:Severe), "binary" for cirrhotic vs healthy')
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='UMAP min_dist parameter')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.latent_file).exists():
        print(f"Error: File {args.latent_file} not found!")
        return
    
    if not Path(args.clinical_file).exists():
        print(f"Error: File {args.clinical_file} not found!")
        return
    
    try:
        # Load data
        filenames, latent_vectors = load_latent_vectors(args.latent_file)
        
        # Load clinical data
        clinical_map = load_clinical_data(args.clinical_file)
        
        # Prepare data for UMAP
        latent_array = prepare_latent_data(latent_vectors)
        
        # Get labels with clinical data
        labels, severity_labels = get_sample_labels(filenames, clinical_map)
        
        # Print some statistics
        print("\nSample distribution by severity:")
        unique_severities, counts = np.unique(severity_labels, return_counts=True)
        severity_names = {0: 'Healthy', 1: 'Mild Cirrhosis', 2: 'Moderate Cirrhosis', 3: 'Severe Cirrhosis'}
        for severity, count in zip(unique_severities, counts):
            print(f"  {severity_names.get(severity, f'Unknown ({severity})')}: {count}")
        
        # Perform UMAP
        embedding = perform_umap(latent_array, 
                               n_neighbors=args.n_neighbors,
                               min_dist=args.min_dist,
                               random_state=args.random_state)
        
        # Create plot based on selected mode
        if args.plot_mode == 'severity':
            print(f"\nCreating severity-based plot...")
            create_scatter_plot(embedding, labels, severity_labels, filenames, args.output)
        else:  # binary mode
            print(f"\nCreating binary plot (cirrhotic vs healthy)...")
            create_binary_plot(embedding, labels, filenames, args.output)
        
        print(f"\nVisualization complete! Plot saved as {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
