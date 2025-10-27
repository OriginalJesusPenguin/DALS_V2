#!/usr/bin/env python3
"""
Script to visualize latent vectors using UMAP projection.
Reads from model checkpoint and test data to create UMAP plots with:
- Option 1: Binary labels (cirrhotic vs healthy)
- Option 2: Severity labels (0: healthy, 1: mild, 2: moderate, 3: severe)
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

def load_data():
    """Load model checkpoint and test data."""
    print("Loading data...")
    
    # Load model checkpoint
    model_checkpoint = '/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-10-24_14-38-31.ckpt'
    model_data = torch.load(model_checkpoint, map_location='cpu')
    
    # Load test latent vectors
    test_data_path = '/home/ralbe/DALS/mesh_autodecoder/inference_results/latents_MeshDecoderTrainer_2025-10-24_14-38-31/all_latent_vectors.pt'
    test_data = torch.load(test_data_path, map_location='cpu')
    
    # Load CSV datasets
    cirrhotic_df = pd.read_csv('/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/cirrhotic_dataset.csv')
    healthy_df = pd.read_csv('/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/healthy_dataset.csv')
    
    print(f"Loaded cirrhotic dataset: {len(cirrhotic_df)} samples")
    print(f"Loaded healthy dataset: {len(healthy_df)} samples")
    print(f"Loaded test data: {len(test_data)} samples")
    
    return model_data, test_data, cirrhotic_df, healthy_df

def extract_patient_id_from_filename(filename):
    """Extract patient ID from mesh filename."""
    # Extract number from filename like "liver_003.obj" -> 3
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def create_disease_mapping(cirrhotic_df, healthy_df):
    """Create mapping from Patient ID to disease severity."""
    print("Creating disease severity mapping...")
    
    # Create mapping for cirrhotic patients
    cirrhotic_mapping = {}
    for _, row in cirrhotic_df.iterrows():
        patient_id = row['Patient ID']
        severity = row['Radiological Evaluation']
        cirrhotic_mapping[patient_id] = severity
    
    # Create mapping for healthy patients (severity = 0)
    healthy_mapping = {}
    for _, row in healthy_df.iterrows():
        patient_id = row['Patient ID']
        healthy_mapping[patient_id] = 0  # Healthy = 0
    
    # Combine mappings
    disease_mapping = {**cirrhotic_mapping, **healthy_mapping}
    
    print(f"Created disease mapping for {len(disease_mapping)} patients")
    print(f"Severity distribution: {pd.Series(list(disease_mapping.values())).value_counts().sort_index()}")
    
    return disease_mapping

def process_latent_vectors(model_data, test_data):
    """Process latent vectors for UMAP analysis."""
    print("Processing latent vectors...")
    
    # Extract training latent vectors and filenames
    train_latent_vectors = model_data['latent_vectors'].weight.detach().cpu().numpy()  # [N_train, latent_dim]
    train_filenames = model_data['train_filenames']
    
    # Extract test latent vectors and filenames
    test_latent_vectors = torch.stack([entry['latent_vectors'] for entry in test_data])  # [N_test, 642, latent_dim]
    test_filenames = [entry['test_filename'] for entry in test_data]
    
    # Compute mean across vertices for test data
    test_latent_vectors_mean = test_latent_vectors.mean(dim=1).detach().cpu().numpy()  # [N_test, latent_dim]
    
    print(f"Train latent vectors shape: {train_latent_vectors.shape}")
    print(f"Test latent vectors shape: {test_latent_vectors_mean.shape}")
    
    return train_latent_vectors, train_filenames, test_latent_vectors_mean, test_filenames

def assign_labels(filenames, disease_mapping, label_type='severity'):
    """Assign labels to samples based on filenames and disease mapping."""
    print(f"Assigning {label_type} labels...")
    
    labels = []
    patient_ids = []
    
    for filename in filenames:
        patient_id = extract_patient_id_from_filename(filename)
        patient_ids.append(patient_id)
        
        if patient_id in disease_mapping:
            severity = disease_mapping[patient_id]
            
            if label_type == 'severity':
                labels.append(severity)
            elif label_type == 'cirrhosis':
                # Binary: 0 for healthy, 1 for cirrhotic (any severity > 0)
                labels.append(1 if severity > 0 else 0)
        else:
            print(f"Warning: Patient ID {patient_id} not found in disease mapping")
            labels.append(-1)  # Unknown
    
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    print(f"Label distribution: {np.bincount(labels[labels >= 0])}")
    print(f"Unknown labels: {np.sum(labels == -1)}")
    
    return labels, patient_ids

def combine_data(train_latent_vectors, train_filenames, test_latent_vectors, test_filenames):
    """Combine train and test data for UMAP visualization."""
    print("Combining train and test data...")
    
    # Combine latent vectors
    all_latent_vectors = np.vstack([train_latent_vectors, test_latent_vectors])
    
    # Combine filenames
    all_filenames = train_filenames + test_filenames
    
    # Create data type labels (train vs test)
    train_labels = ['train'] * len(train_filenames)
    test_labels = ['test'] * len(test_filenames)
    data_type_labels = train_labels + test_labels
    
    print(f"Combined data: {all_latent_vectors.shape[0]} samples")
    print(f"Train samples: {len(train_filenames)}, Test samples: {len(test_filenames)}")
    
    return all_latent_vectors, all_filenames, data_type_labels

def evaluate_umap_separability(embedding, labels, metric='silhouette'):
    """Evaluate how well separated the classes are in the UMAP embedding."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Filter out unknown labels (-1) for evaluation
    valid_mask = labels >= 0
    if np.sum(valid_mask) < 2:
        return 0.0
    
    valid_embedding = embedding[valid_mask]
    valid_labels = labels[valid_mask]
    
    if metric == 'silhouette':
        # Higher is better (range: -1 to 1)
        score = silhouette_score(valid_embedding, valid_labels)
    elif metric == 'calinski_harabasz':
        # Higher is better
        score = calinski_harabasz_score(valid_embedding, valid_labels)
    elif metric == 'davies_bouldin':
        # Lower is better, so we negate it
        score = -davies_bouldin_score(valid_embedding, valid_labels)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return score

def optimize_umap_hyperparameters(train_latent_vectors, train_labels, val_latent_vectors, val_labels, 
                                 supervised=False, random_state=42):
    """Optimize UMAP hyperparameters using validation set separability."""
    print("Optimizing UMAP hyperparameters using validation set...")
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [5, 10, 15, 20, 30],
        'min_dist': [0.01, 0.1, 0.3, 0.5, 0.8]
    }
    
    best_score = -np.inf
    best_params = None
    best_reducer = None
    results = []
    
    print(f"Testing {len(param_grid['n_neighbors']) * len(param_grid['min_dist'])} parameter combinations...")
    
    for n_neighbors in param_grid['n_neighbors']:
        for min_dist in param_grid['min_dist']:
            try:
                # Fit UMAP on training data
                if supervised and train_labels is not None:
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                        random_state=random_state,
                        verbose=False
                    )
                    train_embedding = reducer.fit_transform(train_latent_vectors, y=train_labels)
                else:
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                        random_state=random_state,
                        verbose=False
                    )
                    train_embedding = reducer.fit_transform(train_latent_vectors)
                
                # Transform validation data
                val_embedding = reducer.transform(val_latent_vectors)
                
                # Evaluate separability on validation set
                score = evaluate_umap_separability(val_embedding, val_labels, metric='silhouette')
                
                results.append({
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'score': score
                })
                
                print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}: score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist}
                    best_reducer = reducer
                    
            except Exception as e:
                print(f"  Error with n_neighbors={n_neighbors}, min_dist={min_dist}: {e}")
                continue
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation score: {best_score:.4f}")
    
    return best_reducer, best_params, best_score, results

def fit_umap_on_train(train_latent_vectors, train_labels=None, n_neighbors=15, min_dist=0.1, random_state=42, supervised=False):
    """Fit UMAP on training data only."""
    if supervised and train_labels is not None:
        print("Fitting SUPERVISED UMAP on training data only...")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, supervised=True")
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            verbose=True
        )
        
        # Fit supervised UMAP on training data with labels
        train_embedding = reducer.fit_transform(train_latent_vectors, y=train_labels)
        print(f"Supervised UMAP fitted on training data: {train_embedding.shape}")
        
    else:
        print("Fitting UNSUPERVISED UMAP on training data only...")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, supervised=False")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=True
    )
    
    # Fit unsupervised UMAP on training data
    train_embedding = reducer.fit_transform(train_latent_vectors)
    print(f"Unsupervised UMAP fitted on training data: {train_embedding.shape}")
    
    return reducer, train_embedding

def transform_test_data(reducer, test_latent_vectors):
    """Transform test data using the fitted UMAP reducer."""
    print("Transforming test data using fitted UMAP...")
    test_embedding = reducer.transform(test_latent_vectors)
    print(f"Test data transformed: {test_embedding.shape}")
    return test_embedding

def create_severity_plot(embedding, labels, data_type_labels, filenames, output_path):
    """Create scatter plot colored by disease severity (0: healthy, 1: mild, 2: moderate, 3: severe)."""
    print("Creating severity-based scatter plot...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Separate data by severity
    healthy_mask = np.array(labels) == 0
    mild_mask = np.array(labels) == 1
    moderate_mask = np.array(labels) == 2
    severe_mask = np.array(labels) == 3
    unknown_mask = np.array(labels) == -1
    
    # Separate by data type (train vs test)
    train_mask = np.array(data_type_labels) == 'train'
    test_mask = np.array(data_type_labels) == 'test'
    
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
        
        # Create green colormap for healthy
        green_cmap = LinearSegmentedColormap.from_list('green_density', 
                                                     ['white', 'lightgreen', 'green', 'darkgreen'], N=256)
        plt.contourf(xx, yy, healthy_density, levels=20, cmap=green_cmap, alpha=0.3)
        plt.contour(xx, yy, healthy_density, levels=10, colors='green', alpha=0.6, linewidths=0.5)
    
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
    # Train points (circles)
    if np.any(healthy_mask & train_mask):
        plt.scatter(embedding[healthy_mask & train_mask, 0], embedding[healthy_mask & train_mask, 1], 
                   c='green', label=f'Train: Healthy (n={np.sum(healthy_mask & train_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkgreen', linewidth=1.0, zorder=5, marker='o')
    
    if np.any(mild_mask & train_mask):
        plt.scatter(embedding[mild_mask & train_mask, 0], embedding[mild_mask & train_mask, 1], 
                   c='yellow', label=f'Train: Mild (n={np.sum(mild_mask & train_mask)})', 
                   alpha=0.8, s=60, edgecolors='orange', linewidth=1.0, zorder=5, marker='o')
    
    if np.any(moderate_mask & train_mask):
        plt.scatter(embedding[moderate_mask & train_mask, 0], embedding[moderate_mask & train_mask, 1], 
                   c='orange', label=f'Train: Moderate (n={np.sum(moderate_mask & train_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkorange', linewidth=1.0, zorder=5, marker='o')
    
    if np.any(severe_mask & train_mask):
        plt.scatter(embedding[severe_mask & train_mask, 0], embedding[severe_mask & train_mask, 1], 
                   c='red', label=f'Train: Severe (n={np.sum(severe_mask & train_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkred', linewidth=1.0, zorder=5, marker='o')
    
    # Test points (diamonds)
    if np.any(healthy_mask & test_mask):
        plt.scatter(embedding[healthy_mask & test_mask, 0], embedding[healthy_mask & test_mask, 1], 
                   c='green', label=f'Test: Healthy (n={np.sum(healthy_mask & test_mask)})', 
                   alpha=0.9, s=80, edgecolors='darkgreen', linewidth=1.5, zorder=6, marker='D')
    
    if np.any(mild_mask & test_mask):
        plt.scatter(embedding[mild_mask & test_mask, 0], embedding[mild_mask & test_mask, 1], 
                   c='yellow', label=f'Test: Mild (n={np.sum(mild_mask & test_mask)})', 
                   alpha=0.9, s=80, edgecolors='orange', linewidth=1.5, zorder=6, marker='D')
    
    if np.any(moderate_mask & test_mask):
        plt.scatter(embedding[moderate_mask & test_mask, 0], embedding[moderate_mask & test_mask, 1], 
                   c='orange', label=f'Test: Moderate (n={np.sum(moderate_mask & test_mask)})', 
                   alpha=0.9, s=80, edgecolors='darkorange', linewidth=1.5, zorder=6, marker='D')
    
    if np.any(severe_mask & test_mask):
        plt.scatter(embedding[severe_mask & test_mask, 0], embedding[severe_mask & test_mask, 1], 
                   c='red', label=f'Test: Severe (n={np.sum(severe_mask & test_mask)})', 
                   alpha=0.9, s=80, edgecolors='darkred', linewidth=1.5, zorder=6, marker='D')
    
    if np.any(unknown_mask):
        plt.scatter(embedding[unknown_mask, 0], embedding[unknown_mask, 1], 
                   c='gray', label=f'Unknown (n={np.sum(unknown_mask)})', 
                   alpha=0.8, s=60, edgecolors='black', linewidth=1.0, zorder=5)
    
    # Customize plot
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('Latent Vector UMAP Projection - Disease Severity', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show plot
    plt.show()

def create_binary_plot(embedding, labels, data_type_labels, filenames, output_path):
    """Create a binary plot showing cirrhotic vs healthy with probability density backgrounds."""
    print("Creating binary plot (cirrhotic vs healthy)...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Separate data by binary labels (0: healthy, 1: cirrhotic)
    healthy_mask = np.array(labels) == 0
    cirrhotic_mask = np.array(labels) == 1
    unknown_mask = np.array(labels) == -1
    
    # Separate by data type (train vs test)
    train_mask = np.array(data_type_labels) == 'train'
    test_mask = np.array(data_type_labels) == 'test'
    
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
    # Train points (circles)
    if np.any(healthy_mask & train_mask):
        plt.scatter(embedding[healthy_mask & train_mask, 0], embedding[healthy_mask & train_mask, 1], 
                   c='blue', label=f'Train: Healthy (n={np.sum(healthy_mask & train_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkblue', linewidth=1.0, zorder=5, marker='o')
    
    if np.any(cirrhotic_mask & train_mask):
        plt.scatter(embedding[cirrhotic_mask & train_mask, 0], embedding[cirrhotic_mask & train_mask, 1], 
                   c='red', label=f'Train: Cirrhotic (n={np.sum(cirrhotic_mask & train_mask)})', 
                   alpha=0.8, s=60, edgecolors='darkred', linewidth=1.0, zorder=5, marker='o')
    
    # Test points (diamonds)
    if np.any(healthy_mask & test_mask):
        plt.scatter(embedding[healthy_mask & test_mask, 0], embedding[healthy_mask & test_mask, 1], 
                   c='blue', label=f'Test: Healthy (n={np.sum(healthy_mask & test_mask)})', 
                   alpha=0.9, s=80, edgecolors='darkblue', linewidth=1.5, zorder=6, marker='D')
    
    if np.any(cirrhotic_mask & test_mask):
        plt.scatter(embedding[cirrhotic_mask & test_mask, 0], embedding[cirrhotic_mask & test_mask, 1], 
                   c='red', label=f'Test: Cirrhotic (n={np.sum(cirrhotic_mask & test_mask)})', 
                   alpha=0.9, s=80, edgecolors='darkred', linewidth=1.5, zorder=6, marker='D')
    
    if np.any(unknown_mask):
        plt.scatter(embedding[unknown_mask, 0], embedding[unknown_mask, 1], 
                   c='gray', label=f'Unknown (n={np.sum(unknown_mask)})', 
                   alpha=0.8, s=60, edgecolors='black', linewidth=1.0, zorder=5)
    
    # Customize plot
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('Latent Vector UMAP Projection: Cirrhotic vs Healthy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Binary plot saved to: {output_path}")
    
    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize latent vectors using UMAP with clinical data')
    parser.add_argument('--output', type=str, default='latent_umap_plot.png',
                       help='Output plot filename')
    parser.add_argument('--plot_mode', type=str, choices=['severity', 'binary'], default='severity',
                       help='Plot mode: "severity" for cirrhosis severity levels (0:Healthy, 1:Mild, 2:Moderate, 3:Severe), "binary" for cirrhotic vs healthy')
    parser.add_argument('--supervised', action='store_true',
                       help='Use supervised UMAP (uses disease labels to guide the embedding)')
    parser.add_argument('--optimize_hyperparams', action='store_true',
                       help='Optimize UMAP hyperparameters using validation set separability')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Fraction of training data to use for validation (default: 0.2)')
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter (used if not optimizing)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='UMAP min_dist parameter (used if not optimizing)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    try:
        # Load data
        model_data, test_data, cirrhotic_df, healthy_df = load_data()
        
        # Create disease mapping
        disease_mapping = create_disease_mapping(cirrhotic_df, healthy_df)
        
        # Process latent vectors
        train_latent_vectors, train_filenames, test_latent_vectors, test_filenames = process_latent_vectors(model_data, test_data)
        
        # Assign labels to training data
        if args.plot_mode == 'severity':
            train_labels, train_patient_ids = assign_labels(train_filenames, disease_mapping, 'severity')
        else:  # binary mode
            train_labels, train_patient_ids = assign_labels(train_filenames, disease_mapping, 'cirrhosis')
        
        # Split training data into train/validation if optimizing hyperparameters
        if args.optimize_hyperparams:
            from sklearn.model_selection import train_test_split
            
            print(f"\nSplitting training data: {args.val_split*100:.1f}% for validation...")
            train_latent_vectors_split, val_latent_vectors, train_filenames_split, val_filenames = train_test_split(
                train_latent_vectors, train_filenames, test_size=args.val_split, random_state=args.random_state, stratify=train_labels
            )
            train_labels_split, val_labels = train_test_split(
                train_labels, test_size=args.val_split, random_state=args.random_state, stratify=train_labels
            )
            
            print(f"Training set: {len(train_filenames_split)} samples")
            print(f"Validation set: {len(val_filenames)} samples")
            
            # Use the split data for optimization
            train_latent_vectors_opt = train_latent_vectors_split
            train_labels_opt = train_labels_split
        else:
            # Use full training data
            train_latent_vectors_opt = train_latent_vectors
            train_labels_opt = train_labels
        
        # Assign labels to test data (for visualization only)
        if args.plot_mode == 'severity':
            test_labels, test_patient_ids = assign_labels(test_filenames, disease_mapping, 'severity')
        else:  # binary mode
            test_labels, test_patient_ids = assign_labels(test_filenames, disease_mapping, 'cirrhosis')
        
        # Print some statistics
        print(f"\nTraining data distribution by {args.plot_mode}:")
        unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
        if args.plot_mode == 'severity':
            label_names = {0: 'Healthy', 1: 'Mild', 2: 'Moderate', 3: 'Severe', -1: 'Unknown'}
        else:
            label_names = {0: 'Healthy', 1: 'Cirrhotic', -1: 'Unknown'}
        
        for label, count in zip(unique_train_labels, train_counts):
            print(f"  {label_names.get(label, f'Unknown ({label})')}: {count}")
        
        print(f"\nTest data distribution by {args.plot_mode}:")
        unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)
        for label, count in zip(unique_test_labels, test_counts):
            print(f"  {label_names.get(label, f'Unknown ({label})')}: {count}")
        
        # Fit UMAP on training data only
        if args.optimize_hyperparams:
            # Optimize hyperparameters using validation set
            reducer, best_params, best_score, optimization_results = optimize_umap_hyperparameters(
                train_latent_vectors_opt, train_labels_opt, val_latent_vectors, val_labels,
                supervised=args.supervised, random_state=args.random_state
            )
            
            # Refit on full training data with best parameters
            print(f"\nRefitting UMAP on full training data with best parameters: {best_params}")
            if args.supervised:
                reducer = umap.UMAP(
                    n_neighbors=best_params['n_neighbors'],
                    min_dist=best_params['min_dist'],
                    n_components=2,
                    random_state=args.random_state,
                    verbose=True
                )
                train_embedding = reducer.fit_transform(train_latent_vectors, y=train_labels)
            else:
                reducer = umap.UMAP(
                    n_neighbors=best_params['n_neighbors'],
                    min_dist=best_params['min_dist'],
                    n_components=2,
                    random_state=args.random_state,
                    verbose=True
                )
                train_embedding = reducer.fit_transform(train_latent_vectors)
        else:
            # Use provided parameters
            reducer, train_embedding = fit_umap_on_train(
                train_latent_vectors, 
                train_labels=train_labels,
                               n_neighbors=args.n_neighbors,
                               min_dist=args.min_dist,
                random_state=args.random_state,
                supervised=args.supervised
            )
        
        # Transform test data using the fitted reducer
        test_embedding = transform_test_data(reducer, test_latent_vectors)
        
        # Combine embeddings for visualization
        all_embedding = np.vstack([train_embedding, test_embedding])
        
        # Combine labels for visualization
        all_labels = np.concatenate([train_labels, test_labels])
        
        # Create data type labels (train vs test)
        train_data_type_labels = ['train'] * len(train_filenames)
        test_data_type_labels = ['test'] * len(test_filenames)
        all_data_type_labels = train_data_type_labels + test_data_type_labels
        
        # Combine filenames
        all_filenames = train_filenames + test_filenames
        
        # Create plot based on selected mode
        if args.plot_mode == 'severity':
            print(f"\nCreating severity-based plot...")
            create_severity_plot(all_embedding, all_labels, all_data_type_labels, all_filenames, args.output)
        else:  # binary mode
            print(f"\nCreating binary plot (cirrhotic vs healthy)...")
            create_binary_plot(all_embedding, all_labels, all_data_type_labels, all_filenames, args.output)
        
        method_type = "Supervised" if args.supervised else "Unsupervised"
        print(f"\nVisualization complete! {method_type} UMAP plot saved as {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
