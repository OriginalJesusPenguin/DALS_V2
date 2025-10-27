#!/usr/bin/env python3
"""
UMAP Latent Space Analysis and Disease Prediction
==================================================

This script performs UMAP visualization of latent vectors and implements
distance-weighted voting for disease stage prediction.

Author: AI Assistant
Date: 2025
"""

import torch
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import umap
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def extract_patient_id_from_filename(filename):
    """Extract patient ID from mesh filename."""
    # Extract number from filename like "liver_003.obj" -> 3
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def load_data():
    """Load all required data files."""
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

def assign_disease_labels(filenames, disease_mapping):
    """Assign disease severity labels to samples based on filenames."""
    print("Assigning disease severity labels...")
    
    labels = []
    patient_ids = []
    
    for filename in filenames:
        patient_id = extract_patient_id_from_filename(filename)
        patient_ids.append(patient_id)
        
        if patient_id in disease_mapping:
            labels.append(disease_mapping[patient_id])
        else:
            print(f"Warning: Patient ID {patient_id} not found in disease mapping")
            labels.append(-1)  # Unknown
    
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    print(f"Label distribution: {np.bincount(labels[labels >= 0])}")
    print(f"Unknown labels: {np.sum(labels == -1)}")
    
    return labels, patient_ids

def fit_umap(train_latent_vectors):
    """Fit UMAP on training data."""
    print("Fitting UMAP...")
    
    # Fit UMAP on training data only
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric='euclidean'
    )
    
    train_umap = umap_reducer.fit_transform(train_latent_vectors)
    
    print(f"UMAP fitted on training data: {train_umap.shape}")
    
    return umap_reducer, train_umap

def create_umap_plot(train_umap, train_labels, train_patient_ids, test_umap, test_labels, test_patient_ids):
    """Create interactive UMAP visualization."""
    print("Creating UMAP visualization...")
    
    # Define colors for disease severity
    severity_colors = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red', -1: 'gray'}
    severity_names = {0: 'Healthy', 1: 'Mild', 2: 'Moderate', 3: 'Severe', -1: 'Unknown'}
    
    # Create subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=['UMAP Latent Space - Disease Severity'])
    
    # Plot training data
    for severity in sorted(np.unique(train_labels)):
        if severity == -1:
            continue
            
        mask = train_labels == severity
        fig.add_trace(
            go.Scatter(
                x=train_umap[mask, 0],
                y=train_umap[mask, 1],
                mode='markers',
                marker=dict(
                    color=severity_colors[severity],
                    size=8,
                    opacity=0.7
                ),
                name=f'Train: {severity_names[severity]}',
                text=[f"Patient {pid}" for pid in train_patient_ids[mask]],
                hovertemplate='<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>'
            )
        )
    
    # Plot test data
    for severity in sorted(np.unique(test_labels)):
        if severity == -1:
            continue
            
        mask = test_labels == severity
        fig.add_trace(
            go.Scatter(
                x=test_umap[mask, 0],
                y=test_umap[mask, 1],
                mode='markers',
                marker=dict(
                    color=severity_colors[severity],
                    size=12,
                    opacity=0.9,
                    symbol='diamond'
                ),
                name=f'Test: {severity_names[severity]}',
                text=[f"Patient {pid}" for pid in test_patient_ids[mask]],
                hovertemplate='<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='UMAP Visualization of Latent Space with Disease Severity',
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        width=1000,
        height=800,
        showlegend=True
    )
    
    # Save plot
    output_path = '/home/ralbe/DALS/mesh_autodecoder/umap_latent_space.html'
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"UMAP plot saved to: {output_path}")
    
    return fig

def distance_weighted_voting_original_space(train_latent_vectors, train_labels, test_latent_vectors, k_values=[5, 10, 15]):
    """Implement distance-weighted voting in original latent space (128D)."""
    print("Implementing distance-weighted voting in original latent space...")
    
    predictions = {}
    confidences = {}
    
    for k in k_values:
        print(f"Computing predictions for k={k} in original space...")
        
        pred_labels = []
        pred_confidences = []
        
        for i, test_point in enumerate(test_latent_vectors):
            # Compute distances to all training points in original space
            distances = np.linalg.norm(train_latent_vectors - test_point, axis=1)
            
            # Get k nearest neighbors
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]
            nearest_labels = train_labels[nearest_indices]
            
            # Compute weighted votes
            weights = 1.0 / (nearest_distances + 1e-8)  # Add small epsilon to avoid division by zero
            weighted_votes = {}
            
            for label, weight in zip(nearest_labels, weights):
                if label == -1:  # Skip unknown labels
                    continue
                if label not in weighted_votes:
                    weighted_votes[label] = 0
                weighted_votes[label] += weight
            
            # Predict as label with highest weighted vote
            if weighted_votes:
                predicted_label = max(weighted_votes, key=weighted_votes.get)
                total_weight = sum(weighted_votes.values())
                confidence = weighted_votes[predicted_label] / total_weight
            else:
                predicted_label = -1
                confidence = 0.0
            
            pred_labels.append(predicted_label)
            pred_confidences.append(confidence)
        
        predictions[f'k_{k}_original'] = np.array(pred_labels)
        confidences[f'k_{k}_original'] = np.array(pred_confidences)
    
    return predictions, confidences

def distance_weighted_voting(train_umap, train_labels, test_umap, k_values=[5, 10, 15]):
    """Implement distance-weighted voting for disease prediction in UMAP space."""
    print("Implementing distance-weighted voting in UMAP space...")
    
    predictions = {}
    confidences = {}
    
    for k in k_values:
        print(f"Computing predictions for k={k} in UMAP space...")
        
        pred_labels = []
        pred_confidences = []
        
        for i, test_point in enumerate(test_umap):
            # Compute distances to all training points
            distances = np.linalg.norm(train_umap - test_point, axis=1)
            
            # Get k nearest neighbors
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]
            nearest_labels = train_labels[nearest_indices]
            
            # Compute weighted votes
            weights = 1.0 / (nearest_distances + 1e-8)  # Add small epsilon to avoid division by zero
            weighted_votes = {}
            
            for label, weight in zip(nearest_labels, weights):
                if label == -1:  # Skip unknown labels
                    continue
                if label not in weighted_votes:
                    weighted_votes[label] = 0
                weighted_votes[label] += weight
            
            # Predict as label with highest weighted vote
            if weighted_votes:
                predicted_label = max(weighted_votes, key=weighted_votes.get)
                total_weight = sum(weighted_votes.values())
                confidence = weighted_votes[predicted_label] / total_weight
            else:
                predicted_label = -1
                confidence = 0.0
            
            pred_labels.append(predicted_label)
            pred_confidences.append(confidence)
        
        predictions[f'k_{k}_umap'] = np.array(pred_labels)
        confidences[f'k_{k}_umap'] = np.array(pred_confidences)
    
    return predictions, confidences

def evaluate_predictions(test_labels, predictions, confidences):
    """Evaluate prediction performance."""
    print("Evaluating predictions...")
    
    results = {}
    
    for k_name, pred_labels in predictions.items():
        # Filter out unknown labels for evaluation
        valid_mask = (test_labels >= 0) & (pred_labels >= 0)
        if np.sum(valid_mask) == 0:
            print(f"No valid predictions for {k_name}")
            continue
            
        y_true = test_labels[valid_mask]
        y_pred = pred_labels[valid_mask]
        y_conf = confidences[k_name][valid_mask]
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        results[k_name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'confidences': y_conf,
            'n_valid': len(y_true)
        }
        
        print(f"{k_name}: Accuracy = {accuracy:.3f}, Valid samples = {len(y_true)}")
    
    return results

def save_results(test_patient_ids, test_labels, predictions, confidences, results):
    """Save prediction results to CSV."""
    print("Saving results...")
    
    # Create results DataFrame
    df_results = pd.DataFrame({
        'patient_id': test_patient_ids,
        'true_severity': test_labels
    })
    
    # Add predictions for each k value
    for k_name, pred_labels in predictions.items():
        df_results[f'predicted_severity_{k_name}'] = pred_labels
        df_results[f'confidence_{k_name}'] = confidences[k_name]
    
    # Save to CSV
    output_path = '/home/ralbe/DALS/mesh_autodecoder/disease_predictions.csv'
    df_results.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    
    # Save metrics
    metrics_path = '/home/ralbe/DALS/mesh_autodecoder/prediction_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("Disease Prediction Results\n")
        f.write("=" * 50 + "\n\n")
        
        for k_name, result in results.items():
            f.write(f"{k_name.upper()}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.3f}\n")
            f.write(f"  Valid samples: {result['n_valid']}\n")
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    {result['confusion_matrix']}\n\n")
    
    print(f"Metrics saved to: {metrics_path}")
    
    return df_results

def main():
    """Main analysis pipeline."""
    print("Starting UMAP Disease Analysis...")
    print("=" * 50)
    
    # Load data
    model_data, test_data, cirrhotic_df, healthy_df = load_data()
    
    # Create disease mapping
    disease_mapping = create_disease_mapping(cirrhotic_df, healthy_df)
    
    # Process latent vectors
    train_latent_vectors, train_filenames, test_latent_vectors, test_filenames = process_latent_vectors(model_data, test_data)
    
    # Assign disease labels
    train_labels, train_patient_ids = assign_disease_labels(train_filenames, disease_mapping)
    test_labels, test_patient_ids = assign_disease_labels(test_filenames, disease_mapping)
    
    # Fit UMAP
    umap_reducer, train_umap = fit_umap(train_latent_vectors)
    test_umap = umap_reducer.transform(test_latent_vectors)
    
    # Create visualization
    fig = create_umap_plot(train_umap, train_labels, train_patient_ids, test_umap, test_labels, test_patient_ids)
    
    # Implement distance-weighted voting in both spaces
    print("\n" + "="*50)
    print("PREDICTION COMPARISON: Original Space vs UMAP Space")
    print("="*50)
    
    # Predictions in original 128D latent space
    predictions_original, confidences_original = distance_weighted_voting_original_space(
        train_latent_vectors, train_labels, test_latent_vectors)
    
    # Predictions in UMAP 2D space
    predictions_umap, confidences_umap = distance_weighted_voting(train_umap, train_labels, test_umap)
    
    # Combine predictions from both methods
    all_predictions = {**predictions_original, **predictions_umap}
    all_confidences = {**confidences_original, **confidences_umap}
    
    # Evaluate predictions
    results = evaluate_predictions(test_labels, all_predictions, all_confidences)
    
    # Save results
    df_results = save_results(test_patient_ids, test_labels, all_predictions, all_confidences, results)
    
    print("\nAnalysis complete!")
    print("=" * 50)
    print("Output files:")
    print("- umap_latent_space.html: Interactive UMAP visualization")
    print("- disease_predictions.csv: Prediction results with confidence scores")
    print("- prediction_metrics.txt: Accuracy and confusion matrices")
    print("\nPrediction methods compared:")
    print("- Original 128D latent space (k_5_original, k_10_original, k_15_original)")
    print("- UMAP 2D space (k_5_umap, k_10_umap, k_15_umap)")

if __name__ == "__main__":
    main()

