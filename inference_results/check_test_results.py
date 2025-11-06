import torch
import numpy as np
import os
import trimesh
import plotly.graph_objs as go
import pandas as pd
import re
from scipy.spatial.distance import cdist

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

def find_closest_training_latent(test_vertex_latent, train_latent_vectors, train_labels):
    """Find the closest training latent vector for a test vertex."""
    # Compute distances to all training latent vectors
    distances = cdist([test_vertex_latent], train_latent_vectors, metric='euclidean')[0]
    
    # Find closest training sample
    closest_idx = np.argmin(distances)
    closest_distance = distances[closest_idx]
    closest_label = train_labels[closest_idx]
    
    return closest_idx, closest_distance, closest_label

def visualize_test_mesh_with_disease_coloring(test_idx, test_mesh_path, test_vertex_latents, train_latent_vectors, train_labels):
    """Visualize a test mesh with vertices colored by closest training disease severity."""
    print(f"Visualizing test mesh {test_idx}...")
    
    # Load the test mesh
    mesh = trimesh.load(test_mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces
    
    print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Latent vectors: {len(test_vertex_latents)} (should match template vertices)")
    
    # The latent vectors correspond to the template mesh vertices (642), not the test mesh vertices
    # We need to map the latent vectors to the test mesh vertices
    # For now, let's assume we need to sample or map the latent vectors to the test mesh
    
    # If test mesh has more vertices than latent vectors, we need to handle this
    if len(vertices) != len(test_vertex_latents):
        print(f"Warning: Test mesh has {len(vertices)} vertices but only {len(test_vertex_latents)} latent vectors")
        print("This suggests the test mesh and template mesh have different vertex counts.")
        print("We'll use the latent vectors as-is and map them to the first N vertices of the test mesh.")
        
        # Use only the first N vertices where N = number of latent vectors
        n_latent_vectors = len(test_vertex_latents)
        vertices_to_use = vertices[:n_latent_vectors]
        print(f"Using first {n_latent_vectors} vertices of test mesh")
    else:
        vertices_to_use = vertices
    
    # For each latent vector, find closest training latent and get disease severity
    vertex_info = []
    vertex_colors = []
    
    for i, vertex_latent in enumerate(test_vertex_latents):
        closest_idx, distance, severity = find_closest_training_latent(
            vertex_latent, train_latent_vectors, train_labels
        )
        
        # Color mapping: 0=green, 1=yellow, 2=orange, 3=red
        color_map = {0: 'rgb(0,255,0)', 1: 'rgb(255,255,0)', 2: 'rgb(255,165,0)', 3: 'rgb(255,0,0)'}
        color = color_map.get(severity, 'rgb(128,128,128)')  # gray for unknown
        
        vertex_info.append({
            'vertex_idx': i,
            'closest_train_idx': closest_idx,
            'distance': distance,
            'severity': severity,
            'color': color
        })
        vertex_colors.append(color)
    
    # Create wireframe mesh visualization
    fig = go.Figure()
    
    # Filter faces to only include those with vertices in our latent vector range
    valid_faces = []
    for face in faces:
        if all(v < len(vertices_to_use) for v in face):
            valid_faces.append(face)
    
    print(f"Using {len(valid_faces)} faces out of {len(faces)} total faces")
    
    # Create mesh3d with wireframe mode
    fig.add_trace(go.Mesh3d(
        x=vertices_to_use[:, 0],
        y=vertices_to_use[:, 1], 
        z=vertices_to_use[:, 2],
        i=[face[0] for face in valid_faces],
        j=[face[1] for face in valid_faces],
        k=[face[2] for face in valid_faces],
        vertexcolor=vertex_colors,
        lighting=dict(ambient=0.3, diffuse=0.8, specular=0.1),
        lightposition=dict(x=100, y=100, z=100),
        flatshading=True,
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Update layout for better visualization
    fig.update_layout(
        title=f'Test Mesh {test_idx} - Disease Severity Wireframe (First {len(vertices_to_use)} vertices)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=900,
        showlegend=False
    )
    
    return fig, vertex_info

# Load data
model_checkpoint = '/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-10-24_14-38-31.ckpt'
model_data = torch.load(model_checkpoint, map_location='cpu')

test_data_path = '/home/ralbe/DALS/mesh_autodecoder/inference_results/latents_MeshDecoderTrainer_2025-10-24_14-38-31/all_latent_vectors.pt'
test_data = torch.load(test_data_path, map_location='cpu')

# Load CSV datasets
cirrhotic_df = pd.read_csv('/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/cirrhotic_dataset.csv')
healthy_df = pd.read_csv('/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/healthy_dataset.csv')

# Create disease mapping
disease_mapping = create_disease_mapping(cirrhotic_df, healthy_df)

# Extract training data
train_latent_vectors = model_data['latent_vectors'].weight.detach().cpu().numpy()  # N_templates x latent_dim
train_filenames = model_data['train_filenames']

# Extract test data
test_latent_vectors = torch.stack([entry['latent_vectors'] for entry in test_data])  # N_test_meshes x latent_per_vertex x latent_dim
test_filenames = [entry['test_filename'] for entry in test_data]  # list of N_test_meshes
base_path = '/home/ralbe/DALS/mesh_autodecoder/data/test_meshes'
test_mesh_paths = [os.path.join(base_path, filename) for filename in test_filenames]

print(f"Training data: {len(train_filenames)} samples")
print(f"Test data: {len(test_filenames)} samples")
print(f"Latent dimension: {train_latent_vectors.shape[1]}")

# Assign disease labels to training data
train_labels = []
for filename in train_filenames:
    patient_id = extract_patient_id_from_filename(filename)
    if patient_id in disease_mapping:
        train_labels.append(disease_mapping[patient_id])
    else:
        print(f"Warning: Patient ID {patient_id} not found in disease mapping")
        train_labels.append(-1)  # Unknown

train_labels = np.array(train_labels)
print(f"Training labels distribution: {np.bincount(train_labels[train_labels >= 0])}")

# Assign disease labels to test data
test_labels = []
for filename in test_filenames:
    patient_id = extract_patient_id_from_filename(filename)
    if patient_id in disease_mapping:
        test_labels.append(disease_mapping[patient_id])
    else:
        print(f"Warning: Patient ID {patient_id} not found in disease mapping")
        test_labels.append(-1)  # Unknown

test_labels = np.array(test_labels)
print(f"Test labels distribution: {np.bincount(test_labels[test_labels >= 0])}")

print("\nData loaded successfully!")
print("Ready to visualize test meshes with disease severity coloring.")

# Example: Visualize the first test mesh
if len(test_mesh_paths) > 0:
    test_idx = 0  # Change this to visualize different test meshes
    test_mesh_path = test_mesh_paths[test_idx]
    test_vertex_latents = test_latent_vectors[test_idx].detach().cpu().numpy()  # 642 x latent_dim
    
    print(f"\nVisualizing test mesh {test_idx}: {test_filenames[test_idx]}")
    print(f"True disease severity: {test_labels[test_idx]}")
    print(f"Processing {len(test_vertex_latents)} vertices...")
    
    # Create visualization
    fig, vertex_info = visualize_test_mesh_with_disease_coloring(
        test_idx, test_mesh_path, test_vertex_latents, train_latent_vectors, train_labels
    )
    
    # Print summary statistics
    print(f"\nVertex Disease Severity Summary:")
    severity_counts = {}
    for info in vertex_info:
        severity = info['severity']
        if severity not in severity_counts:
            severity_counts[severity] = 0
        severity_counts[severity] += 1
    
    for severity in sorted(severity_counts.keys()):
        count = severity_counts[severity]
        percentage = (count / len(vertex_info)) * 100
        severity_name = {0: 'Healthy', 1: 'Mild', 2: 'Moderate', 3: 'Severe', -1: 'Unknown'}[severity]
        print(f"  {severity_name}: {count} vertices ({percentage:.1f}%)")
    
    # Save the plot as PNG
    output_path = f'test_mesh_{test_idx}_disease_visualization.png'
    fig.write_image(output_path, width=1200, height=900, scale=2)
    print(f"\nVisualization saved to: {output_path}")
    
else:
    print("No test meshes found!")




