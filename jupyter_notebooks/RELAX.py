#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:


# Create a mask for each latent vector with a given random seed:
masks = []  
for seed in range(n_masks):
    torch.manual_seed(seed)
    mask = torch.bernoulli(torch.ones(latent_dim) * p)
    masks.append(mask)









# In[ ]:


# load the checkpoint:


import sys
sys.path.append('/home/ralbe/DALS/mesh_autodecoder')

import os
import time

import numpy as np
import pandas as pd
import torch
import trimesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes, SubdivideMeshes
from pytorch3d.loss import chamfer_distance

from model.mesh_decoder import MeshDecoder
from model.loss import mesh_bl_quality_loss
from util.metrics import point_metrics, self_intersections

from scipy.spatial import cKDTree  # Compute distance from each predicted (decoded) vertex to closest GT (target) vertex

model_checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_12-00-26.ckpt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(model_checkpoint_path,map_location='cpu')

keys = list(checkpoint.keys())

for key in keys:
    
    if key!='decoder_state_dict':
        print(key)
        
hparams=checkpoint['hparams']
latent_vectors = checkpoint['latent_vectors']
best_epoch = checkpoint['best_epoch']
best_loss = checkpoint['best_loss']
train_data_path = checkpoint['train_data_path']
val_data_path = checkpoint['val_data_path']
train_file_names = checkpoint['train_filenames']
latent_features = checkpoint['latent_features']
decoder_mode = checkpoint['decoder_mode']
template = checkpoint['template']




# AND NOW PRINT THEM ALL

for key in hparams.keys():
    print(f"{key}: {hparams[key]}")
print(f"latent_vectors: {latent_vectors}")
print(f"best_epoch: {best_epoch}")
print(f"best_loss: {best_loss}")
print(f"train_data_path: {train_data_path}")
print(f"val_data_path: {val_data_path}")



# --- Instantiate decoder and template ---
hparams = checkpoint["hparams"]
decoder = MeshDecoder(
    hparams["latent_features"],
    hparams["steps"],
    hparams["hidden_features"],
    hparams["subdivide"],
    mode=hparams["decoder_mode"],
    norm=hparams["normalization"][0],
).to(device).eval()
decoder.load_state_dict(checkpoint["decoder_state_dict"])

template = checkpoint["template"].to(device)
template = SubdivideMeshes()(template)


# In[ ]:


import tqdm
masked_latent_vectors = torch.zeros(n_masks, len(test_latent_vectors), latent_dim)

print(masked_latent_vectors.shape)

masked_decoded_meshes = np.zeros(n_masks, len(test_latent_vectors), 1)


for i in tqdm.tqdm(range(n_masks)):
    for j in range(len(test_latent_vectors)):
        # print(i,j)
        masked_latent_vectors[i,j,:] = test_latent_vectors[j] * masks[i]

        decoded_mesh = decoder(template.clone(), masked_latent_vectors[i,j,:].unsqueeze(0))[-1]
        masked_decoded_meshes[i,j,:] = decoded_mesh



print(masked_latent_vectors[i,j,:].shape)
print((test_latent_vectors[j]*masks[i]).shape)


print(masked_latent_vectors[0,0,:].shape)

latent_vector = masked_latent_vectors[0,0,:]


with torch.no_grad():
    decoded_mesh = decoder(template.clone(), latent_vector.unsqueeze(0))[-1]







# In[ ]:


print(len(test_objs))
metric_samples = 1000
for i in range(len(test_objs)):
    # print(test_objs[i])
    # print(os.path.exists(test_objs[i]))
    target_mesh = load_objs_as_meshes([test_objs[i]], device=device)
    print(target_mesh)
    decoded_mesh = 
    # --- Metric computation ---
    with torch.no_grad():
        pred_samples = sample_points_from_meshes(decoded_mesh, metric_samples)
        true_samples = sample_points_from_meshes(target_mesh, metric_samples)
        chamfer_val = chamfer_distance(true_samples, pred_samples)[0] * 10000
        metric_dict = point_metrics(true_samples, pred_samples, [0.01, 0.02])
        bl_quality = (1.0 - mesh_bl_quality_loss(decoded_mesh)).item()



        





# In[ ]:


import torch 


masks = torch.load('/home/ralbe/DALS/mesh_autodecoder/relax_explanations/relax_masks.pt',map_location='cpu')
print(masks.shape)

print(torch.mean(masks[0,:]))
print(torch.std(masks[0,:]))


# In[ ]:





# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# csv_path = '/home/ralbe/DALS/mesh_autodecoder/relax_explanations/relax_metrics.csv'
csv_path = '/home/ralbe/DALS/mesh_autodecoder/relax_explanations/relax_metrics_old.csv'

df = pd.read_csv(csv_path)

# Metrics where smaller means more similar.
distance_metrics = [
    "emd",
    "ChamferL2_x_10000",
    "Hausdorff",
    "lb_l2_delta",
]

# Metrics already bounded in [0, 1] where larger means more similar.
score_metrics = sorted(
    [
        col
        for col in df.columns
        if col.startswith("Precision@")
        or col.startswith("Recall@")
        or col.startswith("F1@")
    ]
)

similarity_pairs = []
median_scales = {}

def balanced_inverse(values, series):
    m = np.median(values)
    if not np.isfinite(m) or m <= 0:
        positive = values[values > 0]
        m = np.median(positive) if positive.size else 1.0
    sim = (m - series) / (m + series + 1e-12)
    return np.clip(sim, -1.0, 1.0), m

for col in distance_metrics:
    if col not in df.columns:
        continue
    column_vals = df[col].dropna().to_numpy()
    if column_vals.size == 0:
        continue
    sim_values, neutral_scale = balanced_inverse(column_vals, df[col])
    sim_col = f"{col}_similarity"
    df[sim_col] = sim_values
    median_scales[col] = neutral_scale
    similarity_pairs.append((col, sim_col))

for col in score_metrics:
    sim_col = f"{col}_similarity"
    df[sim_col] = (2.0 * df[col]) - 1.0
    df[sim_col] = df[sim_col].clip(-1.0, 1.0)
    similarity_pairs.append((col, sim_col))

if median_scales:
    print("Balanced inverse neutral scales (median m):")
    for metric, m in median_scales.items():
        print(f"  {metric}: {m:.4f}")

if similarity_pairs:
    n_pairs = len(similarity_pairs)
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, (orig_col, sim_col) in zip(axes, similarity_pairs):
        subset = df[[orig_col, sim_col]].dropna()
        ax.scatter(subset[orig_col], subset[sim_col], alpha=0.4, s=12)
        ax.set_xlabel(orig_col)
        ax.set_ylabel(sim_col)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.6)
        ax.set_title(f"{orig_col} vs similarity")
    for ax in axes[len(similarity_pairs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No metrics available for similarity computation.")


# In[30]:


import matplotlib.pyplot as plt
import pandas as pd 
import torch
import numpy as np

csv_path = '/home/ralbe/DALS/mesh_autodecoder/relax_explanations/relax_metrics_old.csv'
df = pd.read_csv(csv_path)
masks_path = '/home/ralbe/DALS/mesh_autodecoder/relax_explanations/relax_masks_old.pt'
masks = torch.load(masks_path, map_location='cpu')

print(df.columns)
print(masks.shape)

# Print the first row for context
print(df.iloc[0])

# Create 'similarity' column in df based on ChamferL2_x_10000 as before
chamfer = df["ChamferL2_x_10000"].to_numpy()
similarity_max = chamfer.max()
similarity_min = chamfer.min()
similarity = (chamfer - similarity_max) / (similarity_min - similarity_max)
df["similarity"] = similarity

# Plot histogram of ChamferL2_x_10000
plt.figure(figsize=(8, 4))
plt.hist(df["ChamferL2_x_10000"].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.xlabel("ChamferL2_x_10000")
plt.ylabel("Frequency")
plt.title("Histogram of ChamferL2_x_10000")
plt.tight_layout()
plt.show()

# Plot histogram of similarity
plt.figure(figsize=(8, 4))
plt.hist(df["similarity"].dropna(), bins=30, color='orange', edgecolor='black')
plt.xlabel("similarity")
plt.ylabel("Frequency")
plt.title("Histogram of similarity (from ChamferL2_x_10000)")
plt.tight_layout()
plt.show()

# Scatter plot: similarity vs ChamferL2_x_10000
plt.figure(figsize=(8, 4))
plt.scatter(df["similarity"], df["ChamferL2_x_10000"], alpha=0.5)
plt.xlabel('similarity')
plt.ylabel('ChamferL2_x_10000')
plt.title('ChamferL2_x_10000 vs similarity')
plt.tight_layout()
plt.show()

# For debugging: print min/max/shape
print("ChamferL2_x_10000 min:", similarity_min)
print("ChamferL2_x_10000 max:", similarity_max)
print("similarity shape:", similarity.shape)

# --- Compute similarity R_k for each k=1...num_latent ---
# R_k = (1/num_masks) * SUM_u=1^num_masks [similarity[u] * mask[u, k]]

# Convert masks to numpy (if not already)
if isinstance(masks, torch.Tensor):
    masks_np = masks.numpy()
else:
    masks_np = masks

# Determine dimensions
num_masks, num_latent = masks_np.shape
print(f"num_masks: {num_masks}, num_latent: {num_latent}")

# If length of similarity matches num_masks, i.e., each row is a mask, multiply directly:
if len(similarity) == num_masks:
    similarity_col = similarity.reshape(-1, 1)  # (num_masks, 1)
    R_k = (similarity_col * masks_np).mean(axis=0)  # (num_latent,)
else:
    # Check if df contains 'mask_idx' column to relate which mask was used for each similarity entry
    if 'mask_idx' in df.columns:
        # Group by mask indices and aggregate
        sim_for_mask = np.zeros(num_masks)
        counts = np.zeros(num_masks)
        for idx, sim_val in zip(df['mask_idx'], similarity):
            sim_for_mask[idx] += sim_val
            counts[idx] += 1
        sim_for_mask = sim_for_mask / np.maximum(counts, 1)
        similarity_col = sim_for_mask.reshape(-1, 1)
        R_k = (similarity_col * masks_np).mean(axis=0)
    else:
        raise ValueError("Length of similarity does not match num_masks and no mask grouping info is available.")

print("R_k shape:", R_k.shape)
print("First 10 R_k values:", R_k[:10])

# Optional: plot R_k
plt.figure(figsize=(10, 4))
plt.plot(R_k, marker='o')
plt.xlabel("Latent element k")
plt.ylabel("R_k")
plt.title("Per-latent similarity R_k (average similarity * mask_k)")
plt.tight_layout()
plt.show()


# In[22]:



        
        


# In[ ]:




