import torch 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go 
import glob



# Define parameters:
import re 

n_masks = 10
latent_dim = 128
p=0.5 # bernouli masking probability
test_latent_vector_files = glob.glob('/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26/latents/*.pt')
print((test_latent_vector_files))
test_latent_vector_files_disease_state= []
test_latent_vector_patient_ids = []

patient_id_pattern = re.compile(r'.*latents/(?:.*_)?(\d+)[_.]')

for file in test_latent_vector_files:
    # Determine disease state
    if 'healthy' in file:
        test_latent_vector_files_disease_state.append('healthy')
        status = 'healthy'
    else:
        test_latent_vector_files_disease_state.append('cirrhotic')
        status = 'cirrhotic'
    # Extract patient ID using regex
    basename = file.split('/')[-1]
    match = patient_id_pattern.match(file)
    if match:
        patient_id = match.group(1)
    else:
        patient_id = None
    print(patient_id)
    test_latent_vector_patient_ids.append(patient_id)
    print(status)
    print(file)


test_objs = []

for patient_id,status in zip(test_latent_vector_patient_ids,test_latent_vector_files_disease_state):
    print(patient_id,status)
    test_objs.append(f'/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26/{status}_{patient_id}_testing_target.obj')
    print(test_objs[-1])
    print('does a file exist at this path?')
    print(os.path.exists(test_objs[-1]))






test_latent_vectors = [torch.load(file) for file in test_latent_vector_files]

print(test_latent_vectors[0].shape)




# Create a mask for each latent vector with a given random seed:
masks = []  
for seed in range(n_masks):
    torch.manual_seed(seed)
    mask = torch.bernoulli(torch.ones(latent_dim) * p)
    masks.append(mask)


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






print(decoded_mesh.verts_packed().shape)



import trimesh 


mesh = trimesh.Trimesh(decoded_mesh.verts_packed().cpu().numpy(), decoded_mesh.faces_packed().cpu().numpy())



test_metrics = {}

for i in range(n_masks):
    for j in range(len(test_latent_vectors)):
        decoded_mesh = masked_decoded_meshes[i,j,:]
        target_mesh = test_objs[j]
        # --- Metric computation ---
        with torch.no_grad():
            pred_samples = sample_points_from_meshes(decoded_mesh, metric_samples)
            true_samples = sample_points_from_meshes(target_mesh, metric_samples)
            chamfer_val = chamfer_distance(true_samples, pred_samples)[0] * 10000
            metric_dict = point_metrics(true_samples, pred_samples, [0.01, 0.02])
            bl_quality = (1.0 - mesh_bl_quality_loss(decoded_mesh)).item()






