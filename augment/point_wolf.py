"""
MIT License

Copyright (c) 2021 MLV Lab (Machine Learning and Vision Lab at Korea University)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Sequence, List

from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_farthest_points


def augment_points(
    points: torch.Tensor,
    num_augment: int = 100,
    num_anchor: int = 4,
    sample_type: str = 'fps',
    sigma: float = 0.5,
    R_range: float = 10,
    S_range: float = 3,
    T_range: float = 0.25,
    loadbar=False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Augment points with PointWOLF.
    """
    assert len(points.shape) == 3, 'points must be B x N x D'
    if len(points) == 0:
        # Nothing to do, just return
        return []

    # Force CPU for augmentation to avoid GPU compatibility issues
    device = torch.device('cpu')
    pw = PointWOLF(
        num_anchor=num_anchor,
        sample_type=sample_type,
        sigma=sigma,
        R_range=R_range,
        S_range=S_range,
        T_range=T_range,
    )

    aug_points = []

    if loadbar:
        points = tqdm(points)
    for p in points:
        p_d = p.to(device)
        for _ in range(num_augment):
            aug_points.append(pw(p_d)[1].to(p.device))

    return torch.stack(aug_points)


def augment_meshes(
    meshes: Sequence[Meshes],
    num_augment: int = 100,
    num_anchor: int = 4,
    sample_type: str = 'fps',
    sigma: float = 0.5,
    R_range: float = 10,
    S_range: float = 3,
    T_range: float = 0.25,
    loadbar=False,
    device: torch.device = None,
) -> List[Meshes]:
    """
    Augment meshes with PointWOLF.
    """
    if len(meshes) == 0:
        # Nothing to do, just return
        return []

    # Force CPU for augmentation to avoid GPU compatibility issues
    device = torch.device('cpu')
    pw = PointWOLF(
        num_anchor=num_anchor,
        sample_type=sample_type,
        sigma=sigma,
        R_range=R_range,
        S_range=S_range,
        T_range=T_range,
    )

    aug_meshes = []

    if loadbar:
        meshes = tqdm(meshes)
    for mesh in meshes:
        verts = mesh.verts_packed().to(device)
        faces = mesh.faces_packed().unsqueeze(0)

        for _ in range(num_augment):
            aug_meshes.append(Meshes(
                verts=pw(verts)[1].unsqueeze(0).to(mesh.device),
                faces=faces,
            ))

    return aug_meshes
    

def uniform(low, high, size, device=None, dtype=None):
    """Return uniformaly random numbers between low and high."""
    a = torch.rand(*size, device=device, dtype=dtype)
    return low + (high - low) * a
    

class PointWOLF(object):
    def __init__(
        self,
        num_anchor: int = 4,
        sample_type: str = 'fps',
        sigma: float = 0.5,
        R_range: float = 10,
        S_range: float = 3,
        T_range: float = 0.25,
    ):
        """
        input:
            w_num_anchor: Num of anchor point 
            sample_type: Sampling method for anchor point, option : (fps, random) 
            sigma: Kernel bandwidth  
            R_range: Maximum rotation range of local transformation
            S_range: Maximum scailing range of local transformation
            T_range: Maximum translation range of local transformation
        """
        self.num_anchor = num_anchor
        self.sample_type = sample_type
        self.sigma = sigma

        self.R_range = (-abs(R_range), abs(R_range))
        self.S_range = (1., S_range)
        self.T_range = (-abs(T_range), abs(T_range))
        
        
    def __call__(self, pos):
        """
        input :
            pos([N,3])
            
        output : 
            pos([N,3]) : original pointcloud
            pos_new([N,3]) : Pointcloud augmneted by PointWOLF
        """
        device = pos.device
        M = self.num_anchor  # (M x 3)
        N, _=pos.shape  # (N)
        
        if self.sample_type == 'random':
            idx = torch.multinomial(
                torch.ones(1, device=device).expand(N, 1),
                M,
            )
            pos_anchor = pos[idx]  # (M, 3), anchor point
        elif self.sample_type == 'fps':
            pos_anchor = sample_farthest_points(
                pos.unsqueeze(0),
                torch.tensor([M], device=device),
                M
            )[0][0]  # Select first output and remove batch dimension.
        
        
        pos_repeat = pos.unsqueeze(0).expand(M, -1, -1) # (M, N, 3)
        pos_normalize = torch.zeros_like(pos_repeat)  # (M, N, 3)
        
        #Move to canonical space
        pos_normalize = pos_repeat - pos_anchor.view(M, -1, 3)
        
        #Local transformation at anchor point
        pos_transformed = self.local_transformaton(pos_normalize)  # (M, N, 3)
        
        #Move to origin space
        pos_transformed = pos_transformed + pos_anchor.view(M, -1, 3) #(M, N, 3)
        
        pos_new = self.kernel_regression(pos, pos_anchor, pos_transformed)
        pos_new = self.normalize(pos_new)
        
        # Final safety check - if augmentation failed, return original
        if torch.isnan(pos_new).any() or torch.isinf(pos_new).any():
            print("Warning: Augmentation produced NaN/Inf, using original points")
            return pos.float(), pos.float()
        
        return pos.float(), pos_new.float()
        

    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([N,3])
            pos_anchor([M,3])
            pos_transformed([M,N,3])
            
        output : 
            pos_new([N,3]) : Pointcloud after weighted local transformation 
        """
        M, N, _ = pos_transformed.shape
        device = pos_transformed.device
        
        # Distance between anchor points & entire points
        sub = pos_anchor.unsqueeze(1).expand(-1, N, -1) - pos.unsqueeze(0).expand(M, -1, -1)  # (M, N, 3), d
        
        project_axis = self.get_random_axis(1, device=device)

        eye = torch.eye(3, device=sub.device, dtype=sub.dtype)
        projection = project_axis.unsqueeze(1) * eye  # (1, 3, 3)
        
        # Project distance
        sub = sub @ projection  # (M, N, 3)
        sub = torch.sum(sub ** 2, dim=2).sqrt()  # (M, N)
        
        # Kernel regression
        weight = torch.exp(-0.5 * (sub ** 2) / (self.sigma ** 2))  # (M, N) 
        pos_new = torch.sum(weight.unsqueeze(2).expand(-1, -1, 3) * pos_transformed, dim=0)  # (N, 3)
        
        # Avoid division by zero
        weight_sum = weight.sum(dim=0, keepdim=True).T
        weight_sum = torch.clamp(weight_sum, min=1e-8)  # Prevent division by zero
        pos_new = pos_new / weight_sum
        
        # Check for NaN/Inf and replace with original if needed
        if torch.isnan(pos_new).any() or torch.isinf(pos_new).any():
            print("Warning: NaN/Inf detected in kernel regression, using original points")
            return pos
            
        return pos_new

    
    def fps(self, pos, npoint):
        """
        input : 
            pos([N,3])
            npoint(int)
            
        output : 
            centroids([npoints]) : index list for fps
        """
        N, _ = pos.shape
        device = pos.device
        centroids = torch.zeros(npoint, dtype=torch.long, device=device)  # (M)
        distance = torch.ones(N, dtype=pos.dtype, device=device) * 1e10  # (N)
        farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = pos[farthest, :]
            dist = torch.sum((pos - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = distance.argmax()
        return centroids

    
    def local_transformaton(self, pos_normalize):
        """
        input :
            pos([N,3]) 
            pos_normalize([M,N,3])
            
        output :
            pos_normalize([M,N,3]) : Pointclouds after local transformation centered at M anchor points.
        """
        M,N,_ = pos_normalize.shape
        device = pos_normalize.device
        transformation_dropout = torch.multinomial(torch.tensor([1.0, 1.0], device=device), M*3, replacement=True).view(M, 3)
        transformation_axis = self.get_random_axis(M, device=device)  # (M, 3)

        degree = np.pi * uniform(*self.R_range, size=(M, 3), device=device) / 180.0 * transformation_dropout[:, 0:1]  # (M, 3), sampling from (-R_range, R_range) 
        
        scale = uniform(*self.S_range, size=(M, 3), device=device) * transformation_dropout[:, 1:2]  # (M, 3), sampling from (1, S_range)
        scale = scale * transformation_axis
        scale = scale + 1 * (scale == 0) #Scaling factor must be larger than 1
        
        trl = uniform(*self.T_range, size=(M, 3), device=device) * transformation_dropout[:, 2:3]  # (M, 3), sampling from (1, T_range)
        trl = trl * transformation_axis
        
        # Scaling Matrix
        eye = torch.eye(3, device=scale.device)
        S = scale.unsqueeze(1) * eye  # Scaling factor to diagonal matrix (M, 3) -> (M, 3, 3)

        # Rotation Matrix
        sins = torch.sin(degree)
        coss = torch.cos(degree)
        sx, sy, sz = sins[:,0], sins[:,1], sins[:,2]
        cx, cy, cz = coss[:,0], coss[:,1], coss[:,2]
        R = torch.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
             sz*cy, sz*sy*sx + cz*cy, sz*sy*cx - cz*sx,
             -sy, cy*sx, cy*cx], dim=1).view(M, 3, 3)
        
        pos_normalize = pos_normalize @ R @ S + trl.view(M, 1, 3)
        return pos_normalize
    

    def get_random_axis(self, n_axis, device=None):
        """
        input :
            n_axis(int)
            
        output :
            axis([n_axis,3]) : projection axis   
        """
        axis = torch.randint(1, 8, (n_axis,), device=device) # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz    
        m = 3 
        axis = (((axis[:, None] & (1 << torch.arange(m, device=device)))) > 0).to(int)
        return axis
    

    def normalize(self, pos):
        """
        input :
            pos([N,3])
        
        output :
            pos([N,3]) : normalized Pointcloud
        """
        pos = pos - pos.mean(dim=-2, keepdim=True)
        
        # Calculate scale with safety checks
        pos_norm = torch.sqrt((pos ** 2).sum(1))
        max_norm = pos_norm.max()
        
        # Avoid division by zero
        if max_norm < 1e-8:
            print("Warning: Very small norm detected, skipping normalization")
            return pos
            
        scale = (1 / max_norm) * 0.999999
        pos = scale * pos
        
        # Final safety check
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            print("Warning: NaN/Inf detected after normalization, returning original")
            return pos - pos.mean(dim=-2, keepdim=True)
            
        return pos

