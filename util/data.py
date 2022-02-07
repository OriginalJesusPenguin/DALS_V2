import os
from typing import Tuple, Sequence, Union, List, Optional, Dict, Any
from collections import defaultdict

from tqdm import tqdm, trange

import numpy as np

import torch
import pytorch3d.io
from pytorch3d.structures import (
    Meshes,
    join_meshes_as_batch,
)
from pytorch3d.ops import sample_points_from_meshes


def load_npz_in_dir(
    path: Union[str, bytes, os.PathLike],
    keys: Optional[Sequence[str]] = None,
    loadbar: bool = False,
    stack: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load all npz files in dir and collect fields in dicitonary

    Args:
        path: Path-like giving directory to load from.
        keys: Fields to collect. If None, collect all fields.
        loadbar: Show progress with loading bar.
        stack: Stack tensors along new dimension before returning.

    Returns:
        data: dictionary with stacked tensors.
    """
    fnames = sorted(os.listdir(path))
    num_files = len(fnames)

    data = defaultdict(list)
    if loadbar:
        fnames = tqdm(fnames)
    # Load all fields into lists
    for fname in fnames:
        with np.load(os.path.join(path, fname)) as fdata:
            load_keys = fdata.files if keys is None else keys
            for key in load_keys:
                data[key].append(torch.from_numpy(fdata[key]))
    # Stack fields
    if stack:
        for key in data.keys():
            data[key] = torch.stack(data[key])

    return data


def split_dict_data(
    data: Dict[Any, torch.Tensor],
    index: int
) -> Tuple[Dict[Any, torch.Tensor], Dict[Any, torch.Tensor]]:
    """
    Split each tensor in dict at specified index
    """
    lhs = dict()
    rhs = dict()
    for key, tensor in data.items():
        lhs[key] = tensor[:index]
        rhs[key] = tensor[index:]

    return lhs, rhs


def load_meshes_in_dir(
    path: Union[str, bytes, os.PathLike],
    loadbar: bool = False,
) -> List[Meshes]:
    """
    Load all meshes in directory

    Args:
        path: Path-like giving directory to load from.
        loadbar: Show progress with loading bar.

    Returns:
        meshes: list of loaded Meshes objects.
    """
    mesh_fnames = sorted(os.listdir(path))
    num_meshes = len(mesh_fnames)

    meshes = []
    io = pytorch3d.io.IO()
    if loadbar:
        mesh_fnames = tqdm(mesh_fnames)
    for fname in mesh_fnames:
        meshes.append(io.load_mesh(os.path.join(path, fname), 
                                   include_textures=False))

    return meshes


def sample_meshes(
    meshes: Sequence[Meshes],
    num_samples_per_mesh: int = 10000,
    return_normals: bool = False,
    sampling_batch_size: int = 100,
    device: Optional[Union[str, torch.device]] = None,
    loadbar: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample points and normals from meshes in batches

    Args:
        meshes: Indexable sequence of Meshes objects to sample form.
        num_samples_per_mesh: Number of samples from each mesh.
        return_normals: Also return normal samples.
        sampling_batch_size: Batch size for mesh sampling
        device: Device to use for sampling. Default is device of first mesh.
        loadbar: Show progress with loading bar.

    Returns:
        point_samples: (N, 3) Tensor with point samples.
        normal_samples: (N, 3) Tensor with normal samples.
            Only returned if return_normals is True.
    """
    orig_device = meshes[0].device
    if device is None:
        device = orig_device
    point_samples = []
    normal_samples = []
    num_meshes = len(meshes)
    range_fn = trange if loadbar else range
    for ib in range_fn(0, num_meshes, sampling_batch_size):
        ie = min(num_meshes, ib + sampling_batch_size)
        samples = sample_points_from_meshes(
            join_meshes_as_batch([m.to(device) for m in meshes[ib:ie]]),
            num_samples=num_samples_per_mesh,
            return_normals=return_normals,
        )
        if return_normals:
            point_samples.append(samples[0].to(orig_device))
            normal_samples.append(samples[1].to(orig_device))
        else:
            point_samples.append(samples.to(orig_device))

    if return_normals:
        return torch.cat(point_samples), torch.cat(normal_samples)
    else:
        return torch.cat(point_samples)
