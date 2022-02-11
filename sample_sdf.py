import os
import sys
from os.path import join
from glob import glob
from time import perf_counter as time

from tqdm import tqdm

import numpy as np

import trimesh

import numpy as np

from mesh_to_sdf import surface_point_cloud
from mesh_to_sdf.surface_point_cloud import BadMeshException
from mesh_to_sdf.utils import scale_to_unit_sphere, sample_uniform_points_in_unit_sphere

# Sampling code from https://github.com/marian42/mesh_to_sdf/

def sample_sdf_near_pointcloud_surface(
    pcl,
    number_of_points=500000,
    use_scans=True,
    sign_method='normal',
    normal_sample_count=11,
    min_size=0,
    return_gradients=False,
    sigma1=0.0025,
    sigma2=0.00025,
):
    """From mesh_to_sdf/surface_point_cloud.py"""
    query_points = []
    surface_sample_count = int(number_of_points * 47 / 50) // 2
    surface_points = pcl.get_random_surface_points(surface_sample_count, use_scans=use_scans)
    query_points.append(surface_points + np.random.normal(scale=sigma1, size=(surface_sample_count, 3)))
    query_points.append(surface_points + np.random.normal(scale=sigma2, size=(surface_sample_count, 3)))
    
    unit_sphere_sample_count = number_of_points - surface_points.shape[0] * 2
    unit_sphere_points = sample_uniform_points_in_unit_sphere(unit_sphere_sample_count)
    query_points.append(unit_sphere_points)
    query_points = np.concatenate(query_points).astype(np.float32)

    if sign_method == 'normal':
        sdf = pcl.get_sdf_in_batches(query_points, use_depth_buffer=False, sample_count=normal_sample_count, return_gradients=return_gradients)
    elif sign_method == 'depth':
        sdf = pcl.get_sdf_in_batches(query_points, use_depth_buffer=True, return_gradients=return_gradients)
    else:
        raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
    if return_gradients:
        sdf, gradients = sdf

    if min_size > 0:
        model_size = np.count_nonzero(sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
        if model_size < min_size:
            raise BadMeshException()

    if return_gradients:
        return query_points, sdf, gradients
    else:
        return query_points, sdf


def get_surface_point_cloud(
    mesh,
    surface_point_method='scan',
    bounding_radius=None,
    scan_count=100,
    scan_resolution=400,
    sample_point_count=10000000,
    calculate_normals=True
):
    """From mesh_to_sdf/__init__.py"""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    if bounding_radius is None:
        bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1
        
    if surface_point_method == 'scan':
        return surface_point_cloud.create_from_scans(mesh, bounding_radius=bounding_radius, scan_count=scan_count, scan_resolution=scan_resolution, calculate_normals=calculate_normals)
    elif surface_point_method == 'sample':
        return surface_point_cloud.sample_from_mesh(mesh, sample_point_count=sample_point_count, calculate_normals=calculate_normals)        
    else:
        raise ValueError('Unknown surface point sampling method: {:s}'.format(surface_point_method))


def sample_sdf_near_surface(
    mesh,
    number_of_points = 500000,
    surface_point_method='scan',
    sign_method='normal',
    scan_count=100,
    scan_resolution=400,
    sample_point_count=10000000,
    normal_sample_count=11,
    min_size=0,
    return_gradients=False,
    sigma1=0.0025,
    sigma2=0.00025,
):
    """From mesh_to_sdf/__init__.py"""
    mesh = scale_to_unit_sphere(mesh)
    
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal' or return_gradients)

    return sample_sdf_near_pointcloud_surface(surface_point_cloud, number_of_points, surface_point_method=='scan', sign_method, normal_sample_count, min_size, return_gradients, sigma1=sigma1, sigma2=sigma2)


os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Enable the script to be used with no screen

# TODO: Maybe find a way to take this from command line too
base_dir = '/work1/patmjen/meshfit/datasets/shapes/liver/raw'
out_dir = '/work1/patmjen/meshfit/datasets/sdf/liver3/raw'

# mesh_fnames = sorted(glob(join(base_dir, '*/model.obj')))
mesh_fnames = sorted(glob(join(base_dir, '*.ply')))

argc = len(sys.argv)
start = int(sys.argv[1]) if argc > 1 else 0
sigma1 = float(sys.argv[2]) if argc > 2 else 0.0025
sigma2 = float(sys.argv[3]) if argc > 3 else sigma1 / 10

for fname in tqdm(mesh_fnames[start:]):
    model_name = fname.split('/')[-1][:-4]
    out_fname = join(out_dir, model_name) + '.npz'
    if os.path.isfile(out_fname):
        continue
    
    mesh = trimesh.load(fname)
    points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000, sign_method='depth')
    np.savez(out_fname, points=points, sdf=sdf)
