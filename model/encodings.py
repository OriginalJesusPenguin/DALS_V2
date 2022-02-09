from typing import Tuple

import numpy as np
import torch

from scipy.special import sph_harm


def cart2sph(x, y, z):
    """
    Transform Cartesian to spherical coordinates.
    
    From: https://geospace-code.github.io/pymap3d/utils.html
    
    Args:
        x, y, z: Array-like with input coordinates.
        
    Returns:
        az: np.array with azimuth angles.
        el: np.array with elevation angles.
        r: np.array with radii.
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph_encoding(directions, degree):
    """
    Compute real-valued spherical harmonics encoding for direction vectors.
    
    Spherical harmonics are computed by first computing azimuth and elevation
    for each direction (length is ignored) and then calling 
    scipy.special.sph_harm. For each degree we use all orders.
    
    Args:
        directions: N x 3 Array-like with direction vectors.
        degree: Max. degree of spherical harmonics functions.
        
    Returns:
        out: N x D Real-valued spherical harmonic encoding.
    """
    az, el, _ = cart2sph(directions[:, 0], directions[:, 1], directions[:, 2])
    out = []
    # Add 1 since range is exclusive
    for n in range(degree + 1):
        for m in range(-n, n + 1):
            s = sph_harm(m, n, az, el)
            # Convert to real-valued version as described in
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
            if m < 0:
                out.append(np.sqrt(2) * (-1 ** m) * np.imag(s))
            elif m > 0:
                out.append(np.sqrt(2) * (-1 ** m) * np.real(s))
            else:  # m == 0
                out.append(s)
                
    out = np.stack(out).T
    assert np.all(np.imag(out) == 0)  # Make sure no imaginary values snuck in
    return np.real(out).astype('float32')


def pos_encoding(x, degree):
    """
    Positional encoding with sines and cosines from the NERF paper.

    Uses the encoding from Mildenhall et al., 2020.
    Encodes 3D positions (assumed normalzed to be in [-1, 1]^3) as
    (sin(2^0*pi*x), cos(2^0*pi*x), ..., sin(2^(L-1)*pi*x), cos(2^(L-1)*pi*x))
    where L = degree.

    Args:
        x: (N, D) Array-like with positions.
        degree: Max. frequency of encoding.

    Returns:
        out: (N, D, 2 * degree) Tensor with encoded positions
    """
    out = []
    for d in range(degree):
        w = np.pi * 2 ** d
        out.append(torch.sin(w * x))
        out.append(torch.cos(w * x))

    return torch.stack(out, dim=-1)

