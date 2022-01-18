from typing import Tuple

import numpy as np

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
    for n in range(degree):
        for m in range(-n, n):
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
    return np.real(out)
