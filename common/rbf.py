from typing import Callable

import numpy as np


def gaussian_kernel(sigma):
    def gaussian_kernel_impl(x):
        return np.exp(-x**2 / (2 * sigma**2))

    return gaussian_kernel_impl


def rbf(coords: np.ndarray,
        values: np.ndarray,
        kernel: Callable[[np.ndarray], np.ndarray] = lambda x: x) -> Callable[[np.ndarray], np.ndarray]:
    """
    coords: [n, dims_coords]
    values: [n, dims_values]
    """
    # TODO remove duplicate points or this will break

    # [1, n, dims_coords] x [n, 1, dims_coords] -> [n, n]
    dists = kernel(np.linalg.norm(
        # [1, n, dims_coords] - [n, 1, dims_coords] -> [n, n, dims_coords]
        coords[np.newaxis, ...] - coords[:, np.newaxis, :], axis=2
    ))

    # [n, dims_values]
    weights = np.matmul(np.linalg.pinv(dists), values)

    # [n_s, dims_coords] -> [n_s, dims_values]
    def sample(coord):
        # Compute (kerneled) distance of query to all points
        dist = kernel(np.linalg.norm(
            coords[np.newaxis, ...] - coord[:, np.newaxis, :], axis=2
        ))

        # Compute weighted average using dist and weights
        return np.matmul(dist, weights)

    return sample
