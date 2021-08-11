#!/usr/bin/env python
from typing import Optional, List, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from common.plot import ISequencePlotter, InteractivePlotter, GifPlotter


def generate_circle(count: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Debugging utility
    """
    angles = sorted(np.random.uniform(size=(count, 1)) * 2 * np.pi)

    pts = np.concatenate([np.sin(angles), np.cos(angles)], 1)
    norma = lambda x: x / np.sqrt(np.sum(x**2, 1, keepdims=True))
    norms = -norma(pts)
    return pts, norms


def generate_sphere(count: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Debugging utility
    """
    angles_a = np.random.uniform(size=(count, 1)) * 2 * np.pi
    angles_b = np.random.uniform(size=(count, 1), low=-1, high=1) * np.pi / 2.0

    pts = np.concatenate([np.sin(angles_a), np.cos(angles_a)], 1)
    pts = np.concatenate([
        np.sin(angles_b), np.cos(angles_b) * pts
    ], 1)
    norma = lambda x: x / np.sqrt(np.sum(x**2, 1, keepdims=True))
    norms = -norma(pts)

    return pts, norms


def solve_poisson(tgt_laplacian: np.ndarray,
                  initial_solution: np.ndarray = None,
                  num_its: int = 500,
                  plotter: ISequencePlotter = None) -> np.ndarray:
    """
    Iterative poisson solver for 2d grids.
    Returns a grid with the same dimensions as tgt_laplacian such that
    laplacian(result) ≃ tgt_laplacian.
    Will start with `initial_solution`, if provided; otherwise, the initial
    solution will be `tgt_laplacian`.
    """
    if initial_solution is None:
        sol = tgt_laplacian.copy()
    else:
        sol = initial_solution.copy()
    
    dt2 = 1
    skip_rate = max(1, num_its // 100)

    for i in tqdm(range(num_its)):
        neighborhood_average = (sol[0:-2, 1:-1] + sol[2:, 1:-1] +
                                sol[1:-1, 0:-2] + sol[1:-1, 2:]) / 4.0
        laplacian_update = neighborhood_average - tgt_laplacian[1:-1, 1:-1] * dt2 / 4.0

        sol[1:-1, 1:-1] = laplacian_update

        if plotter and (i % skip_rate) == 0:
            plotter.update(sol, cmap='PuRd', origin='lower')

    return sol


def solve_poisson_3d(tgt_laplacian: np.ndarray,
                     initial_solution: np.ndarray = None,
                     num_its: int = 500,
                     plotter: ISequencePlotter = None) -> np.ndarray:
    """
    Iterative poisson solver for 3d volumes.
    Returns a volume with the same dimensions as tgt_laplacian such that
    laplacian(result) ≃ tgt_laplacian.
    Will start with `initial_solution`, if provided; otherwise, the initial
    solution will be `tgt_laplacian`.
    """
    if initial_solution is None:
        sol = tgt_laplacian.copy()
    else:
        sol = initial_solution.copy()

    dt2 = 1
    skip_rate = max(1, num_its // 100)

    for i in tqdm(range(num_its)):
        neighborhood_average = (sol[0:-2, 1:-1, 1:-1] + sol[2:, 1:-1, 1:-1] +
                                sol[1:-1, 0:-2, 1:-1] + sol[1:-1, 2:, 1:-1] +
                                sol[1:-1, 1:-1, 0:-2] + sol[1:-1, 1:-1, 2:]) / 6.0
        laplacian_update = neighborhood_average - tgt_laplacian[1:-1, 1:-1, 1:-1] * dt2 / 6.0

        sol[1:-1, 1:-1, 1:-1] = laplacian_update

        if plotter and (i % skip_rate == 0):
            plotter.update(sol[:, :, sol.shape[0] // 2].T, cmap='PuRd', origin='lower')

    return sol


def poisson_reconstruct_level_2d(points: np.ndarray,
                                 norms: np.ndarray,
                                 res: int = 128,
                                 initial_solution: np.ndarray = None,
                                 num_its: int = 500,
                                 plotter: ISequencePlotter = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single pass of the Poisson Surface Reconstruction for 2d grids

    `points` are the point cloud positions in world coordinates. This is a rank
    2 tensor with each point being specified at dim 0 and the xy coordinates in dim 1.
    The same applies to `norms`.

    Returns a tuple (scalar_field, edge_levels). Given enough points, the level
    set at `mean(edge_levels)` of `scalar_field` should be the desired surface.
    """
    if initial_solution is not None:
        res = initial_solution.shape[0]

    # Transform from world coordinates to grid coordinates
    world2grid = lambda p: ((p/1.5)*0.5 + 0.5) * (res - 1)

    yx = world2grid(points).astype(np.int64)

    # Raster normals
    n_grid = np.zeros((res, res, 2))

    # Nearest Neighbor raster:
    n_grid[yx[:, 0], yx[:, 1], :] = norms

    n_grid = cv2.GaussianBlur(n_grid, (17, 17), 0.3)

    # Compute divergent of rasterized normals
    d_grid = np.zeros((res, res))
    d_grid[1:-1, 1:-1] = n_grid[  2:, 1:-1, 0] - n_grid[0:-2, 1:-1, 0] + \
                         n_grid[1:-1,   2:, 1] - n_grid[1:-1, 0:-2, 1]

    scalar_field = solve_poisson(
        d_grid, initial_solution=initial_solution, num_its=num_its, plotter=plotter)
    edge_levels = scalar_field[yx[:, 0], yx[:, 1]]

    return scalar_field, edge_levels


def poisson_reconstruct_level_3d(points: np.ndarray,
                                 norms: np.ndarray,
                                 res: int = 128,
                                 initial_solution: np.ndarray = None,
                                 num_its: int = 500,
                                 plotter: ISequencePlotter = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single pass of the Poisson Surface Reconstruction for 3d volumes

    `points` are the point cloud positions in world coordinates. This is a rank
    2 tensor with each point being specified at dim 0 and the xyz coordinates in dim 1.
    The same applies to `norms`.

    Returns a tuple (scalar_field, edge_levels). Given enough points, the level
    set at `mean(edge_levels)` of `scalar_field` should be the desired surface.
    """
    if initial_solution is not None:
        res = initial_solution.shape[0]

    # Transform from world coordinates to grid coordinates
    xyz_range = 0.8
    clamp = lambda p, low, high: np.maximum(np.minimum(p, high), low)
    world2grid = lambda p: clamp(((p/xyz_range)*0.5 + 0.5) * (res - 1), 0, res - 1)

    zyx = world2grid(points).astype(np.int64)

    # Raster normals
    n_grid = np.zeros((res, res, res, 3))

    # Nearest Neighbor raster:
    n_grid[zyx[:, 0], zyx[:, 1], zyx[:, 2], :] = norms

    # TODO Apply 3d blur to n_grid

    # Compute divergent of rasterized normals
    d_grid = np.zeros((res, res, res))
    d_grid[1:-1, 1:-1, 1:-1] = n_grid[  2:, 1:-1, 1:-1, 0] - n_grid[0:-2, 1:-1, 1:-1, 0] + \
                               n_grid[1:-1,   2:, 1:-1, 1] - n_grid[1:-1, 0:-2, 1:-1, 1] + \
                               n_grid[1:-1, 1:-1,   2:, 2] - n_grid[1:-1, 1:-1, 0:-2, 2]

    scalar_field = solve_poisson_3d(
        d_grid, initial_solution=initial_solution, num_its=num_its, plotter=plotter)
    edge_levels = scalar_field[zyx[:, 0], zyx[:, 1], zyx[:, 2]]

    return scalar_field, edge_levels


def upsample_nn(volume: np.ndarray,
                factor: int = 2) -> np.ndarray:
    """
    Nearest neighbor upsample of tensor by integer factor
    """
    resize_matrix_shape = [factor for _ in range(volume.ndim)]
    return np.kron(volume, np.ones(resize_matrix_shape))


def reconstruct_2d(points: np.ndarray,
                   normals: np.ndarray,
                   initial_resolution: int,
                   iterations: List[int],
                   save_gif: Optional[str]):
    """
    Entry point for the 2D surface reconstruction demo.
    Performs multiple iterations of reconstruction steps at grids of increasingly
    higher resolution to speed up convergence of the Poisson solver.
    """
    if save_gif:
        plotter = GifPlotter(save_gif, tgt_resolution=(256, 256))
    else:
        plotter = InteractivePlotter(tgt_resolution=(512, 512))

    points = (points - np.mean(points, 0)) / (np.max(points) - np.min(points)) * 2

    scalar_field, edge_levels = poisson_reconstruct_level_2d(
        points, normals, res=initial_resolution, num_its=iterations[0], plotter=plotter)

    for its in iterations[1:]:
        scalar_field, edge_levels = poisson_reconstruct_level_2d(
            points, normals, initial_solution=upsample_nn(scalar_field),
            num_its=its, plotter=plotter)

    plt.figure('Isocurve for Sample Points')
    plt.contour(scalar_field[::-1, :], [np.mean(edge_levels)], origin='upper')
    plt.gca().set_aspect('equal')
    plt.show(block=True)

    plotter.finalize()


def reconstruct_3d(points: np.ndarray,
                   normals: np.ndarray,
                   output_name: str,
                   initial_resolution: int,
                   iterations: List[int],
                   save_gif: Optional[str]):
    """
    Entry point for the 3D surface reconstruction demo.
    Performs multiple iterations of reconstruction steps at grids of increasingly
    higher resolution to speed up convergence of the Poisson solver.
    """
    from common.plot import contour3d
    if save_gif:
        plotter = GifPlotter(save_gif, tgt_resolution=(256, 256))
    else:
        plotter = InteractivePlotter(tgt_resolution=(512, 512))

    if not os.path.isfile(output_name):
        points = (points - np.mean(points, 0)) / (np.max(points) - np.min(points)) * 2

        scalar_field, edge_levels = poisson_reconstruct_level_3d(
            points, normals, res=initial_resolution, num_its=iterations[0], plotter=plotter)

        for its in iterations[1:]:
            scalar_field, edge_levels = poisson_reconstruct_level_3d(
                points, normals, initial_solution=upsample_nn(scalar_field),
                num_its=its, plotter=plotter)

        target_level = np.mean(edge_levels)
        np.savez_compressed(
            output_name, scalar_field=scalar_field, target_level=target_level)
    else:
        data = np.load(output_name)
        scalar_field = data['scalar_field']
        target_level = data['target_level']

    contour3d(scalar_field, contours=[target_level])
    plotter.finalize()
