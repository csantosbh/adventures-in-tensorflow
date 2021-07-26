#!/usr/bin/env python

from typing import Dict, Tuple, Union
import os

import numpy as np
import icecream
import quaternion
from plyfile import PlyData
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def load_conf(fname: Union[Path, str]) -> Dict[str, np.ndarray]:
    """
    Load stanford .conf file, returning a dict of file names -> 4x4 transform local2world matrices
    """

    with open(fname, 'r') as fhandle:
        lines = fhandle.readlines()

    # Get translations
    translations = {
        line.split(' ')[1]: np.array([[float(v) for v in line.split(' ')[2:2+3]]]).T
        for line in lines if line.split(' ')[0] == 'bmesh'
    }

    # Get rotations
    def rearrange_quaternion(x):
        return [x[3], x[0], x[1], x[2]]

    rotations = {
        line.split(' ')[1]: quaternion.as_rotation_matrix(
            quaternion.as_quat_array(
                rearrange_quaternion([float(v) for v in line.split(' ')[2+3:]])
            )
        ).T
        for line in lines if line.split(' ')[0] == 'bmesh'
    }

    # Convert translations and rotations to 4x4 matrices
    transforms = {
        fname: np.concatenate([
            np.concatenate([rotations[fname], translations[fname]], 1),
            [[0, 0, 0, 1]]
        ], 0)
        for fname in translations
    }

    return transforms


def load_point_cloud(fname: Union[str, Path],
                     local2world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Stanford .ply point cloud and returns (positions_w, scanner_dir_w), where:
    * positions_w is the set of points in world coordinates; each point is on dim 0 and its xyz coords are on dim 1
    * scanner_dir_w is a per-point direction to the scanner, which is later used to determine the normal direction
    """
    plydata = PlyData.read(fname)
    positions = np.concatenate(
        [np.array([plydata.elements[0].data[k]]).T for k in ['x', 'y', 'z']],
        1
    )
    positions_homog = np.concatenate([positions, np.ones_like(positions)[:, 0:1]], 1)

    # Remove points too far away from the sensor
    threshold = 1e-3
    positions_homog = positions_homog[positions_homog[:, 2] >= threshold, :]

    # Discard outlier points
    outlier_threshold = 0.02
    neighbor_search = NearestNeighbors(n_neighbors=20, n_jobs=-1)
    neighbor_search.fit(positions_homog[:, 0:3])
    dist, _ = neighbor_search.kneighbors(positions_homog[:, 0:3])
    inliers = dist[:, -1] < outlier_threshold
    positions_homog = positions_homog[inliers, :]

    # Compute direction to camera
    camera_dir = np.zeros_like(positions_homog)
    camera_dir[:, 2] = 1

    # Transform to world coordinates
    positions_w = np.matmul(local2world, positions_homog.T).T[:, 0:3]
    scanner_dir_w = np.matmul(local2world, camera_dir.T).T[:, 0:3]

    return positions_w, scanner_dir_w


def compute_normals(positions: np.ndarray,
                    scanner_dirs: np.ndarray,
                    num_neighbors: int = 50) -> np.ndarray:
    """
    Estimates the inwards normal vectors of the point cloud.
    This is computed by the cross product of the two highest components of the PCA components for points around a local
    neighborhood of each point.
    """
    # Get neighborhood around each point
    neighbor_search = NearestNeighbors(n_neighbors=num_neighbors, n_jobs=-1)
    neighbor_search.fit(positions)

    dist, idx = neighbor_search.kneighbors(positions)

    # Run PCA to get largest dispersion directions
    # [b, n_neigh, 3]
    samples = positions[idx]
    samples = samples - np.mean(samples, axis=1, keepdims=True)
    # [b, 3, 3]
    covariance = np.einsum('bki,bkj->bij', samples, samples) / (num_neighbors - 1)
    # [b, 3, 3]
    w, v = np.linalg.eigh(covariance)

    # The normal is the cross of the two largest principal components
    norm = np.cross(v[:, :, -2], v[:, :, -1])
    norm = norm / np.linalg.norm(norm, ord=2, axis=1, keepdims=True)

    # Invert normals that are pointing towards the camera
    alpha = (np.sum(norm * scanner_dirs, 1, keepdims=True) >= 0).astype(np.float32)
    norm = norm * (1 - alpha) - alpha * norm

    return norm


def combine_point_clouds(folder: str,
                         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all .ply point clouds from specified folder and combines them into a single representation.
    This returns the point cloud positions and the normals.
    """
    if verbose:
        print('Loading transforms from .conf files')

    ply_transforms = {
        ply_name: transform for conf_name in tqdm(list(Path(folder).rglob('*.conf')))
        for ply_name, transform in load_conf(conf_name).items()
    }

    if verbose:
        print('Loading individual scan poses')

    positions = []
    scanner_dirs = []
    ply_paths = list(Path(folder).rglob('*.ply'))
    for ply_name, transform in tqdm(ply_transforms.items()):
        ply_path = [path for path in ply_paths if path.name == ply_name][0]
        pos_i, norms_i = load_point_cloud(ply_path, transform)
        positions.append(pos_i)
        scanner_dirs.append(norms_i)

    positions = np.concatenate(positions, 0)
    scanner_dirs = np.concatenate(scanner_dirs, 0)

    if verbose:
        print('Computing normals')

    normals = compute_normals(positions, scanner_dirs)

    return positions, normals


def main():
    import mayavi.mlab as mlab

    icecream.install()
    cloud_name = 'armadillo.npz'

    if not os.path.isfile(cloud_name):
        positions, normals = combine_point_clouds('./')
        np.savez_compressed(cloud_name, **{'positions': positions, 'normals': normals})
    else:
        data = np.load(cloud_name)
        positions = data['positions']
        normals = data['normals']

    mlab.points3d(positions[::15, 0],  positions[::15, 1],  positions[::15, 2], scale_factor=4e-4)
    mlab.quiver3d(positions[::15, 0],  positions[::15, 1],  positions[::15, 2],
                    normals[::15, 0],    normals[::15, 1],    normals[::15, 2], scale_factor=8e-4)
    mlab.show()


if __name__ == '__main__':
    main()
