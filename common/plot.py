import os
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import cv2


class ISequencePlotter(object):
    """
    Common interface for our plotters of sequences of images
    """
    def update(self, image, **kwargs):
        raise NotImplementedError()

    def finalize(self):
        pass


class InteractivePlotter(ISequencePlotter):
    """
    Plotter that allows updating the image dynamically
    """
    def __init__(self, name='InteractivePlotter', tgt_resolution=None):
        plt.ion()
        self.name = name
        self.tgt_resolution = tgt_resolution

    def update(self, image: np.ndarray, **kwargs):
        if kwargs.get('origin', None) == 'lower':
            image = image[::-1, ...]

        if kwargs.get('norm', False):
            upper = np.max(image)
            lower = np.min(image)
            image = (image - lower) / (upper - lower)

        if self.tgt_resolution:
            image = cv2.resize(image, self.tgt_resolution)

        if image.ndim != 3 or image.shape[2] == 1:
            # Apply color map (only for single-channel images)
            cmap = cm.get_cmap(kwargs.get('cmap', 'gray'))
            image = cmap(np.squeeze(image))

        if image.ndim == 3:
            # RGBA to BGR
            image = image[:, :, 2::-1]

        cv2.imshow(self.name, image)
        cv2.waitKey(1)

    def finalize(self):
        cv2.waitKey()


class GifPlotter(ISequencePlotter):
    """
    Plotter that converts the sequence of provided images to a gif file
    """
    def __init__(self, output_path, tgt_resolution=None):
        self.dst_dir = tempfile.TemporaryDirectory()
        self.output_path = output_path
        self.counter = 0
        self.tgt_resolution = tgt_resolution

    def update(self, image, **kwargs):
        if kwargs.get('origin', None) == 'lower':
            image = image[::-1, ...]

        if image.ndim != 3 or image.shape[2] == 1:
            # Apply color map (only for single-channel images)
            cmap = cm.get_cmap(kwargs.get('cmap', 'gray'))
            image = cmap(np.squeeze(image))

        if self.tgt_resolution:
            image = cv2.resize(image, self.tgt_resolution)

        if image.ndim == 3:
            # RGBA to BGR
            image = image[:, :, 2::-1]

        # Compose file name
        filename = Path(self.dst_dir.name) / f'frame_{self.counter:04d}.png'

        # Discretize
        image = np.clip(image * 255, 0, 255).astype(np.int)

        cv2.imwrite(str(filename), image)
        self.counter += 1

    def finalize(self):
        converter = subprocess.Popen([
            'ffmpeg',
            '-y',
            '-f', 'image2',
            '-i', f'{self.dst_dir.name}/frame_%04d.png',
            '-pix_fmt', 'rgb8',
            self.output_path
        ])
        converter.wait()


def contour3d(volume: np.ndarray,
              contours: np.ndarray,
              size: Optional[Tuple[int, int]] = None):
    """
    Plot the isosurfaces from the given 3D `volume` at the specified `contours`
    """
    import mayavi.mlab as mlab

    if size is None:
        size = (768, 768)

    mlab.figure(bgcolor=(1, 1, 1), size=size)
    scalar = mlab.pipeline.scalar_field(volume)
    voi = mlab.pipeline.extract_grid(scalar)
    mlab.pipeline.iso_surface(
        voi, contours=contours, color=(230/255.0, 99/255.0, 90/255.0))

    mlab.view(azimuth=15.0, elevation=150.0, distance=1.5*volume.shape[0],
              focalpoint=[volume.shape[0]/2, volume.shape[1]/2.2, volume.shape[2]/2],
              roll=0)

    mlab.show()


def plot_points_norms(points: np.ndarray,
                      norms: np.ndarray):
    """
    Quiver of points with given normal vectors
    """
    plt.quiver(points[:, 1], points[:, 0], norms[:, 1], norms[:, 0])
    plt.plot(points[:, 1], points[:, 0], 'o')
    plt.gca().set_aspect('equal')
    plt.show()


def plot_points_norms_3d(positions: np.ndarray,
                         normals: np.ndarray,
                         num_points: int = 5000,
                         size: Tuple[int, int] = None):
    """
    3D Quiver of points with given normal vectors
    """
    import mayavi.mlab as mlab
    if size is None:
        size = (768, 768)

    mlab.figure(bgcolor=(1, 1, 1), size=size)
    skip_rate = max(1, positions.shape[0] // num_points)

    mlab.points3d(
        positions[::skip_rate, 0], positions[::skip_rate, 1], positions[::skip_rate, 2],
        color=(31/255.0, 119/255.0, 180/255.0),
        scale_factor=1e-3)
    mlab.quiver3d(
        positions[::skip_rate, 0], positions[::skip_rate, 1], positions[::skip_rate, 2],
        normals[::skip_rate, 0], normals[::skip_rate, 1], normals[::skip_rate, 2],
        color=(0, 0, 0),
        scale_factor=1e-3)

    bounding_box_size = np.linalg.norm(np.max(positions, 0)-np.min(positions, 0), ord=2)
    mlab.view(azimuth=15.0, elevation=150.0, distance=1.5*0.8*bounding_box_size,
              focalpoint=np.mean(positions, 0) * np.array([1., 2/2.2, 1.0]),
              roll=0)

    mlab.show()
