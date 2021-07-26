import os
import tempfile
import subprocess
from pathlib import Path

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


def contour3d(img, contours, size=None):
    import mayavi.mlab as mlab

    if size is None:
        size = (768, 768)

    mlab.figure(bgcolor=(1, 1, 1), size=size)
    scalar = mlab.pipeline.scalar_field(img)
    voi = mlab.pipeline.extract_grid(scalar)
    mlab.pipeline.iso_surface(
        voi, contours=contours, color=(230/255.0, 99/255.0, 90/255.0))

    mlab.view(15.0, 150.0, 1.5*img.shape[0],
              focalpoint=[img.shape[0]/2, img.shape[1]/2.2, img.shape[2]/2],
              roll=0)

    mlab.show()


def plot_points_norms(points, norms):
    plt.quiver(points[:, 1], points[:, 0], norms[:, 1], norms[:, 0])
    plt.plot(points[:, 1], points[:, 0])
    plt.gca().set_aspect('equal')
    plt.show()
