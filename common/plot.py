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
    def __init__(self, name='InteractivePlotter'):
        plt.ion()
        self.name = name

    def update(self, image: np.ndarray, **kwargs):
        if kwargs.get('norm', False):
            upper = np.max(image)
            lower = np.min(image)
            image = (image - lower) / (upper - lower)

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
    def __init__(self, output_path):
        self.dst_dir = tempfile.TemporaryDirectory()
        self.output_path = output_path
        self.counter = 0

    def update(self, image, **kwargs):
        if image.ndim != 3 or image.shape[2] == 1:
            # Apply color map (only for single-channel images)
            cmap = cm.get_cmap(kwargs.get('cmap', 'gray'))
            image = cmap(np.squeeze(image))

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
