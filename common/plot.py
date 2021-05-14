import os
import tempfile
import subprocess
from pathlib import Path

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
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.axim = None

    def update(self, image, **kwargs):
        if self.axim is None:
            self.axim = self.ax.imshow(image, cmap=kwargs.get('cmap', 'gray'))
        else:
            self.axim.set_data(image)
            self.fig.canvas.flush_events()

    def finalize(self):
        plt.show(block=True)


class GifPlotter(ISequencePlotter):
    """
    Plotter that converts the sequence of provided images to a gif file
    """
    def __init__(self, output_path):
        self.dst_dir = tempfile.TemporaryDirectory()
        self.output_path = output_path
        self.counter = 0

    def update(self, image, **kwargs):
        filename = Path(self.dst_dir.name) / f'frame_{self.counter:04d}.png'
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
