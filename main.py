#!/usr/bin/env python
import click
import icecream
import numpy as np
import tensorflow as tf

from common.transforms import Homography, Affine
from common.plot import InteractivePlotter, GifPlotter
from common.io import load_img
from image_registration import image_registration

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command()
@click.argument('img_a_path', type=click.Path())
@click.argument('img_b_path', type=click.Path())
@click.option('--save_gif', default=None, type=click.Path())
def align(img_a_path, img_b_path, save_gif):
    """
    Entry point for the image registration demo.
    Uses a coarse-to-fine approach for aligning the images. This consists of
    running the alignment algorithm multiple times; initially with the images in
    lower resolutions and blurred, with the amount of blur decreasing as the
    alignment .
    """
    warper = Homography()

    blur_levels = [3, 2, 1, 0.5]
    iterations = [600, 300, 100, 10]
    optimizers = [tf.optimizers.Adam(1e-3), tf.optimizers.Adam(1e-3),
                  tf.optimizers.Adam(1e-3), tf.optimizers.Adam(1e-5)]
    image_sizes = [256, 256, 256, 512]

    if save_gif is None:
        plotter = InteractivePlotter()
    else:
        plotter = GifPlotter(save_gif)

    red_i: np.ndarray
    blue_i: np.ndarray

    # Will be called after every alignment iteration
    def update_plot(target, transformed, transform_mask):
        plotter.update(
            transform_mask * (transformed * 0.7 * red_i +
                              target * 0.3 * blue_i) +
            (1-transform_mask) * target * blue_i
        )

    # Iterate in coarse-to-fine representations of the images
    for blur_level, optimizer, its, size in zip(
            blur_levels, optimizers, iterations, image_sizes):
        img_a = load_img(img_a_path, blur_level, size)
        img_b = load_img(img_b_path, blur_level, size)

        height = img_a.shape[0]
        width = img_a.shape[1]

        red_i = np.ones((height, width, 3)) * [[[1, 0.55, 0.55]]]
        blue_i = np.ones((height, width, 3)) * [[[0.55, 0.55, 1]]]

        image_registration.align_to(
            img_b, img_a, warper, iterations=its, callbacks=[update_plot],
            optimizer=optimizer)

    # We are done!
    plotter.finalize()

    print(f'Computed transform parameters:\n{warper.variables[0]}')


@cli.command()
def debug():
    image_registration.debug()


if __name__ == '__main__':
    icecream.install()
    icecream.ic.configureOutput(includeContext=True)
    cli()
