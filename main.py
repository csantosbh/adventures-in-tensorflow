#!/usr/bin/env python
import os
from pathlib import Path

import click
import cv2
import icecream
import numpy as np
import tensorflow as tf

from common.transforms import Homography, Affine
from common.io import load_img
from registration import image_registration as img_reg


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command()
@click.argument('img_a_path', type=click.Path(exists=True))
@click.argument('img_b_path', type=click.Path(exists=True))
@click.option('--save_gif', default=None, type=click.Path())
def image_registration(img_a_path, img_b_path, save_gif):
    """
    Performs rigid registration between two images using homography transform
    Uses a coarse-to-fine approach for aligning the images. This consists of
    running the alignment algorithm multiple times; initially with the images in
    lower resolutions and blurred, with the amount of blur decreasing as the
    alignment .
    """
    from common.plot import InteractivePlotter, GifPlotter

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
        print(f'[Epoch] blur: {blur_level}, size: {size}')
        img_a = load_img(img_a_path, blur_level, size)
        img_b = load_img(img_b_path, blur_level, size)

        height = img_a.shape[0]
        width = img_a.shape[1]

        red_i = np.ones((height, width, 3)) * [[[1, 0.55, 0.55]]]
        blue_i = np.ones((height, width, 3)) * [[[0.55, 0.55, 1]]]

        img_reg.align_to(
            img_b, img_a, warper, iterations=its, callbacks=[update_plot],
            optimizer=optimizer)

    # We are done!
    plotter.finalize()

    print(f'Computed transform parameters:\n{warper.variables[0].numpy()}')


@cli.command()
@click.option('--left', 'left_path', type=click.Path(exists=True),
              required=True)
@click.option('--right', 'right_path', type=click.Path(exists=True),
              required=True)
@click.option('--groundtruth', 'gt_path', type=click.Path(exists=True),
              required=True)
@click.option('--save-gif', default=None, type=click.Path())
def rectified_stereo(left_path, right_path, gt_path, save_gif):
    """
    Compute the disparity map for a rectified stereo pair
    """
    import optical_flow.stereo_matching as ofsm

    tgt_size = 768
    img_left = load_img(left_path, blur_std=0, cvt_grayscale=False,
                        tgt_size=tgt_size)
    img_right = load_img(right_path, blur_std=0, cvt_grayscale=False,
                         tgt_size=tgt_size)
    img_gt = load_img(gt_path, blur_std=0, cvt_grayscale=False, normalize=False,
                      tgt_size=tgt_size, interpolation=cv2.INTER_NEAREST,
                      replace_inf=-1, archive_index='depth')

    ofsm.rectified_stereo_matching(
        img_left, img_right, img_gt, save_gif=save_gif)


@cli.command()
@click.argument('video_path', type=click.Path(exists=True), required=True)
@click.option('--intensity', default=20)
@click.option('--max-frames', default=None, type=int)
@click.option('--show-progress', is_flag=True)
@click.option('--save-gif', default=None, type=click.Path())
def motion_amplification(video_path,
                         intensity,
                         max_frames,
                         show_progress,
                         save_gif):
    """
    Perform amplification of unnoticeable motion of input video
    """
    import video.motion_amplification as amp

    amp.amplify_motion(
        video_path, intensity, max_frames, show_progress, save_gif)


@cli.command()
@click.argument('point_cloud_folder', type=click.Path(exists=True), required=True)
@click.option('--output', 'output_name', type=click.Path(), default='point_cloud.npz')
def import_point_cloud(point_cloud_folder, output_name):
    """
    Import Stanford point cloud into .npz oriented point clouds
    """
    import tarfile
    import mayavi.mlab as mlab
    from geometry import point_cloud
    from tqdm import tqdm

    # Only process file if output doesn't exist
    if not os.path.isfile(output_name):
        # Extract all files
        print('Extracting compressed files')
        for tar_name in tqdm(list(Path(point_cloud_folder).rglob('*.tar.gz'))):
            tar = tarfile.open(tar_name, "r:gz")
            tar.extractall(point_cloud_folder)
            tar.close()

        positions, normals = point_cloud.combine_point_clouds(point_cloud_folder)
        np.savez_compressed(output_name, positions=positions, normals=normals)
    else:
        print(f'Found file {output_name}; skipping preprocessing...')
        data = np.load(output_name)
        positions = data['positions']
        normals = data['normals']

    mlab.points3d(positions[::15, 0],  positions[::15, 1],  positions[::15, 2], scale_factor=4e-4)
    mlab.quiver3d(positions[::15, 0],  positions[::15, 1],  positions[::15, 2],
                  normals[::15, 0],    normals[::15, 1],    normals[::15, 2], scale_factor=8e-4)
    mlab.show()


@cli.command()
@click.argument("svg_path", type=click.Path(exists=True), required=True)
@click.option('--save-gif', default=None, type=click.Path())
def surface_reconstruction_2d(svg_path, save_gif):
    """
    Perform Poisson Surface Reconstruction of the provided 2D oriented point cloud
    """
    from common.io import load_svg
    from geometry.poisson_reconstruction import reconstruct_2d

    points, normals = load_svg(svg_path, 10000)
    initial_resolution = 32
    iterations = [500, 250, 100, 20, 20]
    reconstruct_2d(points, normals, initial_resolution, iterations, save_gif)


@cli.command()
@click.argument("point_cloud_path", type=click.Path(exists=True), required=True)
@click.option('--output', 'output_name', type=click.Path(), default='volume.npz')
@click.option('--save-gif', default=None, type=click.Path())
def surface_reconstruction_3d(point_cloud_path, output_name, save_gif):
    """
    Perform Poisson Surface Reconstruction of the provided 3D oriented point cloud
    """
    from common.io import load_point_cloud
    from geometry.poisson_reconstruction import reconstruct_3d

    points, normals = load_point_cloud(point_cloud_path)
    iterations = [500, 250, 100, 20, 20]
    reconstruct_3d(points, normals, output_name, 32, iterations, save_gif)


if __name__ == '__main__':
    icecream.install()
    icecream.ic.configureOutput(includeContext=True)
    cli()
