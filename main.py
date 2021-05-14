#!/usr/bin/env python
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Union, List

import click
import icecream
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Affine(tf.Module):
    def __init__(self, name=None):
        super(Affine, self).__init__(name=name)
        self.w = tf.Variable(np.eye(2, 3, dtype=np.float32))

    def __call__(self, xx, yy):
        xx_yy = tf.concat([xx, yy], axis=-1)

        # Apply affine part without translation
        xx_yy = tf.einsum('ij,whj->whi', self.w[:, 0:2], xx_yy)

        # Translate
        xx = tf.expand_dims((xx_yy[..., 0] + self.w[0, 2]), 2)
        yy = tf.expand_dims((xx_yy[..., 1] + self.w[1, 2]), 2)

        return xx, yy


class Homography(tf.Module):
    def __init__(self, name=None):
        super(Homography, self).__init__(name=name)
        self.w = tf.Variable(np.eye(3, dtype=np.float32))

    def __call__(self, xx, yy):
        xx_yy = tf.concat([xx, yy], axis=-1)

        # Apply affine part without translation
        xx_yy = tf.einsum('ij,whj->whi', self.w[:, 0:2], xx_yy)

        # Translate and normalize by homogenous coordinate
        xx = (xx_yy[..., 0:1] + self.w[0, 2]) / (1 + xx_yy[..., 2:3])
        yy = (xx_yy[..., 1:2] + self.w[1, 2]) / (1 + xx_yy[..., 2:3])

        return xx, yy


class BilinearSampler(tf.Module):
    def __init__(self, img: np.ndarray, name=None):
        super(BilinearSampler, self).__init__(name=name)

        tf.assert_rank(img, 3)
        self.img = tf.constant(img)
        channel_pad = [] if img.ndim == 2 else [[0, 0]]

        self.width = tf.cast(tf.shape(img)[1], tf.float32)
        self.height = tf.cast(tf.shape(img)[0], tf.float32)

        delta_x = 1 / (self.width - 1)
        delta_y = 1 / (self.height - 1)

        self.dx = np.pad(
            (self.img[:, 1:] - self.img[:, :-1]),
            [[0, 0], [0, 1]] + channel_pad,
            mode='constant'
        ) / delta_x
        self.dy = np.pad(
            (self.img[1:, :] - self.img[:-1, :]),
            [[0, 1], [0, 0]] + channel_pad,
            mode='constant'
        ) / delta_y

        self.ones = tf.ones_like(img)
        self.zeroes = tf.zeros_like(img)

    def get_xy(self):
        # Generate image coordinates
        xx, yy = tf.meshgrid(tf.range(self.width, dtype=tf.float32),
                             tf.range(self.height, dtype=tf.float32))
        xx = tf.expand_dims(xx, 2) / (self.width - 1)
        yy = tf.expand_dims(yy, 2) / (self.height - 1)

        return xx, yy

    def get_mask(self, xx, yy):
        mask = tf.where((xx >= 0) & (xx <= 1) &
                        (yy >= 0) & (yy <= 1), self.ones, self.zeroes)
        return mask

    @staticmethod
    def remap(img: tf.Tensor,
              xx: tf.Tensor,
              yy: tf.Tensor,
              clamp_borders=True) -> tf.Tensor:
        height = tf.cast(tf.shape(img)[0], tf.float32)
        width = tf.cast(tf.shape(img)[1], tf.float32)

        # Clamp xx and yy
        if clamp_borders:
            xx = tf.clip_by_value(xx, 0, 1)
            yy = tf.clip_by_value(yy, 0, 1)

        xx_zero_to_width = xx * (width - 1)
        yy_zero_to_height = yy * (height - 1)

        # Compute base coordinates for bilinear interpolation
        xx0 = tf.math.floor(xx_zero_to_width)
        xx1 = tf.math.ceil(xx_zero_to_width)
        xx_alpha = xx_zero_to_width - xx0

        yy0 = tf.math.floor(yy_zero_to_height)
        yy1 = tf.math.ceil(yy_zero_to_height)
        yy_alpha = yy_zero_to_height - yy0

        # Cast base indices to integral types
        xx0 = tf.cast(xx0, tf.int32)
        xx1 = tf.cast(xx1, tf.int32)
        yy0 = tf.cast(yy0, tf.int32)
        yy1 = tf.cast(yy1, tf.int32)

        # Perform bilinear interpolation
        img_x0y0 = tf.gather_nd(img, tf.concat([yy0, xx0], axis=2))
        img_x1y0 = tf.gather_nd(img, tf.concat([yy0, xx1], axis=2))

        img_x0y1 = tf.gather_nd(img, tf.concat([yy1, xx0], axis=2))
        img_x1y1 = tf.gather_nd(img, tf.concat([yy1, xx1], axis=2))

        # Bilinear dx
        img_y0 = img_x0y0 * (1 - xx_alpha) + img_x1y0 * xx_alpha
        img_y1 = img_x0y1 * (1 - xx_alpha) + img_x1y1 * xx_alpha

        # Bilinear dy
        result = img_y0 * (1 - yy_alpha) + img_y1 * yy_alpha

        return result

    @tf.custom_gradient
    def __call__(self, xx: tf.Tensor, yy: tf.Tensor):

        def grad(upstream):
            """
            dIdW = Ix*dx/dW + Iy*dy/dW
            """

            result = (
                self.remap(self.dx, xx, yy) * upstream,
                self.remap(self.dy, xx, yy) * upstream,
            )
            return result

        return self.remap(self.img, xx, yy), grad


class IPlotter(object):
    def update(self, image, **kwargs):
        raise NotImplementedError()

    def finalize(self):
        pass


class InteractivePlotter(IPlotter):
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


class GifPlotter(IPlotter):
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
            '-f', 'image2',
            '-i', f'{self.dst_dir.name}/frame_%04d.png',
            '-pix_fmt', 'rgb8',
            self.output_path
        ])
        converter.wait()


def l2_loss(target: tf.Tensor,
            transformed: tf.Tensor,
            transform_mask: tf.Tensor):
    masked_diff = transform_mask * (transformed - target)**2
    return tf.reduce_sum(masked_diff) / tf.reduce_sum(transform_mask)


def adjust_shape_to(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    # Enlarge smaller dimensions
    diff_height = max(0, destination.shape[0] - source.shape[0])
    diff_width = max(0, destination.shape[1] - source.shape[1])

    y_before = int(diff_height // 2)
    y_after = int(diff_height - y_before)

    x_before = int(diff_width // 2)
    x_after = int(diff_width - x_before)

    channel_pad = [] if source.ndim == 2 else [[0, 0]]

    source = np.pad(
        source, [[y_before, y_after], [x_before, x_after]] + channel_pad,
        mode='reflect'
    )

    # Reduce bigger dimensions
    diff_height = max(0, source.shape[0] - destination.shape[0])
    diff_width = max(0, source.shape[1] - destination.shape[1])

    y_before = int(diff_height // 2)
    y_after = int(diff_height - y_before) if y_before > 0 else -source.shape[0]

    x_before = int(diff_width // 2)
    x_after = int(diff_width - x_before) if x_before > 0 else -source.shape[1]

    source = source[y_before:-y_after, x_before:-x_after, ...]

    return source


def align_to(source: np.ndarray,
             destination: np.ndarray,
             warper: Union[tf.Module, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
             iterations: int = 800,
             learning_rate: float = 0.001,
             optimizer: tf.optimizers.Optimizer = None,
             loss_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = None,
             callbacks: List[Callable[[tf.Tensor, tf.Tensor, tf.Tensor], None]] = None,
             ) -> np.ndarray:
    if optimizer is None:
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    if loss_fn is None:
        loss_fn = l2_loss

    if callbacks is None:
        callbacks = []

    if source.shape != destination.shape:
        source = adjust_shape_to(source, destination)

    distorter = BilinearSampler(destination)
    xx_o, yy_o = distorter.get_xy()

    source = tf.constant(source)

    for it in range(iterations):
        with tf.GradientTape() as tape:
            xx, yy = warper(xx_o, yy_o)
            dst = distorter(xx, yy)
            mask = distorter.get_mask(xx, yy)
            loss = loss_fn(source, dst, mask)

        w = warper.variables
        grads = tape.jacobian(loss, w)
        optimizer.apply_gradients(zip(grads, w))
        print(f'[{it}] Loss: {loss.numpy()}')

        # Run post-iteration callbacks
        [cbk(source, dst, mask) for cbk in callbacks]

    return dst


def load_img(img_path: str,
             blur_std: int = 1,
             tgt_size: float = 256) -> np.ndarray:
    # Load image
    img = cv2.imread(img_path).astype(np.float32)
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize to 0...1 range
    lowest = np.min(img)
    highest = np.max(img)
    img = (img - lowest) / (highest - lowest)
    # Resize to maximum dimension of 256
    highest_dim = np.max(img.shape)
    new_size = (np.array(img.shape[::-1]) * tgt_size / highest_dim).astype(
        np.int32)
    img = cv2.resize(img, tuple(new_size), interpolation=cv2.INTER_AREA)
    # Blur image
    if blur_std > 0:
        img = cv2.GaussianBlur(img, (33, 33), blur_std)
    # Add channel rank
    img = np.expand_dims(img, 2)

    return img


@click.group()
def cli():
    pass


@cli.command()
@click.argument('img_a_path', type=click.Path())
@click.argument('img_b_path', type=click.Path())
@click.option('--save_gif', default=None, type=click.Path())
def align(img_a_path, img_b_path, save_gif):
    warper = Homography()

    blur_levels = [3, 2, 1, 0.5]
    iterations = [600, 300, 100, 10]
    learning_rates = [1e-3, 1e-3, 1e-3, 1e-5]

    if save_gif is None:
        plotter = InteractivePlotter()
    else:
        plotter = GifPlotter(save_gif)

    red_i: np.ndarray
    blue_i: np.ndarray

    def update_plot(target, transformed, transform_mask):
        plotter.update(
            transform_mask * (transformed * 0.7 * red_i +
                              target * 0.3 * blue_i) +
            (1-transform_mask) * target * blue_i
        )

    for blur_level, lr, its in zip(blur_levels, learning_rates, iterations):
        img_a = load_img(img_a_path, blur_level)
        img_b = load_img(img_b_path, blur_level)

        height = img_a.shape[0]
        width = img_a.shape[1]

        red_i = np.ones((height, width, 3)) * [[[1, 0.55, 0.55]]]
        blue_i = np.ones((height, width, 3)) * [[[0.55, 0.55, 1]]]

        align_to(img_b, img_a, warper, iterations=its, callbacks=[update_plot],
                 learning_rate=lr)

    plotter.finalize()


@cli.command()
def debug():
    img = np.expand_dims(np.mgrid[0:4, 0:4][0], 2).astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    warper = Homography()
    variables = warper.variables[0]
    distorter = BilinearSampler(img)
    xx, yy = distorter.get_xy()

    def warp_img(xx, yy, w=None):
        if w is not None:
            variables.assign(w)
        xx, yy = warper(xx, yy)
        y = distorter(xx, yy)
        return y

    # Compute derivatives with tensorflow
    with tf.GradientTape() as tape:
        y = warp_img(xx, yy)

    jacobian = tape.jacobian(y, variables)
    inspect_idx = [2, 1]
    ic(np.squeeze(jacobian[..., inspect_idx[0], inspect_idx[1]]))

    # Compute derivatives numerically
    grad = np.zeros_like(jacobian)
    w = variables.numpy()
    eps = 1e-3
    for i in range(grad.shape[-2]):
        for j in range(grad.shape[-1]):
            e = np.zeros(shape=(grad.shape[-2], grad.shape[-1]))
            e[i, j] = eps
            grad[..., i, j] = (warp_img(xx, yy, w+e) - warp_img(xx, yy, w)) / eps

    ic(np.squeeze(grad[..., inspect_idx[0], inspect_idx[1]]))


if __name__ == '__main__':
    icecream.install()
    icecream.ic.configureOutput(includeContext=True)
    cli()
