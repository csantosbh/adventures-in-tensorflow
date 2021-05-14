#!/usr/bin/env python
from typing import Callable

import click
import icecream
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def configure_tensorflow():
    pass


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


class ImageDistorter(tf.Module):
    def __init__(self, img: np.ndarray):
        tf.assert_rank(img, 3)
        self.img = tf.constant(img)
        channel_pad = [] if img.ndim == 2 else [[0, 0]]

        self.width = tf.cast(tf.shape(img)[1], tf.float32)
        self.height = tf.cast(tf.shape(img)[0], tf.float32)

        delta_x = 1 / self.width
        delta_y = 1 / self.height

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

        self.dxdy = tf.concat([self.dx, self.dy], 2)

    def get_xy(self):
        # Generate image coordinates
        xx, yy = tf.meshgrid(tf.range(self.width, dtype=tf.float32),
                             tf.range(self.height, dtype=tf.float32))
        xx = tf.expand_dims(xx, 2) / self.width
        yy = tf.expand_dims(yy, 2) / self.height

        return xx, yy

    @staticmethod
    def remap(img: tf.Tensor,
              xx: tf.Tensor,
              yy: tf.Tensor,
              clamp_borders=True) -> tf.Tensor:
        height = tf.cast(tf.shape(img)[0], tf.float32)
        width = tf.cast(tf.shape(img)[1], tf.float32)

        # Clamp xx and yy
        if clamp_borders:
            xx_max = tf.cast(tf.shape(img)[1], tf.float32) - 1
            yy_max = tf.cast(tf.shape(img)[0], tf.float32) - 1

            xx = tf.clip_by_value(xx, 0, xx_max / width)
            yy = tf.clip_by_value(yy, 0, yy_max / height)

        xx_zero_to_width = xx * width
        yy_zero_to_height = yy * height

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
            I(H(x,y)) = I((x*w00+y*w01+w02)/(x*w20+y*w21+w22),
                          (x*w10+y*w11+w12)/(x*w20+y*w21+w22))
            dIdW = Ix*dx/dW + Iy*dy/dW
            let D = 1/(x*w20+y*w21+w22)

            dx/dw:
            Lx = (x*w00+y*w01+w02)/(x*w20+y*w21+w22)**2
            | D*x    D*y    D|
            |   0      0    0|
            |Lx*x   Lx*y   Lx|

            dy/dw:
            Ly = (x*w10+y*w11+w12)/(x*w20+y*w21+w22)**2
            |   0      0    0|
            | D*x    D*y    D|
            |Ly*x   Ly*y   Ly|
            """

            result = (
                self.remap(self.dx, xx, yy) * upstream,
                self.remap(self.dy, xx, yy) * upstream,
            )
            return result

        return self.remap(self.img, xx, yy), grad


class InteractivePlotter(object):
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.axim = None

    def update(self, image):
        if self.axim is None:
            self.axim = self.ax.imshow(image)
        else:
            self.axim.set_data(image)
            self.fig.canvas.flush_events()

    def hold(self):
        plt.show(block=True)


def align_to(source: np.ndarray, destination: np.ndarray,
             iterations: int = 200,
             optimizer: tf.optimizers.Optimizer = None,
             loss_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = None
             ) -> np.ndarray:
    if optimizer is None:
        optimizer = tf.optimizers.SGD(learning_rate=0.001)

    if loss_fn is None:
        loss_fn = lambda target, transformed, transform_mask: \
            tf.reduce_mean(transform_mask * (transformed - target)**2)

    if source.shape != destination.shape:
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

    #warper = Affine()
    warper = Homography()

    distorter = ImageDistorter(destination)
    xx_o, yy_o = distorter.get_xy()

    source = tf.constant(source)
    ones = tf.ones_like(source)
    zeroes = tf.zeros_like(source)

    plotter = InteractivePlotter()

    for it in range(iterations):
        with tf.GradientTape() as tape:
            xx, yy = warper(xx_o, yy_o)
            dst = distorter(xx, yy)
            mask = tf.where((xx >= 0) & (xx < 1) &
                            (yy >= 0) & (yy < 1), ones, zeroes)
            loss = loss_fn(source, dst, mask)

        w = warper.variables
        grads = tape.jacobian(loss, w)
        optimizer.apply_gradients(zip(grads, w))
        print(f'[{it}] Loss: {loss.numpy()}')
        plotter.update(dst * 0.5 + source * 0.5)

    plotter.hold()

    return dst


def load_img(img_path: str) -> np.ndarray:
    # Load image
    img = cv2.imread(img_path).astype(np.float32)
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize to 0...1 range
    lowest = np.min(img)
    highest = np.max(img)
    img = (img - lowest) / (highest - lowest)
    # Blur image
    img = cv2.GaussianBlur(img, (5, 5), 1)
    # Add channel rank
    img = np.expand_dims(img, 2)

    return img


@click.command()
@click.argument('img_a_path', type=click.Path())
@click.argument('img_b_path', type=click.Path())
def main(img_a_path, img_b_path):
    img_a = load_img(img_a_path)
    img_b = load_img(img_b_path)

    img_b = align_to(img_b, img_a)
    """
    t = lambda *x: np.array([
        [1, 0, x[0]],
        [0, 1, x[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    r = lambda x: np.array([
        [np.cos(x*np.pi/180.0), -np.sin(x*np.pi/180.0), 0],
        [np.sin(x*np.pi/180.0), np.cos(x*np.pi/180.0), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    distorter = ImageDistorter(img_a[..., 0:1])
    xx, yy = distorter.get_xy()
    img = distorter(*homography(xx, yy, t(0.5, 0.5) @ r(45) @ t(-0.5, -0.5)))
    plt.imshow(img)
    plt.show()
    #"""

    """
    img = np.expand_dims(np.mgrid[0:4, 0:4][0], 2).astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    w = tf.Variable(np.eye(3, 3), dtype=np.float32, trainable=True)
    distorter = ImageDistorter(img)
    xx, yy = distorter.get_xy()

    def dist_func(xx, yy, w):
        xx, yy = homography(xx, yy, w)
        y = distorter(xx, yy)
        y = tf.reduce_mean(y, keepdims=True)
        return y

    with tf.GradientTape() as tape:
        tape.watch(yy)
        y = dist_func(xx, yy, w)

    jacobian = tape.jacobian(y, w)
    inspect_idx = [1, 0]
    ic(np.squeeze(jacobian[..., inspect_idx[0], inspect_idx[1]]))
    grad2 = np.zeros_like(jacobian)
    eps = 1e-3
    for i in range(grad2.shape[-2]):
        for j in range(grad2.shape[-1]):
            e = np.zeros(shape=(grad2.shape[-2], grad2.shape[-1]))
            e[i, j] = eps
            grad2[..., i, j] = (dist_func(xx, yy, w+e) - dist_func(xx, yy, w)) / eps

    ic(np.squeeze(grad2[..., inspect_idx[0], inspect_idx[1]]))

    #"""


if __name__ == '__main__':
    icecream.install()
    icecream.ic.configureOutput(includeContext=True)
    main()
