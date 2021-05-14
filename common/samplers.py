from typing import Tuple

import tensorflow as tf
import numpy as np


class BilinearSampler(tf.Module):
    """
    Differentiable bilinear image sampler
    """
    def __init__(self, img: np.ndarray, name=None):
        super(BilinearSampler, self).__init__(name=name)

        tf.assert_rank(img, 3)
        self.img = tf.constant(img)

        self.width = tf.cast(tf.shape(img)[1], tf.float32)
        self.height = tf.cast(tf.shape(img)[0], tf.float32)

        # Adjust the gradients, since we work with normalized coordinates
        delta_x = 1 / (self.width - 1)
        delta_y = 1 / (self.height - 1)

        self.dx = np.pad(
            (self.img[:, 1:] - self.img[:, :-1]),
            [[0, 0], [0, 1], [0, 0]],
            mode='constant'
        ) / delta_x
        self.dy = np.pad(
            (self.img[1:, :] - self.img[:-1, :]),
            [[0, 1], [0, 0], [0, 0]],
            mode='constant'
        ) / delta_y

        self.ones = tf.ones_like(img)
        self.zeroes = tf.zeros_like(img)

    def get_xy(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return the x and y coordinates that can be used for sampling the image.
        These coordinates can also be transformed (see `common.transforms`).

        Note that we use normalized coordinates (i.e. they are remapped to the
        range [0, 1]). This presents some benefits, such as invariance to the
        scale of the image being transformed.

        :return: Normalized (xx, yy) coordinates
        """
        # Generate image coordinates
        xx, yy = tf.meshgrid(tf.range(self.width, dtype=tf.float32),
                             tf.range(self.height, dtype=tf.float32))
        xx = tf.expand_dims(xx, 2) / (self.width - 1)
        yy = tf.expand_dims(yy, 2) / (self.height - 1)

        return xx, yy

    def get_mask(self, xx, yy):
        """
        Get mask of valid image coordinates
        :param xx: Horizontal coordinates (see `get_xy`)
        :param yy: Vertical coordinates (see `get_xy`)
        :return: Mask with same shape as image object, where 1 indicate a valid
                 pixel and 0, an invalid.
        """
        mask = tf.where((xx >= 0) & (xx <= 1) &
                        (yy >= 0) & (yy <= 1), self.ones, self.zeroes)
        return mask

    @staticmethod
    def remap(img: tf.Tensor,
              xx: tf.Tensor,
              yy: tf.Tensor) -> tf.Tensor:
        """
        Sample image at coordinates `xx` and `yy` using bilinear interpolation.

        :param img: Image to be transformed. Must have the same shape as `xx`
                    and `yy`.
        :param xx: Horizontal coordinates (see `get_xy`)
        :param yy: Vertical coordinates (see `get_xy`)
        :return: Sampled image.
        """
        height = tf.cast(tf.shape(img)[0], tf.float32)
        width = tf.cast(tf.shape(img)[1], tf.float32)

        # Clamp xx and yy
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
        """
        Core sampling and custom gradient for the image sampling operation
        """
        def grad(upstream):
            """
            Gradient of the image sampling operation I(x, y).

            When sampling via warped coordinates, I(w(x, p), w(y, p)),
            the Jacobian of I with respect to p is given by

            dI/dp = Ix*dx/dp + Iy*dy/dp,
            where Ix and Iy are the partial derivatives of I w.r.t. x and y,
            respectively.
            """

            result = (
                self.remap(self.dx, xx, yy) * upstream,
                self.remap(self.dy, xx, yy) * upstream,
            )
            return result

        return self.remap(self.img, xx, yy), grad
