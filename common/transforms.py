import tensorflow as tf
import numpy as np


class Affine(tf.Module):
    """
    Affine coordinate transform
    """
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
    """
    Homography coordinate transform
    """
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


