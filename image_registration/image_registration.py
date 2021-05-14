from typing import Callable, Union, List

import numpy as np
import tensorflow as tf

from common.samplers import BilinearSampler
from common.transforms import Homography


def mse_loss(target: tf.Tensor,
             transformed: tf.Tensor,
             transform_mask: tf.Tensor):
    """
    Here we implement the simple Mean Squared Error (MSE) metric. The mean is
    computed for the valid pixels, determined by the `transform_mask`.

    :param target: Reference image to which we want to align the `transformed`
                   image
    :param transformed: Image being modified with the goal of alignment to the
                        `target` image
    :param transform_mask: Mask indicating valid pixels of the `transformed`
                           image.
    :return: A scalar indicating how well the images are aligned. Lower is
             better.
    """
    masked_diff = transform_mask * (transformed - target)**2
    return tf.reduce_sum(masked_diff) / tf.reduce_sum(transform_mask)


def adjust_shape_to(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """
    Make the `source` image match the `destination` in size. Smaller dimensions
    are padded; larger dimensions are clipped.
    """
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
             optimizer: tf.optimizers.Optimizer = None,
             loss_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = None,
             callbacks: List[Callable[[tf.Tensor, tf.Tensor, tf.Tensor], None]] = None,
             ) -> np.ndarray:
    """
    Aligns the image `source` to `destination` using the `warper` transform
    (e.g. `common.transforms.Homography` or `common.transforms.Affine`).

    This will update the parameters of `warper` and return the warped `source`
    image.

    :param source: Image to be warped
    :param destination: Reference image for alignment
    :param warper: Transform object (see `common.transoforms`).
    :param iterations: Number of iterations
    :param optimizer: Optimization algorithm
    :param loss_fn: Loss function. For an example, see `mse_loss`.
    :param callbacks: Functions to be called after each iteration. The
                      parameters are the same as for `loss_fn`.
    :return: Transformed image. Important: The `warper` will have its parameters
             updated in-place!
    """
    if optimizer is None:
        optimizer = tf.optimizers.Adam(learning_rate=1e-3)

    if loss_fn is None:
        loss_fn = mse_loss

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


def debug():
    """
    This function was created for comparing TensorFlow autodiff to numeric
    differentiation of image sampling.
    """
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
            grad[..., i, j] = (warp_img(xx, yy, w + e) - warp_img(xx, yy, w)) / eps

    ic(np.squeeze(grad[..., inspect_idx[0], inspect_idx[1]]))
