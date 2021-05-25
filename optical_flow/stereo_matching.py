import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from common.plot import InteractivePlotter, GifPlotter


def normalize01(x: np.ndarray) -> np.ndarray:
    """
    Normalize tensor `x` to range [0, 1]
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def dot(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    """
    Dot product between x and y tensors. Assumes they have the same shapes.
    :param axis: Summation axis
    """
    length = lambda v: tf.sqrt(tf.reduce_sum(v * v, axis))
    den = length(x) * length(y)
    result = tf.reduce_sum(x * y, axis) / den
    return result


def huber_norm(x: tf.Tensor, eps: float, axis: int = None) -> tf.Tensor:
    """
    Huber norm of vector `x`.
    :param axis: Summation axis
    """
    half = 0.5 * eps

    norm_l2_sqr = x**2.0
    if axis is not None:
        norm_l2_sqr = tf.reduce_sum(x, axis)
    norm_l2 = tf.sqrt(norm_l2_sqr)

    norm_l1 = tf.abs(x)
    if axis is not None:
        norm_l1 = tf.reduce_sum(norm_l1, axis)

    return tf.where(norm_l2 < eps, half * norm_l2_sqr, norm_l1 - half)


def x_coords_i_to_f(x_i: np.ndarray) -> np.ndarray:
    """
    Normalize coordinates x_i to the range [0, 1] by dividing its values by the
    tensor width-1.
    :param x_i: int32 tensor with coordinates in range [0, width-1]
    """
    return tf.cast(x_i, tf.float32) / (x_i.shape[1] - 1.0)


def x_coords_f_to_i(x_f: np.ndarray) -> np.ndarray:
    """
    De-normalize coordinates x_i to the range [0, width-1] by multiplying its
    values by the tensor width-1.
    :param x_f: float32 tensor with coordinates in range [0, 1]
    """
    return tf.cast(x_f * (x_f.shape[1] - 1.0), tf.int32)


def gather_pixel_windows(img: np.ndarray,
                         window_size: int,
                         yx_i_coords: np.ndarray) -> tf.Tensor:
    """
    Given an image `img`, generate a new tensor of shape
    [height, width, channels*(2*window_size+1)**2] where the augmented channels
    contain the channels of neighboring pixels around a square window of side
    [2*window_size+1]

    :param img: Tensor of shape [height, width, channels]
    :param yx_i_coords: Coordinate tensor of shape [2, height, width]
    :return:
    """
    height = img.shape[0]
    width = img.shape[1]

    # Gather windows from images
    kernel_coords = np.mgrid[-window_size:window_size + 1,
                             -window_size:window_size + 1]
    windowed_coords = np.expand_dims(yx_i_coords, [3, 4]) + \
                      np.expand_dims(kernel_coords, [1, 2])
    windowed_coords[0, ...] = np.clip(windowed_coords[0, ...], 0, height - 1)
    windowed_coords[1, ...] = np.clip(windowed_coords[1, ...], 0, width - 1)

    img_windowed = img[windowed_coords[0, ...],
                       windowed_coords[1, ...]]

    # Convert window content to channel values for each pixel
    img_windowed = tf.reshape(img_windowed, [height, width, -1])

    return img_windowed


def rectified_stereo_matching(img_left: np.ndarray,
                              img_right: np.ndarray,
                              img_gt: np.ndarray,
                              window_size: int = 1,
                              max_disparity: float = 0.1,
                              coupling_penalty: float = 100.0,
                              continuity_penalty: float = 320.0,
                              save_gif: str = None) -> np.ndarray:
    """
    Compute disparity values for rectified images. The returned disparity map is
    normalized by the image width and is defined with respect to the right image
    (i.e. how much each pixel of the right image has to be translated to the
    right in order to find its best correspondent).

    >>> flow_f = rectified_stereo_matching()
    >>> flow_i = x_coords_f_to_i(flow_global)
    >>> reconstruct_coords = flow_i[..., 0] + yx_i_coords[1, ...]
    >>> reconstructed = img_left[yx_i_coords[0, ...], reconstruct_coords]

    :param img_left: Left image
    :param img_right: Right image
    :param img_gt: Groundtruth image. Disparity values must be normalized by the
                   image width
    :param window_size: Size of window around each pixel used for matching
                        patches.
    :param max_disparity: Maximum allowed disparity value.
    :param coupling_penalty: Lower values will force quicker convergence
                             between flow variable and auxiliary flow
    :param continuity_penalty: Higher values lead to smoother outputs
    :param save_gif: Path to output gif file that will illustrate optimization
                     progress
    :return: Disparity map
    """

    # TODO adjust images to matching size
    height = img_left.shape[0]
    width = img_left.shape[1]

    # Shape [2, h, w]
    yx_i_coords = np.mgrid[0:height, 0:width]
    # Shape [h, w]
    x_f_coords = x_coords_i_to_f(yx_i_coords[1, ...])

    # Compute all possible flow values
    # Shape [1, w]
    x_f_row = x_f_coords[0:1, ...]
    x_f_col = tf.transpose(x_f_row)
    # Shape [1, w, w]
    all_flows = tf.expand_dims(x_f_row - x_f_col, 0)

    # Generate tensor with windows around each pixel
    # Shape [h, 1, w, c]
    img_left_windowed = tf.expand_dims(
        gather_pixel_windows(img_left, window_size, yx_i_coords),
        1
    )
    # Shape [h, w, 1, c]
    img_right_windowed = tf.expand_dims(
        gather_pixel_windows(img_right, window_size, yx_i_coords),
        2
    )

    tf.debugging.assert_shapes([
        (img_left_windowed, ('h', 1, 'w', 'c')),
        (img_right_windowed, ('h', 'w', 1, 'c')),
    ])

    # Precompute data term
    data_term = np.zeros((height, width, width), dtype=np.float32)
    for y in range(height):
        # Shape [1, w, c]
        img_left_row = img_left_windowed[y, ...]
        # Shape [w, 1, c]
        img_right_col = img_right_windowed[y, ...]

        # Pixel-wise difference. Shape [w, w]
        data_term[y, ...] = -dot(
            img_right_col - 0.5,
            img_left_row - 0.5,
            2
        )

    # Penalize negative disparities. Shape [1, w, w]
    neg_disp_penalty = 1e6 * tf.maximum(0, -all_flows)
    # Penalize large disparities. Shape [1, w, w]
    large_disp_penalty = 1e6 * tf.maximum(0, all_flows - max_disparity) ** 2
    # Penalize saturated colors. Shape [h, w, 1]
    color_penalty_right = np.min(
        np.exp((2 * (img_right - 0.5)) ** 2), axis=2, keepdims=True)
    # Final data term. Shape [h, w, w]
    data_term = (data_term * color_penalty_right
                 + neg_disp_penalty + large_disp_penalty).numpy()

    # Output flow
    flow_global = np.zeros((height, width, 1), np.float32)
    # Auxiliary flow for optimization
    u_global = np.zeros((height, width, 1), np.float32)

    def optimize_v():
        # Shape [h, w, w]
        coupling_term = []
        # Iterate along rows to avoid the need for too much GPU memory
        for y in range(height):
            coupling_term_row = \
                0.5 / coupling_penalty \
                * huber_norm(u_global[y:y+1, ...] - all_flows, 1)
            coupling_term.append(coupling_term_row.numpy())

        coupling_term = np.concatenate(coupling_term)
        loss = data_term + coupling_term
        # Shape [h, w]
        best_idx = np.argmin(loss, 2)
        # Shape [h, w, 1]
        flow = tf.expand_dims(
            x_coords_i_to_f(best_idx - yx_i_coords[1, 0, :]), 2
        )

        return flow

    def optimize_u(iterations: int = 400,
                   learning_rate: float = 0.01):
        laplacian_kernel = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.float32) / 16.0
        laplacian_kernel = tf.constant(np.expand_dims(laplacian_kernel, [2, 3]))

        # Shape [h, w]
        u_local = tf.Variable(u_global, trainable=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate)

        progress = keras.utils.Progbar(iterations - 1)
        tf.debugging.assert_all_finite(u_local, 'nan')
        for iteration in range(iterations):
            with tf.GradientTape() as tape:
                # Shape [h, w]
                coupling_term = 0.5 / coupling_penalty * huber_norm(
                    u_local - flow_global, 1)

                # Shape [h, w-1]
                laplacian = tf.nn.conv2d(
                    u_local[tf.newaxis, ...],
                    laplacian_kernel, (1, 1), "VALID")

                regularization_term = continuity_penalty * laplacian**2

                loss = tf.reduce_mean(coupling_term) + \
                       tf.reduce_mean(regularization_term)

            progress.update(iteration, [
                ('avgerr', error_global),
                ('aux loss', loss.numpy())
            ])
            grads = tape.gradient(loss, u_local)
            optimizer.apply_gradients([[grads, u_local]])

        return u_local

    if save_gif is None:
        plotter = InteractivePlotter()
    else:
        plotter = GifPlotter(save_gif)

    epoch_count = 20
    for epoch in range(epoch_count):
        print(f'Epoch {epoch+1}/{epoch_count}')
        flow_global = optimize_v()
        error_global = np.mean(
            np.abs(flow_global - img_gt)[img_gt != -1]) * flow_global.shape[1]
        u_global = optimize_u()
        coupling_penalty *= 0.5

        # Update flow plot
        plotter.update(normalize01(flow_global[3:-3, 3:-3, 0]), cmap='viridis')

    plotter.finalize()

    return flow_global
