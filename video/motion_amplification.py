import math
from typing import List, Optional

import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm

from common import plot, image as improc


def get_steerable_filters(shape: tf.TensorShape,
                          num_scales: int,
                          num_orients: int,
                          eps=1e-9) -> np.ndarray:
    """
    Create frequency-domain filters for composing a steerable pyramid

    Output has shape [num_filters, h, w], with
    num_filters=2+(num_scales-1)*num_orients, and its coordinates are in the
    order compatible with np.fft.fft (so there is no need to ifftshift it)

    :param shape: Format [n_frames, height, width, channels]
    :param num_scales: Number of pyramid levels
    :param num_orients: Number of desired orientations
    :param eps: Used to prevent numerical errors from log2 operation
    :return:
    """
    n_frames, height, width, channels = shape
    scale_powers = np.arange(num_scales)

    ##
    # Create frequency domain coordinates, normalized to [-1, 1] on each
    # separate axis
    coords = np.mgrid[-1:1:1j*height, -1:1:1j*width]
    # Create polar coordinates
    coords_rad = np.sqrt(coords[0]**2 + coords[1]**2)
    # Shape [1, h, w]
    coords_ang = np.expand_dims(np.arctan2(coords[0], coords[1]), 0)

    ##
    # Build multiple scale band masks. Shape [num_scales, h, w]
    h_mask = np.expand_dims(np.log2(coords_rad + eps), 0) + \
             np.expand_dims(scale_powers, [1, 2])

    h_mask = np.clip(h_mask, -1, 0)
    h_mask = np.cos(h_mask * np.pi / 2.0)
    h_mask[h_mask < eps] = 0
    l_mask = np.sqrt(1 - h_mask**2.0)

    ##
    # Modulate intermediate high-frequency masks. Shape [num_scales-1, h, w]
    rings = h_mask[1:] * l_mask[:-1]

    ##
    # Modulate rings by band orientations
    order = num_orients - 1
    o_intensity = np.sqrt(
        2**(2 * order) * math.factorial(order)**2
        / (num_orients * math.factorial(2*order))
    )
    ang_orients = np.pi * np.arange(num_orients) / num_orients
    # Shape [num_orients, 1, 1]
    ang_orients = np.expand_dims(ang_orients, [1, 2])
    ang_orients = circular_mod(coords_ang - ang_orients)
    # Shape [num_orients, h, w]
    orients = o_intensity * np.cos(ang_orients)**order
    orients = np.maximum(orients, 0)

    ##
    # Combine rings and orients into final filters
    # Shape [num_scales, num_orients, h, w]
    masks_combos = rings[:, np.newaxis, ...] * orients[np.newaxis, ...]

    # Shape [2+(num_scales-1)*num_orients, h, w]
    masks_final = np.concatenate([
        h_mask[0:1],
        np.reshape(masks_combos, [-1, height, width]),
        l_mask[-1][np.newaxis, ...]
    ])

    return masks_final


def get_cropped_filters(filters: np.ndarray) -> List[np.ndarray]:
    """
    Filters has shape [n_filters, h, w]
    Output is a list, each with shape [2, h_filt, w_filt] with the rank 0 being
    (y_coordinate, x_coordinate).
    """
    n_filters, h, w = filters.shape

    # Mask target pixels per row/col
    target_pixels = np.abs(filters) > 0
    tgt_y = np.max(target_pixels, axis=2)
    tgt_x = np.max(target_pixels, axis=1)

    # Make masks symmetric
    tgt_y = tgt_y | tgt_y[:, ::-1]
    tgt_x = tgt_x | tgt_x[:, ::-1]

    # Get indices of target pixels
    range_y = np.mgrid[0:n_filters, 0:h][1]
    range_x = np.mgrid[0:n_filters, 0:w][1]

    # Get the wrapping rectangles around each filter
    min_range_y = np.min(range_y + (~tgt_y) * h, 1)
    max_range_y = np.max(range_y * tgt_y, 1)
    min_range_x = np.min(range_x + (~tgt_x) * w, 1)
    max_range_x = np.max(range_x * tgt_x, 1)

    # Small adjustments needed due to the placement of the zero frequency term
    y_is_odd = (min_range_y % 2) == 1
    x_is_odd = (min_range_x % 2) == 1
    heights = max_range_y - min_range_y
    widths = max_range_x - min_range_x
    min_range_y -= (heights % 2) * y_is_odd
    max_range_y += (heights % 2) * y_is_odd
    min_range_x -= (widths % 2) * x_is_odd
    max_range_x += (widths % 2) * x_is_odd

    # Clip maximum range to valid values
    max_range_y = np.minimum(max_range_y, h-1)
    max_range_x = np.minimum(max_range_x, w-1)

    # Compute final mask indices
    indices = [
        np.mgrid[min_range_y[k]:max_range_y[k]+1,
                 min_range_x[k]:max_range_x[k]+1]
        for k in range(n_filters)
    ]

    return indices


def circular_mod(value: np.ndarray):
    """
    Circular mod of value to range [-pi, pi]
    """
    return (value + np.pi) % (2*np.pi) - np.pi


def get_level_filter(video_f: np.ndarray,
                     level_filter: np.ndarray,
                     filter_masks: np.ndarray) -> np.ndarray:
    """
    Get steerable pyramid for filter level.

    Output in the time domain containing phase information.
    Shape [n_frames, h, w, c]
    """
    axes = [1, 2]
    masked = video_f[:, filter_masks[0], filter_masks[1]] * level_filter

    result = np.fft.ifft2(
        np.fft.ifftshift(masked, axes),
        axes=axes
    ).astype(np.complex64)

    return result


def rebuild_level(level_filtered: np.ndarray,
                  level_filter: np.ndarray) -> np.ndarray:
    """
    Given the time domain filtered image level_filtered, bring it back to the
    frequency domain modulated by the filter of level level_idx
    """
    axes = [1, 2]
    return 2 * level_filter * np.fft.fftshift(
        np.fft.fft2(level_filtered, axes=axes), axes)


def amplify_motion(video_path: str,
                   amplification: float,
                   max_frames: int,
                   show_progress: bool,
                   save_gif: Optional[str]) -> np.ndarray:
    """
    Perform motion amplification of input video

    :param video_path: Path to desired video
    :param amplification: Amplification factor
    :param max_frames: Number of frames to process
    :param show_progress: If set, a window will show the second frame being
                          reconstructed
    :param save_gif: Path to the destination gif file, or None to show the
                     results in a window
    :return: Processed video as a tensor of shape [n_frames, h, w, c]
    """
    video_src = cv2.VideoCapture(video_path)

    # Load video
    video_t = []
    has_next_frame, frame = video_src.read()
    curr_frame = 0
    while has_next_frame:
        # Convert frame from BGR uint8 to RGB normalized float
        frame = improc.im2float(frame[:, :, 2::-1])
        video_t.append(frame)

        # Grab next frame
        has_next_frame, frame = video_src.read()
        curr_frame += 1

        # Halt if maximum frames is specified
        if max_frames is not None and curr_frame == max_frames:
            break

    video_t = np.concatenate(np.expand_dims(video_t, 0))

    plotter = plot.InteractivePlotter()

    # Frequency domain filters. Shape [n_filters, h, w]
    filters = get_steerable_filters(video_t.shape, 7, 4).real.astype(np.float32)
    filter_masks = get_cropped_filters(filters)
    axes = [1, 2]
    video_f = np.fft.fftshift(
        np.fft.fft2(video_t, axes=axes).astype(np.complex64), axes
    )

    amplified_f = np.zeros_like(video_f)

    for filter_idx in tqdm(range(0, filters.shape[0]-1),
                           desc='Processing filter level'):
        curr_fmask = filter_masks[filter_idx]
        level_filter = filters[
            filter_idx, curr_fmask[0], curr_fmask[1], np.newaxis]
        level_filtered = get_level_filter(video_f, level_filter, curr_fmask)
        # Compute spatial phase for current level. Shape [n_frames, h, w, c]
        level_phases = np.angle(level_filtered)

        # Compute phase differences. Shape [n_frames-1, h, w, c]
        phase_delta = circular_mod(level_phases[1:, ...] - level_phases[0, ...])

        # TODO Bandpass Filter delta phases

        # Amplify delta phases.
        phase_delta = phase_delta * amplification
        # Apply amplified delta to frames. Shape [n_frames, h, w, c]
        level_filtered[1:] = np.exp(1j * phase_delta) * level_filtered[1:]
        # Rebuild level. Shape [n_frames, h, w, c]
        amplified_f[:, curr_fmask[0], curr_fmask[1], :] += rebuild_level(
            level_filtered, level_filter)

        if show_progress:
            axes = [0, 1]
            plotter.update(
                np.fft.ifft2(np.fft.ifftshift(amplified_f[1], axes),
                             axes=axes).real,
                norm=True
            )

    # Apply lowpass residual
    curr_fmask = filter_masks[-1]
    amplified_f[:, curr_fmask[0], curr_fmask[1], :] += \
        video_f[:, curr_fmask[0], curr_fmask[1], :] * \
        filters[-1, curr_fmask[0], curr_fmask[1], np.newaxis] ** 2

    # Display results
    if save_gif is not None:
        plotter = plot.GifPlotter(save_gif)

    axes = [1, 2]
    amplified_t = np.fft.ifft2(
        np.fft.ifftshift(amplified_f, axes), axes=axes).real

    for frame_idx, frame_amp in enumerate(amplified_t):
        frame_orig = improc.resize_to_max(video_t[frame_idx], 384)
        frame_amp = improc.resize_to_max(frame_amp, 384)

        side_by_side = np.concatenate([frame_orig, frame_amp], axis=1)

        plotter.update(side_by_side, cmap='viridis')

    # Perform motion amplification
    plotter.finalize()

    return amplified_t
