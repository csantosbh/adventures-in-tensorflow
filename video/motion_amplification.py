import math

import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm

from common import plot, image as improc, profiler as prof


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

    # Reorder filters so we dont have to call [i]fftshift on video
    masks_final = np.fft.ifftshift(masks_final, axes=[1, 2])

    return masks_final


def circular_mod(value: np.ndarray):
    """
    Circular mod of value to range [-pi, pi]
    """
    return (value + np.pi) % (2*np.pi) - np.pi


def get_level_filter(video_f: np.ndarray,
                     filters: np.ndarray,
                     level_idx: int) -> np.ndarray:
    """
    Get steerable pyramid for filter level.

    Output in the time domain containing phase information.
    Shape [n_frames, h, w, c]
    """
    # TODO optimize by using central crops of filters
    # TODO optimize by computing on the GPU (will require multiple passes due to large memory requirements)
    result = np.fft.ifft2(
        video_f * filters[level_idx, ..., np.newaxis],
        axes=[1, 2]
    ).astype(np.complex64)

    return result


def rebuild_level(level_filtered: np.ndarray,
                  filters: np.ndarray,
                  level_idx: int) -> np.ndarray:
    """
    Given the time domain filtered image level_filtered, bring it back to the
    frequency domain modulated by the filter of level level_idx
    """
    return 2 * filters[level_idx, ..., np.newaxis] * \
           np.fft.fft2(level_filtered, axes=[1, 2])


def amplify_motion(video_path: str,
                   amplification: float,
                   max_frames: int,
                   show_progress: bool,
                   save_gif: str):
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
    prof.tic()
    filters = get_steerable_filters(video_t.shape, 7, 4).real.astype(np.float32)
    ic(prof.toc())
    video_f = np.fft.fft2(video_t, axes=[1, 2]).astype(np.complex64)
    ic(prof.toc())

    amplified_f = np.zeros_like(video_f)

    for filter_idx in tqdm(range(0, filters.shape[0]-1),
                           desc='Processing filter level'):
        prof.tic()
        level_filtered = get_level_filter(video_f, filters, filter_idx)
        ic(prof.toc())
        # Compute spatial phase for current level. Shape [n_frames, h, w, c]
        level_phases = np.angle(level_filtered)
        ic(prof.toc())

        # Compute phase differences. Shape [n_frames-1, h, w, c]
        phase_delta = circular_mod(level_phases[1:, ...] - level_phases[0, ...])
        ic(prof.toc())

        # TODO Bandpass Filter delta phases

        # Amplify delta phases.
        phase_delta = phase_delta * amplification
        ic(prof.toc())
        # Apply amplified delta to frames. Shape [n_frames, h, w, c]
        level_filtered[1:] = np.exp(1j * phase_delta) * level_filtered[1:]
        ic(prof.toc())
        # Rebuild level. Shape [n_frames, h, w, c]
        amplified_f += rebuild_level(level_filtered, filters, filter_idx)
        ic(prof.toc())

        if show_progress:
            plotter.update(
                np.fft.ifft2(amplified_f[1], axes=[0, 1]).real,
                cmap='viridis', norm=True
            )

    # Apply lowpass residual
    prof.tic()
    amplified_f += video_f * filters[-1, ..., np.newaxis] ** 2
    ic(prof.toc())

    exit()
    # Display results
    if save_gif is not None:
        plotter = plot.GifPlotter(save_gif)

    amplified_t = np.fft.ifft2(amplified_f, axes=[1, 2]).real
    for frame_idx, frame_amp in enumerate(amplified_t):
        frame_orig = improc.resize_to_max(video_t[frame_idx], 384)
        frame_amp = improc.resize_to_max(frame_amp, 384)

        side_by_side = np.concatenate([frame_orig, frame_amp], axis=1)

        plotter.update(side_by_side, cmap='viridis')

    # Perform motion amplification
    plotter.finalize()
