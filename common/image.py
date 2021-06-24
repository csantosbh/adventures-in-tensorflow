import numpy as np
import cv2


def im2float(img: np.ndarray) -> np.ndarray:
    """
    Converts image to floating point format and normalize it to [0, 1] range
    """
    return img.astype(np.float32) / 255.0


def rgb2gray(img: np.ndarray, keepdims=True) -> np.ndarray:
    """
    Converts image from rgb to grayscale.
    :param img: Image in format [h, w, c]
    :param keepdims: If true, output will have rank 3. Otherwise, it will have
                     rank 2.
    :return: Image in format [h, w, 1] or [h, w]
    """
    # TODO get correct rgb2gray coefficients
    img = (img[..., 0] + img[..., 1] + img[..., 2])/3.0
    if keepdims:
        img = np.expand_dims(img, 2)

    return img


def resize_to_max(img: np.ndarray,
                  tgt_size: int,
                  interpolation: int = cv2.INTER_AREA):
    """
    Resize image such that its maximum dimension becomes tgt_size
    """
    highest_dim = np.max(img.shape)
    new_size = (np.array(img.shape[1::-1]) * tgt_size / highest_dim).astype(
        np.int32)
    img = cv2.resize(img, tuple(new_size), interpolation=interpolation)

    return img