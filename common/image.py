import numpy as np


def im2float(img: np.ndarray) -> np.ndarray:
    """
    Converts image to floating point format and normalize it to [0, 1] range
    """
    return img.astype(np.float32) / 255.0
