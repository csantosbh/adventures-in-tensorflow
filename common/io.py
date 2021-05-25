import cv2
import numpy as np


def load_img(img_path: str,
             blur_std: float = 1,
             tgt_size: float = 256,
             cvt_grayscale: bool = True,
             normalize: bool = True,
             interpolation: int = cv2.INTER_AREA,
             replace_inf: float = None,
             archive_index: str = None) -> np.ndarray:
    # Load image
    if img_path.endswith('.npy'):
        img = np.load(img_path)
    elif img_path.endswith('.npz'):
        assert(archive_index is not None)
        img = np.load(img_path)[archive_index]
    else:
        img = cv2.imread(img_path).astype(np.float32)
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if cvt_grayscale:
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Add channel rank
        img = np.expand_dims(img, 2)

    # Normalize to 0...1 range
    if normalize:
        lowest = np.min(img)
        highest = np.max(img)
        img = (img - lowest) / (highest - lowest)

    # Resize to maximum specified dimension
    highest_dim = np.max(img.shape)
    new_size = (np.array(img.shape[1::-1]) * tgt_size / highest_dim).astype(
        np.int32)
    img = cv2.resize(img, tuple(new_size), interpolation=interpolation)

    # Blur image
    if blur_std > 0:
        img = cv2.GaussianBlur(img, (33, 33), blur_std)

    if img.ndim == 2:
        # Add channel rank
        img = np.expand_dims(img, 2)

    # Replace inf values by float number
    if replace_inf is not None:
        img[img == np.inf] = replace_inf

    return img


