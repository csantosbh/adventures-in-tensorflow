import cv2
import numpy as np


def load_img(img_path: str,
             blur_std: int = 1,
             tgt_size: float = 256) -> np.ndarray:
    # Load image
    img = cv2.imread(img_path).astype(np.float32)
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize to 0...1 range
    lowest = np.min(img)
    highest = np.max(img)
    img = (img - lowest) / (highest - lowest)
    # Resize to maximum dimension of 256
    highest_dim = np.max(img.shape)
    new_size = (np.array(img.shape[::-1]) * tgt_size / highest_dim).astype(
        np.int32)
    img = cv2.resize(img, tuple(new_size), interpolation=cv2.INTER_AREA)
    # Blur image
    if blur_std > 0:
        img = cv2.GaussianBlur(img, (33, 33), blur_std)
    # Add channel rank
    img = np.expand_dims(img, 2)

    return img


