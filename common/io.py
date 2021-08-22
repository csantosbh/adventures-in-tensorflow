import cv2
import numpy as np
from typing import Tuple

from common import image as improc


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
    img = improc.resize_to_max(img, tgt_size, interpolation)

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


def load_svg(path='fox.svg', num_samples=100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load svg file, returning set of points and their inwards normals
    """
    from svgpathtools import svg2paths

    paths, attributes = svg2paths(path)
    main_curve = paths[0]
    points = []
    norms = []

    for t in sorted(np.random.uniform(size=(num_samples,))):
        pt = np.conj(main_curve.point(t))
        pt_next = np.conj(main_curve.point((t + 1e-3) % 1.0))

        grad = pt_next - pt
        grad = grad / abs(grad)

        points.append([pt.imag, pt.real])
        norms.append([grad.real, -grad.imag])

    points = np.array(points)
    norma = lambda x: (x - (np.max(x, 0) + np.min(x, 0))/2) / \
                      (np.max(x) - np.min(x)) * 4
    points = norma(points)
    norms = np.array(norms)

    return points, norms


def load_point_cloud(npz_cloud_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NPZ file containing point cloud positions and inwards normals.
    Both have points in dim 0 and xyz coordinates in dim 1.
    These points can be generated with the command `import_point_cloud`.
    """
    data = np.load(npz_cloud_name)
    return data['positions'], data['normals']
