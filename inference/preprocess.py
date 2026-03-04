from pathlib import Path

import cv2
import numpy as np

from .constants import IMAGE_H, IMAGE_W, IMAGENET_MEAN, IMAGENET_STD
"""
Image preprocessing for MambaVision inference.

All functions are stateless and operate on numpy arrays.
Input:  HxWxC uint8 BGR (OpenCV convention) or RGB
Output: (1, C, H, W) float32 normalized tensor ready for ORT
"""

from pathlib import Path

import cv2
import numpy as np

from .constants import IMAGE_H, IMAGE_W, IMAGENET_MEAN, IMAGENET_STD


def load_image_bgr(path: Path) -> np.ndarray:
    """Load image from disk as HxWxC uint8 BGR (OpenCV native format)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """HxWxC BGR uint8 → HxWxC RGB uint8."""
    return image[:, :, ::-1]


def resize(image: np.ndarray, h: int = IMAGE_H, w: int = IMAGE_W) -> np.ndarray:
    """Resize to (h, w) using bilinear interpolation."""
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize(image: np.ndarray,
              mean: tuple = IMAGENET_MEAN,
              std: tuple  = IMAGENET_STD) -> np.ndarray:
    """
    HxWxC uint8 RGB → HxWxC float32 normalized.
    Applies: (pixel / 255 - mean) / std per channel.
    """
    img = image.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    return (img - mean_arr) / std_arr


def hwc_to_nchw(image: np.ndarray) -> np.ndarray:
    """HxWxC float32 → 1xCxHxW float32 (adds batch dim, transposes)."""
    return np.transpose(image, (2, 0, 1))[np.newaxis, ...]  # (1, C, H, W)


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline: BGR uint8 → (1, 3, 224, 224) float32.
    This is the canonical entry point for inference.
    """
    rgb     = bgr_to_rgb(image_bgr)
    resized = resize(rgb)
    normed  = normalize(resized)
    return hwc_to_nchw(normed)


def preprocess_from_path(path: Path) -> np.ndarray:
    """Convenience: load + preprocess in one call."""
    return preprocess(load_image_bgr(path))