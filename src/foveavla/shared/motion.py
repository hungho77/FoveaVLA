"""Motion image of the full view: what changed since an earlier frame, with no colour and little detail.

The image is the signed luminance difference between the current letterboxed full view and the same view a few frames
earlier, 3x3 box-blurred, with small differences (sensor and codec noise) set to zero. 128 means no change; brighter or
darker means the pixel got brighter or darker. A moving arm shows as a bright/dark pair of edges whose order gives the
direction, and a still scene is uniform grey.
"""

import numpy as np

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
# Static frames 8 apart differ by up to ~6 levels after the blur; a moving arm by 50-110.
DEAD_ZONE = 4.0
GAIN = 2.0


def _box_blur(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    h, w = x.shape[-2:]
    return sum(p[..., i : i + h, j : j + w] for i in range(3) for j in range(3)) / 9.0


def motion_image(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """uint8 [..., H, W, 3] current and earlier RGB view -> uint8 [..., H, W, 3] motion image (grey, 128 = still)."""
    if current.shape != previous.shape or current.shape[-1] != 3:
        raise ValueError(f"Expected two RGB images of the same shape, got {current.shape} and {previous.shape}")
    diff = _box_blur(current.astype(np.float32) @ _LUMA - previous.astype(np.float32) @ _LUMA)
    diff = np.sign(diff) * np.maximum(np.abs(diff) - DEAD_ZONE, 0.0)
    grey = np.clip(128.0 + GAIN * diff, 0, 255).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=-1)


def still_image(shape: tuple[int, ...]) -> np.ndarray:
    """The motion image of a scene that has not changed (also used when there is no earlier frame)."""
    return np.full(shape, 128, np.uint8)
