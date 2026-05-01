"""
Minimal Grad-CAM placeholder.

This file is intentionally lightweight at the start. First get training and real-time
classification working. Then implement Grad-CAM here.

Planned flow:
1. Register hooks on the final convolution layer.
2. Run a forward pass.
3. Backpropagate the target class score.
4. Weight activation maps by gradients.
5. Overlay the heatmap on the original image/frame.
"""

import cv2
import numpy as np


def overlay_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a normalized heatmap on a BGR frame."""
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def dummy_heatmap(frame_bgr: np.ndarray) -> np.ndarray:
    """Temporary heatmap for UI testing before real Grad-CAM is implemented."""
    h, w = frame_bgr.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) / 4
    heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * radius ** 2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap
