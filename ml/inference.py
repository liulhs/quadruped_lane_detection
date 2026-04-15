"""
Inference utilities for ContactPointNet heatmaps.

Peak extraction, sub-pixel refinement, left/right classification, line fitting.
All functions use NumPy/OpenCV only (no PyTorch) so they work at runtime on edge
hardware without a PyTorch installation.
"""

import cv2
import numpy as np

# Match the project standard (640×480 frames)
IMG_W, IMG_H = 640, 480
HM_W, HM_H = 160, 120
SCALE_X = IMG_W / HM_W  # 4.0
SCALE_Y = IMG_H / HM_H  # 4.0

# ROI: top 1/3 of image is sky/canopy (matching Options A/B)
ROI_TOP_FRAC = 1 / 3
ROI_TOP = int(IMG_H * ROI_TOP_FRAC)  # 160


def extract_peaks(heatmap, threshold=0.3, nms_radius=3):
    """
    Extract contact point peaks from a heatmap using NMS + thresholding.

    Args:
        heatmap: (H, W) float32 array in [0, 1] at heatmap resolution.
        threshold: minimum confidence to keep a peak.
        nms_radius: radius for non-maximum suppression (max-pool kernel).

    Returns:
        List of (x, y, score) in full image coordinates (640×480).
    """
    h, w = heatmap.shape

    # NMS via dilation (equivalent to max-pool with kernel=2*radius+1)
    kernel_size = 2 * nms_radius + 1
    dilated = cv2.dilate(heatmap, np.ones((kernel_size, kernel_size), np.uint8))
    peaks_mask = (heatmap == dilated) & (heatmap >= threshold)

    ys, xs = np.where(peaks_mask)
    scores = heatmap[ys, xs]

    # Sub-pixel refinement via weighted centroid in 3×3 neighborhood
    points = []
    for y, x, s in zip(ys, xs, scores):
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        patch = heatmap[y0:y1, x0:x1]

        if patch.sum() < 1e-6:
            cx, cy = float(x), float(y)
        else:
            local_ys, local_xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            cx = x0 + np.average(local_xs, weights=patch)
            cy = y0 + np.average(local_ys, weights=patch)

        # Scale to image coordinates
        img_x = cx * SCALE_X
        img_y = cy * SCALE_Y
        points.append((img_x, img_y, float(s)))

    # Sort by y descending (closest trees first, bottom of image)
    points.sort(key=lambda p: -p[1])
    return points


def classify_left_right(points, img_cx=IMG_W / 2):
    """
    Split contact points into left and right groups based on x-position.

    Args:
        points: list of (x, y, score) from extract_peaks.
        img_cx: image center x (default 320).

    Returns:
        left_pts: list of (x, y) for trees left of center.
        right_pts: list of (x, y) for trees right of center.
    """
    left = [(x, y) for x, y, _ in points if x < img_cx]
    right = [(x, y) for x, y, _ in points if x >= img_cx]
    return left, right


def fit_line_through_points(pts, roi_top=ROI_TOP, roi_bot=IMG_H - 1):
    """
    Fit a robust line through contact points and return endpoints clipped to ROI.

    Args:
        pts: list of (x, y) — at least 2 points required.
        roi_top: y-coordinate of ROI top.
        roi_bot: y-coordinate of ROI bottom.

    Returns:
        ((x1, y1), (x2, y2)) line endpoints, or None if too few points.
    """
    if len(pts) < 2:
        return None

    pts_arr = np.array(pts, dtype=np.float32)
    # cv2.fitLine returns (vx, vy, x0, y0) — unit direction + point on line
    line = cv2.fitLine(pts_arr, cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()

    if abs(vy) < 1e-6:
        return None

    # Compute x at roi_top and roi_bot
    t_top = (roi_top - y0) / vy
    x_top = x0 + t_top * vx

    t_bot = (roi_bot - y0) / vy
    x_bot = x0 + t_bot * vx

    return ((float(x_top), float(roi_top)), (float(x_bot), float(roi_bot)))


def heatmap_to_detections(heatmap, threshold=0.3, nms_radius=3, min_points=2):
    """
    Full inference pipeline: heatmap → contact points → left/right lines.

    Args:
        heatmap: (H, W) or (1, H, W) or (1, 1, H, W) float32 array.
        threshold: peak detection threshold.
        nms_radius: NMS kernel radius.
        min_points: minimum points per side to fit a line.

    Returns:
        contact_points: list of (x, y, score) in image coords.
        left_line: ((x1,y1),(x2,y2)) or None.
        right_line: ((x1,y1),(x2,y2)) or None.
    """
    # Handle batch / channel dimensions from ONNX output
    hm = np.squeeze(heatmap).astype(np.float32)
    assert hm.ndim == 2, f"Expected 2D heatmap, got shape {hm.shape}"

    peaks = extract_peaks(hm, threshold=threshold, nms_radius=nms_radius)
    left_pts, right_pts = classify_left_right(peaks)

    left_line = fit_line_through_points(left_pts) if len(left_pts) >= min_points else None
    right_line = fit_line_through_points(right_pts) if len(right_pts) >= min_points else None

    return peaks, left_line, right_line
