"""
Option B — HSV Soil Segmentation + RANSAC Line Fitting
Finds the aisle as the largest connected soil blob, then fits boundary lines.
More robust to aisle clutter (~15-25 FPS on RPi4 @ 640x480).
"""

import cv2
import numpy as np
from utils import undistort_stub, line_intersect, lateral_state, draw_overlay

TARGET_W, TARGET_H = 640, 480
ROI_TOP_FRAC = 1 / 3   # discard sky/upper canopy
ROW_STEP = 4            # sample every Nth row for boundary extraction
MIN_INLIERS = 4         # minimum points for a valid line fit

# HSV range for sandy/tan orchard soil (calibrated from aisle images, H mean ~17-21)
SOIL_H_LO, SOIL_H_HI = 5,   35
SOIL_S_LO, SOIL_S_HI = 10, 140
SOIL_V_LO, SOIL_V_HI = 90, 255


def _soil_mask(frame_hsv):
    lo = np.array([SOIL_H_LO, SOIL_S_LO, SOIL_V_LO], dtype=np.uint8)
    hi = np.array([SOIL_H_HI, SOIL_S_HI, SOIL_V_HI], dtype=np.uint8)
    mask = cv2.inRange(frame_hsv, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask


def _aisle_mask(mask, roi_top, cx):
    """
    Isolate the aisle from tree-base soil blobs by finding the largest
    connected component that contains the bottom-center seed point.
    Falls back to column-profile crop if flood-fill fails.
    """
    roi_mask = mask.copy()
    roi_mask[:roi_top] = 0  # zero out sky region

    # Find connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        roi_mask, connectivity=8)

    if n_labels < 2:
        return roi_mask  # only background — return as-is

    # Seed: bottom center of image should be aisle
    seed_y = int(roi_mask.shape[0] * 0.92)
    seed_x = cx

    # Walk up from seed_y until we hit a soil pixel
    aisle_label = 0
    for y in range(seed_y, roi_top, -1):
        if roi_mask[y, seed_x] == 255:
            aisle_label = labels[y, seed_x]
            break

    if aisle_label == 0:
        # Fallback: use largest component (by area) below roi_top
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
        aisle_label = int(np.argmax(areas)) + 1

    aisle_only = np.zeros_like(roi_mask)
    aisle_only[labels == aisle_label] = 255
    return aisle_only


def _boundary_points(aisle_mask, roi_top, cx):
    """
    Scan each sampled row for the left and right outer edges of the aisle blob.
    Skips rows where the boundary is clamped to the image edge (aisle wider than FOV).
    Returns two lists of (x, y) points.
    """
    h, w = aisle_mask.shape[:2]
    EDGE_MARGIN = 8  # px — ignore boundary points this close to image border
    left_pts, right_pts = [], []

    for y in range(roi_top, h, ROW_STEP):
        row = aisle_mask[y]
        soil_cols = np.where(row == 255)[0]
        if len(soil_cols) == 0:
            continue
        lx = int(soil_cols[0])
        rx = int(soil_cols[-1])
        if lx > EDGE_MARGIN:
            left_pts.append((lx, y))
        if rx < w - EDGE_MARGIN:
            right_pts.append((rx, y))

    return left_pts, right_pts


def _fit_line(points, roi_top, roi_bot, img_w):
    """
    RANSAC-style robust line fit via cv2.fitLine (DIST_HUBER).
    Returns ((x1,y1),(x2,y2)) or None.
    """
    if len(points) < MIN_INLIERS:
        return None

    pts = np.array(points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

    if abs(vy) < 1e-6:
        return None

    x_top = x0 + (vx / vy) * (roi_top - y0)
    x_bot = x0 + (vx / vy) * (roi_bot - y0)

    x_top = int(np.clip(x_top, 0, img_w - 1))
    x_bot = int(np.clip(x_bot, 0, img_w - 1))

    return ((x_top, roi_top), (x_bot, roi_bot))


_prev_left = None
_prev_right = None


def detect(frame, K=None, D=None, debug=False):
    """
    Run Option B detection on a single frame.

    Returns:
        vp      : (x, y) vanishing point or None
        state   : 'centered' | 'drift_left' | 'drift_right' | 'no_detection'
        overlay : annotated BGR frame
        left_line, right_line : ((x1,y1),(x2,y2)) or None each
    """
    global _prev_left, _prev_right

    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    frame = undistort_stub(frame, K, D)
    h, w = frame.shape[:2]
    cx = w // 2
    roi_top = int(h * ROI_TOP_FRAC)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw_mask = _soil_mask(hsv)
    aisle = _aisle_mask(raw_mask, roi_top, cx)

    if debug:
        cv2.imwrite("/tmp/optb_raw_mask.png", raw_mask)
        cv2.imwrite("/tmp/optb_aisle_mask.png", aisle)

    left_pts, right_pts = _boundary_points(aisle, roi_top, cx)

    left_line = _fit_line(left_pts, roi_top, h - 1, w)
    right_line = _fit_line(right_pts, roi_top, h - 1, w)

    if left_line is None:
        left_line = _prev_left
    else:
        _prev_left = left_line

    if right_line is None:
        right_line = _prev_right
    else:
        _prev_right = right_line

    # Symmetry fallback: if one side is still missing, mirror the other about cx
    if left_line is None and right_line is not None:
        left_line = ((w - right_line[0][0], right_line[0][1]),
                     (w - right_line[1][0], right_line[1][1]))
    elif right_line is None and left_line is not None:
        right_line = ((w - left_line[0][0], left_line[0][1]),
                      (w - left_line[1][0], left_line[1][1]))

    vp = None
    state = "no_detection"

    if left_line and right_line:
        vp = line_intersect(left_line[0], left_line[1],
                            right_line[0], right_line[1])
        if vp is not None and vp[1] < h * 0.75 and -w < vp[0] < 2 * w:
            state = lateral_state(vp[0], w)
        else:
            vp = None

    overlay = draw_overlay(frame, vp, left_line, right_line, state)
    return vp, state, overlay, left_line, right_line


if __name__ == "__main__":
    import sys
    import os

    path = sys.argv[1] if len(sys.argv) > 1 else "aisle_1.jpg"
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read {path}")
        sys.exit(1)

    vp, state, overlay, _, _ = detect(img, debug=True)
    print(f"VP: {vp}  State: {state}")
    os.makedirs("output/optionB", exist_ok=True)
    out_path = os.path.join("output/optionB", os.path.basename(path))
    cv2.imwrite(out_path, overlay)
    print(f"Saved: {out_path}")
