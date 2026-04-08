"""
Option A — Probabilistic Hough Line Transform
Lightweight, fast (~25-40 FPS on RPi4 @ 640x480).
"""

import cv2
import numpy as np
from utils import undistort_stub, line_intersect, lateral_state, draw_overlay

TARGET_W, TARGET_H = 640, 480
ROI_TOP_FRAC = 1 / 3           # discard top third (sky + upper canopy)

# Slope (dx/dy) filter — tan(15°)≈0.27, tan(75°)≈3.73
# Left row: x decreases going down (slope < 0)
# Right row: x increases going down (slope > 0)
SLOPE_MIN, SLOPE_MAX = 0.27, 3.73

FLARE_HALF_WIDTH = 50          # px to mask around brightest column
FLARE_PERCENTILE = 80          # mask column if brighter than this percentile


def _mask_lens_flare(gray, roi_top):
    roi = gray[roi_top:]
    col_brightness = roi.mean(axis=0)
    bright_col = int(np.argmax(col_brightness))
    mask = np.ones_like(gray, dtype=np.uint8) * 255
    if col_brightness[bright_col] > np.percentile(col_brightness, FLARE_PERCENTILE):
        lo = max(0, bright_col - FLARE_HALF_WIDTH)
        hi = min(gray.shape[1], bright_col + FLARE_HALF_WIDTH)
        mask[:, lo:hi] = 0
    return mask


def _representative_line(segs, roi_top, roi_bot, img_w):
    """
    Fit one representative line through a set of normalized Hough segments.
    Segments are in (x1, y1, x2, y2) form with y1 < y2.
    Returns ((x1,y1),(x2,y2)) extended to ROI height, or None.
    """
    if not segs:
        return None

    # Collect all endpoints and fit with cv2.fitLine
    pts = []
    for x1, y1, x2, y2 in segs:
        pts.append([x1, y1])
        pts.append([x2, y2])
    pts = np.array(pts, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None

    x_top = x0 + (vx / vy) * (roi_top - y0)
    x_bot = x0 + (vx / vy) * (roi_bot - y0)
    x_top = int(np.clip(x_top, 0, img_w - 1))
    x_bot = int(np.clip(x_bot, 0, img_w - 1))

    return ((x_top, roi_top), (x_bot, roi_bot))


def detect(frame, K=None, D=None):
    """
    Run Option A detection on a single frame.

    Returns:
        vp      : (x, y) vanishing point or None
        state   : 'centered' | 'drift_left' | 'drift_right' | 'no_detection'
        overlay : annotated BGR frame
        left_line, right_line : ((x1,y1),(x2,y2)) or None each
    """
    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    frame = undistort_stub(frame, K, D)
    h, w = frame.shape[:2]
    cx = w // 2
    roi_top = int(h * ROI_TOP_FRAC)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flare_mask = _mask_lens_flare(gray, roi_top)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.bitwise_and(edges, flare_mask)
    roi_edges = edges[roi_top:]

    raw_lines = cv2.HoughLinesP(roi_edges, rho=1, theta=np.pi / 180,
                                 threshold=40, minLineLength=60, maxLineGap=20)

    left_segs, right_segs = [], []

    if raw_lines is not None:
        for seg in raw_lines:
            x1, y1, x2, y2 = seg[0]
            y1 += roi_top
            y2 += roi_top

            # Normalize: y1 is always the TOP point (smaller y in image coords)
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            # Slope: dx per unit dy (how much x changes as we go DOWN the image)
            dy = y2 - y1
            if dy < 1:
                continue
            slope = (x2 - x1) / dy  # negative for left row, positive for right row

            if not (SLOPE_MIN <= abs(slope) <= SLOPE_MAX):
                continue

            mid_x = (x1 + x2) / 2
            # Left tree row: x decreases going down (slope < 0), line on left side
            if slope < 0 and mid_x < cx:
                left_segs.append((x1, y1, x2, y2))
            # Right tree row: x increases going down (slope > 0), line on right side
            elif slope > 0 and mid_x > cx:
                right_segs.append((x1, y1, x2, y2))

    left_line = _representative_line(left_segs, roi_top, h - 1, w)
    right_line = _representative_line(right_segs, roi_top, h - 1, w)

    vp = None
    state = "no_detection"

    if left_line and right_line:
        vp = line_intersect(left_line[0], left_line[1],
                            right_line[0], right_line[1])
        # Valid VP must be in the upper image (y < roi_top + some margin)
        # and within reasonable horizontal range — otherwise lines are diverging
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

    vp, state, overlay, _, _ = detect(img)
    print(f"VP: {vp}  State: {state}")
    os.makedirs("output/optionA", exist_ok=True)
    out_path = os.path.join("output/optionA", os.path.basename(path))
    cv2.imwrite(out_path, overlay)
    print(f"Saved: {out_path}")
