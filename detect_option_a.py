"""
Option A — Probabilistic Hough Line Transform
Upper-ROI-first strategy: fit lines from the clean upper portion (near horizon)
where weeds, people, and ground clutter don't reach, then extrapolate down.
"""

import cv2
import numpy as np
from utils import undistort_stub, line_intersect, lateral_state, draw_overlay

TARGET_W, TARGET_H = 640, 480
ROI_TOP_FRAC  = 1 / 3   # discard sky (top third)
UPPER_ROI_FRAC = 0.45   # top 45% of remaining ROI is "upper zone" (cleaner signal)

# Tighter slope window — tan(22°)≈0.40, tan(68°)≈2.48
# Rejects near-horizontal (drip lines, wheel tracks) and near-vertical (fence posts)
SLOPE_MIN, SLOPE_MAX = 0.40, 2.48

# Hough parameters
MIN_LINE_LEN         = 100  # primary pass — only long structural lines survive
MIN_LINE_LEN_FALLBACK = 60  # fallback pass — used per-side if primary finds nothing
MAX_LINE_GAP  = 10          # tighter gap — don't bridge across unrelated segments
HOUGH_THRESH  = 35

FLARE_HALF_WIDTH  = 50
FLARE_PERCENTILE  = 80


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


def _fit_line(segs, roi_top, roi_bot, img_w):
    """
    Fit one line through all segment endpoints using HUBER-robust cv2.fitLine.
    Returns ((x_top, roi_top), (x_bot, roi_bot)) or None.
    """
    if not segs:
        return None
    pts = np.array([[x, y] for x1, y1, x2, y2 in segs
                            for x, y in ((x1, y1), (x2, y2))], dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None
    x_top = x0 + (vx / vy) * (roi_top - y0)
    x_bot = x0 + (vx / vy) * (roi_bot - y0)
    return ((int(np.clip(x_top, 0, img_w - 1)), roi_top),
            (int(np.clip(x_bot, 0, img_w - 1)), roi_bot))


def _classify(raw_lines, roi_top, cx, roi_mid):
    """
    Normalize each Hough segment and classify into:
      left_upper / left_lower / right_upper / right_lower
    based on slope sign, mid-x vs cx, and mid-y vs roi_mid.
    """
    lu, ll, ru, rl = [], [], [], []
    if raw_lines is None:
        return lu, ll, ru, rl

    for seg in raw_lines:
        x1, y1, x2, y2 = seg[0]
        y1 += roi_top
        y2 += roi_top

        # Normalize so y1 is always the TOP point
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        dy = y2 - y1
        if dy < 1:
            continue
        slope = (x2 - x1) / dy  # dx/dy: neg = left row, pos = right row

        if not (SLOPE_MIN <= abs(slope) <= SLOPE_MAX):
            continue

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        upper = mid_y < roi_mid

        if slope < 0 and mid_x < cx:       # left tree row
            (lu if upper else ll).append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > cx:     # right tree row
            (ru if upper else rl).append((x1, y1, x2, y2))

    return lu, ll, ru, rl


def detect(frame, K=None, D=None):
    """
    Run Option A detection on a single frame.

    Strategy:
      1. Run Hough on the full ROI with a long minimum line length.
      2. Prefer segments whose midpoint falls in the upper ROI zone
         (near horizon — less clutter from weeds, wheel tracks, people).
      3. Fall back to lower segments only if upper zone lacks coverage.
      4. Extrapolate the fitted line across the full ROI height.

    Returns:
        vp           : (x, y) vanishing point or None
        state        : 'centered' | 'drift_left' | 'drift_right' | 'no_detection'
        overlay      : annotated BGR frame
        left_line    : ((x1,y1),(x2,y2)) or None
        right_line   : ((x1,y1),(x2,y2)) or None
    """
    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    frame = undistort_stub(frame, K, D)
    h, w = frame.shape[:2]
    cx = w // 2
    roi_top = int(h * ROI_TOP_FRAC)
    roi_mid = roi_top + int((h - roi_top) * UPPER_ROI_FRAC)  # upper/lower split

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flare_mask = _mask_lens_flare(gray, roi_top)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.bitwise_and(edges, flare_mask)

    # Primary pass — long lines only
    raw_lines = cv2.HoughLinesP(edges[roi_top:], rho=1, theta=np.pi / 180,
                                 threshold=HOUGH_THRESH,
                                 minLineLength=MIN_LINE_LEN,
                                 maxLineGap=MAX_LINE_GAP)
    lu, ll, ru, rl = _classify(raw_lines, roi_top, cx, roi_mid)

    # Upper zone first — prefer near-horizon segments (less clutter)
    left_segs  = lu if len(lu) >= 2 else lu + ll
    right_segs = ru if len(ru) >= 2 else ru + rl

    # Fallback pass — shorter minimum length, used only for sides still empty
    if not left_segs or not right_segs:
        raw_fb = cv2.HoughLinesP(edges[roi_top:], rho=1, theta=np.pi / 180,
                                  threshold=HOUGH_THRESH,
                                  minLineLength=MIN_LINE_LEN_FALLBACK,
                                  maxLineGap=MAX_LINE_GAP)
        fb_lu, fb_ll, fb_ru, fb_rl = _classify(raw_fb, roi_top, cx, roi_mid)
        if not left_segs:
            left_segs  = fb_lu if len(fb_lu) >= 2 else fb_lu + fb_ll
        if not right_segs:
            right_segs = fb_ru if len(fb_ru) >= 2 else fb_ru + fb_rl

    left_line  = _fit_line(left_segs,  roi_top, h - 1, w)
    right_line = _fit_line(right_segs, roi_top, h - 1, w)

    vp    = None
    state = "no_detection"

    if left_line and right_line:
        vp = line_intersect(left_line[0], left_line[1],
                            right_line[0], right_line[1])
        # Reject VPs that are below the image (lines diverging) or wildly off-center
        if vp is not None and vp[1] < h * 0.75 and -w < vp[0] < 2 * w:
            state = lateral_state(vp[0], w)
        else:
            vp = None

    overlay = draw_overlay(frame, vp, left_line, right_line, state)
    return vp, state, overlay, left_line, right_line


if __name__ == "__main__":
    import sys, os
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
