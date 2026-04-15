import cv2
import numpy as np


def undistort_stub(frame, K=None, D=None):
    """Identity undistortion — swap in real K, D once camera is calibrated."""
    if K is None or D is None:
        return frame
    return cv2.undistort(frame, K, D)


def line_intersect(p1, p2, p3, p4):
    """
    Intersect line (p1->p2) with line (p3->p4) using homogeneous coordinates.
    Returns (x, y) float tuple or None if lines are parallel.
    """
    def to_hom(a, b):
        return np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])

    l1 = to_hom(p1, p2)
    l2 = to_hom(p3, p4)
    pt = np.cross(l1, l2)
    if abs(pt[2]) < 1e-6:
        return None
    return (pt[0] / pt[2], pt[1] / pt[2])


def lateral_state(vp_x, img_w, threshold=0.06):
    """
    Compare vanishing point x to image center.
    threshold: fraction of image width defining the dead zone (±6% default).
    Returns 'centered', 'drift_left', or 'drift_right'.
    Note: VP to the right of center → robot drifted left, and vice versa.
    """
    cx = img_w / 2.0
    offset = (vp_x - cx) / img_w  # negative = VP left of center
    if abs(offset) <= threshold:
        return "centered"
    # VP right of center → robot is left of centerline
    return "drift_left" if offset > 0 else "drift_right"


def draw_overlay(frame, vp, left_line, right_line, state):
    """
    Draw the desired-result style overlay:
      - Red lines along detected row boundaries
      - White arrow from VP down to bottom-center
      - State text in top-left
    left_line / right_line: each is ((x1,y1),(x2,y2)) in image coords, or None.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    cx = w // 2

    # Draw boundary lines
    for line in (left_line, right_line):
        if line is not None:
            pt1 = (int(line[0][0]), int(line[0][1]))
            pt2 = (int(line[1][0]), int(line[1][1]))
            cv2.line(out, pt1, pt2, (0, 0, 255), 2)

    # Draw VP marker
    if vp is not None:
        vx, vy = int(vp[0]), int(vp[1])
        cv2.circle(out, (vx, vy), 6, (0, 0, 255), -1)

        # Arrow from VP to bottom-center
        arrow_tip = (cx, h - 20)
        cv2.arrowedLine(out, (vx, vy), arrow_tip,
                        (255, 255, 255), 2, tipLength=0.08)

    # State text
    color_map = {
        "centered":    (0, 255, 0),
        "drift_left":  (0, 165, 255),
        "drift_right": (0, 165, 255),
    }
    color = color_map.get(state, (255, 255, 255))
    cv2.putText(out, state.upper(), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return out
