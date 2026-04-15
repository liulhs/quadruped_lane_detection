"""
ML Keypoint Heatmap detector (ContactPointNet).
Detects tree-ground contact points via a lightweight CNN, fits boundary lines
through left/right groups, computes vanishing point and lateral drift.

Requires a trained ONNX model at models/contact_net.onnx.
Runtime dependencies: Python 3, OpenCV, NumPy (no PyTorch needed).
"""

import os
import cv2
import numpy as np
from utils import undistort_stub, line_intersect, lateral_state, draw_overlay
from ml.inference import heatmap_to_detections

TARGET_W, TARGET_H = 640, 480

# ImageNet normalization (must match training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "contact_net.onnx")

# Peak detection parameters
PEAK_THRESHOLD = 0.3
NMS_RADIUS = 3

# Module-level model cache (loaded once)
_net = None


def _load_model(model_path=None):
    """Load the ONNX model via OpenCV DNN (cached)."""
    global _net
    if _net is not None:
        return _net

    path = model_path or MODEL_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ONNX model not found at {path}. "
            "Train first: python -m ml.train, then export: python -m ml.export_onnx"
        )

    _net = cv2.dnn.readNetFromONNX(path)

    # Use CUDA on Jetson Nano, otherwise OpenCV default (CPU).
    # OpenCV DNN has no MPS backend — CPU is fine for this small model.
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return _net


def _draw_contact_points(overlay, contact_points):
    """Draw red vertical tick marks at each detected contact point."""
    for (x, y, score) in contact_points:
        ix, iy = int(x), int(y)
        cv2.line(overlay, (ix, iy - 20), (ix, iy + 5), (0, 0, 255), 2)
    return overlay


def detect(frame, K=None, D=None, model_path=None):
    """
    Detect tree-ground contact points and compute vanishing point.

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

    net = _load_model(model_path)

    # Preprocess: BGR → RGB, normalize to ImageNet stats, reshape to NCHW blob
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for c in range(3):
        img_rgb[:, :, c] = (img_rgb[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    blob = img_rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # Forward pass
    net.setInput(blob)
    heatmap = net.forward()  # (1, 1, 120, 160)

    # Extract peaks and fit lines
    contact_points, left_line, right_line = heatmap_to_detections(
        heatmap, threshold=PEAK_THRESHOLD, nms_radius=NMS_RADIUS,
    )

    # Compute vanishing point
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
    _draw_contact_points(overlay, contact_points)

    return vp, state, overlay, left_line, right_line


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/aisle_1.jpg"
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read {path}")
        sys.exit(1)

    vp, state, overlay, _, _ = detect(img)
    print(f"VP: {vp}  State: {state}")

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", os.path.basename(path))
    cv2.imwrite(out_path, overlay)
    print(f"Saved: {out_path}")
