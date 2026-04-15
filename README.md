# Quadruped Lane Detection

Vision-based aisle navigation for a quadruped robot operating in row-crop orchards. The system detects tree-ground contact points from a forward-facing camera using a lightweight ML model, fits boundary lines through left/right tree rows, computes a vanishing point, and determines the robot's lateral position relative to the aisle centerline — enabling closed-loop steering corrections without GPS or LiDAR.

---

## Overview

Navigating structured agricultural environments like orchards presents a well-constrained visual problem: parallel rows of trees form strong perspective lines that converge at a vanishing point. This project exploits that geometry to:

1. Detect tree-ground contact points (where each trunk meets the soil) using a CNN
2. Fit lines through the left and right contact point groups
3. Intersect those lines to find the vanishing point (VP)
4. Use the VP position relative to the image center to estimate heading error and lateral offset
5. Output a corrective signal: **centered**, **drift left**, or **drift right**

Detection uses a lightweight MobileNetV2-based model (ContactPointNet) exported to ONNX and run via OpenCV DNN — suitable for edge hardware.

---

## Approach — ML Keypoint Heatmap Detection

- ContactPointNet: MobileNetV2 encoder → transposed-conv decoder → 160×120 heatmap
- Each heatmap peak = one tree-ground contact point
- Peaks extracted via NMS, classified left/right by x-position
- Robust line fit (`cv2.fitLine` with DIST_HUBER) per side → intersect for VP
- Trained with focal loss + heavy augmentation for small-dataset regime
- Runtime: ONNX model loaded via `cv2.dnn` (no PyTorch needed at inference)

---

## Repository Structure

```
quadruped_lane_detection/
├── detect.py                    # Main detector (loads ONNX, runs inference)
├── benchmark.py                 # FPS + accuracy benchmark
├── utils.py                     # Shared utilities (VP, overlay, lateral state)
├── ml/
│   ├── model.py                 # ContactPointNet architecture
│   ├── dataset.py               # Dataset, augmentation, heatmap generation
│   ├── train.py                 # Training loop
│   ├── inference.py             # Heatmap post-processing (PyTorch-free)
│   └── export_onnx.py           # PyTorch → ONNX export
├── annotations/                 # Keypoint labels (LabelMe JSON or consolidated)
├── models/                      # Trained checkpoints + ONNX exports
├── aisle_1.jpg … aisle_9.jpg    # Raw test images captured 2026-04-06
├── Desired result.png           # Target visualization
└── README.md
```

---

## Usage

```bash
# Detect on a single image
python detect.py aisle_3.jpg

# Benchmark all test images
python benchmark.py
python benchmark.py --reps 50    # more reps for stable FPS
```

### Training

```bash
# 1. Label images with LabelMe (click tree-ground contact points)
pip install labelme
labelme aisle_1.jpg

# 2. Install training dependencies
pip install torch torchvision albumentations onnx

# 3. Train
python -m ml.train --image-dir . --epochs 200

# 4. Export to ONNX
python -m ml.export_onnx
```

---

## Target Hardware

- **Robot platform:** Quadruped (legged robot)
- **Camera:** Forward-facing, wide-angle (fisheye distortion present — undistortion required)
- **Compute:** Edge device (Raspberry Pi 4 / NVIDIA Jetson Nano or equivalent)
- **Runtime deps:** Python 3, OpenCV, NumPy

---

## Environment

Test data was collected in a drip-irrigated tree orchard with:
- Sandy, high-contrast soil aisle surface
- Young trees (~1–2m tall) planted in uniform rows
- Bright, direct sunlight (lens flare mitigation needed)
- Aisle width approximately 3–4m

---

## Desired Output

![Desired result](Desired%20result.png)

Red vertical marks at each detected tree-ground contact point. Red lines trace the fitted boundary lines. The centerline arrow runs from the vanishing point to the robot's ground position. Arrow deviation from vertical encodes combined heading and lateral error.

---

## Roadmap

- [ ] Camera intrinsic calibration and undistortion
- [x] ML-based tree-ground contact point detection (ContactPointNet)
- [ ] Collect and label 150+ training images for robust model
- [ ] Benchmark FPS on target edge hardware (RPi4 / Jetson Nano)
- [ ] Integrate with robot locomotion controller
- [ ] Evaluate on additional lighting conditions (dawn, dusk, overcast)

---

## Author

**Haosong Liu** — UCI Graduate Researcher  
Project: RobotX Quadruped Navigation  
