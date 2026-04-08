# Quadruped Lane Detection

Vision-based aisle navigation for a quadruped robot operating in row-crop orchards. The system detects tree-row boundaries from a forward-facing camera, computes a vanishing point, and determines the robot's lateral position relative to the aisle centerline — enabling closed-loop steering corrections without GPS or LiDAR.

---

## Overview

Navigating structured agricultural environments like orchards presents a well-constrained visual problem: parallel rows of trees form strong perspective lines that converge at a vanishing point. This project exploits that geometry to:

1. Detect tree trunks / row boundaries on both sides of the aisle
2. Fit lines through the left and right boundary sets
3. Intersect those lines to find the vanishing point (VP)
4. Use the VP position relative to the image center to estimate heading error and lateral offset
5. Output a corrective signal: **centered**, **drift left**, or **drift right**

All processing runs on-device using classical computer vision (OpenCV) — no neural network inference required, making it suitable for resource-constrained edge hardware.

---

## Approach

Two implementation strategies are under evaluation:

### Option A — Hough Line Transform
- Undistort → Grayscale → ROI crop → Canny edge detection
- Probabilistic Hough transform → angle-filter to isolate row lines
- Cluster lines into left/right groups → intersect for VP
- Target: 30+ FPS on Raspberry Pi 4 / Jetson Nano

### Option B — Aisle Color Segmentation + RANSAC (recommended)
- Undistort → HSV colorspace → threshold sandy-soil color
- Morphological cleanup → extract left/right aisle boundary points
- RANSAC line fit per side → intersect for VP
- More robust to aisle clutter (weeds, drip lines, wheel tracks)
- Target: 15–25 FPS on edge hardware

Both approaches output the vanishing point, a rendered centerline arrow, and a discrete lateral state.

---

## Repository Structure

```
quadruped_lane_detection/
├── aisle_1.jpg … aisle_9.jpg   # Raw test images captured 2026-04-06
├── Desired result.png           # Target visualization (VP lines + centerline arrow)
└── README.md
```

> Source code coming soon.

---

## Target Hardware

- **Robot platform:** Quadruped (legged robot)
- **Camera:** Forward-facing, wide-angle (fisheye distortion present — undistortion required)
- **Compute:** Edge device (Raspberry Pi 4 / NVIDIA Jetson Nano or equivalent)
- **Language:** Python 3, OpenCV

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

Red lines trace the detected tree row boundaries. The centerline arrow runs from the vanishing point to the robot's ground position. Arrow deviation from vertical encodes combined heading and lateral error.

---

## Roadmap

- [ ] Camera intrinsic calibration and undistortion
- [ ] Implement Option A (Hough)
- [ ] Implement Option B (Color seg + RANSAC)
- [ ] Benchmark FPS and accuracy on test images
- [ ] Integrate with robot locomotion controller
- [ ] Evaluate on additional lighting conditions (dawn, dusk, overcast)

---

## Author

**Haosong Liu** — UCI Graduate Researcher  
Project: RobotX Quadruped Navigation  
