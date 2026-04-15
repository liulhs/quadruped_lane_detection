"""
ContactPointDataset — loads images + keypoint annotations, generates Gaussian heatmaps.

Supports two annotation formats:
  1. LabelMe per-image JSON files (aisle_1.json beside aisle_1.jpg)
  2. Consolidated annotations/annotations.json

Augmentations via albumentations with keypoint-aware transforms.
"""

import json
import os
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:
    A = None


# --- Constants ---
IMG_W, IMG_H = 640, 480
HM_W, HM_H = 160, 120          # heatmap = 1/4 resolution
GAUSSIAN_SIGMA = 3.0            # sigma in heatmap pixel coords
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_augmentation():
    """Heavy augmentation for small-dataset regime."""
    if A is None:
        raise ImportError("albumentations is required: pip install albumentations")
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.7),
            A.GaussNoise(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=5,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.RandomResizedCrop(size=(IMG_H, IMG_W), scale=(0.8, 1.0), ratio=(1.3, 1.4), p=0.3),
            A.Resize(IMG_H, IMG_W),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )


def build_val_augmentation():
    """Deterministic resize only."""
    if A is None:
        return None
    return A.Compose(
        [A.Resize(IMG_H, IMG_W)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )


def _gaussian_2d(shape, sigma):
    """Generate a 2D Gaussian kernel (used once, cached)."""
    h, w = shape
    ys = np.arange(h, dtype=np.float32) - h // 2
    xs = np.arange(w, dtype=np.float32) - w // 2
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g


def render_heatmap(points, hm_h=HM_H, hm_w=HM_W, sigma=GAUSSIAN_SIGMA):
    """
    Render a ground-truth heatmap from a list of (x, y) image-coordinate points.
    Points are scaled to heatmap resolution; each gets a Gaussian blob.
    Multiple blobs are merged with element-wise max.
    """
    heatmap = np.zeros((hm_h, hm_w), dtype=np.float32)
    radius = int(3 * sigma)  # 3-sigma covers 99.7%
    kernel_size = 2 * radius + 1
    g = _gaussian_2d((kernel_size, kernel_size), sigma)

    for (px, py) in points:
        # Scale from image coords to heatmap coords
        cx = px * hm_w / IMG_W
        cy = py * hm_h / IMG_H
        ix, iy = int(round(cx)), int(round(cy))

        # Clamp the Gaussian patch to heatmap boundaries
        y0 = max(0, iy - radius)
        y1 = min(hm_h, iy + radius + 1)
        x0 = max(0, ix - radius)
        x1 = min(hm_w, ix + radius + 1)

        gy0 = max(0, radius - iy)
        gy1 = gy0 + (y1 - y0)
        gx0 = max(0, radius - ix)
        gx1 = gx0 + (x1 - x0)

        if y1 > y0 and x1 > x0:
            np.maximum(heatmap[y0:y1, x0:x1], g[gy0:gy1, gx0:gx1], out=heatmap[y0:y1, x0:x1])

    return heatmap


# --- Annotation loading ---

def _load_labelme_json(json_path):
    """Extract point annotations from a LabelMe JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    points = []
    for shape in data.get("shapes", []):
        if shape["shape_type"] == "point":
            x, y = shape["points"][0]
            points.append((float(x), float(y)))
    return points


def _load_consolidated_json(json_path):
    """Load the consolidated annotations/annotations.json format."""
    with open(json_path) as f:
        data = json.load(f)
    # {filename: {"points": [[x,y], ...]}}
    out = {}
    for fname, ann in data.items():
        pts = [(float(p[0]), float(p[1])) for p in ann["points"]]
        out[fname] = pts
    return out


def discover_annotations(image_dir, annotations_dir=None):
    """
    Find all (image_path, points_list) pairs.
    Checks for:
      1. consolidated annotations/annotations.json
      2. per-image LabelMe JSONs next to the images
    """
    pairs = []

    # Try consolidated JSON first
    if annotations_dir is None:
        annotations_dir = os.path.join(os.path.dirname(image_dir), "annotations")
    consolidated = os.path.join(annotations_dir, "annotations.json")
    if os.path.isfile(consolidated):
        ann_map = _load_consolidated_json(consolidated)
        for fname, pts in ann_map.items():
            img_path = os.path.join(image_dir, fname)
            if os.path.isfile(img_path):
                pairs.append((img_path, pts))
        if pairs:
            return pairs

    # Fall back to per-image LabelMe JSONs
    for img_path in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
        json_path = os.path.splitext(img_path)[0] + ".json"
        if os.path.isfile(json_path):
            pts = _load_labelme_json(json_path)
            if pts:
                pairs.append((img_path, pts))

    return pairs


class ContactPointDataset(Dataset):
    """
    PyTorch dataset for contact-point heatmap regression.

    Args:
        image_dir:  directory containing aisle_*.jpg images.
        transform:  albumentations Compose with keypoint_params.
        annotations_dir:  optional path to annotations/ folder.
    """

    def __init__(self, image_dir, transform=None, annotations_dir=None):
        self.pairs = discover_annotations(image_dir, annotations_dir)
        if not self.pairs:
            raise FileNotFoundError(
                f"No annotated images found. Place LabelMe JSONs beside images in {image_dir} "
                f"or create annotations/annotations.json."
            )
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, points = self.pairs[idx]

        # Load and resize to standard dimensions
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Cannot read image: {img_path}")
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scale annotation points to 640×480 if image was a different size
        # (LabelMe saves coords at original resolution)
        sx, sy = IMG_W / orig_w, IMG_H / orig_h
        points = [(x * sx, y * sy) for (x, y) in points]

        # Apply augmentations
        if self.transform is not None:
            result = self.transform(image=img, keypoints=points)
            img = result["image"]
            points = result["keypoints"]

        # Convert to tensor + normalize
        img_t = img.astype(np.float32) / 255.0
        for c in range(3):
            img_t[:, :, c] = (img_t[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        img_t = torch.from_numpy(img_t.transpose(2, 0, 1))  # (3, H, W)

        # Render heatmap
        heatmap = render_heatmap(points)
        hm_t = torch.from_numpy(heatmap).unsqueeze(0)  # (1, HM_H, HM_W)

        return img_t, hm_t


if __name__ == "__main__":
    # Quick sanity check: discover annotations and render one heatmap
    import sys

    img_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    pairs = discover_annotations(img_dir)
    print(f"Found {len(pairs)} annotated images")
    for path, pts in pairs:
        print(f"  {os.path.basename(path)}: {len(pts)} points")
        hm = render_heatmap(pts)
        print(f"    heatmap range: [{hm.min():.3f}, {hm.max():.3f}]")
