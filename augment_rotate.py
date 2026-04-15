"""
Move original images + annotations into data/, then generate ±5° rotated
copies with transformed annotations.

Produces 9 originals + 9 CW + 9 CCW = 27 labeled images in data/.
"""

import json
import math
import os
import shutil

import cv2
import numpy as np

SRC_DIR = "."
DST_DIR = "data"
ANGLE = 5  # degrees


def rotate_point(x, y, cx, cy, angle_rad):
    """Rotate point (x, y) around center (cx, cy) by angle_rad."""
    dx, dy = x - cx, y - cy
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rx = cos_a * dx - sin_a * dy + cx
    ry = sin_a * dx + cos_a * dy + cy
    return rx, ry


def rotate_image_and_annotations(img_path, json_path, dst_dir, suffix, angle_deg):
    """
    Rotate image by angle_deg and transform all point annotations.
    Positive angle = counter-clockwise in OpenCV convention.
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Rotation matrix (OpenCV rotates CCW for positive angles)
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Save rotated image
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_img = os.path.join(dst_dir, f"{base}_{suffix}.jpg")
    cv2.imwrite(out_img, rotated)

    # Transform annotations
    with open(json_path) as f:
        ann = json.load(f)

    angle_rad = math.radians(angle_deg)
    new_shapes = []
    for shape in ann.get("shapes", []):
        if shape["shape_type"] == "point":
            ox, oy = shape["points"][0]
            rx, ry = rotate_point(ox, oy, cx, cy, angle_rad)
            # Discard points that land outside the image
            if 0 <= rx < w and 0 <= ry < h:
                new_shape = dict(shape)
                new_shape["points"] = [[rx, ry]]
                new_shapes.append(new_shape)

    new_ann = dict(ann)
    new_ann["shapes"] = new_shapes
    new_ann["imagePath"] = f"{base}_{suffix}.jpg"

    out_json = os.path.join(dst_dir, f"{base}_{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(new_ann, f, indent=2)

    return len(new_shapes)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    originals = sorted(
        f for f in os.listdir(SRC_DIR)
        if f.startswith("aisle_") and f.endswith(".jpg")
    )

    total = 0
    for img_name in originals:
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(SRC_DIR, img_name)
        json_path = os.path.join(SRC_DIR, f"{base}.json")

        if not os.path.isfile(json_path):
            print(f"  SKIP {img_name} — no annotation")
            continue

        # Copy original
        shutil.copy2(img_path, os.path.join(DST_DIR, img_name))
        shutil.copy2(json_path, os.path.join(DST_DIR, f"{base}.json"))
        print(f"  {img_name} → copied original")
        total += 1

        # CW rotation (+5° in image space = -5° in OpenCV convention)
        n = rotate_image_and_annotations(img_path, json_path, DST_DIR, "cw5", -ANGLE)
        print(f"  {base}_cw5.jpg → {n} points")
        total += 1

        # CCW rotation (-5° in image space = +5° in OpenCV convention)
        n = rotate_image_and_annotations(img_path, json_path, DST_DIR, "ccw5", ANGLE)
        print(f"  {base}_ccw5.jpg → {n} points")
        total += 1

    print(f"\nDone. {total} images in {DST_DIR}/")


if __name__ == "__main__":
    main()
