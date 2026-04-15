"""
Benchmark the ML detector on all aisle_*.jpg images.
Prints FPS and detection results, saves annotated images to output/.

Usage:
    python benchmark.py              # default 20 reps per image
    python benchmark.py --reps 50    # more reps for stable FPS
"""

import argparse
import glob
import os
import time

import cv2

from detect import detect

REPS = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args()

    images = glob.glob("data/**/aisle_*.jpg", recursive=True)
    if not images:
        print("No aisle_*.jpg images found in data/.")
        return

    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ML Keypoint Heatmap Detector")
    print(f"{'='*60}")
    print(f"  {'Image':<18} {'FPS':>7}  {'VP':^22}  {'State'}")
    print(f"  {'-'*65}")

    for img_path in sorted(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  {os.path.basename(img_path):<18}  [cannot read]")
            continue

        # Warm-up
        detect(img.copy())

        # Timed loop
        t0 = time.perf_counter()
        for _ in range(args.reps):
            vp, state, overlay, _, _ = detect(img.copy())
        elapsed = time.perf_counter() - t0

        fps = args.reps / elapsed
        vp_str = f"({vp[0]:6.1f}, {vp[1]:6.1f})" if vp else "       None      "

        print(f"  {os.path.basename(img_path):<18} {fps:>7.1f}  {vp_str}  {state}")

        out_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, overlay)

    print(f"\n  Annotated images saved to: {save_dir}/")


if __name__ == "__main__":
    main()
