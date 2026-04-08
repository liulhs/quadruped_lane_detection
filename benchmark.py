"""
Benchmark both detectors on all aisle_*.jpg images.
Prints FPS and detection results, saves annotated images to output/.

Usage:
    python benchmark.py              # runs both options
    python benchmark.py --option a   # option A only
    python benchmark.py --option b   # option B only
    python benchmark.py --reps 50    # more reps for stable FPS (default 20)
"""

import argparse
import glob
import os
import time

import cv2

import detect_option_a as opt_a
import detect_option_b as opt_b

REPS = 20  # loop repetitions per image for stable timing


def run_detector(name, detect_fn, images, reps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  {'Image':<18} {'FPS':>7}  {'VP':^22}  {'State'}")
    print(f"  {'-'*65}")

    for img_path in sorted(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  {os.path.basename(img_path):<18}  [cannot read]")
            continue

        # Warm-up
        detect_fn(img.copy())

        # Timed loop
        t0 = time.perf_counter()
        for _ in range(reps):
            vp, state, overlay, _, _ = detect_fn(img.copy())
        elapsed = time.perf_counter() - t0

        fps = reps / elapsed
        vp_str = f"({vp[0]:6.1f}, {vp[1]:6.1f})" if vp else "       None      "

        print(f"  {os.path.basename(img_path):<18} {fps:>7.1f}  {vp_str}  {state}")

        out_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, overlay)

    print(f"\n  Annotated images saved to: {save_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["a", "b", "both"], default="both")
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args()

    images = glob.glob("aisle_*.jpg")
    if not images:
        print("No aisle_*.jpg images found in current directory.")
        return

    if args.option in ("a", "both"):
        run_detector("Option A — Hough Line Transform",
                     opt_a.detect, images, args.reps, "output/optionA")

    if args.option in ("b", "both"):
        run_detector("Option B — HSV Segmentation + RANSAC",
                     opt_b.detect, images, args.reps, "output/optionB")


if __name__ == "__main__":
    main()
