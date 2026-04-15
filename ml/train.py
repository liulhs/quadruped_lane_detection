"""
Training script for ContactPointNet.

Usage:
    python -m ml.train                          # train with defaults
    python -m ml.train --epochs 300 --lr 5e-4   # override hyperparams
    python -m ml.train --image-dir /path/to/imgs --annotations-dir /path/to/anns
"""

import argparse
import os
import sys
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ml.model import ContactPointNet, count_params
from ml.dataset import (
    ContactPointDataset,
    build_train_augmentation,
    build_val_augmentation,
    IMG_W, IMG_H, HM_W, HM_H,
    IMAGENET_MEAN, IMAGENET_STD,
)


# ---------------------------------------------------------------------------
# Focal loss for heatmap regression (from CenterNet / CornerNet)
# ---------------------------------------------------------------------------

def focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """
    Modified focal loss for dense keypoint heatmaps.
    Handles the extreme foreground/background imbalance (~0.1% positive pixels).
    """
    pos_mask = gt.ge(0.99).float()
    neg_mask = gt.lt(0.99).float()

    neg_weights = torch.pow(1.0 - gt, beta)

    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask

    num_pos = pos_mask.sum().clamp(min=1)
    loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def visualize_prediction(img_tensor, gt_hm, pred_hm, save_path):
    """Save a side-by-side visualization: image | ground truth heatmap | predicted heatmap."""
    # Denormalize image
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Heatmaps to color
    gt = gt_hm.cpu().numpy().squeeze()
    pr = pred_hm.cpu().detach().numpy().squeeze()

    gt_color = cv2.applyColorMap((gt * 255).astype(np.uint8), cv2.COLORMAP_JET)
    gt_color = cv2.resize(gt_color, (IMG_W, IMG_H))

    pr_color = cv2.applyColorMap((pr * 255).astype(np.uint8), cv2.COLORMAP_JET)
    pr_color = cv2.resize(pr_color, (IMG_W, IMG_H))

    canvas = np.hstack([img, gt_color, pr_color])
    cv2.imwrite(save_path, canvas)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError("No GPU found. Requires CUDA or MPS (Apple Silicon).")
    print(f"Device: {device}")

    # --- Data ---
    if args.val_dir:
        # Explicit train/val directories
        train_dataset = ContactPointDataset(
            args.image_dir,
            transform=build_train_augmentation(),
            annotations_dir=args.annotations_dir,
        )
        val_dataset = ContactPointDataset(
            args.val_dir,
            transform=build_val_augmentation(),
            annotations_dir=args.annotations_dir,
        )
        train_set = train_dataset
        val_set = val_dataset
    else:
        # Auto split from a single directory
        full_dataset = ContactPointDataset(
            args.image_dir,
            transform=build_train_augmentation(),
            annotations_dir=args.annotations_dir,
        )
        n = len(full_dataset)
        n_val = max(1, int(n * 0.2))
        n_train = n - n_val

        gen = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=gen).tolist()
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]

        val_dataset = ContactPointDataset(
            args.image_dir,
            transform=build_val_augmentation(),
            annotations_dir=args.annotations_dir,
        )
        train_set = torch.utils.data.Subset(full_dataset, train_indices)
        val_set = torch.utils.data.Subset(val_dataset, val_indices)

    n_train = len(train_set)
    n_val = len(val_set)
    print(f"Total: {n_train + n_val} images (Train: {n_train}, Val: {n_val})")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # --- Model ---
    model = ContactPointNet(pretrained=True).to(device)
    total, trainable = count_params(model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training ---
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, hms in train_loader:
            imgs, hms = imgs.to(device), hms.to(device)
            pred = model(imgs)
            loss = focal_loss(pred, hms)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, hms in val_loader:
                imgs, hms = imgs.to(device), hms.to(device)
                pred = model(imgs)
                loss = focal_loss(pred, hms)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= n_val

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr:.2e}")

        # Save visualization every 50 epochs
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample_img, sample_hm = val_set[0]
                pred_hm = model(sample_img.unsqueeze(0).to(device)).squeeze(0)
                visualize_prediction(
                    sample_img, sample_hm, pred_hm,
                    os.path.join(vis_dir, f"epoch_{epoch:04d}.jpg"),
                )

        # Checkpointing + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pth"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pth"))
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train ContactPointNet")
    parser.add_argument("--image-dir", default="data", help="Directory with training images")
    parser.add_argument("--val-dir", default=None, help="Separate directory for validation images (if not set, auto-splits from image-dir)")
    parser.add_argument("--annotations-dir", default=None, help="Directory with annotations.json")
    parser.add_argument("--output-dir", default="models", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
