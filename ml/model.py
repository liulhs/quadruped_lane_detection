"""
ContactPointNet — lightweight keypoint heatmap model for tree-ground contact detection.

Architecture:
  MobileNetV2 (pretrained) encoder with stride-8/16/32 feature taps
  → 3-stage transposed-conv decoder with skip connections
  → 1-channel sigmoid heatmap at 1/4 input resolution (160×120)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ContactPointNet(nn.Module):
    """Predict a heatmap of tree-ground contact points from a 640×480 RGB image."""

    def __init__(self, pretrained=True):
        super().__init__()

        # --- Encoder: MobileNetV2 backbone (width_mult=1.0, pretrained) ---
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None,
        )
        # MobileNetV2 feature block layout (width_mult=1.0):
        #   Blocks 0-6   → stride  8,  32 ch  (block 4 introduces stride 8)
        #   Blocks 7-13  → stride 16,  96 ch  (block 7 introduces stride 16)
        #   Blocks 14-17 → stride 32, 320 ch  (block 14 introduces stride 32)
        #
        # With 640×480 input:
        #   After block 6:   80×60,   32 ch
        #   After block 13:  40×30,   96 ch
        #   After block 17:  20×15,  320 ch

        feats = list(backbone.features.children())
        self.enc_s8 = nn.Sequential(*feats[:7])      # → stride  8,  32 ch
        self.enc_s16 = nn.Sequential(*feats[7:14])    # → stride 16,  96 ch
        self.enc_s32 = nn.Sequential(*feats[14:18])   # → stride 32, 320 ch

        # Freeze early layers to reduce overfitting on tiny datasets
        for p in self.enc_s8.parameters():
            p.requires_grad = False

        # --- Decoder ---
        # Stage 1: s32 (320ch) → s16, fuse with s16 skip (96ch)
        self.up1 = nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(128 + 96, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Stage 2: s16 (128ch) → s8, fuse with s8 skip (32ch)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 3: s8 (64ch) → s4 (16ch) — final 1/4 resolution
        self.up3 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Head: 1×1 conv → single-channel heatmap
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 480, 640) RGB tensor, normalized to ImageNet stats.
        Returns:
            heatmap: (B, 1, 120, 160) float tensor in [0, 1].
        """
        # Encoder
        s8 = self.enc_s8(x)       # (B,  32, 60, 80)
        s16 = self.enc_s16(s8)     # (B,  96, 30, 40)
        s32 = self.enc_s32(s16)    # (B, 320, 15, 20)

        # Decoder stage 1: s32 → s16
        d1 = self.relu(self.bn1(self.up1(s32)))    # (B, 128, 30, 40)
        d1 = self.fuse1(torch.cat([d1, s16], dim=1))

        # Decoder stage 2: s16 → s8
        d2 = self.relu(self.bn2(self.up2(d1)))     # (B, 64, 60, 80)
        d2 = self.fuse2(torch.cat([d2, s8], dim=1))

        # Decoder stage 3: s8 → s4
        d3 = self.relu(self.bn3(self.up3(d2)))     # (B, 16, 120, 160)

        # Head
        heatmap = torch.sigmoid(self.head(d3))     # (B, 1, 120, 160)
        return heatmap


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    net = ContactPointNet(pretrained=False)
    total, trainable = count_params(net)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    dummy = torch.randn(1, 3, 480, 640)
    out = net(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (1, 1, 120, 160), f"Unexpected output shape: {out.shape}"
    print("OK")
