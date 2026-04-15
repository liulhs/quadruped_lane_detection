"""
Export trained ContactPointNet to ONNX for deployment with OpenCV DNN.

Usage:
    python -m ml.export_onnx                              # default paths
    python -m ml.export_onnx --checkpoint models/best.pth --output models/contact_net.onnx
"""

import argparse
import os

import torch

from ml.model import ContactPointNet


def export(checkpoint_path, output_path, opset_version=11):
    """Export a trained ContactPointNet checkpoint to ONNX."""
    model = ContactPointNet(pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Fixed input size matching the project standard
    dummy = torch.randn(1, 3, 480, 640)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["heatmap"],
        opset_version=opset_version,
        dynamic_axes=None,  # fixed size for edge optimization
        dynamo=False,  # legacy TorchScript exporter — compatible with OpenCV DNN
    )

    # Verify the export
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported ONNX model to {output_path} ({size_mb:.1f} MB, opset {opset_version})")

    # Quick sanity check with ONNX runtime if available
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed")
    except ImportError:
        print("(install onnx to verify: pip install onnx)")


def main():
    parser = argparse.ArgumentParser(description="Export ContactPointNet to ONNX")
    parser.add_argument("--checkpoint", default="models/best.pth",
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--output", default="models/contact_net.onnx",
                        help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset version (11 works well with OpenCV DNN)")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
