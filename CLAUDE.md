# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Vision-based aisle navigation for a quadruped robot in row-crop orchards. Uses an ML model (ContactPointNet) to detect tree-ground contact points from a forward-facing camera, fits boundary lines through left/right tree rows, computes a vanishing point (VP), and determines lateral drift relative to the aisle centerline — enabling steering corrections without GPS or LiDAR. Runs on edge hardware (RPi4 / Jetson Nano) via ONNX + OpenCV DNN.

## Commands

```bash
# Run detector on one image
python detect.py aisle_3.jpg              # saves to output/

# Benchmark across all test images (FPS + accuracy)
python benchmark.py                       # 20 reps per image
python benchmark.py --reps 50             # more reps for stable FPS

# Train the model (requires PyTorch + labeled data)
python -m ml.train --image-dir . --epochs 200

# Export trained model to ONNX
python -m ml.export_onnx
```

Runtime dependencies: Python 3, OpenCV (`cv2`), NumPy.
Training dependencies: PyTorch, torchvision, albumentations, onnx.

## Architecture

ML-based detection pipeline:

- **`detect.py`** — Main detector. Loads ONNX model via `cv2.dnn`, runs ContactPointNet to produce a heatmap, extracts contact point peaks, classifies left/right, fits boundary lines, computes VP and lateral state. Exposes `detect(frame, K=None, D=None) -> (vp, state, overlay, left_line, right_line)`.

- **`ml/model.py`** — ContactPointNet architecture. MobileNetV2 (pretrained) encoder with stride-8/16/32 feature taps, 3-stage transposed-conv decoder with skip connections, outputs 1-channel 160×120 heatmap.

- **`ml/dataset.py`** — Dataset class. Reads LabelMe per-image JSONs or consolidated `annotations/annotations.json`. Generates Gaussian heatmaps as supervision targets. Heavy albumentations augmentation for small-dataset training.

- **`ml/train.py`** — Training loop. Modified focal loss, AdamW optimizer, cosine LR scheduler, early stopping, validation visualization.

- **`ml/inference.py`** — Heatmap post-processing (PyTorch-free). NMS-based peak extraction, sub-pixel refinement, left/right classification, `cv2.fitLine(DIST_HUBER)` line fitting.

- **`ml/export_onnx.py`** — PyTorch → ONNX export for edge deployment.

- **`utils.py`** — Shared functions:
  - `undistort_stub` — identity pass-through until real camera calibration (K, D matrices) is available
  - `line_intersect` — homogeneous-coordinate line intersection
  - `lateral_state` — classifies VP x-offset as `centered`/`drift_left`/`drift_right` (6% dead zone)
  - `draw_overlay` — renders red boundary lines, VP marker, white centerline arrow, state text

## Git Policy

All commits must use the repository owner's identity (`liulhs`). Do NOT include `Co-Authored-By` lines for Claude. Do NOT modify git config. The goal is that only `liulhs` appears as a contributor.

## Key Design Decisions

- All frames are resized to 640x480 before processing.
- Model outputs heatmap at 1/4 resolution (160×120). Peaks are scaled back to image coords with sub-pixel refinement.
- Contact points are classified left/right by x-position relative to image center (320px).
- VP is rejected if below 75% of image height or wildly off-center — prevents diverging-line false positives.
- Camera undistortion is stubbed out — `K` and `D` are not yet calibrated.
- Edge deployment uses ONNX + OpenCV DNN only (no PyTorch at runtime).
- Training uses ImageNet-pretrained MobileNetV2 backbone with early layers frozen to prevent overfitting on small datasets.

## Data Labeling

Use LabelMe (`pip install labelme`) to annotate tree-ground contact points:
- Open each image, use "Create Point" tool, click where each tree trunk meets the soil
- Label each point as `contact`
- LabelMe saves per-image JSON files beside the images
- The training pipeline discovers these automatically

## Plugins & MCP Servers

Three plugin servers are available. Use them proactively — don't wait for the user to ask.

### Context7 — Live Documentation Lookup

Use Context7 whenever working with OpenCV, NumPy, PyTorch, or any library API — even if you think you know the answer. Your training data may be stale.

- **Workflow**: Always call `resolve-library-id` first to get the Context7 library ID, then `query-docs` with that ID.
- **When to use**: Any OpenCV function usage (e.g., `cv2.dnn` API, `cv2.fitLine` flags), PyTorch model operations, NumPy array operations, Python stdlib edge cases, or if the user introduces a new dependency.
- **Limit**: Max 3 calls per question. If you can't find what you need after 3, use the best result.

### Notion — Project Documentation & Task Tracking

Full read/write access to the team's Notion workspace. Use it to document decisions, track work, and share results.

- **Search** (`notion-search`): Find existing pages, databases, or content across the workspace and connected sources (Slack, Google Drive, GitHub, Jira, Linear). Supports date and creator filters.
- **Fetch** (`notion-fetch`): Read full page/database content by URL or ID. Always fetch a database before creating pages in it (to get the schema and data source IDs).
- **Create pages** (`notion-create-pages`): Create documentation pages or database entries. Always include a title property. For database rows, match property names to the fetched schema.
- **Create databases** (`notion-create-database`): Create structured databases with SQL DDL schema syntax (e.g., task boards, test result trackers).
- **Create views** (`notion-create-view`): Add table/board/calendar/chart views to databases. Fetch the database first to get `database_id` and `data_source_id`.
- **Update pages** (`notion-update-page`): Edit page properties or content. Always fetch the page first to get current content for accurate find-and-replace.
- **Update schemas** (`notion-update-data-source`): Add/remove/rename columns in database schemas using SQL DDL.
- **Update views** (`notion-update-view`): Modify view filters, sorts, grouping, or display settings.
- **Comments** (`notion-create-comment`, `notion-get-comments`): Add or read discussion threads on pages.
- **Users & Teams** (`notion-get-users`, `notion-get-teams`): Look up workspace members and teamspaces.
- **Move/Duplicate** (`notion-move-pages`, `notion-duplicate-page`): Reorganize or clone pages.

**Key patterns**:
- Always `fetch` before `update` — you need the current content for accurate edits.
- For databases with multiple data sources, use `data_source_id` (from `collection://` URLs in fetch output), not `database_id`.
- For the Notion Markdown spec, fetch the MCP resource at `notion://docs/enhanced-markdown-spec` before writing rich content.

### GitHub — Authenticated GitHub Access

Provides authenticated access to GitHub via OAuth. Must call `authenticate` first to start the OAuth flow, then `complete_authentication` after the user authorizes in their browser. Once authenticated, GitHub-specific tools become available.

- **When to use**: For GitHub operations that need authentication beyond what the `gh` CLI provides, or when `gh` is not configured.
- **Prefer `gh` CLI** (via Bash) for standard operations like `gh pr view`, `gh issue view`, `gh api` — it's faster and doesn't require the OAuth dance.
