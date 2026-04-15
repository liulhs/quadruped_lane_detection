# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Vision-based aisle navigation for a quadruped robot in row-crop orchards. Detects tree-row boundaries from a forward-facing camera, computes a vanishing point (VP), and determines lateral drift relative to the aisle centerline — enabling steering corrections without GPS or LiDAR. All processing uses classical OpenCV (no neural networks) to run on edge hardware (RPi4 / Jetson Nano).

## Commands

```bash
# Run a single detector on one image
python detect_option_a.py aisle_3.jpg        # Hough-based, saves to output/optionA/
python detect_option_b.py aisle_3.jpg        # HSV+RANSAC, saves to output/optionB/

# Benchmark both detectors across all test images (FPS + accuracy)
python benchmark.py                          # both options, 20 reps
python benchmark.py --option a --reps 50     # option A only, 50 reps for stable FPS
```

No build step, no package manager, no tests. Dependencies: Python 3, OpenCV (`cv2`), NumPy.

## Architecture

Two competing detection pipelines share a common interface and utility layer:

- **`detect_option_a.py`** — Hough Line Transform. Upper-ROI-first strategy: runs Hough on the full ROI with long `minLineLength`, prefers segments near the horizon (less clutter), falls back to shorter segments per-side. Two-pass approach (primary + fallback) with slope filtering to reject drip lines and fence posts.

- **`detect_option_b.py`** — HSV soil segmentation + boundary fitting. Thresholds sandy-soil color in HSV, isolates the aisle via connected components (seeds from bottom-center), scans rows for left/right boundary points, fits lines with DIST_HUBER. Includes temporal smoothing (carries previous frame's lines) and symmetry fallback (mirrors one side if the other is missing).

- **`utils.py`** — Shared functions used by both detectors:
  - `undistort_stub` — identity pass-through until real camera calibration (K, D matrices) is available
  - `line_intersect` — homogeneous-coordinate line intersection
  - `lateral_state` — classifies VP x-offset as `centered`/`drift_left`/`drift_right` (6% dead zone)
  - `draw_overlay` — renders red boundary lines, VP marker, white centerline arrow, state text

Both detectors expose `detect(frame, K=None, D=None) -> (vp, state, overlay, left_line, right_line)` with the same return signature.

## Git Policy

All commits must use the repository owner's identity (`liulhs`). Do NOT include `Co-Authored-By` lines for Claude. Do NOT modify git config. The goal is that only `liulhs` appears as a contributor.

## Key Design Decisions

- All frames are resized to 640x480 before processing. The top 1/3 is discarded as sky/canopy ROI.
- VP is rejected if below 75% of image height or wildly off-center — prevents diverging-line false positives.
- Option A uses dx/dy slope convention (not dy/dx) so negative slope = left row, positive = right row.
- Option B's HSV soil range (`H:5-35, S:10-140, V:90-255`) is calibrated for sandy orchard soil in direct sunlight.
- Camera undistortion is stubbed out — `K` and `D` are not yet calibrated. First roadmap item.

## Plugins & MCP Servers

Three plugin servers are available. Use them proactively — don't wait for the user to ask.

### Context7 — Live Documentation Lookup

Use Context7 whenever working with OpenCV, NumPy, or any library API — even if you think you know the answer. Your training data may be stale.

- **Workflow**: Always call `resolve-library-id` first to get the Context7 library ID, then `query-docs` with that ID.
- **When to use**: Any OpenCV function usage (e.g., `cv2.HoughLinesP` parameters, `cv2.fitLine` flags), NumPy array operations, Python stdlib edge cases, or if the user introduces a new dependency.
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
