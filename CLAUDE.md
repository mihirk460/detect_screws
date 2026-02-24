# CLAUDE.md

## Project Overview

Screw detection system using YOLO and RF-DETR models to identify screw types from camera images.

## Scripts

- `img_capture.py` — Capture images from camera and save to labeled folders. Run with `--stereo` for stereo camera (uses left 1920x1080 half of 3840x1080 frame).
- `run_inference.py` — Run detection inference. Use `--image <path>` for single image or `--live [--stereo]` for live camera feed.
- `run_gui.py` — PySide6 operator GUI for assembly line screw tracking. Confirms 28 screws on desk before build starts, confirms 0 screws after build completes. Run with `python run_gui.py`.

## Models

Stored in `models/`. Naming: `{type}_{mAP}_{dataset_size}.pt`

| File | Type | Input |
|---|---|---|
| `rfdetr_96_108.pt` | RF-DETR | 704px |
| `rfdetr_65_200.pt` | RF-DETR | 704px |
| `yolo26_94_108.pt` | YOLO | 640px |
| `yolo26_44_200.pt` | YOLO | 640px |

RF-DETR loaded via `RFDETRLarge(pretrain_weights=..., resolution=704, num_classes=2)` + `optimize_for_inference()`. YOLO loaded via `YOLO(path)`.

## Image Data

`images/` subfolders by screw type. Unversioned folders are used by `img_capture.py` at capture time; versioned `_v{n}` folders are for training datasets.

| Folder | Contents |
|---|---|
| `2x6/` | 2x6 screw images |
| `3x8/` | 3x8 screw images |
| `3x8loctite/` | 3x8 with Loctite |
| `2x6and3x8/` | Mixed 2x6 and 3x8 |
| `all_screws/` | All screw types |
| `test/` | Test images (not for training) |

**V2 dataset structure** (per `images/data_struct.md`): 20 images per folder — 10 lower light + 10 brighter light. Each set: 3 single + 3 paired + 4 multiple screws.

## Notes

- Inference crops the center square (704px for RFDETR, 640px for YOLO) from the input image/frame. A red outline marks this crop region in both scripts.
- `img_capture.py` saves raw frames (no post-processing); displays a lighting score (0–100) based on mean brightness and contrast.
