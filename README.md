# Screw Detection

Computer vision system using RF-DETR and YOLO models for screw detection and assembly line poka-yoke (error-proofing). Includes image capture, inference CLI, and operator GUI.

## Setup

```bash
pip install opencv-python ultralytics rfdetr pillow numpy pyside6
```

## Usage

### Capture Images

```bash
python img_capture.py             # webcam
python img_capture.py --stereo    # stereo camera (3840x1080, uses left half)
```

| Key | Saves to |
|-----|----------|
| `1` | `images/2x6/` |
| `2` | `images/3x8/` |
| `3` | `images/3x8loctite/` |
| `4` | `images/all_screws/` |
| `5` | `images/2x6and3x8/` |
| `c` | `images/test/` |
| `q` | Quit |

A **lighting score (0–100)** is shown live — aim for green (≥60) before capturing.
A red rectangle marks the **704×704 center crop** region used by the model.

### Run Inference

```bash
# Single image
python run_inference.py --image path/to/image.jpg

# Live camera
python run_inference.py --live
python run_inference.py --live --stereo
```

Select a model at startup, then use the window controls:

| Key | Action |
|-----|--------|
| `b` | Toggle bounding boxes |
| `[` / `]` | Decrease / Increase confidence threshold |
| `,` / `.` | Decrease / Increase IoU threshold |
| `q` | Quit |

Bounding box color indicates confidence: blue (low) → green (mid) → red (high).

### Assembly Line Operator GUI

```bash
python run_gui.py
```

**Workflow:**
1. Enter serial #, hardware revision, operator name, checked-by name
2. Click **Count Screws** → camera confirms 28 screws on desk (5 stable frames required)
3. Install screws one by one → live count shows progress
4. Camera auto-detects when all screws are removed (5 stable frames of 0 screws)
5. Click **Submit** → build data logged

**Controls:**
- **Bypass Camera Check** — skip count verification (for testing)
- **Progress bar** — shows screws installed (0 → 28) during the build
- Camera feed shows live count and red crop outline

---

## Models

Stored in `models/`. RF-DETR optimized for inference; YOLO for comparison. Currently trained to detect screws in general (screw type classification planned for future model).

| Model | Type | mAP | Dataset |
|-------|------|-----|---------|
| `rfdetr_96_108.pt` | RF-DETR | 96 | 108 images |
| `rfdetr_65_200.pt` | RF-DETR | 65 | 200 images |
| `yolo26_94_108.pt` | YOLO | 94 | 108 images |
| `yolo26_44_200.pt` | YOLO | 44 | 200 images |

## Project Structure

```
detect_screws/
├── img_capture.py       # Image capture tool
├── run_inference.py     # Inference tool (CLI)
├── run_gui.py           # Assembly line operator GUI (PySide6)
├── models/              # Trained model weights (.pt)
└── images/              # Captured image datasets
    ├── 2x6/
    ├── 3x8/
    ├── 3x8loctite/
    ├── 2x6and3x8/
    ├── all_screws/
    └── test/
```
