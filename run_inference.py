"""
Screw Detection Inference

Usage:
  python run_inference.py --image path/to/image.jpg
  python run_inference.py --live [--stereo]

Controls:
  b     Toggle bounding boxes
  [ ]   Decrease / Increase confidence threshold (0.05 steps)
  , .   Decrease / Increase IoU threshold (0.05 steps)
  q     Quit
"""

import cv2
import numpy as np
import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

MODELS = {
    1: {"name": "RFDETR 96 mAP (108 imgs)", "file": "rfdetr_96_108.pt", "type": "rfdetr", "input_size": 704},
    2: {"name": "RFDETR 65 mAP (200 imgs)", "file": "rfdetr_65_200.pt", "type": "rfdetr", "input_size": 704},
    3: {"name": "YOLO 94 mAP (108 imgs)",   "file": "yolo26_94_108.pt", "type": "yolo",   "input_size": 640},
    4: {"name": "YOLO 44 mAP (200 imgs)",   "file": "yolo26_44_200.pt", "type": "yolo",   "input_size": 640},
}

_RAW_CONF_FLOOR = 0.01  # Low floor so cached detections cover all threshold adjustments


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def select_model():
    print("\n" + "=" * 50)
    print("AVAILABLE MODELS:")
    for num, m in MODELS.items():
        print(f"  {num}. {m['name']}  ({m['type'].upper()}, {m['input_size']}px)")
    print("=" * 50)
    while True:
        try:
            choice = int(input("Select model (1-4): "))
            if choice in MODELS:
                print(f"Selected: {MODELS[choice]['name']}")
                return MODELS[choice]
            print("Enter a number from 1-4.")
        except ValueError:
            print("Enter a number.")


def load_model(config):
    path = os.path.join(MODELS_DIR, config["file"])
    if not os.path.isfile(path):
        print(f"Model not found: {path}")
        sys.exit(1)

    if config["type"] == "yolo":
        from ultralytics import YOLO
        model = YOLO(path)
        print(f"Loaded YOLO: {path}")
        return model

    from rfdetr import RFDETRLarge
    model = RFDETRLarge(pretrain_weights=path, resolution=config["input_size"], num_classes=2)
    model.optimize_for_inference()
    print(f"Loaded RFDETR: {path}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def center_crop(img, size):
    h, w = img.shape[:2]
    scale = size / min(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def run_raw(model, img, config):
    """Forward pass at low conf floor. Returns (N,5) array [x1,y1,x2,y2,conf]."""
    if config["type"] == "yolo":
        results = model.predict(
            source=img, conf=_RAW_CONF_FLOOR, iou=1.0,
            imgsz=config["input_size"], verbose=False,
        )
        if results and hasattr(results[0], "boxes"):
            b = results[0].boxes
            return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy().reshape(-1, 1)])
        return np.empty((0, 5), dtype=np.float32)

    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dets = model.predict(pil, threshold=_RAW_CONF_FLOOR)
    if dets is not None and len(dets) > 0:
        return np.hstack([dets.xyxy, dets.confidence.reshape(-1, 1)]).astype(np.float32)
    return np.empty((0, 5), dtype=np.float32)


def apply_nms(raw, conf_thresh, iou_thresh):
    """Filter cached raw detections by confidence then apply NMS. No re-inference needed."""
    if len(raw) == 0:
        return []
    filtered = raw[raw[:, 4] >= conf_thresh]
    if len(filtered) == 0:
        return []
    xywh = np.column_stack([
        filtered[:, 0], filtered[:, 1],
        filtered[:, 2] - filtered[:, 0],
        filtered[:, 3] - filtered[:, 1],
    ])
    indices = cv2.dnn.NMSBoxes(xywh.tolist(), filtered[:, 4].tolist(), conf_thresh, iou_thresh)
    if indices is None or len(indices) == 0:
        return []
    kept = filtered[np.asarray(indices).flatten()]
    return [{"x1": int(r[0]), "y1": int(r[1]), "x2": int(r[2]), "y2": int(r[3]), "conf": float(r[4])} for r in kept]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def conf_color(c):
    """Confidence -> BGR color: blue (low) -> green (mid) -> red (high)."""
    c = max(0.0, min(1.0, c))
    if c < 0.5:
        return (255, int(255 * c * 2), 0)
    return (0, int(255 * 2 * (1 - c)), int(255 * (c - 0.5) * 2))


def draw_frame(img, dets, show_boxes, conf, iou, model_name):
    out = img.copy()
    if show_boxes:
        for d in dets:
            color = conf_color(d["conf"])
            cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
            cv2.putText(out, f"{d['conf']:.0%}", (d["x1"], max(d["y1"] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    hud = [
        f"Model: {model_name}",
        f"Detections: {len(dets)}",
        f"Conf: {conf:.2f}  [/]",
        f"IoU:  {iou:.2f}  ,/.",
        f"Boxes: {'ON' if show_boxes else 'OFF'}  (b)",
        "Quit: q",
    ]
    for i, line in enumerate(hud):
        cv2.putText(out, line, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def image_mode(model, config, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    img = center_crop(img, config["input_size"])
    print(f"Cropped to {img.shape[1]}x{img.shape[0]}")

    print("Running inference...")
    raw = run_raw(model, img, config)
    print(f"Raw candidates: {len(raw)}")

    conf, iou = 0.50, 0.50
    dets = apply_nms(raw, conf, iou)
    show_boxes = True

    win = "Screw Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 900)

    while True:
        cv2.imshow(win, draw_frame(img, dets, show_boxes, conf, iou, config["name"]))
        key = cv2.waitKey(0) & 0xFF
        refilter = False

        if key == ord("q"):
            break
        elif key == ord("b"):
            show_boxes = not show_boxes
        elif key == ord("["):
            conf = round(max(0.0, conf - 0.05), 2); refilter = True
        elif key == ord("]"):
            conf = round(min(1.0, conf + 0.05), 2); refilter = True
        elif key == ord(","):
            iou = round(max(0.0, iou - 0.05), 2); refilter = True
        elif key == ord("."):
            iou = round(min(1.0, iou + 0.05), 2); refilter = True

        if refilter:
            dets = apply_nms(raw, conf, iou)
            print(f"Conf={conf:.2f}  IoU={iou:.2f}  Detections={len(dets)}")

    cv2.destroyAllWindows()
    print(f"Final: {len(dets)} detections (conf={conf:.2f}, iou={iou:.2f})")


def live_mode(model, config, stereo=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    if stereo:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print("Stereo camera connected — using left view")
    else:
        print("Webcam connected")

    conf, iou = 0.50, 0.50
    show_boxes = True
    size = config["input_size"]

    win = "Screw Detection — Live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        if stereo:
            frame = frame[:, :1920]

        # Crop region for inference, draw outline on full frame
        h, w = frame.shape[:2]
        cx1, cy1 = (w - size) // 2, (h - size) // 2
        crop = frame[cy1:cy1 + size, cx1:cx1 + size]

        raw = run_raw(model, crop, config)
        dets = apply_nms(raw, conf, iou)

        display = frame.copy()
        cv2.rectangle(display, (cx1, cy1), (cx1 + size, cy1 + size), (0, 0, 255), 2)

        if show_boxes:
            for d in dets:
                # Offset box coords from crop space to full frame space
                x1, y1 = d["x1"] + cx1, d["y1"] + cy1
                x2, y2 = d["x2"] + cx1, d["y2"] + cy1
                color = conf_color(d["conf"])
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{d['conf']:.0%}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        hud = [
            f"Model: {config['name']}",
            f"Detections: {len(dets)}",
            f"Conf: {conf:.2f}  [/]",
            f"IoU:  {iou:.2f}  ,/.",
            f"Boxes: {'ON' if show_boxes else 'OFF'}  (b)",
            "Quit: q",
        ]
        for i, line in enumerate(hud):
            cv2.putText(display, line, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("b"):
            show_boxes = not show_boxes
        elif key == ord("["):
            conf = round(max(0.0, conf - 0.05), 2)
            print(f"Conf={conf:.2f}")
        elif key == ord("]"):
            conf = round(min(1.0, conf + 0.05), 2)
            print(f"Conf={conf:.2f}")
        elif key == ord(","):
            iou = round(max(0.0, iou - 0.05), 2)
            print(f"IoU={iou:.2f}")
        elif key == ord("."):
            iou = round(min(1.0, iou + 0.05), 2)
            print(f"IoU={iou:.2f}")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Screw Detection Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="Run inference on a single image")
    group.add_argument("--live", action="store_true", help="Live camera feed inference")
    parser.add_argument("--stereo", action="store_true", help="Use stereo camera in live mode (3840x1080, left view)")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  SCREW DETECTION")
    print("=" * 50)

    config = select_model()
    print("\nLoading model...")
    model = load_model(config)

    if args.image:
        image_mode(model, config, args.image)
    else:
        live_mode(model, config, stereo=args.stereo)


if __name__ == "__main__":
    main()
