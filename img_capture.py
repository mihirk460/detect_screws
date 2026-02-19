# Captures images from camera and saves them to labeled subfolders

import cv2
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Image Capture Tool')
parser.add_argument('--stereo', action='store_true', help='Stereo camera mode (3840x1080, uses left camera)')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

images_dir = os.path.join(SCRIPT_DIR, 'images')
folder_map = {
    '1': os.path.join(images_dir, '2x6'),
    '2': os.path.join(images_dir, '3x8'),
    '3': os.path.join(images_dir, '3x8loctite'),
    '4': os.path.join(images_dir, 'all_screws'),
    '5': os.path.join(images_dir, '2x6and3x8'),
    'c': os.path.join(images_dir, 'test'),
}
for folder in folder_map.values():
    os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

if args.stereo:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("Stereo camera (3840x1080) connected — using left view")
else:
    print("Webcam connected")

print("\nCapture:")
print("  1 : 2x6       2 : 3x8       3 : 3x8loctite")
print("  4 : all_screws  5 : 2x6and3x8  c : test")
print("  q : Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    if args.stereo:
        left_frame = frame[:, :1920]
        display_frame = left_frame.copy()
    else:
        display_frame = frame.copy()
        left_frame = None

    # Lighting score: grayscale mean measures overall brightness (0=black, 255=white),
    # std dev measures contrast/texture. Score peaks when mean is near 128 (mid-exposure)
    # and std dev is high (scene has detail). Weighted 60/40 brightness/contrast.
    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    mean, std = gray.mean(), gray.std()
    brightness_score = max(0.0, 1.0 - abs(mean - 128) / 128)
    contrast_score = min(std / 60.0, 1.0)
    score = int(brightness_score * 0.6 * 100 + contrast_score * 0.4 * 100)

    score_color = (0, 255, 0) if score >= 60 else (0, 165, 255) if score >= 35 else (0, 0, 255)

    # Draw 704x704 center crop outline
    h, w = display_frame.shape[:2]
    crop = 704
    x1, y1 = (w - crop) // 2, (h - crop) // 2
    cv2.rectangle(display_frame, (x1, y1), (x1 + crop, y1 + crop), (0, 0, 255), 2)

    cv2.putText(display_frame, f"Lighting: {score}/100", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

    cv2.imshow('Image Capture', display_frame)
    key = cv2.waitKey(1) & 0xFF

    if chr(key) in folder_map:
        folder = folder_map[chr(key)]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_frame = (left_frame if args.stereo else frame).copy()
        filename = os.path.join(folder, f"img_{timestamp}.jpg")
        cv2.imwrite(filename, save_frame)
        print(f"Saved: {filename}")
    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Program terminated successfully")
