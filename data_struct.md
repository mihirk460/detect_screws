# Dataset Documentation

## V2 Dataset Structure

### Capture Plan
For the V2 dataset, **20 images per folder** (2x6, 3x8, 3x8loctite, all_screws, 2x6and3x8, etc.) will be captured with the following breakdown:

- **10 images in lower light conditions**
- **10 images in brighter light conditions**

### Image Composition per Light Condition (10 images each)
From each set of 10 images (per light condition):
- **3 images**: Single screw
- **3 images**: 2 screws placed closely together
- **4 images**: Multiple screws together

### Total Per Folder
- **2x6 folder**: 20 images
  - Lower light: 10 images (3 single + 3 paired + 4 multiple)
  - Brighter light: 10 images (3 single + 3 paired + 4 multiple)
- Same structure for 3x8, 3x8loctite, all_screws, 2x6and3x8, and any other screw type folders

### File Hierarchy

```
Matic_Stage_3/
├── images/
│   ├── dataset.md                    (this file)
│   ├── 2x6/
│   │   ├── img_lower_light_single_001.jpg
│   │   ├── img_lower_light_single_002.jpg
│   │   ├── img_lower_light_single_003.jpg
│   │   ├── img_lower_light_paired_001.jpg
│   │   ├── img_lower_light_paired_002.jpg
│   │   ├── img_lower_light_paired_003.jpg
│   │   ├── img_lower_light_multiple_001.jpg
│   │   ├── img_lower_light_multiple_002.jpg
│   │   ├── img_lower_light_multiple_003.jpg
│   │   ├── img_lower_light_multiple_004.jpg
│   │   ├── img_bright_light_single_001.jpg
│   │   ├── img_bright_light_single_002.jpg
│   │   ├── img_bright_light_single_003.jpg
│   │   ├── img_bright_light_paired_001.jpg
│   │   ├── img_bright_light_paired_002.jpg
│   │   ├── img_bright_light_paired_003.jpg
│   │   ├── img_bright_light_multiple_001.jpg
│   │   ├── img_bright_light_multiple_002.jpg
│   │   ├── img_bright_light_multiple_003.jpg
│   │   └── img_bright_light_multiple_004.jpg
│   ├── 3x8/
│   │   └── (same structure as 2x6 - 20 images)
│   ├── 3x8loctite/
│   │   └── (same structure as 2x6 - 20 images)
│   ├── all_screws/
│   │   └── (same structure as 2x6 - 20 images)
│   ├── 2x6and3x8/
│   │   └── (same structure as 2x6 - 20 images)
│   └── test/
│       └── (test images)
│
├── image_capture/
│   ├── main.py
│   ├── find_focus.py
│   └── cam_settings.yaml
│
├── run_inference/
│   ├── test_rfdetr.py
│   ├── models/
│   └── screenshots/
│
└── screw_detection/
    └── (screw detection implementation)
```

### Capture Notes
- Use the **704x704 red outline** in both `image_capture/main.py` and `test_rfdetr.py` (video mode) to ensure proper framing
- Adjust lighting between lower and brighter conditions using the **Lighting Scene Presets** (keys 7/8/9 to load, Shift+7/8/9 to save)
- Capture images using the stereo camera (3840x1080) which extracts the left 1920x1080 view
- Use the appropriate folder keys to save images to corresponding screw type folders
