# Real-Time AI Object Detection Demo

A Tkinter desktop application that streams live webcam footage through a **YOLOv8** model and overlays bounding boxes, labels, and confidence scores in real time.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## Features

- Live webcam object detection at up to ~60 FPS (hardware dependent)
- Switch between all five YOLOv8 model sizes (`n`, `s`, `m`, `l`, `x`) at runtime
- Adjustable confidence threshold slider (0.05 – 0.95)
- FPS counter and live detection log in the sidebar
- One-click annotated screenshot save
- Dark Catppuccin-inspired colour theme
- Fully decoupled detector module — easy to swap for a different backbone

---

## Requirements

- Python 3.10 or later
- A working webcam (device index `0`)
- ~6 MB (yolov8n) to ~130 MB (yolov8x) of disk space for model weights  
  *(downloaded automatically on first use)*

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/python-yolo-demo.git
cd python-yolo-demo

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python app.py
```

1. Select a model from the dropdown (`yolov8n` is fastest, `yolov8x` is most accurate).
2. Click **Start Camera** — the model is downloaded automatically on first use.
3. Adjust the **Confidence Threshold** slider to filter weak detections.
4. Click **Save Screenshot** to write the current annotated frame to a PNG.
5. Click **Stop Camera** or close the window to exit cleanly.

---

## Project Structure

```
python-yolo-demo/
├── app.py          # Tkinter GUI — App class, threading, frame consumer
├── detector.py     # ObjectDetector — YOLOv8 wrapper, inference, rendering
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── .github/
    ├── workflows/
    │   └── ci.yml
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── feature_request.md
```

---

## Configuration

All tuneable constants live at the top of [app.py](app.py):

| Constant   | Default | Description                          |
|------------|---------|--------------------------------------|
| `_VIDEO_W` | `800`   | Display width in pixels              |
| `_VIDEO_H` | `600`   | Display height in pixels             |
| `_CTRL_W`  | `230`   | Control panel width in pixels        |
| `_POLL_MS` | `15`    | GUI update interval (~67 FPS cap)    |
| `_Q_SIZE`  | `2`     | Frame queue depth (latency vs. drop) |

---

## Contributing

Pull requests are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

---

## License

[GPL-3.0 License](LICENSE) Copyright (c) 2026 Sean Morley. All rights reserved.
