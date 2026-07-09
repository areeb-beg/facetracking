# Face Detection — School Reception Robot

Real-time face detection using a webcam, built for a school reception robot system. Detects multiple faces, draws bounding boxes, and displays confidence scores.

## Project Overview

This project provides a lightweight, single-script face detection module for a robot stationed at a school reception. The robot uses a webcam to detect visitors' faces as they approach, forming the first layer of perception before any downstream tasks (e.g., greeting, attendance, or routing).

The system runs at interactive frame rates on modest hardware and is designed to be integrated into a larger robotics pipeline.

## Features

- **Real-time detection** — processes each webcam frame with minimal latency
- **Multiple face support** — detects and annotates all visible faces simultaneously
- **Confidence display** — shows detection confidence percentage above each bounding box
- **Mirrored feed** — horizontally flipped for natural interaction
- **Short-range indoor model** — optimised for desk / reception distances via `blaze_face_short_range.tflite`
- **Clean exit** — press `q` to quit gracefully

## Installation

### Prerequisites

- Python 3.10+
- A webcam connected to the system

### Setup

```bash
# Clone or enter the project directory
cd face-traking

# (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe
```

The face detection model (`blaze_face_short_range.tflite`) downloads automatically on first run.

## How to Run

```bash
python face_detection.py
```

A window titled **School Reception Robot - Face Detection** will open showing the webcam feed. Detected faces are outlined in green with confidence percentages.

- Press **`q`** to exit.

## Technologies Used

| Library       | Version | Purpose                         |
|---------------|---------|---------------------------------|
| OpenCV        | 5.x     | Video capture, frame processing, display |
| MediaPipe     | 0.10.x  | Face detection via tasks API    |
| NumPy         | 2.x     | Array / image data handling     |
| Python        | 3.10+   | Runtime                         |

## Project Structure

```
face-traking/
├── face_detection.py          # Main entry point
├── blaze_face_short_range.tflite  # Detection model (auto-downloaded)
└── README.md                  # This file
```

## Future Improvements

- Add person counting and logging for reception analytics
- Integrate with a greeting or registration system
- Support additional camera types (IP / USB / Raspberry Pi camera module)
- Add config file for confidence threshold, resolution, and model selection
- Package as a reusable Python module for the broader robot software stack


## License

MIT License — see the [LICENSE](LICENSE) file for details.
