# PALMGATE — Palmprint Recognition Preview

Web-based preview of a smart door lock with palmprint recognition.

## Requirements

- Python 3.10+
- `final1.tflite` in the project root
- `hand_landmarker.task` in the project root (auto-downloaded on first run)

## Setup

```bash
pip install -r requirements.txt
```

If `hand_landmarker.task` is missing:

```python
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    'hand_landmarker.task'
)
```

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Features

- **Scan Palm** — Live webcam + palm recognition with ALLOWED/DENIED status
- **Register** — Capture 5 palm samples to enroll a new user
- **Access Log** — Full timestamped history of scan attempts

## How it works

1. MediaPipe Hands detects your palm and extracts a square ROI
2. CLAHE enhances the ROI, resized to 224×224
3. `final1.tflite` (EfficientNetB0) generates a 1280-dim embedding
4. Cosine similarity compares against registered user embeddings
5. Result: ALLOWED (≥ 0.65 similarity) or DENIED

## Tests

```bash
python -m pytest tests/ -v
```
