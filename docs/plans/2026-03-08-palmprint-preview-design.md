# Palmprint Recognition Preview — Design Document

**Date:** 2026-03-08
**Status:** Approved

## Purpose

Build a web-based preview/demo of a smart door lock with palmprint recognition. The system captures a user's palm via webcam, identifies them using a TFLite model, and shows an ALLOWED/DENIED status. It also supports registering new users and viewing access logs.

The preview runs on a PC; the final product targets a Raspberry Pi.

## Stack

- **Backend:** FastAPI (Python)
- **Frontend:** HTML/CSS/JavaScript (live webcam via WebRTC)
- **Model:** `final1.tflite` — EfficientNetB0 classifier, repurposed as an embedding extractor
- **Database:** SQLite
- **Hand Detection:** MediaPipe Hands

## Architecture

```
Browser (HTML/JS)              FastAPI Backend (Python)
┌──────────────┐               ┌────────────────────────────┐
│ WebRTC Camera│──base64 img──▶│ /api/recognize             │
│ + UI Screens │◀──JSON result─│ /api/register              │
│              │               │ /api/users, /api/logs      │
└──────────────┘               ├────────────────────────────┤
                               │ Palm Processing Pipeline   │
                               │ MediaPipe → ROI → CLAHE    │
                               │ → TFLite → 1280-dim embed  │
                               │ → Cosine Similarity        │
                               ├────────────────────────────┤
                               │ SQLite (palmprint.db)      │
                               └────────────────────────────┘
```

## Palm Processing Pipeline

1. **Hand Detection:** MediaPipe Hands detects 21 keypoints from the webcam frame.
2. **Palm ROI Extraction:** Use WRIST, INDEX_FINGER_MCP, and PINKY_MCP landmarks to define a square palm region.
3. **Enhancement:** Convert to grayscale, apply CLAHE (clipLimit=2.0, tileGridSize=8x8), convert back to 3-channel.
4. **Resize:** Scale to 224x224 pixels.
5. **Embedding Extraction:** Feed into `final1.tflite`. Read the GlobalAveragePooling2D output (1280-dim) instead of the softmax output.
6. **Matching:** Compute cosine similarity against all stored user embeddings.
7. **Decision:** If best similarity > threshold (default 0.75) → ALLOWED. Otherwise → DENIED.

### Accuracy Caveat

The model was trained with a different preprocessing pipeline (rembg background removal + FFT-based ROI extraction). Using MediaPipe-based ROI instead means accuracy may differ from notebook results. This is a known trade-off for real-time performance on Pi.

## Registration Flow

1. User enters their name.
2. Captures 5 palm frames via webcam (one at a time, pressing a capture button).
3. Each frame is processed through the pipeline to extract a 1280-dim embedding.
4. The 5 embeddings are averaged into a single stored embedding per user.
5. The averaged embedding and name are saved to SQLite.

## Database Schema

```sql
CREATE TABLE users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE access_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER,
    matched_name    TEXT NOT NULL,
    status          TEXT NOT NULL,        -- "ALLOWED" or "DENIED"
    similarity      REAL NOT NULL,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

## API Endpoints

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| POST | `/api/recognize` | `{ image: base64 }` | `{ status, name, similarity, closest_match }` |
| POST | `/api/register` | `{ name, images: [base64...] }` | `{ success, user_id, name }` |
| GET | `/api/users` | — | `[{ id, name, created_at }]` |
| DELETE | `/api/users/{id}` | — | `{ success }` |
| GET | `/api/logs` | `?limit=50` | `[{ timestamp, name, status, similarity }]` |

### Response Examples

**Recognized:**
```json
{ "status": "ALLOWED", "name": "Naufal", "similarity": 0.94 }
```

**Denied:**
```json
{ "status": "DENIED", "name": "Unknown", "similarity": 0.45, "closest_match": "Abi" }
```

**Error:**
```json
{ "error": "No hand detected" }
```

## UI Screens

### 1. Scan Palm (Home)
- Live webcam feed with a dashed palm guide overlay.
- "SCAN PALM" button captures a frame and sends to `/api/recognize`.
- Result panel shows ALLOWED (green) with user name, or DENIED (red) with closest match and score.

### 2. Register New User
- Text input for user name.
- Live webcam feed.
- "CAPTURE PALM" button — press 5 times. Progress shown as filled/empty dots (e.g., ◉◉◉○○).
- "REGISTER USER" button enabled only after 5 captures.

### 3. Access Log
- Table showing: timestamp, name, status (ALLOWED/DENIED), similarity score.
- Refresh button to reload.
- Sorted by most recent first.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| No hand detected | Return error — frontend shows positioning guidance |
| Multiple hands | Use highest-confidence hand |
| Palm too far/close | Warn via landmark spread analysis |
| Registration < 5 captures | Block register button |
| Duplicate user name | Allow (re-registration updates embedding) |
| No registered users | Recognize returns DENIED, closest_match: null |
| TFLite load failure | App fails to start with clear error |
| DB doesn't exist | Auto-create on startup |

## Project Structure

```
preview-palm/
├── final1.tflite
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── config.py                   # threshold, model path, DB path
│   ├── database.py                 # SQLite setup + CRUD
│   ├── palm_processor.py           # MediaPipe + ROI + CLAHE + TFLite
│   ├── routes/
│   │   ├── recognize.py
│   │   ├── register.py
│   │   ├── users.py
│   │   └── logs.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── palmprint.db                    # auto-created at runtime
├── requirements.txt
└── README.md
```

## Dependencies

- fastapi
- uvicorn
- mediapipe
- opencv-python-headless
- numpy
- tflite-runtime (or tensorflow-lite)
- python-multipart
- scikit-learn (for cosine similarity)
