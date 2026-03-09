# Palmprint Recognition Preview — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web-based palmprint recognition preview with live webcam, user registration, and access logging using FastAPI + MediaPipe + TFLite.

**Architecture:** FastAPI serves both the REST API and static frontend files. The palm processor uses MediaPipe Hands for hand detection, extracts a palm ROI from landmarks, enhances it with CLAHE, and feeds it to `final1.tflite` to get a 1280-dim embedding. Cosine similarity compares embeddings for recognition. SQLite stores user embeddings and access logs.

**Tech Stack:** FastAPI, uvicorn, MediaPipe, OpenCV, TFLite Runtime, SQLite, vanilla HTML/CSS/JS with WebRTC.

**Design Doc:** `docs/plans/2026-03-08-palmprint-preview-design.md`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `app/main.py`

**Step 1: Create `requirements.txt`**

```
fastapi==0.115.*
uvicorn[standard]==0.34.*
mediapipe==0.10.*
opencv-python-headless==4.11.*
numpy>=1.26,<2.0
tflite-runtime==2.16.*
python-multipart==0.0.*
scikit-learn==1.6.*
```

> Note: If `tflite-runtime` is not available for your platform, replace with `tensorflow>=2.16,<2.17` and use `tf.lite.Interpreter` instead.

**Step 2: Create `app/config.py`**

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "final1.tflite"
DB_PATH = BASE_DIR / "palmprint.db"

SIMILARITY_THRESHOLD = 0.75
REGISTRATION_CAPTURES = 5

IMG_SIZE = (224, 224)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
```

**Step 3: Create `app/main.py` skeleton**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Palmprint Recognition Preview")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
```

**Step 4: Create `app/__init__.py`**

Empty file.

**Step 5: Install dependencies**

Run: `pip install -r requirements.txt`

**Step 6: Verify server starts**

Run: `uvicorn app.main:app --reload`
Expected: Server starts on `http://127.0.0.1:8000`, no import errors.

**Step 7: Commit**

```bash
git init
git add requirements.txt app/__init__.py app/config.py app/main.py
git commit -m "feat: project scaffolding with FastAPI + dependencies"
```

---

### Task 2: Database Layer

**Files:**
- Create: `app/database.py`
- Create: `tests/__init__.py`
- Create: `tests/test_database.py`

**Step 1: Write failing tests for database**

Create `tests/test_database.py`:

```python
import os
import tempfile
import numpy as np
import pytest
from app.database import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    yield database
    database.close()
    os.unlink(path)


def test_add_user(db):
    embedding = np.random.rand(1280).astype(np.float32)
    user_id = db.add_user("TestUser", embedding)
    assert user_id is not None
    assert user_id > 0


def test_get_all_users(db):
    emb1 = np.random.rand(1280).astype(np.float32)
    emb2 = np.random.rand(1280).astype(np.float32)
    db.add_user("Alice", emb1)
    db.add_user("Bob", emb2)
    users = db.get_all_users()
    assert len(users) == 2
    assert users[0]["name"] == "Alice"
    assert users[1]["name"] == "Bob"


def test_get_all_embeddings(db):
    emb = np.random.rand(1280).astype(np.float32)
    db.add_user("Alice", emb)
    embeddings = db.get_all_embeddings()
    assert len(embeddings) == 1
    assert embeddings[0]["name"] == "Alice"
    np.testing.assert_array_almost_equal(embeddings[0]["embedding"], emb, decimal=5)


def test_delete_user(db):
    emb = np.random.rand(1280).astype(np.float32)
    user_id = db.add_user("ToDelete", emb)
    assert db.delete_user(user_id) is True
    assert len(db.get_all_users()) == 0


def test_add_access_log(db):
    db.add_access_log(user_id=None, matched_name="Unknown", status="DENIED", similarity=0.3)
    logs = db.get_access_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["status"] == "DENIED"
    assert logs[0]["matched_name"] == "Unknown"


def test_get_access_logs_ordered(db):
    db.add_access_log(None, "First", "DENIED", 0.1)
    db.add_access_log(None, "Second", "DENIED", 0.2)
    logs = db.get_access_logs(limit=10)
    assert logs[0]["matched_name"] == "Second"  # most recent first
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_database.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.database'`

**Step 3: Implement `app/database.py`**

```python
import sqlite3
import numpy as np
from pathlib import Path


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS access_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER,
                matched_name    TEXT NOT NULL,
                status          TEXT NOT NULL,
                similarity      REAL NOT NULL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        self.conn.commit()

    def add_user(self, name: str, embedding: np.ndarray) -> int:
        cursor = self.conn.execute(
            "INSERT INTO users (name, embedding) VALUES (?, ?)",
            (name, embedding.tobytes()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_all_users(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_embeddings(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, embedding FROM users ORDER BY id"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
            }
            for r in rows
        ]

    def delete_user(self, user_id: int) -> bool:
        cursor = self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def add_access_log(
        self,
        user_id: int | None,
        matched_name: str,
        status: str,
        similarity: float,
    ):
        self.conn.execute(
            "INSERT INTO access_logs (user_id, matched_name, status, similarity) VALUES (?, ?, ?, ?)",
            (user_id, matched_name, status, similarity),
        )
        self.conn.commit()

    def get_access_logs(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, user_id, matched_name, status, similarity, timestamp FROM access_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_database.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add app/database.py tests/
git commit -m "feat: SQLite database layer with user + access log CRUD"
```

---

### Task 3: Palm Processor — MediaPipe Hand Detection + ROI Extraction

**Files:**
- Create: `app/palm_processor.py`
- Create: `tests/test_palm_processor.py`

**Step 1: Write failing test for hand detection + ROI**

Create `tests/test_palm_processor.py`:

```python
import numpy as np
import pytest
from app.palm_processor import PalmProcessor


@pytest.fixture
def processor():
    proc = PalmProcessor(model_path=None)  # skip TFLite loading for unit tests
    yield proc
    proc.close()


def test_extract_palm_roi_no_hand(processor):
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.extract_palm_roi(black_frame)
    assert result is None


def test_apply_clahe(processor):
    gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    enhanced = processor.apply_clahe(gray_img)
    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8


def test_preprocess_roi(processor):
    roi = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
    processed = processor.preprocess_roi(roi)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_palm_processor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.palm_processor'`

**Step 3: Implement palm detection + ROI in `app/palm_processor.py`**

```python
import cv2
import numpy as np
import mediapipe as mp
from app.config import IMG_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, MODEL_PATH


class PalmProcessor:
    def __init__(self, model_path=MODEL_PATH):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
        )
        self.interpreter = None
        self._input_index = None
        self._gap_output_index = None

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        self._input_index = input_details[0]["index"]

        tensor_details = self.interpreter.get_tensor_details()
        for t in tensor_details:
            if t["shape"].tolist() == [1, 1280]:
                self._gap_output_index = t["index"]
                break

        if self._gap_output_index is None:
            output_details = self.interpreter.get_output_details()
            self._gap_output_index = output_details[0]["index"]

    def extract_palm_roi(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        h, w, _ = frame_rgb.shape

        lm = hand.landmark
        wrist = lm[mp.solutions.hands.HandLandmark.WRIST]
        index_mcp = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = lm[mp.solutions.hands.HandLandmark.PINKY_MCP]
        middle_mcp = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]

        cx = int(middle_mcp.x * w)
        cy = int(((middle_mcp.y + wrist.y) / 2) * h)

        palm_width = abs(int((index_mcp.x - pinky_mcp.x) * w))
        roi_size = int(palm_width * 1.5)
        half = roi_size // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        return roi

    def apply_clahe(self, gray_img: np.ndarray) -> np.ndarray:
        return self.clahe.apply(gray_img)

    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        enhanced = self.apply_clahe(gray)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)

    def get_embedding(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        roi = self.extract_palm_roi(frame_rgb)
        if roi is None:
            return None

        processed = self.preprocess_roi(roi)
        return self._run_inference(processed)

    def _run_inference(self, processed: np.ndarray) -> np.ndarray:
        if self.interpreter is None:
            raise RuntimeError("TFLite model not loaded")

        input_data = np.expand_dims(processed, axis=0)
        self.interpreter.set_tensor(self._input_index, input_data)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self._gap_output_index)[0]
        return embedding.copy()

    def compute_similarity(
        self, embedding: np.ndarray, stored_embeddings: list[dict], threshold: float
    ) -> dict:
        if not stored_embeddings:
            return {
                "status": "DENIED",
                "name": "Unknown",
                "similarity": 0.0,
                "closest_match": None,
                "user_id": None,
            }

        best_score = -1.0
        best_match = None
        best_user_id = None

        for entry in stored_embeddings:
            stored = entry["embedding"]
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(stored)
            if norm_a == 0 or norm_b == 0:
                continue
            score = float(np.dot(embedding, stored) / (norm_a * norm_b))
            if score > best_score:
                best_score = score
                best_match = entry["name"]
                best_user_id = entry["id"]

        if best_score >= threshold:
            return {
                "status": "ALLOWED",
                "name": best_match,
                "similarity": round(best_score, 4),
                "closest_match": best_match,
                "user_id": best_user_id,
            }

        return {
            "status": "DENIED",
            "name": "Unknown",
            "similarity": round(best_score, 4),
            "closest_match": best_match,
            "user_id": None,
        }

    def close(self):
        self.hands.close()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_palm_processor.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add app/palm_processor.py tests/test_palm_processor.py
git commit -m "feat: palm processor with MediaPipe detection, CLAHE, TFLite embedding"
```

---

### Task 4: API Routes — Recognize + Register

**Files:**
- Create: `app/routes/__init__.py`
- Create: `app/routes/recognize.py`
- Create: `app/routes/register.py`
- Modify: `app/main.py` — add route includes and app state

**Step 1: Create `app/routes/__init__.py`**

Empty file.

**Step 2: Implement `app/routes/recognize.py`**

```python
import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import SIMILARITY_THRESHOLD

router = APIRouter()


class RecognizeRequest(BaseModel):
    image: str  # base64-encoded image


class RecognizeResponse(BaseModel):
    status: str
    name: str
    similarity: float
    closest_match: str | None = None


def decode_base64_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@router.post("/api/recognize", response_model=RecognizeResponse)
async def recognize(req: RecognizeRequest):
    from app.main import palm_processor, db

    try:
        frame = decode_base64_image(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    embedding = palm_processor.get_embedding(frame)
    if embedding is None:
        raise HTTPException(status_code=422, detail="No hand detected")

    stored = db.get_all_embeddings()
    result = palm_processor.compute_similarity(
        embedding, stored, SIMILARITY_THRESHOLD
    )

    db.add_access_log(
        user_id=result["user_id"],
        matched_name=result["name"] if result["status"] == "ALLOWED" else "Unknown",
        status=result["status"],
        similarity=result["similarity"],
    )

    return RecognizeResponse(**result)
```

**Step 3: Implement `app/routes/register.py`**

```python
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import REGISTRATION_CAPTURES
from app.routes.recognize import decode_base64_image

router = APIRouter()


class RegisterRequest(BaseModel):
    name: str
    images: list[str]  # list of base64-encoded images


class RegisterResponse(BaseModel):
    success: bool
    user_id: int
    name: str


@router.post("/api/register", response_model=RegisterResponse)
async def register(req: RegisterRequest):
    from app.main import palm_processor, db

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    if len(req.images) < REGISTRATION_CAPTURES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {REGISTRATION_CAPTURES} palm images",
        )

    embeddings = []
    for i, img_b64 in enumerate(req.images):
        try:
            frame = decode_base64_image(img_b64)
        except Exception:
            raise HTTPException(
                status_code=400, detail=f"Invalid image at index {i}"
            )

        emb = palm_processor.get_embedding(frame)
        if emb is None:
            raise HTTPException(
                status_code=422,
                detail=f"No hand detected in image {i + 1}",
            )
        embeddings.append(emb)

    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    user_id = db.add_user(req.name.strip(), avg_embedding)

    return RegisterResponse(success=True, user_id=user_id, name=req.name.strip())
```

**Step 4: Create `app/routes/users.py` and `app/routes/logs.py`**

`app/routes/users.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class UserOut(BaseModel):
    id: int
    name: str
    created_at: str


@router.get("/api/users", response_model=list[UserOut])
async def list_users():
    from app.main import db
    return db.get_all_users()


@router.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    from app.main import db
    if not db.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True}
```

`app/routes/logs.py`:

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class LogOut(BaseModel):
    id: int
    user_id: int | None
    matched_name: str
    status: str
    similarity: float
    timestamp: str


@router.get("/api/logs", response_model=list[LogOut])
async def get_logs(limit: int = 50):
    from app.main import db
    return db.get_access_logs(limit=limit)
```

**Step 5: Update `app/main.py` with routes and app state**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.config import MODEL_PATH, DB_PATH
from app.database import Database
from app.palm_processor import PalmProcessor
from app.routes import recognize, register, users, logs

db: Database = None
palm_processor: PalmProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, palm_processor
    db = Database(DB_PATH)
    palm_processor = PalmProcessor(MODEL_PATH)
    yield
    palm_processor.close()
    db.close()


app = FastAPI(title="Palmprint Recognition Preview", lifespan=lifespan)

app.include_router(recognize.router)
app.include_router(register.router)
app.include_router(users.router)
app.include_router(logs.router)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")
```

**Step 6: Verify server starts**

Run: `uvicorn app.main:app --reload`
Expected: Server starts, visit `http://127.0.0.1:8000/docs` to see auto-generated API documentation.

**Step 7: Commit**

```bash
git add app/routes/ app/main.py
git commit -m "feat: API routes for recognize, register, users, and logs"
```

---

### Task 5: Frontend — HTML Structure

**Files:**
- Create: `app/static/index.html`

**Step 1: Create `app/static/index.html`**

Build a single-page app with three tab panels: Scan Palm, Register, Access Log. The HTML structure includes:

- A navigation bar with three tab buttons
- Scan panel: video element for webcam, scan button, result display area
- Register panel: name input, video element, capture button with progress dots, register button
- Log panel: table with refresh button

> **Use the @frontend-design skill for styling and UX polish.**

**Step 2: Verify page loads**

Run: `uvicorn app.main:app --reload`
Visit: `http://127.0.0.1:8000/`
Expected: Page renders with all three tabs navigable.

**Step 3: Commit**

```bash
git add app/static/index.html
git commit -m "feat: frontend HTML structure with scan, register, and log panels"
```

---

### Task 6: Frontend — CSS Styling

**Files:**
- Create: `app/static/style.css`

**Step 1: Style the app**

Apply clean, modern styling:
- Dark theme (fits door-lock aesthetic)
- Green for ALLOWED, red for DENIED status panels
- Webcam video element centered with a dashed palm guide overlay
- Responsive layout
- Tab navigation with active state indicators

> **Use the @frontend-design skill for production-quality design.**

**Step 2: Verify styling**

Visit: `http://127.0.0.1:8000/`
Expected: Styled interface matching the design wireframes.

**Step 3: Commit**

```bash
git add app/static/style.css
git commit -m "feat: frontend CSS styling with dark theme"
```

---

### Task 7: Frontend — JavaScript (Webcam + API Integration)

**Files:**
- Create: `app/static/app.js`

**Step 1: Implement `app/static/app.js`**

The JavaScript handles:

1. **Tab navigation** — show/hide panels, maintain webcam stream across tabs
2. **Webcam initialization** — `navigator.mediaDevices.getUserMedia` for live feed
3. **Scan Palm flow:**
   - On "Scan" click, capture frame from `<video>` to `<canvas>`
   - Convert to base64
   - POST to `/api/recognize`
   - Display result (ALLOWED/DENIED) with name and score
4. **Register flow:**
   - Capture 5 palm images one at a time (store base64 in array)
   - Update progress dots after each capture
   - On "Register" click (enabled at 5/5), POST all to `/api/register`
5. **Access Log flow:**
   - Fetch `/api/logs` on tab switch and on refresh click
   - Populate HTML table with results

**Step 2: Verify full flow**

1. Open `http://127.0.0.1:8000/`
2. Allow camera access
3. Navigate to Register tab, enter a name, capture 5 palms, register
4. Navigate to Scan tab, scan palm, verify recognition result
5. Navigate to Access Log tab, verify log entry appears

**Step 3: Commit**

```bash
git add app/static/app.js
git commit -m "feat: frontend JS with webcam, scan, register, and log integration"
```

---

### Task 8: End-to-End Verification

**Step 1: Start the server**

Run: `uvicorn app.main:app --reload`

**Step 2: Test full workflow**

1. Open browser to `http://127.0.0.1:8000/`
2. Register a user: go to Register tab, enter name, capture 5 palm images, click Register
3. Recognize the user: go to Scan tab, scan palm, verify ALLOWED status
4. Check logs: go to Access Log tab, verify entry
5. Test denial: cover camera or use different hand, scan, verify DENIED status

**Step 3: Test API docs**

Visit `http://127.0.0.1:8000/docs` — verify all endpoints are documented.

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete palmprint recognition preview MVP"
```
