import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "final1.tflite"
HAND_LANDMARKER_PATH = BASE_DIR / "hand_landmarker.task"

# DB_PATH can be overridden via environment variable for Docker deployments
# e.g. DB_PATH=/data/palmprint.db → mount a named volume at /data
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "palmprint.db")))

SIMILARITY_THRESHOLD = 0.75
DUPLICATE_THRESHOLD  = 0.75   # block registration if palm already matches at this level
REGISTRATION_CAPTURES = 5

IMG_SIZE = (224, 224)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
