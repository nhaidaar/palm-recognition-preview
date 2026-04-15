from fastapi import APIRouter

from app.config import DB_PATH

router = APIRouter()


@router.get("/api/status")
async def status():
    from app.main import db

    row = db.get_device_status() if db is not None else None
    return {
        "app": {"mode": "hybrid", "version": "local"},
        "database": {"path": str(DB_PATH)},
        "device": row or {
            "worker_state": "disabled",
            "camera_connected": 0,
            "last_error": None,
            "fps": None,
            "last_inference_ms": None,
            "last_recognition_at": None,
        },
    }
