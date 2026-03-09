from fastapi import APIRouter

router = APIRouter()


@router.get("/api/logs")
async def get_logs(limit: int = 50):
    from app.main import db
    return db.get_access_logs(limit=limit)
