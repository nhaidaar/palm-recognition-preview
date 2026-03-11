from fastapi import APIRouter

router = APIRouter()


@router.get("/api/logs/count")
async def get_logs_count():
    from app.main import db
    return {"count": db.count_access_logs()}


@router.get("/api/logs")
async def get_logs(limit: int = 20, offset: int = 0):
    from app.main import db
    return db.get_access_logs(limit=limit, offset=offset)
