from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class UserOut(BaseModel):
    id: int
    name: str
    created_at: str


@router.get("/api/users", response_model=list)
async def list_users():
    from app.main import db
    return db.get_all_users()


@router.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    from app.main import db
    if not db.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True}
