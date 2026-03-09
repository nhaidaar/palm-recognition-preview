import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import REGISTRATION_CAPTURES, DUPLICATE_THRESHOLD
from app.routes.recognize import decode_base64_image

router = APIRouter()


class RegisterRequest(BaseModel):
    name: str
    images: list
    is_roi: bool = False  # True when the browser pre-cropped all palm ROIs


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
            raise HTTPException(status_code=400, detail=f"Invalid image at index {i}")

        if req.is_roi:
            emb = palm_processor.get_embedding_from_roi(frame)
        else:
            emb = palm_processor.get_embedding(frame)

        if emb is None:
            raise HTTPException(
                status_code=422,
                detail=f"No hand detected in image {i + 1}",
            )
        embeddings.append(emb)

    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

    # Reject if this palm is already registered under a different name
    stored = db.get_all_embeddings()
    dupe = palm_processor.compute_similarity(avg_embedding, stored, DUPLICATE_THRESHOLD)
    if dupe["status"] == "ALLOWED":
        raise HTTPException(
            status_code=409,
            detail=f"This palm is already registered as '{dupe['name']}' "
                   f"(similarity {dupe['similarity'] * 100:.0f}%). "
                   "Use a different palm or remove the existing user first.",
        )

    user_id = db.add_user(req.name.strip(), avg_embedding)
    return RegisterResponse(success=True, user_id=user_id, name=req.name.strip())
