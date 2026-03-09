import base64
import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class DebugRequest(BaseModel):
    image: str


@router.post("/api/debug/detect")
async def debug_detect(req: DebugRequest):
    """Returns hand detection result and image shape for debugging."""
    from app.main import palm_processor
    from app.routes.recognize import decode_base64_image

    try:
        frame = decode_base64_image(req.image)
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    result = {
        "image_shape": list(frame.shape),
        "image_dtype": str(frame.dtype),
        "hand_detected": False,
        "landmarks_count": 0,
        "roi_shape": None,
    }

    if palm_processor._hand_landmarker is None:
        result["error"] = "Hand landmarker not loaded"
        return result

    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection = palm_processor._hand_landmarker.detect(mp_image)

    if detection.hand_landmarks:
        result["hand_detected"] = True
        result["landmarks_count"] = len(detection.hand_landmarks[0])
        wrist = detection.hand_landmarks[0][0]
        result["wrist_xy"] = {"x": round(wrist.x, 3), "y": round(wrist.y, 3)}

        roi = palm_processor.extract_palm_roi(frame)
        if roi is not None:
            result["roi_shape"] = list(roi.shape)

    return result
