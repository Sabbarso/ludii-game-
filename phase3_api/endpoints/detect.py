"""Endpoint /detect : image -> pieces detectees."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from phase2_vision.board_detector import BoardGameDetector
from phase3_api.dependencies import get_detector
from phase3_api.schemas import DetectResponse

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
async def detect_game(
    file: UploadFile = File(...),
    detector: BoardGameDetector = Depends(get_detector),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Fichier doit etre une image (recu: {file.content_type})")
    try:
        contents = await file.read()
        result = detector.full_analysis(contents)
        return DetectResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur detection: {e}")