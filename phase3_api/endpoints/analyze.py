"""Endpoint /analyze : pipeline complet image -> vision -> RAG."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from phase1_ludii_rag.ludii_nlp_pipeline import LudiiNLPPipeline
from phase2_vision.board_detector import BoardGameDetector
from phase3_api.dependencies import get_detector, get_pipeline
from phase3_api.schemas import AnalyzeResponse, DetectResponse, RetrievedRule

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    detector: BoardGameDetector = Depends(get_detector),
    pipeline: LudiiNLPPipeline = Depends(get_pipeline),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier doit etre une image")
    try:
        contents = await file.read()
        vision_result = detector.full_analysis(contents)
        vision_response = DetectResponse(**vision_result)

        rules_summary = []
        if vision_result["game_type"] != "unknown":
            retrieved = pipeline.rag.retrieve_similar_rules(
                query=f"Rules and how to play {vision_result['game_type']}",
                top_k=5,
                game_filter=vision_result["game_type"],
            )
            rules_summary = [RetrievedRule(**r) for r in retrieved]

        recommendations = _build_recommendations(vision_result)

        return AnalyzeResponse(
            vision=vision_response,
            rules_summary=rules_summary,
            recommendations=recommendations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur analyse: {e}")


def _build_recommendations(vision: dict) -> list:
    recs = []
    if vision["game_type"] == "unknown":
        recs.append("Aucun jeu reconnu - essaie une image plus claire.")
        return recs
    if vision["missing_pieces"]:
        n = sum(m["missing"] for m in vision["missing_pieces"])
        recs.append(f"{n} piece(s) manquante(s) detectee(s).")
        for missing in vision["missing_pieces"][:3]:
            recs.append(f"Il manque {missing['missing']} x {missing['piece']} (trouve {missing['found']}/{missing['expected']})")
    else:
        recs.append("Le jeu semble complet, pret a jouer!")
    if vision["confidence"] < 0.6:
        recs.append("Confiance faible - reessaie avec une meilleure photo.")
    return recs