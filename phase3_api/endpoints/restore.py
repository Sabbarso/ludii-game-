"""Endpoint /restore : restauration d'une regle endommagee."""

from fastapi import APIRouter, Depends, HTTPException

from phase1_ludii_rag.ludii_nlp_pipeline import LudiiNLPPipeline
from phase3_api.dependencies import get_pipeline
from phase3_api.schemas import RuleRestoreRequest, RuleRestoreResponse, RetrievedRule

router = APIRouter()


@router.post("/restore", response_model=RuleRestoreResponse)
async def restore_rule(
    request: RuleRestoreRequest,
    pipeline: LudiiNLPPipeline = Depends(get_pipeline),
):
    try:
        result = pipeline.restore_rule(damaged_rule=request.damaged_rule, game=request.game)
        return RuleRestoreResponse(
            original_damaged=result["original_damaged"],
            restored=result["restored"],
            confidence=result["confidence"],
            retrieved_context=[RetrievedRule(**c) for c in result["retrieved_context"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur restauration: {e}")