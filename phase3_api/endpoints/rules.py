"""Endpoint /rules : recherche RAG."""

from fastapi import APIRouter, Depends, HTTPException

from phase1_ludii_rag.ludii_nlp_pipeline import LudiiNLPPipeline
from phase3_api.dependencies import get_pipeline
from phase3_api.schemas import RuleQuery, RuleSearchResponse, RetrievedRule

router = APIRouter()


@router.post("/rules", response_model=RuleSearchResponse)
async def search_rules(
    request: RuleQuery,
    pipeline: LudiiNLPPipeline = Depends(get_pipeline),
):
    try:
        retrieved = pipeline.rag.retrieve_similar_rules(
            query=request.query, top_k=request.top_k, game_filter=request.game,
        )
        results = [RetrievedRule(**r) for r in retrieved]
        return RuleSearchResponse(query=request.query, results=results, n_results=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur RAG: {e}")