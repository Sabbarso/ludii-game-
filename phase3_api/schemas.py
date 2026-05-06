"""Schemas Pydantic pour validation request/response."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class MissingPiece(BaseModel):
    piece: str
    expected: int
    found: int
    missing: int


class DetectResponse(BaseModel):
    game_type: str
    confidence: float
    pieces: List[Dict]
    count_by_class: Dict[str, int]
    total_pieces: int
    missing_pieces: List[MissingPiece]
    is_complete: bool


class RuleQuery(BaseModel):
    query: str = Field(..., min_length=1)
    game: Optional[str] = None
    top_k: int = Field(3, ge=1, le=10)


class RetrievedRule(BaseModel):
    text: str
    game: Optional[str] = None
    section: Optional[str] = None
    similarity: float


class RuleSearchResponse(BaseModel):
    query: str
    results: List[RetrievedRule]
    n_results: int


class RuleRestoreRequest(BaseModel):
    damaged_rule: str = Field(..., min_length=1)
    game: Optional[str] = None


class RuleRestoreResponse(BaseModel):
    original_damaged: str
    restored: str
    confidence: float
    retrieved_context: List[RetrievedRule]


class AnalyzeResponse(BaseModel):
    vision: DetectResponse
    rules_summary: List[RetrievedRule] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]
    supported_games: List[str]