"""Singletons : detecteur YOLO + pipeline RAG + Neo4j."""

from functools import lru_cache

from phase2_vision.board_detector import BoardGameDetector
from phase1_ludii_rag.ludii_nlp_pipeline import LudiiNLPPipeline
from phase4_neo4j.pipeline import Neo4jPipeline
from phase4_neo4j.analytics import GameAnalytics


@lru_cache(maxsize=1)
def get_detector() -> BoardGameDetector:
    return BoardGameDetector()


@lru_cache(maxsize=1)
def get_pipeline() -> LudiiNLPPipeline:
    return LudiiNLPPipeline()


@lru_cache(maxsize=1)
def get_neo4j_pipeline() -> Neo4jPipeline:
    return Neo4jPipeline()


@lru_cache(maxsize=1)
def get_analytics() -> GameAnalytics:
    return GameAnalytics()