"""Endpoints pour Neo4j : analytics + sessions."""

from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from phase2_vision.board_detector import BoardGameDetector
from phase4_neo4j.pipeline import Neo4jPipeline
from phase4_neo4j.analytics import GameAnalytics
from phase3_api.dependencies import get_detector, get_neo4j_pipeline, get_analytics

router = APIRouter()


@router.get("/graph/games")
async def list_games(analytics: GameAnalytics = Depends(get_analytics)):
    """Liste tous les jeux avec leurs metadata."""
    try:
        return {"games": analytics.list_all_games()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Neo4j: {e}")


@router.get("/graph/games/{game_name}/rules")
async def game_rules(game_name: str, analytics: GameAnalytics = Depends(get_analytics)):
    """Regles d'un jeu groupees par categorie."""
    try:
        return {"game": game_name, "categories": analytics.rules_by_game_and_category(game_name)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Neo4j: {e}")


@router.get("/graph/missing-pieces/top")
async def top_missing_pieces(top_n: int = 5, analytics: GameAnalytics = Depends(get_analytics)):
    """Top N des pieces les plus souvent manquantes."""
    try:
        return {"top_missing": analytics.most_missing_pieces(top_n)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Neo4j: {e}")


@router.get("/graph/sessions")
async def recent_sessions(limit: int = 10, analytics: GameAnalytics = Depends(get_analytics)):
    """Sessions recentes."""
    try:
        return {"sessions": analytics.recent_sessions(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Neo4j: {e}")


@router.get("/graph/shared-mechanics")
async def shared_mechanics(analytics: GameAnalytics = Depends(get_analytics)):
    """Mecaniques partagees entre jeux."""
    try:
        return {"shared": analytics.shared_mechanics()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Neo4j: {e}")


@router.get("/graph/origins")
async def games_by_origin(analytics: GameAnalytics = Depends(get_analytics)):
    """Jeux groupes par pays d'origine."""
    try:
        return {"origins": analytics.games_by_origin()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Neo4j: {e}")


@router.get("/graph/pieces/{game_name}")
async def piece_values(game_name: str, analytics: GameAnalytics = Depends(get_analytics)):
    """Resume des pieces d'un jeu avec leur valeur."""
    try:
        return {"game": game_name, "pieces": analytics.piece_value_summary(game_name)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Neo4j: {e}")


@router.post("/graph/analyze-and-record")
async def analyze_and_record(
    file: UploadFile = File(...),
    players: Optional[str] = None,
    detector: BoardGameDetector = Depends(get_detector),
    neo4j: Neo4jPipeline = Depends(get_neo4j_pipeline),
):
    """Detection + enregistrement Neo4j en une seule requete."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Fichier doit etre une image")
    try:
        contents = await file.read()
        detection = detector.full_analysis(contents)

        if detection["game_type"] == "unknown":
            return {"status": "no_game_detected", "vision": detection}

        players_list = players.split(",") if players else ["Player1", "Player2"]
        session_id = neo4j.create_session(detection["game_type"], players=players_list)
        snapshot_id = neo4j.record_detection(session_id, detection)

        return {
            "status": "recorded",
            "session_id": session_id,
            "snapshot_id": snapshot_id,
            "vision": detection,
        }
    except Exception as e:
        raise HTTPException(500, f"Erreur: {e}")