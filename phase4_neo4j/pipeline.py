"""Pipeline d'ingestion : detections Vision (Phase 2) -> Neo4j."""

from datetime import datetime
from typing import Dict, List
from uuid import uuid4

from phase4_neo4j.neo4j_service import Neo4jService


class Neo4jPipeline:
    """Ingestion des resultats de detection dans Neo4j."""

    def __init__(self):
        self.service = Neo4jService()
        self.service.connect()

    def close(self):
        self.service.close()

    def create_session(self, game_type: str, players: List[str] = None) -> str:
        """Cree une session de jeu et retourne son ID."""
        session_id = f"session_{uuid4().hex[:8]}"
        players = players or ["Player1", "Player2"]

        query = """
        MATCH (g:Game {id: $game_id})
        CREATE (s:GameSession {
            id: $session_id,
            created_at: datetime(),
            status: 'active',
            players: $players
        })
        MERGE (s)-[:PLAYS]->(g)
        RETURN s.id AS id
        """

        result = self.service.run_query(query, {
            "game_id": f"game_{game_type.lower()}",
            "session_id": session_id,
            "players": players,
        })

        print(f"Session creee : {session_id}")
        return session_id

    def record_detection(
        self,
        session_id: str,
        detection_result: Dict,
    ) -> str:
        """Enregistre une detection comme snapshot."""
        snapshot_id = f"snap_{uuid4().hex[:8]}"

        # 1. Creer le snapshot
        snapshot_query = """
        MATCH (s:GameSession {id: $session_id})
        CREATE (snap:Snapshot {
            id: $snap_id,
            timestamp: datetime(),
            game_type: $game_type,
            confidence: $confidence,
            total_pieces: $total_pieces,
            is_complete: $is_complete
        })
        MERGE (s)-[:HAS_SNAPSHOT]->(snap)
        RETURN snap.id AS id
        """

        self.service.run_query(snapshot_query, {
            "session_id": session_id,
            "snap_id": snapshot_id,
            "game_type": detection_result["game_type"],
            "confidence": detection_result["confidence"],
            "total_pieces": detection_result["total_pieces"],
            "is_complete": detection_result.get("is_complete", False),
        })

        # 2. Lier les detections aux pieces du graphe
        for i, piece in enumerate(detection_result.get("pieces", [])):
            detection_id = f"{snapshot_id}_det_{i}"
            piece_class = piece["class"]
            piece_id = self._build_piece_id(detection_result["game_type"], piece_class)

            detection_query = """
            MATCH (snap:Snapshot {id: $snap_id})
            OPTIONAL MATCH (p:Piece {id: $piece_id})
            CREATE (d:Detection {
                id: $det_id,
                class: $piece_class,
                confidence: $confidence,
                bbox_x1: $x1, bbox_y1: $y1, bbox_x2: $x2, bbox_y2: $y2
            })
            MERGE (snap)-[:DETECTED]->(d)
            FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
                MERGE (d)-[:OF_TYPE]->(p)
            )
            """

            bbox = piece.get("bbox", [0, 0, 0, 0])
            self.service.run_query(detection_query, {
                "snap_id": snapshot_id,
                "det_id": detection_id,
                "piece_id": piece_id,
                "piece_class": piece_class,
                "confidence": piece["confidence"],
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
            })

        # 3. Enregistrer les pieces manquantes
        for missing in detection_result.get("missing_pieces", []):
            missing_query = """
            MATCH (snap:Snapshot {id: $snap_id})
            MATCH (p:Piece {id: $piece_id})
            MERGE (snap)-[m:MISSING_PIECE]->(p)
            SET m.expected = $expected,
                m.found = $found,
                m.missing = $missing
            """
            piece_id = self._build_piece_id(detection_result["game_type"], missing["piece"])
            self.service.run_query(missing_query, {
                "snap_id": snapshot_id,
                "piece_id": piece_id,
                "expected": missing["expected"],
                "found": missing["found"],
                "missing": missing["missing"],
            })

        print(f"Snapshot enregistre : {snapshot_id} ({len(detection_result.get('pieces', []))} pieces)")
        return snapshot_id

    @staticmethod
    def _build_piece_id(game_type: str, piece_class: str) -> str:
        """Convertit une classe YOLO en ID de piece du graphe."""
        # Ex: 'white-pawn' + 'chess' -> 'piece_chess_white_pawn'
        normalized = piece_class.lower().replace("-", "_").replace(" ", "_")
        return f"piece_{game_type.lower()}_{normalized}"


# =============== Test ===============
if __name__ == "__main__":
    pipeline = Neo4jPipeline()

    # Simuler une detection
    fake_detection = {
        "game_type": "chess",
        "confidence": 0.87,
        "total_pieces": 30,
        "is_complete": False,
        "pieces": [
            {"class": "white-pawn", "confidence": 0.95, "bbox": [100, 200, 150, 250]},
            {"class": "white-pawn", "confidence": 0.92, "bbox": [160, 200, 210, 250]},
            {"class": "black-king", "confidence": 0.98, "bbox": [400, 100, 450, 150]},
            {"class": "white-queen", "confidence": 0.89, "bbox": [350, 500, 400, 550]},
        ],
        "missing_pieces": [
            {"piece": "black-pawn", "expected": 8, "found": 7, "missing": 1},
            {"piece": "black-rook", "expected": 2, "found": 1, "missing": 1},
        ],
    }

    session_id = pipeline.create_session("chess", players=["Alice", "Bob"])
    snapshot_id = pipeline.record_detection(session_id, fake_detection)

    print(f"\nDans Neo4j Browser, tape :")
    print(f"  MATCH (s:GameSession {{id: '{session_id}'}})-[*1..3]-(n) RETURN s, n")

    pipeline.close()