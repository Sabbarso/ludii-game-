"""Requetes analytiques Cypher."""

from typing import List, Dict
from phase4_neo4j.neo4j_service import Neo4jService


class GameAnalytics:
    """Requetes d'analyse sur le graphe Neo4j."""

    def __init__(self):
        self.service = Neo4jService()

    def list_all_games(self) -> List[Dict]:
        """Q1 : Liste tous les jeux avec leurs caracteristiques."""
        query = """
        MATCH (g:Game)
        OPTIONAL MATCH (g)-[:ORIGINATES_FROM]->(o:Origin)
        OPTIONAL MATCH (g)-[:HAS_MECHANIC]->(m:Mechanic)
        RETURN g.name AS game,
               g.players AS players,
               g.num_pieces_total AS pieces,
               collect(DISTINCT o.name) AS origins,
               collect(DISTINCT m.name) AS mechanics
        ORDER BY g.name
        """
        return self.service.run_query(query)

    def rules_by_game_and_category(self, game_name: str) -> List[Dict]:
        """Q2 : Regles d'un jeu groupees par categorie."""
        query = """
        MATCH (g:Game {name: $game_name})-[:HAS_RULE]->(r:Rule)
        RETURN r.category AS category, collect(r.text) AS rules, count(r) AS total
        ORDER BY total DESC
        """
        return self.service.run_query(query, {"game_name": game_name})

    def most_missing_pieces(self, top_n: int = 5) -> List[Dict]:
        """Q3 : Pieces les plus souvent manquantes a travers toutes les sessions."""
        query = """
        MATCH (snap:Snapshot)-[m:MISSING_PIECE]->(p:Piece)
        RETURN p.name AS piece,
               sum(m.missing) AS total_missing,
               count(snap) AS sessions_affected
        ORDER BY total_missing DESC
        LIMIT $top_n
        """
        return self.service.run_query(query, {"top_n": top_n})

    def game_completion_rate(self) -> List[Dict]:
        """Q4 : Taux de jeux complets vs incomplets par jeu."""
        query = """
        MATCH (g:Game)<-[:PLAYS]-(s:GameSession)-[:HAS_SNAPSHOT]->(snap:Snapshot)
        RETURN g.name AS game,
               count(snap) AS total_snapshots,
               sum(CASE WHEN snap.is_complete THEN 1 ELSE 0 END) AS complete,
               sum(CASE WHEN NOT snap.is_complete THEN 1 ELSE 0 END) AS incomplete,
               round(100.0 * sum(CASE WHEN snap.is_complete THEN 1 ELSE 0 END) / count(snap), 1) AS completion_pct
        """
        return self.service.run_query(query)

    def games_by_origin(self) -> List[Dict]:
        """Q5 : Jeux groupes par pays d'origine."""
        query = """
        MATCH (g:Game)-[:ORIGINATES_FROM]->(o:Origin)
        RETURN o.name AS origin,
               o.continent AS continent,
               collect(g.name) AS games,
               count(g) AS total
        ORDER BY total DESC
        """
        return self.service.run_query(query)

    def shared_mechanics(self) -> List[Dict]:
        """Q6 : Mecaniques partagees entre jeux (pour les recommandations)."""
        query = """
        MATCH (g1:Game)-[:HAS_MECHANIC]->(m:Mechanic)<-[:HAS_MECHANIC]-(g2:Game)
        WHERE g1.name < g2.name
        RETURN g1.name AS game1, g2.name AS game2,
               collect(m.name) AS shared_mechanics,
               count(m) AS similarity_score
        ORDER BY similarity_score DESC
        """
        return self.service.run_query(query)

    def piece_value_summary(self, game_name: str) -> List[Dict]:
        """Q7 : Resume des pieces d'un jeu avec leur valeur."""
        query = """
        MATCH (g:Game {name: $game_name})-[:HAS_PIECE]->(p:Piece)
        RETURN p.color AS color,
               p.name AS piece,
               p.expected_count AS count,
               p.value AS value,
               p.expected_count * p.value AS total_value
        ORDER BY p.color, total_value DESC
        """
        return self.service.run_query(query, {"game_name": game_name})

    def recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Q8 : Sessions recentes avec leurs detections."""
        query = """
        MATCH (s:GameSession)-[:PLAYS]->(g:Game)
        OPTIONAL MATCH (s)-[:HAS_SNAPSHOT]->(snap:Snapshot)
        RETURN s.id AS session_id,
               g.name AS game,
               s.created_at AS created_at,
               s.status AS status,
               count(snap) AS num_snapshots
        ORDER BY s.created_at DESC
        LIMIT $limit
        """
        return self.service.run_query(query, {"limit": limit})


# =============== Test ===============
if __name__ == "__main__":
    analytics = GameAnalytics()

    print("=" * 60)
    print("Q1 : Liste des jeux")
    print("=" * 60)
    for game in analytics.list_all_games():
        print(f"  {game['game']} - {game['players']} joueurs, {game['pieces']} pieces")
        print(f"    Origines : {game['origins']}")
        print(f"    Mecaniques : {game['mechanics']}")

    print("\n" + "=" * 60)
    print("Q2 : Regles de Chess par categorie")
    print("=" * 60)
    for cat in analytics.rules_by_game_and_category("Chess"):
        print(f"  [{cat['category']}] ({cat['total']} regles)")
        for rule in cat["rules"][:2]:
            print(f"    - {rule[:80]}")

    print("\n" + "=" * 60)
    print("Q5 : Jeux par origine")
    print("=" * 60)
    for origin in analytics.games_by_origin():
        print(f"  {origin['origin']} ({origin['continent']}) : {origin['games']}")

    print("\n" + "=" * 60)
    print("Q6 : Mecaniques partagees")
    print("=" * 60)
    for shared in analytics.shared_mechanics():
        print(f"  {shared['game1']} <-> {shared['game2']}")
        print(f"    Partagent : {shared['shared_mechanics']}")

    print("\n" + "=" * 60)
    print("Q7 : Pieces de Chess")
    print("=" * 60)
    for p in analytics.piece_value_summary("Chess"):
        print(f"  {p['color']:6} {p['piece']:20} x{p['count']:2} (val={p['value']}, total={p['total_value']})")