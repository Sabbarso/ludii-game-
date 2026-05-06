"""Client Python pour interagir avec Neo4j."""

import os
from contextlib import contextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class Neo4jService:
    """Wrapper autour du driver Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        self._driver = None

    def connect(self):
        """Etablit la connexion."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            print(f"Connecte a Neo4j sur {self.uri}")

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            print("Connexion Neo4j fermee")

    @contextmanager
    def session(self):
        """Context manager pour les sessions."""
        self.connect()
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

    def run_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute une requete Cypher et retourne les resultats."""
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_write(self, query: str, parameters: Optional[Dict] = None) -> Dict:
        """Execute une requete d'ecriture (CREATE/MERGE/SET/DELETE)."""
        with self.session() as session:
            result = session.run(query, parameters or {})
            summary = result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "relationships_created": summary.counters.relationships_created,
                "properties_set": summary.counters.properties_set,
            }

    def execute_script(self, script_path: str) -> int:
        """Execute un script Cypher complet (separateur ';')."""
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Diviser par ';' tout en gardant les requetes valides
        queries = [q.strip() for q in content.split(";") if q.strip() and not q.strip().startswith("//")]

        executed = 0
        with self.session() as session:
            for query in queries:
                try:
                    session.run(query)
                    executed += 1
                except Exception as e:
                    print(f"Erreur sur requete: {query[:80]}...")
                    print(f"  -> {e}")

        print(f"{executed}/{len(queries)} requetes executees")
        return executed

    def reset_database(self):
        """ATTENTION : supprime tout le graphe."""
        self.run_write("MATCH (n) DETACH DELETE n")
        print("Base Neo4j videe")

    def stats(self) -> Dict:
        """Statistiques du graphe."""
        node_counts = self.run_query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
        )
        rel_counts = self.run_query(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
        )
        return {
            "nodes": node_counts,
            "relationships": rel_counts,
            "total_nodes": sum(r["count"] for r in node_counts),
            "total_relationships": sum(r["count"] for r in rel_counts),
        }


# =============== Test ===============
if __name__ == "__main__":
    service = Neo4jService()
    try:
        service.connect()
        result = service.run_query("RETURN 'Hello Neo4j!' AS message")
        print(f"Test : {result}")

        stats = service.stats()
        print(f"\nStats actuelles :")
        print(f"  Nodes : {stats['total_nodes']}")
        print(f"  Relations : {stats['total_relationships']}")
        for label_info in stats["nodes"]:
            print(f"    - {label_info['label']}: {label_info['count']}")
    except Exception as e:
        print(f"Erreur connexion Neo4j : {e}")
        print("Verifier que Neo4j est lance sur bolt://localhost:7687")
    finally:
        service.close()