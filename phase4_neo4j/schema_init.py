"""Initialise le schema Neo4j et insere les donnees de base."""

from pathlib import Path

from phase4_neo4j.neo4j_service import Neo4jService


CYPHER_DIR = Path("phase4_neo4j/cypher")


def init_schema(reset: bool = False):
    """Initialise le schema (contraintes + index + donnees seed)."""
    service = Neo4jService()
    service.connect()

    if reset:
        print("Reset de la base...")
        service.reset_database()

    print("\n[1/2] Initialisation du schema...")
    service.execute_script(str(CYPHER_DIR / "init_schema.cypher"))

    print("\n[2/2] Insertion des donnees seed...")
    service.execute_script(str(CYPHER_DIR / "seed_games.cypher"))

    # Stats finales
    stats = service.stats()
    print("\n" + "=" * 60)
    print("Schema initialise avec succes")
    print("=" * 60)
    print(f"Total nodes : {stats['total_nodes']}")
    print(f"Total relationships : {stats['total_relationships']}")
    print("\nDetails par type :")
    for n in stats["nodes"]:
        print(f"  - {n['label']}: {n['count']}")

    service.close()


if __name__ == "__main__":
    init_schema(reset=True)