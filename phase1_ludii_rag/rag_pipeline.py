"""RAG Pipeline : embeddings (sentence-transformers) + ChromaDB."""

import os
import requests
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


class LudiiRAG:
    """Systeme RAG pour les regles Ludii."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        persist_directory: str = "phase1_ludii_rag/vector_db",
        collection_name: str = "ludii_rules",
    ):
        print(f"Chargement du modele : {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # =============== Ingestion ===============
    def ingest_ludii_rules(self, games_rules: List[Dict]) -> int:
        documents: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict] = []

        for rule_dict in games_rules:
            game_name = rule_dict.get("game", "Unknown")
            rules_text = rule_dict.get("rules", "")
            source = rule_dict.get("source", "unknown")

            if not rules_text.strip():
                continue

            chunks = self._chunk_rules(rules_text, game_name)

            for i, chunk in enumerate(chunks):
                doc_id = f"{game_name.lower().replace(' ', '_')}_chunk_{i}"
                documents.append(chunk["text"])
                ids.append(doc_id)
                metadatas.append({
                    "game": game_name.lower(),
                    "chunk_index": i,
                    "section": chunk.get("section", "general"),
                    "source": source,
                })

        if not documents:
            print("Aucun document a indexer.")
            return 0

        ids = self._ensure_unique_ids(ids)

        print(f"Generation des embeddings pour {len(documents)} chunks...")
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        print(f"{len(documents)} chunks indexes depuis Ludii")
        return len(documents)

    def _chunk_rules(
        self,
        rules_text: str,
        game_name: str,
        target_words: int = 200,
    ) -> List[Dict]:
        chunks = []
        lines = rules_text.split("\n")
        current_chunk: List[str] = []

        for line in lines:
            current_chunk.append(line)
            word_count = len(" ".join(current_chunk).split())

            if word_count >= target_words:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "game": game_name,
                    "section": self._detect_section(current_chunk),
                })
                current_chunk = []

        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "game": game_name,
                "section": self._detect_section(current_chunk),
            })

        return chunks

    @staticmethod
    def _detect_section(lines: List[str]) -> str:
        text = " ".join(lines).lower()
        if "win" in text or "end" in text or "checkmate" in text:
            return "win_condition"
        if "move" in text or "play" in text or "step" in text:
            return "movement"
        if "setup" in text or "place" in text or "start" in text:
            return "setup"
        if "piece" in text or "pawn" in text or "king" in text:
            return "pieces"
        return "general"

    @staticmethod
    def _ensure_unique_ids(ids: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        unique = []
        for id_ in ids:
            if id_ in seen:
                seen[id_] += 1
                unique.append(f"{id_}_{seen[id_]}")
            else:
                seen[id_] = 0
                unique.append(id_)
        return unique

    # =============== Recherche ===============
    def retrieve_similar_rules(
        self,
        query: str,
        top_k: int = 3,
        game_filter: Optional[str] = None,
    ) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query).tolist()
        where = {"game": game_filter.lower()} if game_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": doc,
                "game": meta.get("game"),
                "section": meta.get("section"),
                "similarity": round(1 - dist, 4),
            })

        return retrieved

    # =============== Completion via LLM ===============
    def complete_damaged_rule(
        self,
        damaged_rule: str,
        game_context: Optional[str] = None,
    ) -> str:
        """
        Restaure une regle endommagee via RAG + Groq LLM.
        Necessite GROQ_API_KEY dans .env.
        """
        token = os.getenv("GROQ_API_KEY")
        if not token:
            return "[GROQ_API_KEY manquant dans .env]"

        try:
            from groq import Groq
            retrieved = self.retrieve_similar_rules(
                damaged_rule, top_k=3, game_filter=game_context
            )
            context = "\n".join([r["text"] for r in retrieved])

            client = Groq(api_key=token)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a board game expert. Restore damaged rules clearly and concisely."
                    },
                    {
                        "role": "user",
                        "content": f"Game: {game_context or 'Unknown'}\n\nContext from similar rules:\n{context}\n\nDamaged rule: {damaged_rule}\n\nRestored rule (one sentence only):"
                    }
                ],
                max_tokens=100,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[Erreur Groq: {str(e)}]"

    # =============== Maintenance ===============
    def stats(self) -> Dict:
        return {
            "total_chunks": self.collection.count(),
            "collection": self.collection.name,
        }

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Collection '{name}' reinitialisee")


# =============== Test ===============
if __name__ == "__main__":
    print("=" * 60)
    print("TEST RAG Pipeline")
    print("=" * 60)

    rag = LudiiRAG()
    rag.reset()

    test_rules = [
        {
            "game": "Chess",
            "rules": (
                "Pawns move forward one square, or two squares on their first move. "
                "Rooks move horizontally or vertically any number of squares. "
                "Bishops move diagonally any number of squares. "
                "The queen combines rook and bishop moves. "
                "The king moves one square in any direction. "
                "Checkmate ends the game with the attacking player winning."
            ),
            "source": "test",
        },
        {
            "game": "Checkers",
            "rules": (
                "Each player starts with 12 pieces on a standard 8x8 board. "
                "Pieces move diagonally one square forward to an empty square. "
                "Captures are made by jumping over opponent pieces. "
                "When a piece reaches the opposite side, it becomes a king. "
                "The first player to capture all opponent pieces wins."
            ),
            "source": "test",
        },
    ]

    rag.ingest_ludii_rules(test_rules)

    print("\n" + "=" * 60)
    print("Recherche : 'How does the king move?'")
    print("=" * 60)
    results = rag.retrieve_similar_rules("How does the king move?", top_k=2)
    for i, r in enumerate(results, 1):
        print(f"\nResultat {i} (similarity={r['similarity']}):")
        print(f"  Game: {r['game']} | Section: {r['section']}")
        print(f"  Text: {r['text'][:200]}...")

    print("\n" + "=" * 60)
    print("Test LLM restore...")
    restored = rag.complete_damaged_rule("Pawns move [MASK] one square", game_context="Chess")
    print(f"Restored: {restored}")

    print("\n" + "=" * 60)
    print(f"Stats: {rag.stats()}")
    print("=" * 60)