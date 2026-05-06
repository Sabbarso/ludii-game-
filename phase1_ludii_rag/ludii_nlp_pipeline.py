"""Pipeline unifié : Parser → RAG → Restoration (sans scraper)."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from phase1_ludii_rag.ludii_parser import LudParser
from phase1_ludii_rag.rag_pipeline import LudiiRAG
from phase1_ludii_rag.damaged_rules_generator import DamagedRulesGenerator, LUDII_GAMES_RULES


class LudiiNLPPipeline:
    def __init__(self, lud_files_dir: str = "phase1_ludii_rag/lud_files"):
        self.lud_files_dir = Path(lud_files_dir)
        self.rag = LudiiRAG()
        self.damager = DamagedRulesGenerator()
        self.parsed_games: List[Dict] = []

    def setup(self, reset_db: bool = True):
        print("=" * 60)
        print("🚀 Setup Ludii NLP Pipeline")
        print("=" * 60)

        if reset_db:
            self.rag.reset()

        all_rules: List[Dict] = []

        # 1. Parser les .lud locaux
        print("\n📄 Parsing des fichiers .lud...")
        for lud_file in self.lud_files_dir.glob("*.lud"):
            try:
                parsed = LudParser.from_file(str(lud_file)).parse()
                self.parsed_games.append(parsed.to_dict())
                rule_text = self._build_rule_text(parsed.to_dict())
                if rule_text:
                    all_rules.append({
                        "game": parsed.game_name or lud_file.stem.capitalize(),
                        "rules": rule_text,
                        "source": "lud_file",
                    })
                print(f"   ✅ {lud_file.name} → {parsed.game_name}")
            except Exception as e:
                print(f"   ❌ {lud_file.name} : {e}")

        # 2. Ajouter les règles de référence (claires) du damager
        print("\n📚 Ajout des règles de référence...")
        for game, rules_list in LUDII_GAMES_RULES.items():
            all_rules.append({
                "game": game,
                "rules": " ".join(rules_list),
                "source": "ludii_reference",
            })

        # 3. Indexation RAG
        print(f"\n💾 Indexation de {len(all_rules)} sources...")
        self.rag.ingest_ludii_rules(all_rules)

        print("\n" + "=" * 60)
        print(f"✅ Pipeline prêt : {self.rag.stats()}")
        print("=" * 60)

    @staticmethod
    def _build_rule_text(parsed: Dict) -> str:
        parts = []
        if parsed.get("description"): parts.append(f"Description: {parsed['description']}")
        if parsed.get("origin"): parts.append(f"Origin: {parsed['origin']}")
        if parsed.get("period"): parts.append(f"Period: {parsed['period']}")
        if parsed.get("rules"): parts.append("Rules: " + " ".join(parsed["rules"]))
        if parsed.get("win_condition"): parts.append(f"Win: {parsed['win_condition']}")
        return "\n".join(parts)

    def restore_rule(self, damaged_rule: str, game: Optional[str] = None) -> Dict:
        retrieved = self.rag.retrieve_similar_rules(damaged_rule, top_k=3, game_filter=game)
        completed = self.rag.complete_damaged_rule(damaged_rule, game_context=game)
        confidence = self._calculate_confidence(retrieved)
        return {
            "original_damaged": damaged_rule,
            "restored": completed,
            "retrieved_context": retrieved,
            "confidence": confidence,
            "source": "ludii_rag",
        }

    @staticmethod
    def _calculate_confidence(retrieved: List[Dict]) -> float:
        if not retrieved: return 0.0
        avg = sum(r["similarity"] for r in retrieved) / len(retrieved)
        return round(min(1.0, avg + 0.2), 3)

    def evaluate_on_damaged_dataset(
        self,
        dataset_path: str = "phase1_ludii_rag/datasets/ludii_rules_damaged/dataset.jsonl",
        max_samples: int = 20,
    ) -> Dict:
        if not Path(dataset_path).exists():
            print(f"⚠️  Dataset introuvable. Lance d'abord generate-dataset.")
            return {}

        damaged = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["type"] == "damaged":
                    damaged.append(item)

        damaged = damaged[:max_samples]
        print(f"\n📊 Évaluation sur {len(damaged)} règles endommagées...")

        scores = [self.restore_rule(s["damaged"], game=s["game"])["confidence"] for s in damaged]
        avg = sum(scores) / len(scores) if scores else 0
        return {
            "samples_evaluated": len(damaged),
            "avg_confidence": round(avg, 3),
            "min_confidence": round(min(scores), 3) if scores else 0,
            "max_confidence": round(max(scores), 3) if scores else 0,
        }