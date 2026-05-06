"""Génère un dataset de règles endommagées pour tester la restauration."""

import json
import random
import re
from pathlib import Path
from typing import Dict, List


LUDII_GAMES_RULES: Dict[str, List[str]] = {
    "Chess": [
        "Pawns move forward one square, or two squares on their first move.",
        "Rooks move horizontally or vertically any number of squares.",
        "Bishops move diagonally any number of squares.",
        "Knights move in an L-shape: 2 squares in one direction and 1 square perpendicular.",
        "The queen combines the powers of rooks and bishops.",
        "The king moves one square in any direction.",
        "Checkmate ends the game with the attacking player winning.",
        "Pawns capture diagonally one square forward.",
        "When a pawn reaches the last rank, it is promoted.",
        "Castling allows the king to move two squares toward a rook.",
    ],
    "Checkers": [
        "Each player starts with 12 pieces placed on dark squares.",
        "Pieces move diagonally one square forward to an empty square.",
        "Captures are mandatory and made by jumping over opponent pieces.",
        "Multiple captures in a single turn are allowed and required.",
        "When a piece reaches the opposite side, it becomes a king.",
        "Kings can move and capture diagonally both forward and backward.",
        "The player who captures all opponent pieces wins.",
        "If a player cannot move, they lose the game.",
        "The board is 8x8 squares, alternating light and dark colors.",
        "Pieces are placed only on dark squares at start.",
    ],
}

KEYWORDS_TO_MASK = [
    "pawn", "rook", "knight", "bishop", "queen", "king",
    "move", "moves", "square", "squares", "diagonally",
    "piece", "pieces", "player", "capture", "captures",
    "forward", "backward", "promotion", "castling",
]


class DamagedRulesGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    @staticmethod
    def truncate(rule: str) -> str:
        cut = int(len(rule) * random.uniform(0.5, 0.8))
        return rule[:cut].rstrip(",.; ") + " [INCOMPLETE]"

    @staticmethod
    def mask_keywords(rule: str, max_masks: int = 2) -> str:
        damaged = rule
        masked = 0
        for kw in KEYWORDS_TO_MASK:
            if masked >= max_masks: break
            pattern = re.compile(rf"\b{kw}\b", re.IGNORECASE)
            if pattern.search(damaged):
                damaged = pattern.sub("[MASK]", damaged, count=1)
                masked += 1
        return damaged

    @staticmethod
    def remove_parts(rule: str) -> str:
        sentences = [s.strip() for s in rule.split(".") if s.strip()]
        if len(sentences) <= 1:
            words = rule.split()
            keep = len(words) // 2
            return " ".join(words[:keep]) + " [...]"
        del sentences[random.randint(0, len(sentences) - 1)]
        return ". ".join(sentences) + "."

    @staticmethod
    def shuffle_words(rule: str) -> str:
        words = rule.split()
        random.shuffle(words)
        return " ".join(words)

    def damage_rule(self, rule: str, damage_type: str) -> str:
        damages = {
            "truncate": self.truncate,
            "mask_keywords": self.mask_keywords,
            "remove_parts": self.remove_parts,
            "shuffle": self.shuffle_words,
        }
        return damages[damage_type](rule)

    def generate_dataset(
        self, games_rules: Dict[str, List[str]] = None,
        output_path: str = "phase1_ludii_rag/datasets/ludii_rules_damaged/dataset.jsonl",
    ) -> List[Dict]:
        games_rules = games_rules or LUDII_GAMES_RULES
        damage_types = ["truncate", "mask_keywords", "remove_parts", "shuffle"]
        dataset = []

        for game, rules in games_rules.items():
            for rule in rules:
                dataset.append({
                    "game": game, "type": "complete",
                    "text": rule, "source": "ludii_reference",
                })
                for dtype in damage_types:
                    dataset.append({
                        "game": game, "type": "damaged",
                        "original": rule,
                        "damaged": self.damage_rule(rule, dtype),
                        "damage_type": dtype, "source": "ludii_damaged",
                    })

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ Dataset créé : {len(dataset)} exemples → {output_path}")
        return dataset


if __name__ == "__main__":
    DamagedRulesGenerator().generate_dataset()