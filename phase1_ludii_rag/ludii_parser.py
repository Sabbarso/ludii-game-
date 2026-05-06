"""Parse le format Ludii Description Language (.lud)."""

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ParsedLud:
    game_name: str = "Unknown Game"
    description: str = ""
    board: Optional[Dict] = None
    pieces: List[Dict] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    win_condition: str = ""
    equipment: Dict = field(default_factory=dict)
    origin: str = ""
    period: str = ""
    raw_content: str = ""
    source_file: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class LudParser:
    def __init__(self, lud_content: str, source_file: str = ""):
        self.content = lud_content
        self.source_file = source_file

    @classmethod
    def from_file(cls, file_path: str) -> "LudParser":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")
        return cls(path.read_text(encoding="utf-8"), source_file=str(path))

    def parse(self) -> ParsedLud:
        return ParsedLud(
            game_name=self._extract_game_name(),
            description=self._extract_metadata("description"),
            board=self._extract_board(),
            pieces=self._extract_pieces(),
            rules=self._extract_rules(),
            win_condition=self._extract_win_condition(),
            equipment=self._extract_equipment(),
            origin=self._extract_metadata("origin"),
            period=self._extract_metadata("date"),
            raw_content=self.content,
            source_file=self.source_file,
        )

    def _extract_game_name(self) -> str:
        match = re.search(r'\(game\s+"([^"]+)"', self.content)
        return match.group(1) if match else "Unknown Game"

    def _extract_metadata(self, field_name: str) -> str:
        block = re.search(
            rf'\({field_name}\s*\{{([^}}]+)\}}',
            self.content, re.IGNORECASE | re.DOTALL,
        )
        if block:
            strings = re.findall(r'"([^"]+)"', block.group(1))
            return " ".join(strings).strip()
        simple = re.search(rf'\({field_name}\s+"([^"]+)"', self.content, re.IGNORECASE)
        return simple.group(1).strip() if simple else ""

    def _extract_board(self) -> Optional[Dict]:
        match = re.search(r'(board\s+.*?\(.*?\))', self.content, re.DOTALL | re.IGNORECASE)
        if match:
            board_text = match.group(1)
            return {"raw": board_text[:500], "type": self._detect_board_type(board_text)}
        return None

    @staticmethod
    def _detect_board_type(board_text: str) -> str:
        text = board_text.lower()
        if "square" in text: return "square"
        if "hex" in text: return "hex"
        if "circular" in text: return "circular"
        return "custom"

    def _extract_pieces(self) -> List[Dict]:
        pieces = []
        for match in re.finditer(r'\(piece\s+"([^"]+)"', self.content, re.IGNORECASE):
            pieces.append({"name": match.group(1), "position": match.start()})
        return pieces

    def _extract_rules(self) -> List[str]:
        rules = []
        match = re.search(r'\(rules\s+(.*?)\)\s*\)', self.content, re.DOTALL | re.IGNORECASE)
        if match:
            for line in match.group(1).split("\n"):
                stripped = line.strip()
                if stripped and not stripped.startswith("//"):
                    rules.append(stripped)
        return rules

    def _extract_win_condition(self) -> str:
        match = re.search(r'\(end\s+(.*?)\)\s*\)', self.content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip()[:500] if match else ""

    def _extract_equipment(self) -> Dict:
        equipment = {}
        text = self.content.lower()
        if "dice" in text: equipment["dice"] = True
        if "card" in text: equipment["cards"] = True
        if "board" in text: equipment["board"] = True
        return equipment


if __name__ == "__main__":
    lud_dir = Path("phase1_ludii_rag/lud_files")
    for lud_file in lud_dir.glob("*.lud"):
        print(f"\n=== {lud_file.name} ===")
        parsed = LudParser.from_file(str(lud_file)).parse()
        print(f"Game: {parsed.game_name}")
        print(f"Origin: {parsed.origin[:100]}")
        print(f"Pieces: {len(parsed.pieces)} found")
        print(f"Rules: {len(parsed.rules)} lines")
        print(f"Equipment: {parsed.equipment}")



        