"""Détecteur unifié Chess + Checkers basé sur YOLO."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class BoardGameDetector:
    # ⚠️ ADAPTE CETTE STRUCTURE selon la sortie de verify_models.py
    EXPECTED_PIECES = {
        "chess": {
            "white-pawn": 8, "black-pawn": 8,
            "white-rook": 2, "black-rook": 2,
            "white-knight": 2, "black-knight": 2,
            "white-bishop": 2, "black-bishop": 2,
            "white-queen": 1, "black-queen": 1,
            "white-king": 1, "black-king": 1,
        },
        "checkers": {
            "white": 12, "black": 12,
        },
    }

    def __init__(
        self,
        chess_model_path: str = "phase2_vision/models/chess_yolo.pt",
        checkers_model_path: str = "phase2_vision/models/checkers_yolo.pt",
        confidence_threshold: float = 0.5,
    ):
        print("Chargement des modeles YOLO...")
        self.chess_model = YOLO(chess_model_path)
        self.checkers_model = YOLO(checkers_model_path)
        self.conf = confidence_threshold
        print(f"Chess classes : {list(self.chess_model.names.values())}")
        print(f"Checkers classes : {list(self.checkers_model.names.values())}")

    def _load_image(self, image_input: Union[str, Path, Image.Image, np.ndarray, bytes]) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Image introuvable : {image_input}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        if isinstance(image_input, np.ndarray):
            return image_input
        if isinstance(image_input, bytes):
            arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Bytes invalides")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise TypeError(f"Type non supporte : {type(image_input)}")

    CHESS_SPECIFIC_CLASSES = ["queen", "bishop", "knight", "rook", "king", "pawn"]

    def detect_game_type(self, image_input) -> Dict:
        img = self._load_image(image_input)

        # Lancer les deux modeles
        chess_results = self.chess_model(img, conf=self.conf, verbose=False)[0]
        checkers_results = self.checkers_model(img, conf=self.conf, verbose=False)[0]

        # Extraire les classes detectees par chess model
        chess_detected_classes = []
        if chess_results.boxes is not None and len(chess_results.boxes) > 0:
            for box in chess_results.boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                cls_name = self.chess_model.names[cls_id].lower()
                chess_detected_classes.append(cls_name)

        # Extraire les classes detectees par checkers model
        checkers_detected_classes = []
        if checkers_results.boxes is not None and len(checkers_results.boxes) > 0:
            for box in checkers_results.boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                cls_name = self.checkers_model.names[cls_id].lower()
                checkers_detected_classes.append(cls_name)

        # Option B : classifier par classes specifiques
        chess_specific_found = [
            cls for cls in chess_detected_classes
            if any(specific in cls for specific in self.CHESS_SPECIFIC_CLASSES)
        ]

        checkers_specific_found = [
            cls for cls in checkers_detected_classes
            if cls in ["black", "white", "board"]
        ]

        # Decision avec seuil de confiance et diversite de classes
        chess_score = self._score_results(chess_results)
        checkers_score = self._score_results(checkers_results)

        nb_classes_chess_differentes = len(set(chess_specific_found))
        chess_conf_ok = chess_score["avg_conf"] >= 0.7
        chess_diversite_ok = nb_classes_chess_differentes >= 2

        if chess_specific_found and chess_conf_ok and chess_diversite_ok:
            return {
                "game_type": "chess",
                "confidence": chess_score["score"],
                "detected_classes": list(set(chess_detected_classes)),
                "details": {
                    "method": "class_based",
                    "chess_specific": chess_specific_found,
                    "nb_classes": nb_classes_chess_differentes,
                }
            }
        elif checkers_specific_found:
            return {
                "game_type": "checkers",
                "confidence": checkers_score["score"],
                "detected_classes": list(set(checkers_detected_classes)),
                "details": {
                    "method": "class_based",
                    "checkers_specific": checkers_specific_found,
                }
            }
        else:
            return {"game_type": "unknown", "confidence": 0.0, "detected_classes": [], "details": {}}

    @staticmethod
    def _score_results(results) -> Dict:
        if results.boxes is None or len(results.boxes) == 0:
            return {"score": 0.0, "n_detections": 0, "avg_conf": 0.0}
        conf = results.boxes.conf.cpu().numpy()
        avg = float(conf.mean())
        n = len(conf)
        score = avg * min(1.0, n / 8.0)
        return {"score": round(score, 3), "n_detections": n, "avg_conf": round(avg, 3)}

    def detect_pieces(self, image_input, game_type: Optional[str] = None) -> Dict:
        img = self._load_image(image_input)

        if game_type is None:
            game_info = self.detect_game_type(img)
            game_type = game_info["game_type"]
            game_confidence = game_info["confidence"]
        else:
            game_confidence = 1.0

        if game_type == "unknown":
            return {"game_type": "unknown", "confidence": 0.0, "pieces": [],
                    "count_by_class": {}, "total_pieces": 0}

        model = self.chess_model if game_type == "chess" else self.checkers_model
        results = model(img, conf=self.conf, verbose=False)[0]

        pieces = []
        count_by_class: Dict[str, int] = {}

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf.cpu().numpy()[0])
                xyxy = box.xyxy.cpu().numpy()[0].tolist()

                pieces.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x, 1) for x in xyxy],
                })
                count_by_class[cls_name] = count_by_class.get(cls_name, 0) + 1

        return {
            "game_type": game_type,
            "confidence": round(game_confidence, 3),
            "pieces": pieces,
            "count_by_class": count_by_class,
            "total_pieces": len(pieces),
        }

    def detect_missing_pieces(self, detection_result: Dict) -> List[Dict]:
        game_type = detection_result["game_type"]
        if game_type not in self.EXPECTED_PIECES:
            return []
        expected = self.EXPECTED_PIECES[game_type]
        actual = detection_result["count_by_class"]
        missing = []
        for piece, exp_count in expected.items():
            actual_count = actual.get(piece, 0)
            if actual_count < exp_count:
                missing.append({
                    "piece": piece, "expected": exp_count,
                    "found": actual_count, "missing": exp_count - actual_count,
                })
        return missing

    def annotate_image(self, image_input, save_path: Optional[str] = None) -> np.ndarray:
        img = self._load_image(image_input)
        game_info = self.detect_game_type(img)
        if game_info["game_type"] == "unknown":
            return img
        model = self.chess_model if game_info["game_type"] == "chess" else self.checkers_model
        results = model(img, conf=self.conf, verbose=False)[0]
        annotated = results.plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if isinstance(annotated, np.ndarray) else np.array(annotated)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        return annotated_rgb

    def full_analysis(self, image_input) -> Dict:
        detection = self.detect_pieces(image_input)
        missing = self.detect_missing_pieces(detection)
        return {
            **detection,
            "missing_pieces": missing,
            "is_complete": len(missing) == 0 and detection["game_type"] != "unknown",
        }


if __name__ == "__main__":
    detector = BoardGameDetector()
    test_image = "phase2_vision/test_images/test_chess.jpg"
    if Path(test_image).exists():
        result = detector.full_analysis(test_image)
        print(f"\nGame type: {result['game_type']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Total pieces: {result['total_pieces']}")
        print(f"Pieces by class: {result['count_by_class']}")
        print(f"Missing: {result['missing_pieces']}")
    else:
        print(f"Place une image dans {test_image}")