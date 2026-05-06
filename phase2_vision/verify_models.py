"""Vérifie les classes connues par les modèles YOLO."""

from ultralytics import YOLO

print("=== Chess model ===")
chess = YOLO("phase2_vision/models/chess_yolo.pt")
print(f"Classes: {chess.names}")
print(f"Total: {len(chess.names)} classes")

print("\n=== Checkers model ===")
checkers = YOLO("phase2_vision/models/checkers_yolo.pt")
print(f"Classes: {checkers.names}")
print(f"Total: {len(checkers.names)} classes")