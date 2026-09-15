from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.ml.training_pipeline import ProductionTrainingPipeline


def main() -> None:
    pipeline = ProductionTrainingPipeline(project_root)
    plan = pipeline.build_training_plan()
    output_path = project_root / "artifacts" / "training_plan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(__import__("json").dumps(plan, indent=2), encoding="utf-8")
    print(f"Training plan saved to: {output_path}")
    print("Next steps:")
    print("1. Collect dataset under datasets/gesture and datasets/expression")
    print("2. Train gesture and expression models using pretrained backbones")
    print("3. Save weights to models/hand_gesture_model.pth and models/facial_expression_model.pth")
    print("4. Replace heuristic fallback with trained checkpoint inference")


if __name__ == "__main__":
    main()
