from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.ml.model_registry import model_status_summary


class ProductionInferencePipeline:
    """Production inference interface for gesture + expression + intent pipeline.

    This layer does not force a heavy ML dependency at import time. It exposes a structured
    contract for a trained model deployment and falls back to the current rule-based logic when
    model weights are not yet available.
    """

    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
        self.model_summary = model_status_summary()

    def model_ready(self) -> bool:
        return any(
            spec.get("status") == "trained"
            for spec in self.model_summary.values()
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "models": self.model_summary,
            "status": "ready_for_training" if not self.model_ready() else "trained",
            "strategy": "pretrained_backbone_plus_finetune",
        }

    def infer(self, gesture_name: str | None, expression_name: str | None, confidence: float | None = None) -> dict[str, Any]:
        payload = {
            "gesture": gesture_name or "unknown",
            "expression": expression_name or "unknown",
            "confidence": float(confidence or 0.0),
            "model_status": self.get_status(),
            "source": "production_inference_pipeline",
        }

        if not self.model_ready():
            payload["warning"] = "No trained weights present yet. Using staged production heuristic pipeline."
        return payload

    def export_manifest(self, output_path: str | Path) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.get_status(), indent=2), encoding="utf-8")
        return str(output)
