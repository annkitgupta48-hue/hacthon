from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    name: str
    model_type: str
    source: str
    weights_path: str | None = None
    input_type: str = "landmarks"
    description: str = ""
    status: str = "not_trained"
    metrics: dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "hand_gesture": ModelSpec(
        name="MediaPipeHandGestureClassifier",
        model_type="gesture",
        source="pretrained + fine-tune",
        weights_path="models/hand_gesture_model.pth",
        input_type="landmarks",
        description="Gesture model for recognizing hand poses and motion patterns using MediaPipe landmarks.",
        status="ready_for_finetuning",
        metrics={"top1": "pending", "dataset": "custom_webcam + HaGRID + ASL"},
    ),
    "facial_expression": ModelSpec(
        name="EfficientNetEmotionClassifier",
        model_type="expression",
        source="pretrained + fine-tune",
        weights_path="models/facial_expression_model.pth",
        input_type="aligned_face_crop",
        description="Emotion model for recognizing neutral, happy, sad, angry, and surprised expressions.",
        status="ready_for_finetuning",
        metrics={"top1": "pending", "dataset": "FER2013 + AffectNet + CK+"},
    ),
    "intent_fusion": ModelSpec(
        name="MultimodalIntentFusion",
        model_type="fusion",
        source="rule_based + sequence_model",
        weights_path=None,
        input_type="gesture+expression+history",
        description="Combines gesture, expression, and temporal context to produce intent and text output.",
        status="active",
        metrics={"rule": "enabled", "sequence_model": "planned"},
    ),
}


def get_model_registry() -> dict[str, ModelSpec]:
    return MODEL_REGISTRY


def resolve_model_paths(project_root: str | Path | None = None) -> dict[str, str | None]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    resolved = {}
    for key, spec in MODEL_REGISTRY.items():
        if not spec.weights_path:
            resolved[key] = None
            continue
        resolved[key] = str((root / spec.weights_path).resolve())
    return resolved


def model_status_summary() -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key, spec in MODEL_REGISTRY.items():
        summary[key] = {
            "name": spec.name,
            "type": spec.model_type,
            "source": spec.source,
            "status": spec.status,
            "weights_path": spec.weights_path,
            "description": spec.description,
        }
    return summary
