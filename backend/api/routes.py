from typing import Any

from fastapi import APIRouter

from backend.ml.inference_pipeline import ProductionInferencePipeline
from backend.vision.live_analyzer import analyze_frame, decode_base64_image

router = APIRouter()
inference_pipeline = ProductionInferencePipeline()


@router.get("/")
def root() -> dict[str, Any]:
    return {"message": "GestureSpeak AI backend is running."}


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@router.get("/models/status")
def models_status() -> dict[str, Any]:
    return inference_pipeline.get_status()


@router.post("/predict/gesture")
def predict_gesture(payload: dict[str, Any]) -> dict[str, Any]:
    return {"gesture": payload.get("gesture") or "unknown", "confidence": payload.get("confidence", 0.0)}


@router.post("/predict/expression")
def predict_expression(payload: dict[str, Any]) -> dict[str, Any]:
    return {"expression": payload.get("expression") or "unknown", "confidence": payload.get("confidence", 0.0)}


@router.post("/predict/intent")
def predict_intent(payload: dict[str, Any]) -> dict[str, Any]:
    gesture = payload.get("gesture")
    expression = payload.get("expression")
    from backend.intent.intent_engine import determine_intent

    return determine_intent(gesture, expression, payload.get("confidence"))


@router.post("/translate")
def translate(payload: dict[str, Any]) -> dict[str, Any]:
    from backend.fusion.multimodal_fusion import fuse_results

    gesture = payload.get("gesture") or {"name": "unknown", "confidence": 0.0}
    expression = payload.get("expression") or {"name": "unknown", "confidence": 0.0}
    return fuse_results(gesture, expression)


@router.post("/analyze-frame")
def analyze_frame_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    frame_data = payload.get("frame")
    frame = decode_base64_image(frame_data) if frame_data else None
    result = analyze_frame(frame)
    return result
