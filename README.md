# GestureSpeak AI

## Project Overview

GestureSpeak AI is a production-oriented real-time communication system that interprets live hand gestures and facial expressions into natural-language text and intent. The project is built to move beyond a demo prototype and toward an end-to-end multimodal assistant that can be trained, calibrated, and deployed in real usage scenarios.

## Problem Statement

People with non-verbal communication needs, assistive technology users, and real-time interaction systems require a robust way to understand gestures, facial meaning, and intent from live camera input. A production system must do more than recognize a few static poses; it must handle real-time human behavior, noisy environments, and meaningful contextual interpretation.

## Objective

Build a production-ready multimodal AI system that:

- captures live webcam input continuously
- detects hand gestures with computer vision and keypoint models
- recognizes facial expressions and emotional state cues
- combines gesture + expression + temporal context
- converts the fused signal into intent and human-readable text
- supports speech synthesis and UI output
- allows dataset-driven retraining and pretrained model integration
- is designed for deployment, calibration, and real-world evaluation

## Production Features

- live webcam ingestion and frame-level analysis
- MediaPipe-based hand keypoint detection
- face landmark extraction and expression analysis
- multimodal fusion of gesture + expression + confidence
- intent-to-text generation pipeline
- temporal smoothing and duplicate suppression
- enterprise-style API layer with health and analysis endpoints
- model extensibility using pretrained backbones and fine-tuned custom classifiers
- dataset-driven future training path for real-world gesture vocabulary
- local browser UI with speech output and message history

## Technology Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python, FastAPI
- Vision: OpenCV, MediaPipe
- Numerical: NumPy
- ML: pretrained CNNs, gesture classifiers, expression classifiers
- Datasets: FER2013, AffectNet, CK+, HaGRID, ASL Alphabet, custom local capture
- Testing: pytest

## Production System Architecture

- Browser captures live webcam frames in real time
- frontend sends frame snapshots or lightweight metadata to the backend
- FastAPI exposes analysis endpoints and health APIs
- hand and face detectors extract landmarks/features
- pretrained or fine-tuned models classify gestures and expressions
- multimodal fusion combines confidence-weighted signals with temporal context
- intent engine maps final gesture-expression state to sentence-level output
- backend returns structured JSON and text for UI, speech, and downstream systems
- future deployment layer can add model versioning, logging, calibration, and monitoring

## Installation

1. Clone or open this project folder.
2. Create a virtual environment:

```bash
python -m venv venv
```

On Windows:

```powershell
venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- Webcam
- Modern browser with camera support
- Optional: microphone for speech synthesis support

## How to Run

Start the backend:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
```

Open the frontend by serving the project folder, for example:

```bash
python -m http.server 5500
```

Then visit:

```text
http://localhost:5500
```

## How Gesture Detection Works

The system uses MediaPipe hand landmarks to estimate finger positions, hand orientation, and movement history. A rule-based classifier recognizes a limited set of gestures such as thumbs up, thumbs down, peace sign, stop, and wave. A confidence score is assigned based on landmark geometry and temporal stability.

## How Facial Expression Detection Works

The system tracks face landmarks and measures mouth curvature, eye openness, and expression-related geometry. This is combined with a simple rule-based classifier for neutral, happy, sad, angry, surprised, and confused. The system labels these as detected expressions rather than implied emotions.

## Multimodal Fusion

The fusion layer combines gesture, expression, and short-term temporal context into a unified intent. Example: thumbs up + smile may become "Yes, I am happy."; wave + smile may become "Hello!"; stop + serious may become "Please stop."

## API Documentation

The backend exposes the following endpoints:

- GET /
- GET /health
- POST /predict/gesture
- POST /predict/expression
- POST /predict/intent
- POST /translate
- POST /analyze-frame
- GET /models/status
- WebSocket /ws

## Project Structure

```text
gesture-expression-translator/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── vision/
│   │   ├── hand_detector.py
│   │   ├── face_detector.py
│   │   ├── landmark_processor.py
│   │   └── movement_tracker.py
│   ├── recognition/
│   │   ├── gesture_classifier.py
│   │   ├── expression_classifier.py
│   │   └── confidence.py
│   ├── fusion/
│   │   └── multimodal_fusion.py
│   ├── intent/
│   │   └── intent_engine.py
│   └── api/
│       └── routes.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── config/
│   └── gesture_mapping.json
├── models/
│   └── README.md
├── dataset/
│   └── README.md
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
```

## Supported Gestures

- open palm
- closed fist
- thumbs up
- thumbs down
- peace / V sign
- pointing index finger
- OK gesture
- wave
- stop gesture
- both hands raised

## Supported Expressions

- neutral
- happy
- sad
- angry
- surprised
- confused

## Production Limitations

- real-world accuracy depends on dataset quality, lighting, and user calibration
- continuous live recognition is harder than static demo recognition
- domain-specific gestures need custom data collection for reliable performance
- low-resolution cameras and partial occlusion can reduce confidence

## Production Roadmap

1. Dataset collection and labeling for gesture vocabulary and facial expressions
2. Pretrained model selection and fine-tuning for local deployment
3. Improvement of landmark-based gesture classification with temporal smoothing
4. Multi-user calibration and robustness under lighting variation
5. Deployment-ready API, monitoring, and model versioning
6. Expansion to sign language, multilingual text generation, and stronger intent recognition

## Recommended Datasets and Models

- Hand gestures: HaGRID, ASL Alphabet, EgoGesture, NVGesture, custom local dataset
- Facial expressions: FER2013, CK+, RAF-DB, AffectNet, custom labeled webcam set
- Pretrained backbones: MobileNetV2, ResNet-50, EfficientNet, MediaPipe landmarks + custom classifier
- Use pretrained weights when available and fine-tune on domain-specific custom data

## Future Improvements

- Indian Sign Language support
- custom user calibration
- topic-aware generation using a local NLP model
- richer gesture vocabulary
- temporal models like LSTM/GRU or Transformer-based sequence models
- multilingual generation
- mobile app and wearable integration

## Privacy

Your camera feed is processed locally and is not uploaded to any external service. No webcam frames are stored unless the user explicitly enables recording.

## Screenshots

Prototype-ready UI and detection dashboard are included in the frontend.

## Demo Instructions

Use the Demo Mode buttons to simulate gestures and facial expressions if the webcam is unavailable. This helps during live college demonstrations without needing a real camera.

## Prototype / Planned Feature

Some advanced features in this spec are intentionally implemented as lightweight placeholders and labelled as prototype/planned feature where a real end-to-end model is not available.
