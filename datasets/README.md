# Datasets

This folder is reserved for production training datasets.

## Gesture data
- datasets/gesture/
- use HaGRID, ASL Alphabet, EgoGesture, NVGesture, or custom webcam capture

## Expression data
- datasets/expression/
- use FER2013, CK+, AffectNet, RAF-DB, or custom labeled face captures

## Recommended structure

```text
datasets/
├── gesture/
│   ├── train/
│   ├── val/
│   └── test/
├── expression/
│   ├── train/
│   ├── val/
│   └── test/
└── README.md
```
