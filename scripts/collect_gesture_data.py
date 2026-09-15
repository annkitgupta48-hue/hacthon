from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2


def collect(label: str, seconds: int, output_root: Path, camera_index: int = 0) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    label_dir = output_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.csv"
    write_header = not metadata_path.exists()

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and camera_index.")

    started = time.time()
    sample_id = int(started * 1000)
    with metadata_path.open("a", newline="", encoding="utf-8") as metadata_file:
        writer = csv.writer(metadata_file)
        if write_header:
            writer.writerow(["file", "label", "timestamp"])

        while time.time() - started < seconds:
            ok, frame = camera.read()
            if not ok:
                continue
            filename = f"{label}_{sample_id}.jpg"
            target = label_dir / filename
            cv2.imwrite(str(target), frame)
            writer.writerow([str(target.relative_to(output_root)), label, time.time()])
            sample_id += 1
            cv2.imshow("GestureSpeak dataset capture - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved samples for '{label}' in {label_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture labeled webcam frames for GestureSpeak training.")
    parser.add_argument("label", help="Gesture label, e.g. wave, thumbs_up, stop")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", default="datasets/gesture/raw")
    args = parser.parse_args()
    collect(args.label, args.seconds, Path(args.output), args.camera_index)


if __name__ == "__main__":
    main()