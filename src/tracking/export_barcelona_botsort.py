from ultralytics import YOLO
from pathlib import Path
import csv


MODEL_PATH = (
    "runs/detect/outputs/training/"
    "yolov8s_soccernet_baseline/weights/best.pt"
)

VIDEO_PATH = Path(
    "data/videos/barcelona_highlight.mp4"
)

OUTPUT_DIR = Path("outputs/tracking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "barcelona_botsort.csv"


model = YOLO(MODEL_PATH)


with open(CSV_PATH, "w", newline="") as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "frame",
        "track_id",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ])

    frame_number = 0

    results = model.track(
        source=str(VIDEO_PATH),

        # BoT-SORT
        tracker="botsort.yaml",

        persist=True,

        # Same threshold as our ByteTrack experiment
        conf=0.30,

        imgsz=640,
        device=0,

        stream=True,
        verbose=False,
    )

    for result in results:

        frame_number += 1

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for box, track_id, cls, confidence in zip(
            boxes,
            track_ids,
            classes,
            confidences,
        ):

            x1, y1, x2, y2 = box

            writer.writerow([
                frame_number,
                track_id,
                cls,
                round(float(confidence), 4),
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2),
            ])


print()
print("======================================")
print("Barcelona BoT-SORT export complete")
print("======================================")
print(f"Frames processed: {frame_number}")
print(f"CSV: {CSV_PATH}")