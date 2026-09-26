from ultralytics import YOLO
from pathlib import Path
import csv

MODEL_PATH = (
    "runs/detect/outputs/training/"
    "yolov8s_soccernet_baseline/weights/best.pt"
)

IMAGE_DIR = Path(
    "data/raw/SoccerNet/tracking-2023/"
    "train/train/SNMOT-113/img1"
)

OUTPUT_DIR = Path("outputs/tracking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "soccernet_botsort.csv"

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

    frame_count = 0

    for image_path in sorted(IMAGE_DIR.glob("*.jpg")):
        frame_count += 1

        results = model.track(
            source=str(image_path),
            tracker="botsort.yaml",
            persist=True,
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        for box, track_id, cls, conf in zip(
            boxes, ids, classes, confs
        ):
            x1, y1, x2, y2 = box

            writer.writerow([
                frame_count,
                track_id,
                cls,
                round(float(conf), 4),
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2),
            ])

print()
print("BoT-SORT SoccerNet tracking complete.")
print(f"Frames processed: {frame_count}")
print(f"CSV: {CSV_PATH}")