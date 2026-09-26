from pathlib import Path
from ultralytics import YOLO
import csv
import time

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    "runs/detect/outputs/training/"
    "yolov8s_soccernet_baseline/weights/best.pt"
)

ROOT = Path("data/raw/SoccerNet/tracking-2023/train/train")

OUTPUT_DIR = Path("outputs/tracking/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCES = [
    "SNMOT-113",
    "SNMOT-157",
    "SNMOT-066",
    "SNMOT-167",
    "SNMOT-068",
    "SNMOT-074",
    "SNMOT-075",
    "SNMOT-077",
    "SNMOT-161",
    "SNMOT-061",
    "SNMOT-067",
    "SNMOT-154",
]

# ============================================================
# Tracker runner
# ============================================================

def run_tracker(sequence_name, tracker_name):

    image_dir = ROOT / sequence_name / "img1"

    output_csv = (
        OUTPUT_DIR /
        f"{sequence_name}_{tracker_name}.csv"
    )

    model = YOLO(MODEL_PATH)

    frame_count = 0
    record_count = 0

    print()
    print("=" * 60)
    print(f"Sequence : {sequence_name}")
    print(f"Tracker  : {tracker_name}")
    print("=" * 60)

    start_time = time.time()

    with open(output_csv, "w", newline="") as csv_file:

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

        for image_path in sorted(image_dir.glob("*.jpg")):

            frame_count += 1

            results = model.track(
                source=str(image_path),
                tracker=f"{tracker_name}.yaml",
                persist=True,
                conf=0.20,
                imgsz=640,
                device=0,
                verbose=False,
            )

            result = results[0]

            if (
                result.boxes is None
                or result.boxes.id is None
            ):
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()

            for box, track_id, cls, conf in zip(
                boxes,
                ids,
                classes,
                confs
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

                record_count += 1

    elapsed = time.time() - start_time

    print(f"Frames processed : {frame_count}")
    print(f"Tracking records : {record_count}")
    print(f"Time              : {elapsed / 60:.2f} min")
    print(f"Output            : {output_csv}")

    return frame_count, record_count


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    trackers = [
    "botsort",
    ]

    total_start = time.time()

    for tracker in trackers:

        print()
        print("#" * 70)
        print(f"RUNNING {tracker.upper()}")
        print("#" * 70)

        for sequence in SEQUENCES:

            run_tracker(
                sequence,
                tracker
            )

    total_time = time.time() - total_start

    print()
    print("=" * 70)
    print("VALIDATION BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"Total runtime: {total_time / 60:.2f} minutes"
    )

    print()
    print("Generated files:")
    
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(path)