from pathlib import Path
import cv2
import csv


# ============================================================
# Paths
# ============================================================

IMAGE_DIR = Path(
    "data/raw/SoccerNet/tracking-2023/"
    "train/train/SNMOT-113/img1"
)

CSV_PATH = Path(
    "outputs/tracking/soccernet_tracking.csv"
)

OUTPUT_DIR = Path("outputs/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "soccernet_bytetrack.mp4"


# ============================================================
# Load tracking CSV
# ============================================================

tracks = {}

with open(CSV_PATH, "r", newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        frame = int(row["frame"])

        tracks.setdefault(frame, []).append({
            "track_id": int(row["track_id"]),
            "confidence": float(row["confidence"]),
            "x1": float(row["x1"]),
            "y1": float(row["y1"]),
            "x2": float(row["x2"]),
            "y2": float(row["y2"]),
        })


# ============================================================
# Read first frame to determine dimensions
# ============================================================

first_frame = IMAGE_DIR / "000001.jpg"

frame = cv2.imread(str(first_frame))

if frame is None:
    raise FileNotFoundError(first_frame)

height, width = frame.shape[:2]


# ============================================================
# Video writer
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    25,
    (width, height)
)


# ============================================================
# Render frames
# ============================================================

image_paths = sorted(
    IMAGE_DIR.glob("*.jpg")
)

for frame_number, image_path in enumerate(
    image_paths,
    start=1
):

    frame = cv2.imread(str(image_path))

    if frame is None:
        continue

    current_tracks = tracks.get(
        frame_number,
        []
    )

    for track in current_tracks:

        track_id = track["track_id"]
        confidence = track["confidence"]

        x1 = int(track["x1"])
        y1 = int(track["y1"])
        x2 = int(track["x2"])
        y2 = int(track["y2"])

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = (
            f"ID {track_id} "
            f"{confidence:.2f}"
        )

        label_y = max(
            y1 - 8,
            20
        )

        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # Frame counter
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    writer.write(frame)


# ============================================================
# Finish
# ============================================================

writer.release()

print()
print("===================================")
print("Tracking visualization complete")
print("===================================")
print(f"Frames rendered: {len(image_paths)}")
print(f"Output: {OUTPUT_VIDEO}")