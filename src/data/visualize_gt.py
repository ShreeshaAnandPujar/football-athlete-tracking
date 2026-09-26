from pathlib import Path
import cv2


SEQ_DIR = Path(
    "data/raw/SoccerNet/tracking-2023/train/train/SNMOT-060"
)

IMAGE_PATH = SEQ_DIR / "img1" / "000001.jpg"
GT_PATH = SEQ_DIR / "gt" / "gt.txt"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Read ground-truth annotations
with open(GT_PATH, "r") as f:
    for line in f:
        parts = line.strip().split(",")

        if len(parts) < 6:
            continue

        frame_id = int(parts[0])

        # We only want frame 1
        if frame_id != 1:
            continue

        track_id = int(parts[1])
        x = int(float(parts[2]))
        y = int(float(parts[3]))
        w = int(float(parts[4]))
        h = int(float(parts[5]))

        x2 = x + w
        y2 = y + h

        cv2.rectangle(
            image,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"ID {track_id}",
            (x, max(y - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

output_path = Path("outputs/gt_visualization.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output_path), image)

print(f"Saved visualization to: {output_path}")