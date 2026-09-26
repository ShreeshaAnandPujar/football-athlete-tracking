from pathlib import Path
import cv2


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

image_path = Path(
    "data/processed/yolo_test/images/000001.jpg"
)

label_path = Path(
    "data/processed/yolo_test/labels/000001.txt"
)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

with open(label_path, "r") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        center_x = float(parts[1]) * IMAGE_WIDTH
        center_y = float(parts[2]) * IMAGE_HEIGHT
        width = float(parts[3]) * IMAGE_WIDTH
        height = float(parts[4]) * IMAGE_HEIGHT

        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"player",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

output_path = Path("outputs/yolo_visualization.jpg")
cv2.imwrite(str(output_path), image)

print(f"Saved: {output_path}")