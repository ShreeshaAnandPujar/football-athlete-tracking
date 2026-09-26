from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = (
    "runs/detect/outputs/training/"
    "yolov8s_soccernet_baseline/weights/best.pt"
)

IMAGE_PATH = (
    "data/processed/yolo/images/val/"
    "SNMOT-113_000001.jpg"
)

OUTPUT_DIR = Path("outputs/detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    imgsz=640,
    device=0,
    save=True,
    project="outputs/detections",
    name="single_frame",
)

print("Detection complete.")
print(f"Results saved to: {OUTPUT_DIR / 'single_frame'}")