from ultralytics import YOLO
from pathlib import Path


# ============================================================
# Paths
# ============================================================

MODEL_PATH = (
    "runs/detect/outputs/training/"
    "yolov8s_soccernet_baseline/weights/best.pt"
)

VIDEO_PATH = Path(
    "data/videos/barcelona_highlight.mp4"
)

OUTPUT_DIR = Path(
    "outputs/videos"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load model
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)


# ============================================================
# Run detection + tracking
# ============================================================

print("Starting Barcelona video tracking...")

results = model.track(
    source=str(VIDEO_PATH),

    # ByteTrack
    tracker="bytetrack.yaml",

    # Keep IDs between frames
    persist=True,

    # Detection threshold
    conf=0.25,

    # Model input size
    imgsz=640,

    # RTX 2050
    device=0,

    # Save annotated video
    save=True,

    project=str(OUTPUT_DIR),

    name="barcelona_bytetrack",

    verbose=True,
)


print()
print("======================================")
print("Barcelona tracking complete")
print("======================================")
print(
    f"Output directory: "
    f"{OUTPUT_DIR / 'barcelona_bytetrack'}"
)