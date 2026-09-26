from ultralytics import YOLO


model = YOLO("yolov8s.pt")

model.train(
    data="data/processed/yolo/data.yaml",

    # Real baseline training
    epochs=20,
    imgsz=640,
    batch=2,

    # RTX 2050
    device=0,

    # Windows stability
    workers=0,

    # Save checkpoints
    save=True,
    save_period=5,

    project="outputs/training",
    name="yolov8s_soccernet_baseline",

    # Reproducibility
    seed=42,

    # Use pretrained weights
    pretrained=True,
)