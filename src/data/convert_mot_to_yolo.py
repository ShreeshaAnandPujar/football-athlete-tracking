from pathlib import Path
import shutil
import random
import configparser


# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = Path(
    "data/raw/SoccerNet/tracking-2023/train/train"
)

OUTPUT_DIR = Path(
    "data/processed/yolo"
)

TRAIN_RATIO = 0.8
SEED = 42


# ============================================================
# Create output directories
# ============================================================

for split in ["train", "val"]:
    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Find sequences
# ============================================================

sequences = sorted(
    [
        p for p in SOURCE_DIR.iterdir()
        if p.is_dir() and p.name.startswith("SNMOT-")
    ]
)

print(f"Found {len(sequences)} sequences.")


# ============================================================
# Train / validation split
# ============================================================

random.seed(SEED)

random.shuffle(sequences)

split_index = int(len(sequences) * TRAIN_RATIO)

train_sequences = sequences[:split_index]
val_sequences = sequences[split_index:]


print(f"Training sequences:   {len(train_sequences)}")
print(f"Validation sequences: {len(val_sequences)}")


# ============================================================
# Conversion function
# ============================================================

def convert_sequence(sequence_dir, split):

    # --------------------------------------------------------
    # Read sequence information
    # --------------------------------------------------------

    seqinfo_path = sequence_dir / "seqinfo.ini"

    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    width = int(config["Sequence"]["imWidth"])
    height = int(config["Sequence"]["imHeight"])

    # --------------------------------------------------------
    # Read ground truth
    # --------------------------------------------------------

    gt_path = sequence_dir / "gt" / "gt.txt"

    annotations = {}

    with open(gt_path, "r") as f:

        for line in f:

            parts = line.strip().split(",")

            if len(parts) < 6:
                continue

            frame_id = int(parts[0])

            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            # Ignore invalid boxes
            if w <= 0 or h <= 0:
                continue

            # ------------------------------------------------
            # MOT → YOLO
            # ------------------------------------------------

            center_x = x + w / 2
            center_y = y + h / 2

            center_x /= width
            center_y /= height
            w /= width
            h /= height

            # Clamp values to valid YOLO range
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            yolo_line = (
                f"0 "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

            annotations.setdefault(
                frame_id,
                []
            ).append(yolo_line)

    # --------------------------------------------------------
    # Copy images + labels
    # --------------------------------------------------------

    image_dir = sequence_dir / "img1"

    converted = 0

    for frame_id, labels in annotations.items():

        image_name = f"{frame_id:06d}.jpg"

        source_image = image_dir / image_name

        if not source_image.exists():
            continue

        destination_image = (
            OUTPUT_DIR
            / "images"
            / split
            / f"{sequence_dir.name}_{image_name}"
        )

        destination_label = (
            OUTPUT_DIR
            / "labels"
            / split
            / f"{sequence_dir.name}_{frame_id:06d}.txt"
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        with open(destination_label, "w") as f:
            f.write("\n".join(labels))

        converted += 1

    return converted


# ============================================================
# Run conversion
# ============================================================

train_frames = 0
val_frames = 0


print("\nConverting training sequences...")

for sequence in train_sequences:

    count = convert_sequence(
        sequence,
        "train"
    )

    train_frames += count

    print(
        f"{sequence.name}: {count} frames"
    )


print("\nConverting validation sequences...")

for sequence in val_sequences:

    count = convert_sequence(
        sequence,
        "val"
    )

    val_frames += count

    print(
        f"{sequence.name}: {count} frames"
    )


# ============================================================
# Summary
# ============================================================

print("\n======================================")
print("Conversion complete")
print("======================================")

print(f"Training frames:   {train_frames}")
print(f"Validation frames: {val_frames}")

print(f"\nDataset saved to:")
print(OUTPUT_DIR)