from pathlib import Path
import zipfile
import shutil
import pandas as pd

ROOT = Path("data/raw/SoccerNet/tracking-2023/train/train")
TRACKING_DIR = Path("outputs/tracking/validation")
EVAL_DIR = Path("outputs/evaluation/validation")
TEMP_DIR = EVAL_DIR / "build_buffer60"

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

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Create tracker files
# ---------------------------------------------------------

tracker_build = TEMP_DIR / "tracker" / "data"
tracker_build.mkdir(parents=True, exist_ok=True)

for sequence in SEQUENCES:

    source = TRACKING_DIR / f"{sequence}_botsort_buffer60.csv"
    destination = tracker_build / f"{sequence}.txt"

    df = pd.read_csv(source)

    with open(destination, "w") as f:

        for _, row in df.iterrows():

            frame = int(row["frame"])
            track_id = int(row["track_id"])

            x = float(row["x1"])
            y = float(row["y1"])
            width = float(row["x2"] - row["x1"])
            height = float(row["y2"] - row["y1"])
            confidence = float(row["confidence"])

            f.write(
                f"{frame},{track_id},{x:.2f},{y:.2f},"
                f"{width:.2f},{height:.2f},{confidence:.4f},-1,-1,-1\n"
            )

tracker_zip = EVAL_DIR / "botsort_buffer60.zip"

with zipfile.ZipFile(
    tracker_zip,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for txt in tracker_build.glob("*.txt"):
        z.write(txt, arcname=txt.name)

print(f"Created tracker ZIP: {tracker_zip}")


# ---------------------------------------------------------
# Create GT ZIP
# ---------------------------------------------------------

gt_build = TEMP_DIR / "gt" / "test-evalAI"

for sequence in SEQUENCES:

    sequence_gt = gt_build / sequence / "gt"
    sequence_gt.mkdir(parents=True, exist_ok=True)

    source_gt = ROOT / sequence / "gt" / "gt.txt"
    source_seqinfo = ROOT / sequence / "seqinfo.ini"

    shutil.copy(source_gt, sequence_gt / "gt.txt")
    shutil.copy(
        source_seqinfo,
        gt_build / sequence / "seqinfo.ini"
    )

gt_zip = EVAL_DIR / "gt_buffer60.zip"

with zipfile.ZipFile(
    gt_zip,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for file in gt_build.rglob("*"):

        if file.is_file():

            z.write(
                file,
                arcname=file.relative_to(
                    TEMP_DIR / "gt"
                )
            )

print(f"Created GT ZIP: {gt_zip}")


# ---------------------------------------------------------
# Sequence map
# ---------------------------------------------------------

seqmap = EVAL_DIR / "SNMOT-test-buffer60.txt"

with open(seqmap, "w") as f:

    f.write("name\n")

    for sequence in SEQUENCES:
        f.write(f"{sequence}\n")

print(f"Created sequence map: {seqmap}")

print("\n" + "=" * 60)
print("Buffer 60 evaluation packages created successfully.")
print("=" * 60)