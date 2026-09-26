from pathlib import Path
import zipfile
import shutil

# ============================================================
# Paths
# ============================================================

ROOT = Path(".")

TRACKING_DIR = ROOT / "outputs" / "tracking" / "validation"
EVAL_DIR = ROOT / "outputs" / "evaluation" / "validation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

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

TRACKER_NAME = "botsort_reid"


# ============================================================
# 1. Create tracker ZIP
# ============================================================

import pandas as pd

tracker_zip = EVAL_DIR / f"{TRACKER_NAME}.zip"

if tracker_zip.exists():
    tracker_zip.unlink()

with zipfile.ZipFile(tracker_zip, "w", zipfile.ZIP_DEFLATED) as z:

    for seq in SEQUENCES:

        csv_file = TRACKING_DIR / f"{seq}_{TRACKER_NAME}.csv"

        if not csv_file.exists():
            raise FileNotFoundError(
                f"Missing tracking file:\n{csv_file}"
            )

        txt_file = EVAL_DIR / f"{seq}.txt"

        # Actual CSV columns:
        # frame, track_id, class, confidence, x1, y1, x2, y2
        df = pd.read_csv(csv_file)

        required_columns = [
            "frame",
            "track_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {csv_file}: {missing}"
            )

        with open(txt_file, "w", encoding="utf-8") as f:

            for _, row in df.iterrows():

                frame = int(row["frame"])
                track_id = int(row["track_id"])

                x = float(row["x1"])
                y = float(row["y1"])

                w = float(row["x2"] - row["x1"])
                h = float(row["y2"] - row["y1"])

                confidence = float(row["confidence"])

                # TrackEval MOT format:
                #
                # frame,id,x,y,w,h,confidence,class,visibility,unused
                #
                # class = 1
                # visibility = 1
                # unused = 1

                f.write(
                    f"{frame},{track_id},"
                    f"{x:.4f},{y:.4f},"
                    f"{w:.4f},{h:.4f},"
                    f"{confidence:.4f},"
                    f"1,1,1\n"
                )

        # Tracker ZIP structure:
        # botsort_match70/SNMOT-113.txt
        z.write(
            txt_file,
            arcname=f"{TRACKER_NAME}/{seq}.txt"
        )

        txt_file.unlink()

print("Created tracker ZIP:")
print(tracker_zip)


# ============================================================
# 2. Create TrackEval-compatible GT ZIP
# ============================================================

gt_zip = EVAL_DIR / "gt_reid.zip"

if gt_zip.exists():
    gt_zip.unlink()

GT_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "SoccerNet"
    / "tracking-2023"
    / "train"
    / "train"
)

with zipfile.ZipFile(gt_zip, "w", zipfile.ZIP_DEFLATED) as z:

    for seq in SEQUENCES:

        seq_root = GT_ROOT / seq

        gt_file = seq_root / "gt" / "gt.txt"
        seqinfo_file = seq_root / "seqinfo.ini"

        if not gt_file.exists():
            raise FileNotFoundError(
                f"Missing GT file:\n{gt_file}"
            )

        if not seqinfo_file.exists():
            raise FileNotFoundError(
                f"Missing seqinfo.ini:\n{seqinfo_file}"
            )

        # Create temporary MOT-compatible GT.
        temp_gt = EVAL_DIR / f"{seq}_gt_match70.txt"

        with open(gt_file, "r", encoding="utf-8") as src:
            with open(temp_gt, "w", encoding="utf-8") as dst:

                for line in src:
                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split(",")

                    if len(parts) < 10:
                        continue

                    frame = parts[0].strip()
                    track_id = parts[1].strip()
                    x = parts[2].strip()
                    y = parts[3].strip()
                    w = parts[4].strip()
                    h = parts[5].strip()
                    confidence = parts[6].strip()

                    # MOT class = 1 (pedestrian/player)
                    # Visibility = 1
                    dst.write(
                        f"{frame},{track_id},{x},{y},{w},{h},"
                        f"{confidence},1,1,1\n"
                    )

        z.write(
            temp_gt,
            arcname=f"test-evalAI/{seq}/gt/gt.txt"
        )

        z.write(
            seqinfo_file,
            arcname=f"test-evalAI/{seq}/seqinfo.ini"
        )

        temp_gt.unlink()

print(f"Created TrackEval-compatible GT ZIP:")
print(gt_zip)


# ============================================================
# 3. Create sequence map
# ============================================================

seqmap_file = EVAL_DIR / "SNMOT-test-reid.txt"

with open(seqmap_file, "w", encoding="utf-8") as f:
    f.write("name\n")

    for seq in SEQUENCES:
        f.write(f"{seq}\n")


print(f"Created sequence map:")
print(seqmap_file)


# ============================================================
# 4. Summary
# ============================================================

print()
print("=" * 60)
print("MATCH70 EVALUATION PACKAGE CREATED")
print("=" * 60)

print(f"Tracker ZIP : {tracker_zip}")
print(f"GT ZIP      : {gt_zip}")
print(f"Seqmap      : {seqmap_file}")

print()
print("Sequences:")
for seq in SEQUENCES:
    print(f"  ✓ {seq}")

print()
print("Ready for TrackEval.")