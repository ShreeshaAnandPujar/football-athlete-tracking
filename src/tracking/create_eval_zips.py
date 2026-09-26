from pathlib import Path
import zipfile
import shutil

ROOT = Path("outputs/evaluation")
TEMP = ROOT / "zip_build"

GT_SOURCE = ROOT / "gt.txt"

# Clean old temporary files
if TEMP.exists():
    shutil.rmtree(TEMP)

# --------------------------------------------------
# Create tracker ZIP
# --------------------------------------------------

def create_tracker_zip(tracker_name):
    tracker_dir = TEMP / tracker_name / "data"
    tracker_dir.mkdir(parents=True)

    source = ROOT / f"{tracker_name}.txt"
    destination = tracker_dir / "SNMOT-113.txt"

    shutil.copy(source, destination)

    zip_path = ROOT / f"{tracker_name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(
            destination,
            arcname=f"{tracker_name}/data/SNMOT-113.txt"
        )

    print(f"Created: {zip_path}")


# --------------------------------------------------
# Create GT ZIP
# --------------------------------------------------

gt_dir = TEMP / "gt" / "test-evalAI" / "SNMOT-113" / "gt"
gt_dir.mkdir(parents=True)

shutil.copy(
    GT_SOURCE,
    gt_dir / "gt.txt"
)

gt_zip = ROOT / "gt.zip"

with zipfile.ZipFile(gt_zip, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(
        gt_dir / "gt.txt",
        arcname="test-evalAI/SNMOT-113/gt/gt.txt"
    )

print(f"Created: {gt_zip}")


# --------------------------------------------------
# Sequence map
# --------------------------------------------------

seqmap = ROOT / "SNMOT-test.txt"

seqmap.write_text(
    "name\n"
    "SNMOT-113\n"
)

print(f"Created: {seqmap}")


# --------------------------------------------------
# Tracker ZIPs
# --------------------------------------------------

create_tracker_zip("bytetrack")
create_tracker_zip("botsort")

print("\nEvaluation package preparation complete.")