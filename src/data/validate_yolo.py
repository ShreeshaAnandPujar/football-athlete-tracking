from pathlib import Path


DATASET_DIR = Path("data/processed/yolo")

total = 0
valid = 0
empty = 0
malformed = 0
out_of_range = 0

for split in ["train", "val"]:

    label_dir = DATASET_DIR / "labels" / split

    for label_file in label_dir.glob("*.txt"):

        total += 1

        with open(label_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty += 1
            continue

        file_valid = True

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                malformed += 1
                file_valid = False
                continue

            try:
                class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]
            except ValueError:
                malformed += 1
                file_valid = False
                continue

            # YOLO normalized coordinates must be 0-1
            if not all(0 <= value <= 1 for value in values):
                out_of_range += 1
                file_valid = False

            if class_id != 0:
                malformed += 1
                file_valid = False

        if file_valid:
            valid += 1


print("\n==============================")
print("YOLO DATASET VALIDATION")
print("==============================")

print(f"Total label files: {total}")
print(f"Valid files:       {valid}")
print(f"Empty files:       {empty}")
print(f"Malformed files:   {malformed}")
print(f"Out-of-range:      {out_of_range}")

print("==============================")

if (
    empty == 0
    and malformed == 0
    and out_of_range == 0
):
    print("DATASET VALIDATION PASSED")
else:
    print("DATASET VALIDATION FAILED")