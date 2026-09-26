from pathlib import Path
import pandas as pd


SEQ = "SNMOT-113"

GT_PATH = Path(
    f"data/raw/SoccerNet/tracking-2023/"
    f"train/train/{SEQ}/gt/gt.txt"
)

OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_prediction(input_csv, output_txt):

    df = pd.read_csv(input_csv)

    with open(output_txt, "w") as f:

        for _, row in df.iterrows():

            frame = int(row["frame"])
            track_id = int(row["track_id"])

            x = float(row["x1"])
            y = float(row["y1"])
            w = float(row["x2"] - row["x1"])
            h = float(row["y2"] - row["y1"])

            confidence = float(row["confidence"])

            # MOT/SoccerNet format:
            # frame,id,x,y,w,h,confidence,-1,-1,-1

            f.write(
                f"{frame},{track_id},"
                f"{x:.2f},{y:.2f},"
                f"{w:.2f},{h:.2f},"
                f"{confidence:.4f},-1,-1,-1\n"
            )


# Copy ground truth into evaluation directory
gt_output = OUTPUT_DIR / "gt.txt"
gt_output.write_text(GT_PATH.read_text())


# ByteTrack
convert_prediction(
    "outputs/tracking/soccernet_tracking.csv",
    OUTPUT_DIR / "bytetrack.txt"
)


# BoT-SORT
convert_prediction(
    "outputs/tracking/soccernet_botsort.csv",
    OUTPUT_DIR / "botsort.txt"
)


print("Evaluation files prepared.")
print(f"Ground truth: {gt_output}")
print(f"ByteTrack:    {OUTPUT_DIR / 'bytetrack.txt'}")
print(f"BoT-SORT:     {OUTPUT_DIR / 'botsort.txt'}")