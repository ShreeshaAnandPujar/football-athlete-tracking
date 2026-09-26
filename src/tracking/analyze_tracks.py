import pandas as pd

CSV_PATH = "outputs/tracking/barcelona_botsort.csv"

df = pd.read_csv(CSV_PATH)

# Number of frames each track appears in
track_lengths = (
    df.groupby("track_id")["frame"]
    .nunique()
    .sort_values(ascending=False)
)

print("\n==============================")
print("BARCELONA TRACK ANALYSIS")
print("==============================")

print(f"Frames: {df['frame'].nunique()}")
print(f"Records: {len(df)}")
print(f"Unique IDs: {df['track_id'].nunique()}")

print("\nTrack duration statistics:")
print(track_lengths.describe())

print("\n==============================")
print("TRACK DURATION BUCKETS")
print("==============================")

bins = [0, 5, 15, 30, 60, 100, 200, float("inf")]
labels = [
    "<=5 frames",
    "6-15 frames",
    "16-30 frames",
    "31-60 frames",
    "61-100 frames",
    "101-200 frames",
    ">200 frames",
]

buckets = pd.cut(
    track_lengths,
    bins=bins,
    labels=labels
)

print(buckets.value_counts().sort_index())

print("\n==============================")
print("LONGEST TRACKS")
print("==============================")

print(track_lengths.head(20))