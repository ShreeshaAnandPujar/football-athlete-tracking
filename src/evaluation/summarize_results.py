import sys
import argparse
from pathlib import Path
import pandas as pd

MASTER_CSV = Path("outputs/evaluation/validation/all_experiments_master.csv")

def summarize_all(sort_by="HOTA", ascending=False):
    if not MASTER_CSV.exists():
        print(f"Master evaluation CSV not found at {MASTER_CSV}")
        return

    df = pd.read_csv(MASTER_CSV)
    cols = ["tracker", "experiment", "description", "HOTA", "AssA", "IDF1", "IDSW", "Frag", "DetA", "MOTA"]
    cols = [c for c in cols if c in df.columns]

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    print("\n" + "=" * 95)
    print("ALL EXPERIMENTS BENCHMARK SUMMARY (SoccerNet Tracking-2023 Validation)")
    print("=" * 95)
    print(df[cols].to_markdown(index=False))
    print("=" * 95)

def main():
    parser = argparse.ArgumentParser(description="Summarize all tracking experiments.")
    parser.add_argument("--sort", default="HOTA", help="Metric to sort by (e.g. HOTA, AssA, IDF1, IDSW)")
    parser.add_argument("--asc", action="store_true", help="Sort in ascending order (useful for IDSW / Frag)")
    args = parser.parse_args()

    summarize_all(sort_by=args.sort, ascending=args.asc)

if __name__ == "__main__":
    main()
