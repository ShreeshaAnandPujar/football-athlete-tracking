import sys
import argparse
from pathlib import Path
import pandas as pd

MASTER_CSV = Path("outputs/evaluation/validation/all_experiments_master.csv")

def compare(baseline_name, candidate_names):
    if not MASTER_CSV.exists():
        print(f"Master evaluation CSV not found at {MASTER_CSV}")
        return

    df = pd.read_csv(MASTER_CSV)
    df.set_index("tracker", inplace=True)

    if baseline_name not in df.index:
        print(f"Baseline tracker '{baseline_name}' not found in master records.")
        print(f"Available trackers: {list(df.index)}")
        return

    base_row = df.loc[baseline_name]
    metrics = ["HOTA", "AssA", "IDF1", "IDSW", "Frag", "DetA", "MOTA"]

    rows = []
    rows.append({
        "Role": "Baseline",
        "Tracker": baseline_name,
        "Description": base_row.get("description", ""),
        **{m: round(float(base_row[m]), 3) for m in metrics}
    })

    for cand in candidate_names:
        if cand not in df.index:
            print(f"Candidate '{cand}' not found in master records. Skipping.")
            continue
        cand_row = df.loc[cand]
        row_dict = {
            "Role": "Candidate",
            "Tracker": cand,
            "Description": cand_row.get("description", "")
        }
        for m in metrics:
            val = float(cand_row[m])
            diff = val - float(base_row[m])
            diff_str = f"{diff:+.3f}" if m not in ["IDSW", "Frag"] else f"{int(diff):+d}"
            val_str = f"{val:.3f}" if m not in ["IDSW", "Frag"] else f"{int(val)}"
            row_dict[m] = f"{val_str} ({diff_str})"
        rows.append(row_dict)

    df_comp = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(f"EXPERIMENT COMPARISON (Baseline: {baseline_name})")
    print("=" * 100)
    print(df_comp.to_markdown(index=False))
    print("=" * 100)

def main():
    parser = argparse.ArgumentParser(description="Compare tracking experiments with delta computation.")
    parser.add_argument("--baseline", default="botsort", help="Baseline tracker name (default: botsort)")
    parser.add_argument("--candidates", nargs="+", default=["botsort_conf35_orb", "botsort_conf35_orb_match85", "botsort_conf35_orb_match85_reid"], help="One or more candidate tracker names to compare against baseline")
    args = parser.parse_args()

    compare(args.baseline, args.candidates)

if __name__ == "__main__":
    main()
