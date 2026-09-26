import sys
import argparse
import subprocess
import json
from pathlib import Path
import pandas as pd

PYTHON_EXE = sys.executable
TRACKEVAL_SCRIPT = Path("tools/sn-trackeval/scripts/run_soccernet_mot.py")
EVAL_DIR = Path("outputs/evaluation/validation")
GT_ZIP = EVAL_DIR / "gt_conf20.zip"
SEQMAP = EVAL_DIR / "SNMOT-test-conf20.txt"

HOTA_HEADERS = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA', 'HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
CLEAR_HEADERS = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA', 'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
ID_HEADERS = ['IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP']

def run_evaluation(experiment_name, output_json=None):
    tracker_zip = EVAL_DIR / f"{experiment_name}.zip"
    if not tracker_zip.exists():
        raise FileNotFoundError(f"Evaluation zip not found: {tracker_zip}. Please ensure tracker predictions have been zipped for TrackEval.")

    if not GT_ZIP.exists():
        raise FileNotFoundError(f"Ground truth zip not found: {GT_ZIP}")

    print(f"\n{'='*70}\n[TRACKEVAL] Evaluating Experiment: {experiment_name}\n{'='*70}")
    cmd = [
        PYTHON_EXE,
        str(TRACKEVAL_SCRIPT),
        "--BENCHMARK", "SNMOT",
        "--SPLIT_TO_EVAL", "test",
        "--DO_PREPROC", "False",
        "--SEQMAP_FILE", str(SEQMAP),
        "--TRACKERS_TO_EVAL", experiment_name,
        "--TRACKERS_FOLDER_ZIP", str(tracker_zip),
        "--GT_FOLDER_ZIP", str(GT_ZIP),
        "--INPUT_AS_ZIP", "True",
        "--PRINT_RESULTS", "True",
        "--PRINT_ONLY_COMBINED", "True"
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[ERROR] TrackEval execution failed:")
        print(res.stderr)
        sys.exit(res.returncode)

    lines = [l.strip() for l in res.stdout.split("\n") if l.strip()]
    metrics = {"tracker": experiment_name}

    for i, l in enumerate(lines):
        if l.startswith("HOTA:"):
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                for h, v in zip(HOTA_HEADERS, lines[i+1].split()[1:]):
                    try: metrics[h] = float(v)
                    except: metrics[h] = v
        elif l.startswith("CLEAR:"):
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                for h, v in zip(CLEAR_HEADERS, lines[i+1].split()[1:]):
                    try: metrics[h] = float(v)
                    except: metrics[h] = v
        elif l.startswith("Identity:"):
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                for h, v in zip(ID_HEADERS, lines[i+1].split()[1:]):
                    try: metrics[h] = float(v)
                    except: metrics[h] = v

    # Print Clean Formatted Result Table
    display_keys = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag"]
    data = {k: [metrics.get(k, "N/A")] for k in display_keys}
    df = pd.DataFrame(data)

    print("\n" + "-"*50)
    print(f"OFFICIAL BENCHMARK RESULTS: {experiment_name}")
    print("-"*50)
    for k in display_keys:
        val = metrics.get(k, 'N/A')
        if isinstance(val, float):
            if k in ["IDSW", "Frag"]:
                print(f"  {k:<10} {int(val)}")
            else:
                print(f"  {k:<10} {val:.3f}")
        else:
            print(f"  {k:<10} {val}")
    print("-" * 50)

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON to: {out_path}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run SoccerNet MOT TrackEval benchmark for an experiment.")
    parser.add_argument("--experiment", required=True, help="Name of the experiment tracker (e.g., botsort_conf35_orb_match85)")
    parser.add_argument("--json", default=None, help="Optional output JSON path for the metrics")
    args = parser.parse_args()

    run_evaluation(args.experiment, output_json=args.json)

if __name__ == "__main__":
    main()
