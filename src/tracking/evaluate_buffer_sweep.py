import subprocess
import pandas as pd
from pathlib import Path

EVAL_DIR = Path("outputs/evaluation/validation")

def evaluate(tracker_name):
    cmd = [
        ".venv/Scripts/python.exe",
        "tools/sn-trackeval/scripts/run_soccernet_mot.py",
        "--BENCHMARK", "SNMOT",
        "--SPLIT_TO_EVAL", "test",
        "--DO_PREPROC", "False",
        "--SEQMAP_FILE", "outputs/evaluation/validation/SNMOT-test-conf20.txt",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--TRACKERS_FOLDER_ZIP", f"outputs/evaluation/validation/{tracker_name}.zip",
        "--GT_FOLDER_ZIP", "outputs/evaluation/validation/gt_conf20.zip",
        "--INPUT_AS_ZIP", "True",
        "--PRINT_RESULTS", "True",
        "--PRINT_ONLY_COMBINED", "True"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout
    lines = [l.strip() for l in out.split("\n") if l.strip()]
    metrics = {"tracker": tracker_name}
    for i, l in enumerate(lines):
        if l.startswith("HOTA:"):
            # Headers are after tracker name
            headers = l.split()[2:]
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                vals = lines[i+1].split()[1:]
                for h, v in zip(headers, vals):
                    try:
                        metrics[h] = float(v)
                    except:
                        metrics[h] = v
        elif l.startswith("CLEAR:"):
            headers = l.split()[2:]
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                vals = lines[i+1].split()[1:]
                for h, v in zip(headers, vals):
                    try:
                        metrics[h] = float(v)
                    except:
                        metrics[h] = v
        elif l.startswith("Identity:"):
            headers = ["IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"]
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                vals = lines[i+1].split()[1:]
                for h, v in zip(headers, vals):
                    try:
                        metrics[h] = float(v)
                    except:
                        metrics[h] = v
    return metrics

if __name__ == "__main__":
    trackers = [
        ("botsort_buffer15", "track_buffer = 15"),
        ("botsort", "track_buffer = 30 (baseline)"),
        ("botsort_buffer45", "track_buffer = 45"),
        ("botsort_buffer60", "track_buffer = 60"),
        ("botsort_buffer90", "track_buffer = 90"),
    ]
    
    results = []
    for t_name, desc in trackers:
        m = evaluate(t_name)
        m["description"] = desc
        results.append(m)
        
    df = pd.DataFrame(results)
    cols = ["description", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]
    print(df[cols].to_markdown(index=False))
