import os
import sys
import time
import csv
import zipfile
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# ============================================================
# Paths & Configuration
# ============================================================
ROOT = Path("data/raw/SoccerNet/tracking-2023/train/train")
MODEL_PATH = "runs/detect/outputs/training/yolov8s_soccernet_baseline/weights/best.pt"
TRACKING_DIR = Path("outputs/tracking/validation")
EVAL_DIR = Path("outputs/evaluation/validation")
RESULTS_CSV = EVAL_DIR / "experiment_summary.csv"

TRACKING_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = sys.executable

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

EXPERIMENTS = {
    # ---------------------------------------------------------
    # Experiment 4: Track Buffer Sweep (baseline buffer=30, conf=0.20)
    # ---------------------------------------------------------
    "botsort_buffer15": {
        "exp": "Exp 4: Buffer Sweep",
        "yaml": "src/tracking/botsort_buffer15.yaml",
        "conf": 0.20,
        "note": "track_buffer=15"
    },
    "botsort_buffer45": {
        "exp": "Exp 4: Buffer Sweep",
        "yaml": "src/tracking/botsort_buffer45.yaml",
        "conf": 0.20,
        "note": "track_buffer=45"
    },
    "botsort_buffer60": {
        "exp": "Exp 4: Buffer Sweep",
        "yaml": "src/tracking/botsort_buffer60.yaml",
        "conf": 0.20,
        "note": "track_buffer=60"
    },
    "botsort_buffer90": {
        "exp": "Exp 4: Buffer Sweep",
        "yaml": "src/tracking/botsort_buffer90.yaml",
        "conf": 0.20,
        "note": "track_buffer=90"
    },

    # ---------------------------------------------------------
    # Experiment 5: Detector Confidence Sweep (buffer=30)
    # ---------------------------------------------------------
    "botsort_conf15": {
        "exp": "Exp 5: Confidence Sweep",
        "yaml": "src/tracking/botsort.yaml",
        "conf": 0.15,
        "note": "conf=0.15"
    },
    "botsort_conf25": {
        "exp": "Exp 5: Confidence Sweep",
        "yaml": "src/tracking/botsort.yaml",
        "conf": 0.25,
        "note": "conf=0.25"
    },
    "botsort_conf30": {
        "exp": "Exp 5: Confidence Sweep",
        "yaml": "src/tracking/botsort.yaml",
        "conf": 0.30,
        "note": "conf=0.30"
    },
    "botsort_conf35": {
        "exp": "Exp 5: Confidence Sweep",
        "yaml": "src/tracking/botsort.yaml",
        "conf": 0.35,
        "note": "conf=0.35"
    },

    # ---------------------------------------------------------
    # Experiment 6: GMC Sweep (buffer=30, conf=0.20)
    # ---------------------------------------------------------
    "botsort_gmc_orb": {
        "exp": "Exp 6: GMC Sweep",
        "yaml": "src/tracking/botsort_gmc_orb.yaml",
        "conf": 0.20,
        "note": "gmc=orb"
    },
    "botsort_gmc_ecc": {
        "exp": "Exp 6: GMC Sweep",
        "yaml": "src/tracking/botsort_gmc_ecc.yaml",
        "conf": 0.20,
        "note": "gmc=ecc"
    },
    "botsort_gmc_none": {
        "exp": "Exp 6: GMC Sweep",
        "yaml": "src/tracking/botsort_gmc_none.yaml",
        "conf": 0.20,
        "note": "gmc=none"
    },
}

def run_tracking_for_experiment(tracker_name, yaml_path, conf=0.20):
    print(f"\n{'='*70}\n[TRACKING] Running {tracker_name} (conf={conf}, yaml={yaml_path})\n{'='*70}")
    
    # Check if all sequence CSVs already exist and are non-empty
    all_exist = True
    for seq in SEQUENCES:
        csv_file = TRACKING_DIR / f"{seq}_{tracker_name}.csv"
        if not (csv_file.exists() and csv_file.stat().st_size > 1000):
            all_exist = False
            break
            
    if all_exist:
        print(f"  All sequence CSVs already exist for {tracker_name}. Skipping inference.")
        return

    model = YOLO(MODEL_PATH)
    start_time = time.time()
    
    for seq in SEQUENCES:
        csv_file = TRACKING_DIR / f"{seq}_{tracker_name}.csv"
        if csv_file.exists() and csv_file.stat().st_size > 1000:
            print(f"  -> {seq}: already exists ({csv_file.stat().st_size} bytes), skipping.")
            continue
            
        img_dir = ROOT / seq / "img1"
        img_paths = sorted(img_dir.glob("*.jpg"))
        if not img_paths:
            print(f"  WARNING: No images found in {img_dir}")
            continue
            
        print(f"  Tracking {seq} ({len(img_paths)} frames)...", end="", flush=True)
        t0 = time.time()
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])
            
            frame_count = 0
            record_count = 0
            for img_path in img_paths:
                frame_count += 1
                results = model.track(
                    source=str(img_path),
                    tracker=yaml_path,
                    persist=True,
                    conf=conf,
                    imgsz=640,
                    device=0,
                    verbose=False,
                )
                res = results[0]
                if res.boxes is None or res.boxes.id is None:
                    continue
                    
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.int().cpu().tolist()
                classes = res.boxes.cls.int().cpu().tolist()
                confs = res.boxes.conf.cpu().tolist()
                
                for box, track_id, cls, c in zip(boxes, ids, classes, confs):
                    x1, y1, x2, y2 = box
                    writer.writerow([
                        frame_count, track_id, cls,
                        round(float(c), 4),
                        round(float(x1), 2), round(float(y1), 2),
                        round(float(x2), 2), round(float(y2), 2)
                    ])
                    record_count += 1
                    
        elapsed = time.time() - t0
        print(f" done in {elapsed:.1f}s ({record_count} records)")
        
    print(f"[TRACKING] Completed {tracker_name} in {(time.time() - start_time)/60:.2f} min")

def create_eval_zip(tracker_name):
    tracker_zip = EVAL_DIR / f"{tracker_name}.zip"
    if tracker_zip.exists() and tracker_zip.stat().st_size > 10000:
        print(f"  -> {tracker_zip} already exists ({tracker_zip.stat().st_size} bytes), skipping creation.")
        return tracker_zip

    print(f"[EVAL-ZIP] Building {tracker_zip}...")
    temp_build = EVAL_DIR / f"temp_{tracker_name}"
    if temp_build.exists():
        shutil.rmtree(temp_build)
    temp_build.mkdir(parents=True, exist_ok=True)
    
    for seq in SEQUENCES:
        src_csv = TRACKING_DIR / f"{seq}_{tracker_name}.csv"
        dst_txt = temp_build / f"{seq}.txt"
        
        df = pd.read_csv(src_csv)
        with open(dst_txt, "w") as f:
            for _, row in df.iterrows():
                frame = int(row["frame"])
                track_id = int(row["track_id"])
                x = float(row["x1"])
                y = float(row["y1"])
                w = float(row["x2"] - row["x1"])
                h = float(row["y2"] - row["y1"])
                conf = float(row["confidence"])
                f.write(f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")
                
    with zipfile.ZipFile(tracker_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for txt in temp_build.glob("*.txt"):
            z.write(txt, arcname=txt.name)
            
    shutil.rmtree(temp_build)
    print(f"[EVAL-ZIP] Created {tracker_zip}")
    return tracker_zip

def evaluate_tracker(tracker_name):
    print(f"\n[TRACKEVAL] Evaluating {tracker_name}...")
    cmd = [
        PYTHON_EXE,
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
    err = res.stderr
    
    if res.returncode != 0:
        print(f"TrackEval failed with error:\n{err}\n{out}")
        return None
        
    metrics = {"tracker": tracker_name}
    lines = [l.strip() for l in out.split("\n") if l.strip()]
    for i, l in enumerate(lines):
        if l.startswith("HOTA:"):
            headers = l.split()[2:]
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                vals = lines[i+1].split()[1:]
                for h, v in zip(headers, vals):
                    try:
                        metrics[f"HOTA_{h}"] = float(v)
                    except:
                        metrics[f"HOTA_{h}"] = v
        elif l.startswith("CLEAR:"):
            headers = l.split()[2:]
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                vals = lines[i+1].split()[1:]
                for h, v in zip(headers, vals):
                    try:
                        metrics[f"CLEAR_{h}"] = float(v)
                    except:
                        metrics[f"CLEAR_{h}"] = v
        elif l.startswith("Identity:"):
            headers = ["IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"]
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                vals = lines[i+1].split()[1:]
                for h, v in zip(headers, vals):
                    try:
                        metrics[f"ID_{h}"] = float(v)
                    except:
                        metrics[f"ID_{h}"] = v
                    
    print(f"Results for {tracker_name}: HOTA={metrics.get('HOTA_HOTA')}, DetA={metrics.get('HOTA_DetA')}, AssA={metrics.get('HOTA_AssA')}, MOTA={metrics.get('CLEAR_MOTA')}, IDF1={metrics.get('ID_IDF1')}, IDSW={metrics.get('CLEAR_IDSW')}")
    return metrics

def run_suite(suite_names):
    all_results = []
    
    # Also include baseline
    baseline_zip = EVAL_DIR / "botsort.zip"
    if baseline_zip.exists():
        print("[BASELINE] Evaluating baseline botsort (conf=0.20, buffer=30, gmc=sparseOptFlow)...")
        b_res = evaluate_tracker("botsort")
        if b_res:
            b_res["experiment"] = "Baseline"
            b_res["notes"] = "buffer=30, conf=0.20, gmc=sparseOptFlow"
            all_results.append(b_res)

    for name in suite_names:
        cfg = EXPERIMENTS[name]
        exp_name = cfg["exp"]
        yaml_path = cfg["yaml"]
        conf = cfg["conf"]
        note = cfg["note"]
        
        # 1. Inference
        run_tracking_for_experiment(name, yaml_path, conf)
        # 2. Build ZIP
        create_eval_zip(name)
        # 3. Evaluate
        res = evaluate_tracker(name)
        if res:
            res["experiment"] = exp_name
            res["notes"] = note
            all_results.append(res)
            
        # Update summary dataframe
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"\n--- Progress saved to {RESULTS_CSV} ---")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "exp4":
        targets = ["botsort_buffer15", "botsort_buffer45", "botsort_buffer60", "botsort_buffer90"]
    elif mode == "exp5":
        targets = ["botsort_conf15", "botsort_conf25", "botsort_conf30", "botsort_conf35"]
    elif mode == "exp6":
        targets = ["botsort_gmc_orb", "botsort_gmc_ecc", "botsort_gmc_none"]
    else:
        targets = list(EXPERIMENTS.keys())
        
    print(f"Target experiments to run ({len(targets)}): {targets}")
    run_suite(targets)
