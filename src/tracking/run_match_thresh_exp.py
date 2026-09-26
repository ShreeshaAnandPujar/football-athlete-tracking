import sys
import time
import csv
import zipfile
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

ROOT = Path("data/raw/SoccerNet/tracking-2023/train/train")
MODEL_PATH = "runs/detect/outputs/training/yolov8s_soccernet_baseline/weights/best.pt"
TRACKING_DIR = Path("outputs/tracking/validation")
EVAL_DIR = Path("outputs/evaluation/validation")

PYTHON_EXE = sys.executable

SEQUENCES = [
    "SNMOT-113", "SNMOT-157", "SNMOT-066", "SNMOT-167",
    "SNMOT-068", "SNMOT-074", "SNMOT-075", "SNMOT-077",
    "SNMOT-161", "SNMOT-061", "SNMOT-067", "SNMOT-154",
]

def run_tracking(tracker_name, yaml_path, conf=0.35):
    print(f"\n{'='*70}\n[TRACKING] Running {tracker_name} (conf={conf}, yaml={yaml_path})\n{'='*70}")
    all_exist = all((TRACKING_DIR / f"{seq}_{tracker_name}.csv").exists() and (TRACKING_DIR / f"{seq}_{tracker_name}.csv").stat().st_size > 1000 for seq in SEQUENCES)
    if all_exist:
        print(f"  All CSVs already exist for {tracker_name}. Skipping inference.")
        return

    model = YOLO(MODEL_PATH)
    start_time = time.time()
    
    for seq in SEQUENCES:
        csv_file = TRACKING_DIR / f"{seq}_{tracker_name}.csv"
        if csv_file.exists() and csv_file.stat().st_size > 1000:
            print(f"  -> {seq}: already exists, skipping.")
            continue
            
        img_dir = ROOT / seq / "img1"
        img_paths = sorted(img_dir.glob("*.jpg"))
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
                    
        print(f" done in {time.time() - t0:.1f}s ({record_count} records)")
        
    print(f"[TRACKING] Completed {tracker_name} in {(time.time() - start_time)/60:.2f} min")

def create_eval_zip(tracker_name):
    tracker_zip = EVAL_DIR / f"{tracker_name}.zip"
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
                f.write(f"{int(row['frame'])},{int(row['track_id'])},{float(row['x1']):.2f},{float(row['y1']):.2f},"
                        f"{float(row['x2']-row['x1']):.2f},{float(row['y2']-row['y1']):.2f},{float(row['confidence']):.4f},-1,-1,-1\n")
                
    with zipfile.ZipFile(tracker_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for txt in temp_build.glob("*.txt"):
            z.write(txt, arcname=txt.name)
            
    shutil.rmtree(temp_build)
    print(f"[EVAL-ZIP] Created {tracker_zip}")

def evaluate(tracker_name):
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
    lines = [l.strip() for l in res.stdout.split("\n") if l.strip()]
    metrics = {"tracker": tracker_name}
    
    hota_headers = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA', 'HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
    clear_headers = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA', 'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
    id_headers = ['IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP']
    
    for i, l in enumerate(lines):
        if l.startswith("HOTA:"):
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                for h, v in zip(hota_headers, lines[i+1].split()[1:]):
                    try: metrics[h] = float(v)
                    except: metrics[h] = v
        elif l.startswith("CLEAR:"):
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                for h, v in zip(clear_headers, lines[i+1].split()[1:]):
                    try: metrics[h] = float(v)
                    except: metrics[h] = v
        elif l.startswith("Identity:"):
            if i + 1 < len(lines) and lines[i+1].startswith("COMBINED"):
                for h, v in zip(id_headers, lines[i+1].split()[1:]):
                    try: metrics[h] = float(v)
                    except: metrics[h] = v
                    
    print(f"Results for {tracker_name}: HOTA={metrics.get('HOTA')}, AssA={metrics.get('AssA')}, IDF1={metrics.get('IDF1')}, IDSW={metrics.get('IDSW')}")
    return metrics

if __name__ == "__main__":
    # 1. match_thresh = 0.75
    run_tracking("botsort_conf35_orb_match75", "src/tracking/botsort_conf35_orb_match75.yaml", conf=0.35)
    create_eval_zip("botsort_conf35_orb_match75")
    m75 = evaluate("botsort_conf35_orb_match75")
    
    # 2. match_thresh = 0.85
    run_tracking("botsort_conf35_orb_match85", "src/tracking/botsort_conf35_orb_match85.yaml", conf=0.35)
    create_eval_zip("botsort_conf35_orb_match85")
    m85 = evaluate("botsort_conf35_orb_match85")
    
    # 3. Evaluate match_thresh = 0.80 (champion) for direct side-by-side comparison
    m80 = evaluate("botsort_conf35_orb")
    
    m75["description"] = "conf=0.35, ORB, match_thresh=0.75"
    m80["description"] = "conf=0.35, ORB, match_thresh=0.80 (Champion)"
    m85["description"] = "conf=0.35, ORB, match_thresh=0.85"
    
    df = pd.DataFrame([m75, m80, m85])
    cols = ["description", "HOTA", "AssA", "IDF1", "IDSW", "Frag", "DetA", "MOTA"]
    print("\n" + "="*80)
    print("MATCH_THRESH EXPERIMENT RESULTS:")
    print("="*80)
    print(df[cols].to_markdown(index=False))
