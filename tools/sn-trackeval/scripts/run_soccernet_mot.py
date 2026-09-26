""" run_soccernet_mot.py

Run example:

python run_soccernet_mot.py \
--BENCHMARK SNMOT \
--DO_PREPROC False \
--SEQMAP_FILE tools/SNMOT-test.txt \
--TRACKERS_TO_EVAL test \
--SPLIT_TO_EVAL test \
--OUTPUT_SUB_FOLDER eval_results \
--TRACKERS_FOLDER_ZIP soccernet_mot_results.zip \
--GT_FOLDER_ZIP gt.zip  

run_soccernet_mot.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL Lif_T

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
        'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
        'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'DO_PREPROC': True,  # Whether to perform preprocessing (never done for 2D_MOT_2015)
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support
import zipfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)

    parser.add_argument('--TRACKERS_FOLDER_ZIP', type=str, default='')
    parser.add_argument('--GT_FOLDER_ZIP', type=str, default='')

    # args = parser.parse_args().__dict__
    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    # if not empty ..., extract and modify trackers folder
    assert len(args.TRACKERS_FOLDER_ZIP) > 0
    assert len(args.GT_FOLDER_ZIP) > 0

    # ---------------------------------------------------------
    # Prepare temporary TrackEval-compatible directory structure
    # ---------------------------------------------------------

    temp_dir = os.path.abspath('./temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Determine split directory name (e.g. SNMOT-test)
    benchmark = args.BENCHMARK if hasattr(args, 'BENCHMARK') and args.BENCHMARK else 'SNMOT'
    split = args.SPLIT_TO_EVAL if hasattr(args, 'SPLIT_TO_EVAL') and args.SPLIT_TO_EVAL else 'test'
    split_fol = f"{benchmark}-{split}"

    gt_target_split = os.path.join(temp_dir, 'gt', split_fol)
    gt_target_root = os.path.join(temp_dir, 'gt')
    os.makedirs(gt_target_split, exist_ok=True)

    tracker_target_split = os.path.join(temp_dir, split_fol)
    tracker_target_root = temp_dir
    os.makedirs(tracker_target_split, exist_ok=True)

    # ---------------------------------------------------------
    # Extract and prepare tracker files
    # ---------------------------------------------------------
    tracker_extract_dir = os.path.join(temp_dir, '_tracker_extract')
    with zipfile.ZipFile(args.TRACKERS_FOLDER_ZIP, 'r') as zip_ref:
        zip_ref.extractall(tracker_extract_dir)

    # Identify tracker names from args or subdirectories
    trackers_to_eval = args.TRACKERS_TO_EVAL
    if isinstance(trackers_to_eval, str):
        trackers_to_eval = [trackers_to_eval]
    elif trackers_to_eval is None:
        trackers_to_eval = []

    # Find all sequence txt files in tracker_extract_dir
    tracker_files = []
    for root, dirs, files in os.walk(tracker_extract_dir):
        for f in files:
            if f.endswith('.txt'):
                tracker_files.append(os.path.join(root, f))

    if not tracker_files:
        raise Exception(f"No tracker .txt files found in {args.TRACKERS_FOLDER_ZIP}")

    # Group tracker files by tracker name
    tracker_seq_files = {}  # {tracker_name: {seq_name: full_path}}
    for fpath in tracker_files:
        fname = os.path.basename(fpath)
        seq_name = os.path.splitext(fname)[0]
        rel_parts = os.path.relpath(fpath, tracker_extract_dir).split(os.sep)
        detected_tracker = None
        if len(rel_parts) > 1:
            for part in rel_parts[:-1]:
                if part.lower() != 'data':
                    detected_tracker = part
                    break
        if not detected_tracker:
            detected_tracker = trackers_to_eval[0] if trackers_to_eval else 'botsort'

        tracker_seq_files.setdefault(detected_tracker, {})[seq_name] = fpath

    if trackers_to_eval:
        for t in trackers_to_eval:
            if t not in tracker_seq_files and len(tracker_seq_files) == 1:
                single_k = list(tracker_seq_files.keys())[0]
                tracker_seq_files[t] = tracker_seq_files[single_k]

    for t_name, seqs in tracker_seq_files.items():
        for base_dir in [tracker_target_split, tracker_target_root]:
            t_data_dir = os.path.join(base_dir, t_name, 'data')
            os.makedirs(t_data_dir, exist_ok=True)
            for seq_name, fpath in seqs.items():
                dst = os.path.join(t_data_dir, f"{seq_name}.txt")
                if not os.path.exists(dst):
                    shutil.copy2(fpath, dst)

            t_zip_path = os.path.join(base_dir, t_name, 'data.zip')
            with zipfile.ZipFile(t_zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
                for seq_name, fpath in seqs.items():
                    z.write(fpath, arcname=f"{seq_name}.txt")

    # ---------------------------------------------------------
    # Extract and prepare GT files
    # ---------------------------------------------------------
    gt_extract_dir = os.path.join(temp_dir, '_gt_extract')
    with zipfile.ZipFile(args.GT_FOLDER_ZIP, 'r') as zip_ref:
        zip_ref.extractall(gt_extract_dir)

    # 1. Find all seqinfo.ini files across all subdirectories
    seqinfo_files = {}
    for root, dirs, files in os.walk(gt_extract_dir):
        if 'seqinfo.ini' in files:
            seq_name = os.path.basename(root)
            seqinfo_files[seq_name] = os.path.join(root, 'seqinfo.ini')

    for seq_name, ini_path in seqinfo_files.items():
        for target_dir in [gt_target_split, gt_target_root]:
            dst_dir = os.path.join(target_dir, seq_name)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(ini_path, os.path.join(dst_dir, 'seqinfo.ini'))

    # 2. Look for data.zip in the extracted GT files
    found_data_zip = None
    for root, dirs, files in os.walk(gt_extract_dir):
        if 'data.zip' in files:
            found_data_zip = os.path.join(root, 'data.zip')
            break

    if found_data_zip:
        for target_dir in [gt_target_split, gt_target_root]:
            shutil.copy2(found_data_zip, os.path.join(target_dir, 'data.zip'))

        with zipfile.ZipFile(found_data_zip, 'r') as z:
            for name in z.namelist():
                if name.endswith('.txt'):
                    seq_name = os.path.splitext(os.path.basename(name))[0]
                    content = z.read(name)
                    for target_dir in [gt_target_split, gt_target_root]:
                        seq_gt_dir = os.path.join(target_dir, seq_name, 'gt')
                        os.makedirs(seq_gt_dir, exist_ok=True)
                        with open(os.path.join(seq_gt_dir, 'gt.txt'), 'wb') as f:
                            f.write(content)
    else:
        gt_txt_files = {}
        for root, dirs, files in os.walk(gt_extract_dir):
            for f in files:
                if f == 'gt.txt':
                    seq_name = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == 'gt' else os.path.basename(root)
                    gt_txt_files[seq_name] = os.path.join(root, f)
                elif f.endswith('.txt') and not f.startswith('seqmap'):
                    seq_name = os.path.splitext(f)[0]
                    gt_txt_files[seq_name] = os.path.join(root, f)

        if not gt_txt_files:
            raise Exception(f"No ground truth annotations (data.zip or gt.txt) found in {args.GT_FOLDER_ZIP}")

        for seq_name, gt_path in gt_txt_files.items():
            for target_dir in [gt_target_split, gt_target_root]:
                seq_gt_dir = os.path.join(target_dir, seq_name, 'gt')
                os.makedirs(seq_gt_dir, exist_ok=True)
                shutil.copy2(gt_path, os.path.join(seq_gt_dir, 'gt.txt'))

        for target_dir in [gt_target_split, gt_target_root]:
            zip_out = os.path.join(target_dir, 'data.zip')
            with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as z:
                for seq_name, gt_path in gt_txt_files.items():
                    z.write(gt_path, arcname=f"{seq_name}.txt")

    # TrackEval locations
    args.TRACKERS_FOLDER = temp_dir
    args.GT_FOLDER = os.path.join(temp_dir, 'gt')

    args = args.__dict__
    for single_str_setting in ['SEQMAP_FILE', 'SEQMAP_FOLDER', 'OUTPUT_FOLDER', 'OUTPUT_SUB_FOLDER']:
        if single_str_setting in args and isinstance(args[single_str_setting], list):
            args[single_str_setting] = args[single_str_setting][0] if len(args[single_str_setting]) > 0 else None

    args.pop('TRACKERS_FOLDER_ZIP', None)
    args.pop('GT_FOLDER_ZIP', None)

    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None] * len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)
