import pandas as pd
from pathlib import Path

MASTER_CSV = Path("outputs/evaluation/validation/all_experiments_master.csv")
OUT_CSV = Path("outputs/evaluation/experiments_table.csv")

def generate_table():
    df = pd.read_csv(MASTER_CSV)
    
    # Extract metadata from tracker name / description
    def parse_meta(row):
        t = row["tracker"]
        desc = str(row["description"])
        
        # Default baseline
        conf = 0.20
        gmc = "sparseOptFlow"
        match = 0.80
        reid = "No"
        
        if "conf15" in t: conf = 0.15
        elif "conf25" in t: conf = 0.25
        elif "conf30" in t: conf = 0.30
        elif "conf35" in t: conf = 0.35
        elif "conf40" in t: conf = 0.40
        elif "conf45" in t: conf = 0.45
        
        if "gmc_none" in t: gmc = "none"
        elif "gmc_ecc" in t: gmc = "ecc"
        elif "gmc_orb" in t or "orb" in t: gmc = "ORB"
        
        if "match75" in t: match = 0.75
        elif "match85" in t: match = 0.85
        elif "match70" in t: match = 0.70
        
        if "reid" in t.lower(): reid = "Yes"
        
        return pd.Series({"Conf": conf, "GMC": gmc, "Match": match, "ReID": reid})

    meta = df.apply(parse_meta, axis=1)
    df["Conf"] = meta["Conf"]
    df["GMC"] = meta["GMC"]
    df["Match"] = meta["Match"]
    df["ReID"] = meta["ReID"]
    
    cols = ["experiment", "tracker", "Conf", "GMC", "Match", "ReID", "HOTA", "AssA", "IDF1", "IDSW", "Frag", "DetA", "MOTA"]
    out_df = df[cols].copy()
    out_df.rename(columns={"experiment": "Experiment", "tracker": "Tracker"}, inplace=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Generated {OUT_CSV}")
    print("\n" + out_df.to_markdown(index=False))

if __name__ == "__main__":
    generate_table()
