import geopandas as gpd
import pandas as pd 
import glob
import os

county_ward = pd.read_pickle('county_ward_list.pkl')

wards_gdf = gpd.read_file("data/Kenya_wards_NDMA.shp")
wards_gdf.set_crs(epsg=4326, inplace=True) 
wards_gdf = wards_gdf.merge(
    county_ward, left_on="Ward", right_on="Ward", how="left")

wards_gdf.dropna(subset=['County'], inplace=True)

counties_gdf = gpd.read_file("data/ken_admbnda_adm1_iebc_20191031.shp")

wards_gdf.to_file("data/Kenya_wards_with_counties.geojson", driver="GeoJSON")


import os, glob
import pandas as pd

# base_dir = "/home/ebenezer/Desktop/Before Changes/dews-flask-application2/Kenya_MUAC_NDMA_implementation/results" #change to where the results are

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
base_dir = os.path.join(PARENT_DIR, "Kenya_MUAC_NDMA_implementation", "results")


def load_latest_prediction_file(horizon: int, target: str):
    """
    Load latest prediction file for a given horizon and target variable.

    Parameters
    ----------
    horizon : int
        Prediction horizon in months (1, 2, or 3).
    target : str
        One of {"wasting_smoothed", "wasting_risk_smoothed"}.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - For horizon==1:
            * observed column: 'observed' (wasting) or 'risk_observed' (risk)
        - For all horizons:
            * predictions: 'pred_{h}mo' or 'risk_pred_{h}mo'
            * CIs: 'lower_bound_{h}mo'/'upper_bound_{h}mo' or risk_*
    """
    assert target in {"wasting_smoothed", "wasting_risk_smoothed"}
    pattern = os.path.join(base_dir, f"{target}_pred_hb_{horizon}_36m_*.csv")
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise FileNotFoundError(f"No matching files found for horizon {horizon} and target '{target}' in {base_dir}")

    matching_files.sort()  # filenames encode chronology
    latest_file = matching_files[-1]
    print(f"[{target}] horizon {horizon}: {latest_file}")

    df = pd.read_csv(latest_file)
    df["time_period"] = pd.to_datetime(df["time_period"])
    df["Ward"] = df["Ward"].astype(str).str.strip()

    col_suffix = f"{horizon}mo"
    is_risk = (target == "wasting_risk_smoothed")
    prefix = "risk_" if is_risk else ""

    # Build rename map
    rename_map = {
        "yhat": f"{prefix}pred_{col_suffix}",
        "lower_bound": f"{prefix}lower_bound_{col_suffix}",
        "upper_bound": f"{prefix}upper_bound_{col_suffix}",
    }
    # Only the 1-month files contain the contemporaneous observed value needed downstream
    if horizon == 1:
        rename_map[target] = f"{prefix}observed"

    df = df.rename(columns=rename_map)
    return df

# --- Load wasting_smoothed  ---
df_1mo = load_latest_prediction_file(1, "wasting_smoothed")
df_2mo = load_latest_prediction_file(2, "wasting_smoothed")
df_3mo = load_latest_prediction_file(3, "wasting_smoothed")

# --- Load wasting_risk_smoothed  ---
risk_1mo = load_latest_prediction_file(1, "wasting_risk_smoothed")
risk_2mo = load_latest_prediction_file(2, "wasting_risk_smoothed")
risk_3mo = load_latest_prediction_file(3, "wasting_risk_smoothed")

# Merge county info (assumes a DataFrame `county_ward` with ['Ward','County'])
df_1mo = df_1mo.merge(county_ward[['Ward', 'County']], on='Ward', how='left')
df_2mo = df_2mo.merge(county_ward[['Ward', 'County']], on='Ward', how='left')
df_3mo = df_3mo.merge(county_ward[['Ward', 'County']], on='Ward', how='left')

risk_1mo = risk_1mo.merge(county_ward[['Ward', 'County']], on='Ward', how='left')
risk_2mo = risk_2mo.merge(county_ward[['Ward', 'County']], on='Ward', how='left')
risk_3mo = risk_3mo.merge(county_ward[['Ward', 'County']], on='Ward', how='left')

# Write outputs 
df_1mo.to_csv("data/Smoothed_wasting_prediction_hb_1.csv", index=False)
df_2mo.to_csv("data/Smoothed_wasting_prediction_hb_2.csv", index=False)
df_3mo.to_csv("data/Smoothed_wasting_prediction_hb_3.csv", index=False)

risk_1mo.to_csv("data/Smoothed_wasting_risk_prediction_hb_1.csv", index=False)
risk_2mo.to_csv("data/Smoothed_wasting_risk_prediction_hb_2.csv", index=False)
risk_3mo.to_csv("data/Smoothed_wasting_risk_prediction_hb_3.csv", index=False)


# Add figures to main folder 

import os, re, glob, shutil
from pathlib import Path
from datetime import datetime

# === Directories ===
# FIG_DIR      = Path("/home/ebenezer/Desktop/Before Changes/dews-flask-application2/Kenya_MUAC_NDMA_implementation/figures")
# COV_DIR      = Path("/home/ebenezer/Desktop/Before Changes/dews-flask-application2/Kenya_MUAC_NDMA_implementation/covariates_graphs")
# ASSETS_DIR   = Path("/home/ebenezer/Desktop/Before Changes/dews-flask-application2/early_warning_dashboard/assets/figures")
FIG_DIR = Path(os.path.join(PARENT_DIR, "Kenya_MUAC_NDMA_implementation", "figures"))
COV_DIR = Path(os.path.join(PARENT_DIR, "Kenya_MUAC_NDMA_implementation", "covariates_graphs"))
ASSETS_DIR = Path(os.path.join(PARENT_DIR, "early_warning_dashboard-main", "assets", "figures"))
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Choose "newest" (for 2025-11) or "oldest"
SELECT = "newest"

# === Patterns for top-feature plots ===
PATTERNS = {
    "wasting_smoothed":      "top_features_wasting_smoothed_*_multiH.png",
    "wasting_risk_smoothed": "top_features_wasting_risk_smoothed_*_multiH.png",
}

# === Covariate plots to always copy ===
COVARIATE_FILES = [
    "precip_zscore_vs_wasting_2021_2025.png",
    "evi_ndvi_zscore_vs_wasting_2021_2025.png",
    "conflict_fatalities_trends.png",
]

DATE_RE = re.compile(r"_(\d{4}-\d{2})_multiH\.png$", re.IGNORECASE)

def parsed_date_from_name(p: Path):
    m = DATE_RE.search(p.name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m")
        except ValueError:
            pass
    return datetime.fromtimestamp(p.stat().st_mtime)

def pick_by_mode(paths, mode="newest"):
    if not paths:
        return None
    ranked = sorted(paths, key=lambda x: (parsed_date_from_name(x), x.stat().st_mtime))
    return ranked[-1] if mode == "newest" else ranked[0]

def copy_one(outcome: str, pattern: str):
    files = [Path(p) for p in glob.glob(str(FIG_DIR / pattern))]
    if not files:
        print(f"[{outcome}] No files found for pattern: {pattern}")
        return None
    chosen = pick_by_mode(files, SELECT)
    dest = ASSETS_DIR / chosen.name
    shutil.copy2(chosen, dest)
    print(f"[{outcome}] {SELECT.capitalize()} -> {chosen.name} copied to {dest}")
    return dest

def copy_covariate_graphs():
    copied = []
    for fname in COVARIATE_FILES:
        src = COV_DIR / fname
        if src.exists():
            dest = ASSETS_DIR / src.name
            shutil.copy2(src, dest)
            copied.append(dest)
            print(f"[covariate] Copied {src} -> {dest}")
        else:
            print(f"[covariate] Missing file: {src}")
    return copied

if __name__ == "__main__":
    results = {k: copy_one(k, pat) for k, pat in PATTERNS.items()}
    cov_results = copy_covariate_graphs()

    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k}: {v if v else 'No file copied'}")
    print("  Covariates:")
    for v in cov_results:
        print(f"   - {v}")

