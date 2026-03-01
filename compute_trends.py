import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================
# Tunables
# =============================
EPS = 0.01                 # deadband for trend classification (1 percentage point)
ALERT_THRESHOLD = 0.10     # 10% alert threshold
EMERGENCY_THRESHOLD = 0.15 # 15% emergency threshold (optional override; see logic)
PLOT_ALERTS = True         # set False if you don’t want figures

# =============================
# Classifiers
# =============================
def classify(delta: float, eps: float = EPS) -> str:
    if pd.isna(delta):
        return "No Data"
    if delta > eps:
        return "Increasing"
    if delta < -eps:
        return "Decreasing"
    return "Stable"

def classify_with_ci(observed, lower, upper, eps: float = EPS) -> str:
    if pd.isna(observed) or pd.isna(lower) or pd.isna(upper):
        return "No Data"
    if lower > observed + eps:
        return "Increasing"
    if upper < observed - eps:
        return "Decreasing"
    return "Stable"

# =============================
# Core pipeline (runs once per target)
# =============================
def run_trend_pipeline_for_target(
    target_name: str,
    file_stub: str,
    col_prefix: str = ""
):
    """
    Inputs (CSV):
      data/Smoothed_{file_stub}_prediction_hb_1.csv
      data/Smoothed_{file_stub}_prediction_hb_2.csv
      data/Smoothed_{file_stub}_prediction_hb_3.csv

    Output CSVs:
      data/clean_trend_long_with_CI{_risk}.csv
      data/clean_trend_wide_with_CI{_risk}.csv
    """

    # === Load data ===
    hb_1 = pd.read_csv(f"data/Smoothed_{file_stub}_prediction_hb_1.csv")
    hb_2 = pd.read_csv(f"data/Smoothed_{file_stub}_prediction_hb_2.csv")
    hb_3 = pd.read_csv(f"data/Smoothed_{file_stub}_prediction_hb_3.csv")

    for df in (hb_1, hb_2, hb_3):
        df["time_period"] = pd.to_datetime(df["time_period"])
        df["Ward"] = df["Ward"].astype(str).str.strip()

    # Column mappings (handle "" vs "risk_" prefixes)
    obs_col  = f"{col_prefix}observed"
    p1_col   = f"{col_prefix}pred_1mo";  lb1_col = f"{col_prefix}lower_bound_1mo";  ub1_col = f"{col_prefix}upper_bound_1mo"
    p2_col   = f"{col_prefix}pred_2mo";  lb2_col = f"{col_prefix}lower_bound_2mo";  ub2_col = f"{col_prefix}upper_bound_2mo"
    p3_col   = f"{col_prefix}pred_3mo";  lb3_col = f"{col_prefix}lower_bound_3mo";  ub3_col = f"{col_prefix}upper_bound_3mo"

    # Normalize names we actually use
    hb_1 = hb_1.rename(columns={obs_col: "observed", p1_col: "pred_1mo", lb1_col: "lower_bound_1mo", ub1_col: "upper_bound_1mo"})
    hb_2 = hb_2.rename(columns={p2_col: "pred_2mo",   lb2_col: "lower_bound_2mo",   ub2_col: "upper_bound_2mo"})
    hb_3 = hb_3.rename(columns={p3_col: "pred_3mo",   lb3_col: "lower_bound_3mo",   ub3_col: "upper_bound_3mo"})

    # --- Build a robust OBS block from hb_1 and (fallback) hb_3 ---
    if col_prefix == "":  # wasting prevalence
        cand_hb1 = ["observed", "wasting_smoothed"]          # typical names in hb_1
        cand_hb3 = ["observed", "wasting_smoothed"]          # typical names in hb_3
    else:                  # wasting risk
        cand_hb1 = ["risk_observed", "wasting_risk_smoothed", "observed"]
        cand_hb3 = ["risk_observed", "wasting_risk_smoothed", "observed"]

    def first_present(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    obs1_col = first_present(hb_1, cand_hb1)
    obs3_col = first_present(hb_3, cand_hb3)

    # Start with hb_1 observed if available
    obs_1 = hb_1[["Ward", "time_period"] + (["County"] if "County" in hb_1.columns else [])].copy()
    if obs1_col is not None:
        obs_1["observed"] = pd.to_numeric(hb_1[obs1_col], errors="coerce")
    else:
        obs_1["observed"] = pd.NA

    # Fallback from hb_3 for missing months
    obs_3 = hb_3[["Ward", "time_period"] + (["County"] if "County" in hb_3.columns else [])].copy()
    if obs3_col is not None:
        obs_3["observed"] = pd.to_numeric(hb_3[obs3_col], errors="coerce")
    else:
        obs_3["observed"] = pd.NA

    # Prefer hb_1 (if present), else take hb_3
    obs = (
        pd.concat([obs_1, obs_3], ignore_index=True)
        .sort_values(["Ward", "time_period"])
        .drop_duplicates(["Ward", "time_period"], keep="first")
    )

    # If County is missing in hb_1, bring it from hb_3
    if "County" not in obs.columns:
        # try to attach County from whichever has it
        for src in (hb_1, hb_3):
            if "County" in src.columns:
                obs = obs.merge(
                    src[["Ward","time_period","County"]].drop_duplicates(["Ward","time_period"]),
                    on=["Ward","time_period"], how="left"
                )
                break


    # Align forecasts to their target month t
    def shift_predictions(df, months_back, horizon_label):
        return (
            df.assign(time_period=df["time_period"] - pd.DateOffset(months=months_back))[
                ["Ward", "time_period", f"pred_{horizon_label}", f"lower_bound_{horizon_label}", f"upper_bound_{horizon_label}"]
            ]
        )

    pred_1mo = shift_predictions(hb_1, 1, "1mo")
    pred_2mo = shift_predictions(hb_2, 2, "2mo")
    pred_3mo = shift_predictions(hb_3, 3, "3mo")

    # Merge all horizons
    merged = (
        obs.merge(pred_1mo, on=["Ward", "time_period"], how="left")
           .merge(pred_2mo, on=["Ward", "time_period"], how="left")
           .merge(pred_3mo, on=["Ward", "time_period"], how="left")
           .sort_values(["Ward", "time_period"])
    )

    # Keep rows where observed + 3mo horizon exist (alerts depend on 3mo)
    merged = merged.dropna(subset=["observed", "pred_3mo", "lower_bound_3mo", "upper_bound_3mo"])

    # ===== Trends (existing) =====
    merged["trend_1mo"] = (merged.get("pred_1mo") - merged["observed"]).apply(classify) if "pred_1mo" in merged else "No Data"
    merged["trend_2mo"] = (merged.get("pred_2mo") - merged["observed"]).apply(classify) if "pred_2mo" in merged else "No Data"
    merged["trend_3mo"] = (merged["pred_3mo"] - merged["observed"]).apply(classify)

    merged["trend_with_CI_1mo"] = merged.apply(lambda r: classify_with_ci(r["observed"], r.get("lower_bound_1mo"), r.get("upper_bound_1mo")), axis=1)
    merged["trend_with_CI_2mo"] = merged.apply(lambda r: classify_with_ci(r["observed"], r.get("lower_bound_2mo"), r.get("upper_bound_2mo")), axis=1)
    merged["trend_with_CI_3mo"] = merged.apply(lambda r: classify_with_ci(r["observed"], r["lower_bound_3mo"], r["upper_bound_3mo"]), axis=1)

    merged["observed_slope_2mo"] = merged.groupby("Ward")["observed"].diff()
    merged["observed_trend_2mo"] = merged["observed_slope_2mo"].apply(classify)

    # ===== NEW: CI deviation fields per horizon =====
    AS_PERCENT = True  # set False if your numbers are already in percentage points

    def ci_dev_block(df, h):
        lo = df[f"lower_bound_{h}"]
        hi = df[f"upper_bound_{h}"]
        obs = df["observed"]

        pos = pd.Series("within", index=df.index)
        pos = pos.mask(obs < lo, "below_lower")
        pos = pos.mask(obs > hi, "above_upper")

        # gap: distance to relevant bound (positive if below lower; negative if above upper; 0 if within)
        gap = pd.Series(0.0, index=df.index)
        gap = gap.mask(pos.eq("below_lower"), lo - obs)   # positive
        gap = gap.mask(pos.eq("above_upper"), hi - obs)   # negative

        # formatted badge
        def fmt_badge(p, g):
            if p == "within" or pd.isna(g):
                return "Within CI"
            if AS_PERCENT:
                val = g * 100.0
                s = f"{abs(val):.1f}%"
            else:
                s = f"{abs(g):.1f}"
            return f"▲ +{s}" if p == "below_lower" else f"▼ -{s}"

        badge = [fmt_badge(p, g) for p, g in zip(pos, gap)]

        return (
            pos.rename(f"ci_position_{h}"),
            gap.rename(f"ci_gap_{h}"),
            pd.Series(badge, index=df.index, name=f"ci_badge_{h}")
        )

    for h in ["1mo", "2mo", "3mo"]:
        if f"lower_bound_{h}" in merged and f"upper_bound_{h}" in merged and f"pred_{h}" in merged:
            pos, gap, badge = ci_dev_block(merged, h)
            merged[pos.name] = pos
            merged[gap.name] = gap
            merged[badge.name] = badge

    # -------- Long format (carry new fields) --------
    def to_long(h_label: str):
        cols = {
            f"trend_{h_label}": "predicted_trend",
            f"pred_{h_label}": "predicted_value",
            f"lower_bound_{h_label}": "lower_bound",
            f"upper_bound_{h_label}": "upper_bound",
            f"trend_with_CI_{h_label}": "predicted_trend_CI",
            f"ci_position_{h_label}": "ci_position",
            f"ci_gap_{h_label}": "ci_gap",
            f"ci_badge_{h_label}": "ci_badge",
        }
        keep = ["Ward", "time_period"] + [c for c in cols.keys() if c in merged.columns]
        df = merged[keep].rename(columns=cols)
        df["horizon"] = h_label
        return df

    trend_long = pd.concat([to_long("1mo"), to_long("2mo"), to_long("3mo")], ignore_index=True, sort=False)

    # Attach observed & county once
    trend_long = trend_long.merge(
        merged[["Ward", "County", "time_period", "observed", "observed_trend_2mo"]],
        on=["Ward", "time_period"],
        how="left"
    )

    # Alert flag (unchanged)
    trend_long["alert_flag"] = False
    is_3mo = trend_long["horizon"].eq("3mo")
    ok_obs = trend_long["observed"].ge(ALERT_THRESHOLD)
    ok_ci  = trend_long["predicted_trend_CI"].isin(["Increasing", "Stable"])
    trend_long.loc[is_3mo & ok_obs & ok_ci, "alert_flag"] = True

    # Save long
    trend_long = trend_long.sort_values(["Ward", "time_period", "horizon"])
    long_path = f"data/clean_trend_long_with_CI{'_risk' if col_prefix=='risk_' else ''}.csv"
    trend_long.to_csv(long_path, index=False)
    print(f"✅ Saved {target_name} trend (long) with CI: {long_path}")

    # === Optional quick plots (unchanged) ===
    if PLOT_ALERTS:
        alerts_3mo = trend_long[(trend_long["horizon"] == "3mo") & (trend_long["alert_flag"])]
        wards = alerts_3mo["Ward"].unique()
        if len(wards) > 0:
            colors = plt.cm.get_cmap("tab10", len(wards))
            fig, ax = plt.subplots(figsize=(14, 6))
            for i, wname in enumerate(wards):
                w = alerts_3mo[alerts_3mo["Ward"] == wname].sort_values("time_period")
                ax.plot(w["time_period"], w["observed"], label=wname, color=colors(i), marker="o", linewidth=2)
            ax.axhline(ALERT_THRESHOLD, linestyle="--", color="red", linewidth=1.5, label=f"{int(ALERT_THRESHOLD*100)}% Threshold")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)
            ax.set_title(f"Observed {target_name} in Wards That Triggered Alerts (3mo rule)", fontsize=14)
            ax.set_ylabel(target_name); ax.set_xlabel("Time")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout(); #plt.show()

    # -------- Wide format (now includes CI deviation fields) --------
    base = trend_long.sort_values(["Ward", "time_period", "horizon"]).drop_duplicates(["Ward", "time_period", "horizon"])

    def pv(col):
        out = base.pivot(index=["Ward","time_period"], columns="horizon", values=col)
        out.columns = [f"{col}_{c}" for c in out.columns]
        return out

    pv_value    = pv("predicted_value")
    pv_trend    = pv("predicted_trend")
    pv_trend_ci = pv("predicted_trend_CI")
    pv_lb       = pv("lower_bound")
    pv_ub       = pv("upper_bound")
    pv_pos      = pv("ci_position")
    pv_gap      = pv("ci_gap")
    pv_badge    = pv("ci_badge")

    observed_block = (base[["Ward","County","time_period","observed","observed_trend_2mo"]]
                      .drop_duplicates(["Ward","time_period"])
                      .set_index(["Ward","time_period"]))
    alert_3mo = base[base["horizon"]=="3mo"][["Ward","time_period","alert_flag"]].set_index(["Ward","time_period"])

    trend_wide = (
        pd.concat([observed_block, alert_3mo,
                   pv_value, pv_trend, pv_trend_ci, pv_lb, pv_ub,
                   pv_pos, pv_gap, pv_badge], axis=1)
          .reset_index()
          .sort_values(["Ward","time_period"])
    )

    # Consecutive alerts (unchanged)
    trend_wide["alert_flag"] = trend_wide["alert_flag"].fillna(False).astype(bool)
    trend_wide["consecutive_alerts"] = 0
    for ward, grp in trend_wide.groupby("Ward", sort=False):
        streak = (grp["alert_flag"].astype(int)
                  .groupby((~grp["alert_flag"]).cumsum()).cumsum())
        trend_wide.loc[grp.index, "consecutive_alerts"] = streak.values

    wide_path = f"data/clean_trend_wide_with_CI{'_risk' if col_prefix=='risk_' else ''}.csv"
    trend_wide.to_csv(wide_path, index=False)
    print(f"✅ Saved {target_name} trend (wide) with CI: {wide_path}")

# =============================
# Run for both targets
# =============================
# 1) Wasting prevalence (no prefix)
run_trend_pipeline_for_target(
    target_name="Wasting Prevalence",
    file_stub="wasting",
    col_prefix=""
)
# 2) Wasting risk (risk_ prefix)
run_trend_pipeline_for_target(
    target_name="Wasting Risk",
    file_stub="wasting_risk",
    col_prefix="risk_"
)

