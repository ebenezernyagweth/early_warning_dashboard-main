import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
import glob
import os
from dash import dash_table
import re

# Optional app-level basic auth (controlled by env var USE_DASH_AUTH)
USE_DASH_AUTH = os.getenv("USE_DASH_AUTH", "false").lower() == "true"
if USE_DASH_AUTH:
    import dash_auth
# ======================
# Setup
# ======================
server = Flask(__name__)
app = dash.Dash(
    __name__, server=server,
    suppress_callback_exceptions=True,
    assets_folder="assets",
    assets_url_path="/assets",
)
#app.title = "Wasting Prediction Dashboard"

# ---- Attach basic auth if enabled ----
if USE_DASH_AUTH:
    # You will set DASH_USER and DASH_PASSWORD in Render
    VALID_USERNAME_PASSWORD_PAIRS = {
        os.getenv("DASH_USER", "user"): os.getenv("DASH_PASSWORD", "password")
    }
    dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

# ======================
# Targets (config)
# ======================
TARGETS = {
    "wasting": {
        "pretty": "Wasting Prevalence",
        "file_stub": "wasting",
        "trend_file": "data/clean_trend_wide_with_CI.csv",
        "prefix": "",            # columns: observed, pred_*, lower_*, upper_*
        "obs_col_in_hb3": "wasting_smoothed"
    },
    "wasting_risk": {
        "pretty": "Wasting Risk",
        "file_stub": "wasting_risk",
        "trend_file": "data/clean_trend_wide_with_CI_risk.csv",
        "prefix": "risk_",       # columns: risk_observed, risk_pred_*, risk_lower_*, risk_upper_*
        "obs_col_in_hb3": "risk_observed"  # if your _hb_3 files have this; otherwise map below
    }
}
ALLOWED_COUNTIES = {"tana river", "marsabit", "isiolo"}

def norm_text(s):
    return s.astype(str).str.strip()

def norm_county(s):
    return s.astype(str).str.strip().str.lower()

# ======================
# Geo data
# ======================
gdf = gpd.read_file("data/Kenya_wards_with_counties.geojson")
gdf["Ward"] = norm_text(gdf["Ward"])
gdf["County"] = norm_county(gdf["County"])
# Keep only wards in the allowed counties
if "County" in gdf.columns:
    gdf = gdf[gdf["County"].isin(ALLOWED_COUNTIES)].copy()
else:
    raise ValueError("Expected 'County' column in Kenya_wards_with_counties.geojson")

gdf["geometry_json"] = gdf["geometry"].apply(lambda g: g.__geo_interface__)


counties = gpd.read_file("data/ken_admbnda_adm1_iebc_20191031.shp").to_crs(gdf.crs)

# ======================
# Helpers: load & standardize
# ======================
def _standardize_hb(df, target_key, horizon):
    """
    Convert hb1/hb2/hb3 input CSV into a fully standardized canonical format:

        pred_{h}mo
        lower_bound_{h}mo
        upper_bound_{h}mo
        observed

    Regardless of prefixes or naming differences in the raw CSVs.
    """

    pref = TARGETS[target_key]["prefix"]  # '' for wasting, 'risk_' for risk

    # --- Normalize Ward ---
    df["Ward"] = df["Ward"].astype(str).str.strip()

    # --- Normalize County (if present) ---
    if "County" in df.columns:
        df["County"] = norm_county(df["County"])

    # --- Standardize dates ---
    df["time_period"] = pd.to_datetime(df["time_period"], errors="coerce") \
                           .dt.to_period("M").dt.start_time

    # ---------------------------------------
    # PRED COLUMNS
    # ---------------------------------------
    pred_raw = f"{pref}pred_{horizon}mo"
    pred_std = f"pred_{horizon}mo"

    if pred_raw in df.columns:
        df = df.rename(columns={pred_raw: pred_std})
    elif pred_std not in df.columns:
        df[pred_std] = pd.NA

    # ---------------------------------------
    # CI COLUMNS
    # ---------------------------------------
    lo_raw = f"{pref}lower_bound_{horizon}mo"
    up_raw = f"{pref}upper_bound_{horizon}mo"
    lo_std = f"lower_bound_{horizon}mo"
    up_std = f"upper_bound_{horizon}mo"

    if lo_raw in df.columns:
        df = df.rename(columns={lo_raw: lo_std})
    elif lo_std not in df.columns:
        df[lo_std] = pd.NA

    if up_raw in df.columns:
        df = df.rename(columns={up_raw: up_std})
    elif up_std not in df.columns:
        df[up_std] = pd.NA

    # ---------------------------------------
    # OBSERVED COLUMN
    # ---------------------------------------
    obs_candidates = [
        TARGETS[target_key]["obs_col_in_hb3"],
        f"{pref}observed",
        "observed",
        "wasting_smoothed",
        "risk_observed",
        "wasting_risk_smoothed",
    ]

    found = next((c for c in obs_candidates if c in df.columns), None)
    if found:
        df = df.rename(columns={found: "observed"})
    elif "observed" not in df.columns:
        df["observed"] = pd.NA

    # ---------------------------------------
    # Ensure numeric types
    # ---------------------------------------
    for col in [pred_std, lo_std, up_std, "observed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------------------------------
    # Guarantee uniqueness (no duplicates)
    # ---------------------------------------
    df = df.drop_duplicates(subset=["Ward", "time_period"], keep="last")

    return df

def _read_hb(h, target_key):
    stub = TARGETS[target_key]["file_stub"]
    path = f"data/Smoothed_{stub}_prediction_hb_{h}.csv"
    df = pd.read_csv(path)
    return _standardize_hb(df, target_key, horizon=h)

def load_all_for_target(target_key):

    hb1 = _read_hb(1, target_key)
    hb2 = _read_hb(2, target_key)
    hb3 = _read_hb(3, target_key)

    allowed_wards = set(gdf["Ward"].unique())

    # ---------------- FILTERS ----------------
    def _filter(df):
        if "County" in df.columns:
            return df[df["County"].isin(ALLOWED_COUNTIES)].copy()
        return df[df["Ward"].isin(allowed_wards)].copy()

    hb1 = _filter(hb1)
    hb2 = _filter(hb2)
    hb3 = _filter(hb3)

    # ---------------- TREND FILE ----------------
    trend_file = TARGETS[target_key]["trend_file"]
    trend_df = pd.read_csv(trend_file)

    trend_df["Ward"] = norm_text(trend_df["Ward"])
    trend_df["time_period"] = pd.to_datetime(trend_df["time_period"]) \
                                .dt.to_period("M").dt.start_time

    if "County" in trend_df.columns:
        trend_df["County"] = norm_county(trend_df["County"])


    trend_df = trend_df[trend_df["County"].isin(ALLOWED_COUNTIES)].copy()
    trend_df = trend_df[trend_df["Ward"].isin(allowed_wards)].copy()

    # hb3 already has observed standardized — do NOT merge again
    if "observed" not in hb3.columns:
        hb3["observed"] = pd.NA

    # ---------------- Last observed month ----------------
    last_obs = trend_df["time_period"].max()

    last_map_month_str = pd.Timestamp(last_obs).strftime("%Y-%m")

    available_months = sorted(
        trend_df["time_period"].dropna().dt.to_period("M").astype(str).unique()
    )

    county_list = sorted(
        set(trend_df["County"].dropna().unique()).intersection(ALLOWED_COUNTIES)
    )

    return hb1, hb2, hb3, trend_df, last_map_month_str, available_months, county_list, TARGETS[target_key]["pretty"]

def _first_tab_like_ts(
    hb1_unused,
    hb2_unused,
    hb3,
    ward,
    title,
    end_month_ts=None,
    months_back=None,
    county=None,
    alert_months=None,
):
    """
    CLEAN VERSION — USE ONLY HB3 (3-MONTH HORIZON) FOR THE ENTIRE GRAPH.
    Includes shading of alert months and a legend entry for alerts.
    """

    import plotly.graph_objects as go
    import pandas as pd
    import math

    # Colors
    CI_FILL      = "rgba(255,165,0,0.20)"
    CI_BORDER    = "rgb(230,120,0)"
    CI_WHISKER   = "rgb(230,120,0)"
    PRED_LINE    = "rgb(33,113,181)"
    OBS_LINE     = "black"
    BAND_FILL    = "rgba(0,0,0,0.035)"
    ALERT_FILL   = "rgba(165, 42, 42, 0.12)"   # brown-red shading

    if not ward:
        return go.Figure()

    w = str(ward).strip()

    # ---------------- SELECT WARD ----------------
    if county is not None and "County" in hb3.columns:
        df = hb3[
            (hb3["Ward"] == w) &
            (hb3["County"] == county)
        ].copy()
    else:
        df = hb3[hb3["Ward"] == w].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for this ward",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="gray")
        )
        return fig

    # Ensure proper datetime
    df["time_period"] = pd.to_datetime(df["time_period"], errors="coerce")
    df = df.sort_values("time_period")

    # Keep only needed cols
    keep = ["Ward", "County", "time_period",
            "observed", "pred_3mo", "lower_bound_3mo", "upper_bound_3mo"]
    df = df[[c for c in df.columns if c in keep]]

    # ---------------- LOOKBACK FILTER ----------------
    anchor_ts = end_month_ts if end_month_ts is not None else df["time_period"].max()

    if end_month_ts is not None:
        df = df[df["time_period"] <= end_month_ts]

    if months_back not in (None, -1):
        start = (pd.Period(anchor_ts, "M") - int(months_back) + 1).to_timestamp()
        df = df[df["time_period"] >= start]

    # ---------------- BUILD FIGURE ----------------
    fig = go.Figure()

    # ---------------- ALERT MONTH SHADING (must come first) ----------------
    if alert_months:
        alerts = sorted(pd.to_datetime(alert_months))
        half_month = pd.Timedelta(days=15)
        tmin, tmax = df["time_period"].min(), df["time_period"].max()

        for ts in alerts:
            if ts < tmin or ts > tmax:
                continue
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=ts - half_month,
                x1=ts + half_month,
                y0=0,
                y1=1,
                fillcolor=ALERT_FILL,
                line=dict(width=0),
                layer="below"
            )

    # ---------------- CI BOXES ----------------
    ci = df.dropna(subset=["lower_bound_3mo", "upper_bound_3mo"])
    half = pd.Timedelta(days=12)

    for ts, lo, hi in zip(ci["time_period"], ci["lower_bound_3mo"], ci["upper_bound_3mo"]):
        fig.add_shape(
            type="rect",
            x0=ts - half, x1=ts + half,
            y0=lo, y1=hi,
            line=dict(color=CI_BORDER, width=1),
            fillcolor=CI_FILL,
            layer="below"
        )

    # ---------------- CI WHISKERS ----------------
    wx, wy = [], []
    for ts, lo, hi in zip(ci["time_period"], ci["lower_bound_3mo"], ci["upper_bound_3mo"]):
        wx += [ts, ts, None]
        wy += [lo, hi, None]

    fig.add_trace(go.Scatter(
        x=wx, y=wy, mode="lines",
        name="3mo CI",
        line=dict(color=CI_WHISKER, width=2),
        hoverinfo="skip"
    ))

    # ---------------- PREDICTED ----------------
    fig.add_trace(go.Scatter(
        x=df["time_period"],
        y=df["pred_3mo"],
        mode="lines+markers",
        name="Predicted (3mo)",
        line=dict(color=PRED_LINE, dash="dot", width=3),
        connectgaps=True
    ))

    # ---------------- OBSERVED ----------------
    fig.add_trace(go.Scatter(
        x=df["time_period"],
        y=df["observed"],
        mode="lines+markers",
        name="Observed",
        line=dict(color=OBS_LINE, width=3)
    ))

    # ---------------- Y–AXIS ----------------
    ymax = max(
        0.05,
        math.ceil(
            max(
                df["observed"].max(skipna=True),
                df["pred_3mo"].max(skipna=True),
                df["upper_bound_3mo"].max(skipna=True),
            ) / 0.05
        ) * 0.05
    )
    fig.update_yaxes(range=[0, ymax], dtick=0.05, tickformat=".2%")

    # ---------------- LIGHT BACKGROUND STRIPS ----------------
    level, toggle = 0, True
    while level < ymax:
        if toggle:
            fig.add_shape(
                type="rect",
                xref="paper", x0=0, x1=1,
                yref="y", y0=level, y1=min(level + 0.05, ymax),
                fillcolor=BAND_FILL, line_width=0, layer="below"
            )
        toggle = not toggle
        level += 0.05

    # ---------------- LEGEND ENTRY FOR ALERT ----------------
    if alert_months:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                size=14,
                color=ALERT_FILL,
                line=dict(width=1, color="rgba(165,42,42,1)")
            ),
            name="Alert month"
        ))

    # ---------------- LAYOUT ----------------
    fig.update_layout(
        title=f"{w} — {title}",
        xaxis_title="Date",
        yaxis_title="Prevalence",
        height=420,
        legend=dict(orientation="h", x=0, y=1.02),
        margin=dict(t=80, r=40, l=70, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    fig.update_xaxes(tickformat="%Y-%m")

    return fig

def _table_last_n_months_ci(hb1_unused, hb2_unused, hb3, ward, end_ts, n=10, county=None):
    """
    Table aligned with the time-series graph:
      • Uses ONLY hb3 (3-month horizon) — same data as the plot.
      • 9-row window by default:
          - Up to 6 historic rows (latest first in window order),
          - 1 summary row,
          - Up to 3 future rows.
      • All prediction / CI info comes from pred_3mo, lower_bound_3mo, upper_bound_3mo.
    """
    import pandas as pd

    if end_ts is None or not ward:
        return []

    w = str(ward).strip()

    # --- Filter hb3 exactly like the graph ---
    if county is not None and "County" in hb3.columns:
        df = hb3[
            (hb3["Ward"].astype(str).str.strip() == w)
            & (hb3["County"] == county)
        ].copy()
    else:
        df = hb3[hb3["Ward"].astype(str).str.strip() == w].copy()

    if df.empty:
        return []

    # Normalize time_period to month start
    df["time_period"] = pd.to_datetime(
        df["time_period"], errors="coerce"
    ).dt.to_period("M").dt.start_time

    # Standardize prediction column name to predicted_value_3mo (from pred_3mo)
    if "predicted_value_3mo" not in df.columns and "pred_3mo" in df.columns:
        df = df.rename(columns={"pred_3mo": "predicted_value_3mo"})

    # Ensure numeric types where relevant
    for c in [
        "observed",
        "predicted_value_3mo",
        "lower_bound_3mo",
        "upper_bound_3mo",
        "ci_badge_3mo",
        "ci_position_3mo",
        "ci_gap_3mo",
    ]:
        if c in df.columns and c not in ("ci_badge_3mo", "ci_position_3mo"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Build the window of months (n rows back including end_ts) ---
    idx = pd.period_range(
        end_ts.to_period("M") - (n - 1),
        end_ts.to_period("M"),
        freq="M",
    ).to_timestamp()

    keep = [
        c
        for c in [
            "time_period",
            "observed",
            "predicted_value_3mo",
            "lower_bound_3mo",
            "upper_bound_3mo",
            "ci_badge_3mo",
            "ci_position_3mo",
            "ci_gap_3mo",
        ]
        if c in df.columns
    ]

    base = df[keep].drop_duplicates(subset=["time_period"]).set_index("time_period")
    aligned = pd.DataFrame(index=idx).join(base, how="left")

    # Last observed month (for splitting history vs future)
    last_obs_ts = (
        df.loc[df["observed"].notna(), "time_period"].max()
        if "observed" in df.columns
        else pd.NaT
    )

    pct = lambda v: "" if pd.isna(v) else f"{v:.1%}"

    def fmt_pred_ci(p, lo, hi):
        has_p = pd.notna(p)
        has_ci = pd.notna(lo) and pd.notna(hi)
        if has_p and has_ci:
            return f"{pct(p)}({pct(lo)}–{pct(hi)})"
        if has_p:
            return pct(p)
        if has_ci:
            return f"({pct(lo)}–{pct(hi)})"
        return ""

    month_rows, observed_flags = [], []
    for _, r in aligned.reset_index(names="time_period").iterrows():
        ts = r["time_period"]
        obs = r.get("observed")
        pred3 = r.get("predicted_value_3mo")
        lo3 = r.get("lower_bound_3mo")
        hi3 = r.get("upper_bound_3mo")

        is_future = pd.isna(obs) or (pd.notna(last_obs_ts) and ts > last_obs_ts)

        # --- CI badge/position: only for historical months (3-mo horizon) ---
        if is_future:
            ci_dev = ""
            ci_pos = ""
        else:
            if pd.notna(r.get("ci_badge_3mo", pd.NA)):
                ci_dev = r["ci_badge_3mo"]
                ci_pos = (r.get("ci_position_3mo") or "")
            else:
                # Compute deviation relative to 3-month CI
                if pd.isna(obs) or pd.isna(lo3) or pd.isna(hi3):
                    ci_dev, ci_pos = "", ""
                elif obs < lo3:
                    ci_dev = f"▲ +{(lo3 - obs) * 100:.1f}%"
                    ci_pos = "below_lower"
                elif obs > hi3:
                    ci_dev = f"▼ {(hi3 - obs) * 100:.1f}%"
                    ci_pos = "above_upper"
                else:
                    ci_dev, ci_pos = "Within CI", "within"

        # Always use the 3-month horizon from hb3 for prediction + CI
        pred_ci = fmt_pred_ci(pred3, lo3, hi3)

        month_rows.append(
            {
                "ts": ts,
                "obs": obs,
                "lo3": lo3,
                "hi3": hi3,
                "ci_pos": ci_pos,
                "row": {
                    "month": ts.strftime("%Y-%m"),
                    "observed": pct(obs),
                    "pred_ci": pred_ci,
                    "ci_dev": ci_dev,
                    "ci_pos": ci_pos,
                },
            }
        )
        observed_flags.append(pd.notna(obs))

    # If no observed months → just return rows (no summary / future separation)
    if not any(observed_flags):
        return [m["row"] for m in month_rows]

    # --- Split history / future at last observed month ---
    last_obs_idx = max(i for i, f in enumerate(observed_flags) if f)
    hist_all = month_rows[: last_obs_idx + 1]
    fut_all = month_rows[last_obs_idx + 1 :]

    # Target counts
    hist_target = min(6, len(hist_all))
    fut_target = min(3, len(fut_all))

    # Fit into n rows (n = hist + summary + fut)
    if hist_target + 1 + fut_target > n:
        overflow = hist_target + 1 + fut_target - n
        cut_fut = min(fut_target, overflow)
        fut_target -= cut_fut
        overflow -= cut_fut
        if overflow > 0:
            hist_target = max(0, hist_target - overflow)

    hist_keep = hist_all[-hist_target:] if hist_target > 0 else []
    fut_keep = fut_all[:fut_target] if fut_target > 0 else []

    # Summary average: ONLY over kept historic rows that are OUTSIDE 3-mo CI
    diffs = []
    for m in hist_keep:
        obs = m["obs"]
        lo3 = m["lo3"]
        hi3 = m["hi3"]
        if pd.notna(obs) and pd.notna(lo3) and pd.notna(hi3):
            if obs < lo3:
                diffs.append(float(lo3 - obs))
            elif obs > hi3:
                diffs.append(float(obs - hi3))

    avg_err = (sum(diffs) / len(diffs)) if diffs else None
    n_used = len(diffs)

    summary_txt = (
        "Avg. 3-month prediction error (past 6 mo): — (n=0)"
        if avg_err is None
        else f"Avg. 3-month prediction error (past 6 mo): {avg_err:.1%} (n={n_used})"
    )

    summary_row = {
        "month": "",
        "observed": "",
        "pred_ci": summary_txt,
        "ci_dev": "",
        "ci_pos": "summary",
    }

    return [h["row"] for h in hist_keep] + [summary_row] + [f["row"] for f in fut_keep]

def build_ward_monthly_frame(hb3, trend_df, ward, county=None, end_month=None):
    """
    Month grid for one ward:
      - observed comes from trend_df (same as maps)
      - pred/CI comes from hb3
      - months exist even if hb3 is missing them
      - timeline spans from earliest available month to latest (obs/pred/ref)
    """
    import pandas as pd

    w = str(ward).strip()

    # ---- observed (source of truth for observed map) ----
    obs = trend_df[["Ward","County","time_period","observed"]].copy()
    obs["Ward"] = obs["Ward"].astype(str).str.strip()
    obs["time_period"] = pd.to_datetime(obs["time_period"], errors="coerce") \
                           .dt.to_period("M").dt.start_time

    if county is not None and "County" in obs.columns:
        obs = obs[(obs["Ward"] == w) & (obs["County"] == county)]
    else:
        obs = obs[obs["Ward"] == w]

    obs = obs[["time_period", "observed"]].drop_duplicates("time_period", keep="last")
    obs["observed"] = pd.to_numeric(obs["observed"], errors="coerce")

    # ---- hb3 predictions/CIs ----
    pred = hb3[["Ward","County","time_period","pred_3mo","lower_bound_3mo","upper_bound_3mo"]].copy()
    pred["Ward"] = pred["Ward"].astype(str).str.strip()
    pred["time_period"] = pd.to_datetime(pred["time_period"], errors="coerce") \
                            .dt.to_period("M").dt.start_time

    if county is not None and "County" in pred.columns:
        pred = pred[(pred["Ward"] == w) & (pred["County"] == county)]
    else:
        pred = pred[pred["Ward"] == w]

    pred = pred[["time_period", "pred_3mo", "lower_bound_3mo", "upper_bound_3mo"]] \
              .drop_duplicates("time_period", keep="last")

    for c in ["pred_3mo", "lower_bound_3mo", "upper_bound_3mo"]:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    # ---- decide calendar start/end correctly ----
    min_obs  = obs["time_period"].min() if not obs.empty else pd.NaT
    max_obs  = obs["time_period"].max() if not obs.empty else pd.NaT
    min_pred = pred["time_period"].min() if not pred.empty else pd.NaT
    max_pred = pred["time_period"].max() if not pred.empty else pd.NaT

    start_candidates = [t for t in [min_obs, min_pred] if pd.notna(t)]
    end_candidates   = [t for t in [max_obs, max_pred, end_month] if pd.notna(t)]

    if not start_candidates or not end_candidates:
        return pd.DataFrame(columns=["time_period", "observed", "pred_3mo", "lower_bound_3mo", "upper_bound_3mo"])

    start_month = min(start_candidates)
    end_month   = max(end_candidates)

    start_month = pd.Period(start_month, "M").to_timestamp()
    end_month   = pd.Period(end_month, "M").to_timestamp()

    idx = pd.period_range(start_month.to_period("M"), end_month.to_period("M"), freq="M").to_timestamp()
    base = pd.DataFrame({"time_period": idx})

    out = (base
           .merge(obs,  on="time_period", how="left")
           .merge(pred, on="time_period", how="left"))

    # add labels back (useful for downstream funcs / debugging)
    out["Ward"] = w
    if county is not None:
        out["County"] = county

    return out.sort_values("time_period")


COLOR_WITHIN = "#EEF7EE"   # within CI
COLOR_BELOW  = "#E6F0FF"   # observed below lower → ▲
TEXT_BELOW   = "#0B5ED7"
COLOR_ABOVE  = "#FFEAEA"   # observed above upper → ▼
TEXT_ABOVE   = "#C1121F"

def make_ci_table(table_id: str):
    return dash_table.DataTable(
        id=table_id,
        columns=[
            {"name": "Month", "id": "month", "type": "text"},
            {"name": "Observed", "id": "observed", "type": "text"},
            {"name": "Pred (3 mo) + CI", "id": "pred_ci", "type": "text"},
            {"name": "Prediction error", "id": "ci_dev", "type": "text"},
            {"name": "pos", "id": "ci_pos", "type": "text"},  # helper for styling; we'll hide it via CSS
        ],
        data=[],
        style_as_list_view=True,
        style_cell={"padding": "6px", "fontSize": 12, "textAlign": "center"},
        style_header={"fontWeight": "bold"},
        style_table={"overflowX": "auto"},

        # Hide ci_pos column (both header and cells)
        style_cell_conditional=[
            {"if": {"column_id": "ci_pos"}, "display": "none"}
        ],
        style_header_conditional=[
            {"if": {"column_id": "ci_pos"}, "display": "none"}
        ],

        # Color the badge cell based on ci_pos
        style_data_conditional=[
            # --- summary row: visually span across by hiding other cells ---
            {"if": {"filter_query": "{ci_pos} = 'summary'", "column_id": "month"},    "display": "none"},
            {"if": {"filter_query": "{ci_pos} = 'summary'", "column_id": "observed"}, "display": "none"},
            {"if": {"filter_query": "{ci_pos} = 'summary'", "column_id": "ci_dev"},   "display": "none"},
            {
                "if": {"filter_query": "{ci_pos} = 'summary'", "column_id": "pred_ci"},
                "backgroundColor": "#F4F4F5",
                "fontWeight": "700",
                "textAlign": "left",
                "whiteSpace": "normal",
            },

            # --- existing badge color rules ---
            {"if": {"filter_query": "{ci_pos} = 'within'", "column_id": "ci_dev"},
            "backgroundColor": COLOR_WITHIN, "color": "black", "fontWeight": 500},
            {"if": {"filter_query": "{ci_pos} = 'below_lower'", "column_id": "ci_dev"},
            "backgroundColor": COLOR_BELOW, "color": TEXT_BELOW, "fontWeight": 600},
            {"if": {"filter_query": "{ci_pos} = 'above_upper'", "column_id": "ci_dev"},
            "backgroundColor": COLOR_ABOVE, "color": TEXT_ABOVE, "fontWeight": 600},
        ]
            ,
        page_size=10,
    )


# Preload both targets in memory (fast switching)
CACHE = {}
for k in TARGETS.keys():
    CACHE[k] = {}
    (CACHE[k]["hb1"], CACHE[k]["hb2"], CACHE[k]["hb3"],
    CACHE[k]["trend_df"],
    CACHE[k]["last_map_month_str"],
    CACHE[k]["available_months"],
    CACHE[k]["county_list"],
    CACHE[k]["pretty"]) = load_all_for_target(k)


# Default target
DEFAULT_TARGET = "wasting"

def _build_alert_map_from_df(gdf_, counties_gdf, trend_df, month):
    month_df = trend_df[trend_df["time_period"] == month][[
        "Ward","County","predicted_trend_CI_3mo","predicted_value_3mo",
        "lower_bound_3mo","upper_bound_3mo","alert_flag"
    ]].copy()
    month_df["alert_flag"] = month_df["alert_flag"].fillna(False)
    month_df["alert_status"] = month_df["alert_flag"].map({True: "active", False: "inactive"})

    merged = gdf_[["Ward","geometry_json"]].merge(month_df, on="Ward", how="left")

    def fmt_hover(r):
        val = r.get("predicted_value_3mo")
        lb  = r.get("lower_bound_3mo")
        ub  = r.get("upper_bound_3mo")
        val_str = "N/A" if pd.isna(val) else f"{val:.3f}"
        ci_str  = "N/A" if pd.isna(lb) or pd.isna(ub) else f"[{lb:.3f} – {ub:.3f}]"
        trig    = r.get("alert_status", "inactive")
        return (
            f"<b>County:</b> {r.get('County','')}"
            f"<br><b>Ward:</b> {r.get('Ward','')}"
            f"<br><b>trigger:</b> {trig}"
            f"<br><b>Predicted Trend (3mo):</b> {r.get('predicted_trend_CI_3mo','N/A')}"
            f"<br><b>Predicted (3mo):</b> {val_str}"
            f"<br><b>95% CI:</b> {ci_str}"
        )

    merged["hover_label"] = merged.apply(fmt_hover, axis=1)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": geom,
             "properties": {"Ward": ward, "hover_label": label}}
            for ward, geom, label in zip(merged["Ward"], merged["geometry_json"], merged["hover_label"])
        ]
    }

    fig = px.choropleth_mapbox(
        merged,
        geojson=geojson,
        locations="Ward",
        featureidkey="properties.Ward",
        color="alert_status",
        color_discrete_map={"inactive": "lightblue", "active": "brown"},
        custom_data=["hover_label"],
        mapbox_style="carto-positron",
        zoom=5.5,
        center={"lat": 0.5, "lon": 37},
        height=420,
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(legend_title_text="")

    # county outlines
    for _, row in counties_gdf.iterrows():
        for poly in getattr(row.geometry, "geoms", [row.geometry]):
            x, y = poly.exterior.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(x), lat=list(y), mode="lines",
                line=dict(color="black"), hoverinfo="skip", showlegend=False
            ))
    fig.update_layout(margin=dict(r=0, t=0, l=0, b=0))
    return fig


def _build_obs_map_from_df(gdf_, counties_gdf, trend_df, month):
    # Observed for the selected month
    month_df = trend_df[trend_df["time_period"] == month][["Ward", "observed"]].copy()

    # Merge Ward + County + geometry
    merged = gdf_[["Ward", "County", "geometry"]].merge(month_df, on="Ward", how="left")

    # Buckets (legend labels in percent)
    def bucket(v):
        if pd.isna(v): return "No Data"
        if v <= 0.10:  return "0–10%"
        if v <= 0.15:  return "10–15%"
        return ">15%"

    order = ["No Data", "0–10%", "10–15%", ">15%"]
    color_map = {
        "No Data": "lightgray",
        "0–10%":  "lightblue",
        "10–15%": "lightcoral",
        ">15%":   "brown",
    }

    merged["group"] = merged["observed"].apply(bucket)
    merged["group"] = pd.Categorical(merged["group"], categories=order, ordered=True)

    fig = px.choropleth_mapbox(
        merged,
        geojson=merged.geometry,
        locations=merged.index,  # we fully control hover via custom_data
        color="group",
        color_discrete_map=color_map,
        category_orders={"group": order},
        mapbox_style="carto-positron",
        zoom=5.5,
        center={"lat": 0.5, "lon": 37},
        height=420,
        custom_data=["Ward", "County", "observed"]
    )

    # Consistent hover
    fig.update_traces(
        hovertemplate=(
            "<b>County:</b> %{customdata[1]}<br>"
            "<b>Ward:</b> %{customdata[0]}<br>"
            "<b>Observed:</b> %{customdata[2]:.1%}"
            "<extra></extra>"
        )
    )

    # County outlines
    for _, row in counties_gdf.iterrows():
        geom = row.geometry
        for poly in getattr(geom, "geoms", [geom]):
            x, y = poly.exterior.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(x), lat=list(y), mode="lines",
                line=dict(color="black"),
                hoverinfo="skip",
                showlegend=False
            ))

    # --- Ensure ALL legend items show every month ---
    present = set(merged["group"].dropna().astype(str).unique())
    missing = [c for c in order if c not in present]
    for cat in missing:
        # Add a legend-only dummy trace in the right color
        fig.add_trace(go.Scattermapbox(
            lon=[37], lat=[0.5],  # any point; hidden from view
            mode="markers",
            marker=dict(size=10, color=color_map[cat], opacity=0),
            name=cat,
            hoverinfo="skip",
            visible="legendonly",
            showlegend=True
        ))

    # Clean legend
    fig.update_layout(
        legend_title_text="",
        legend_traceorder="normal",
        margin=dict(r=0, t=0, l=0, b=0)
    )
    return fig

# ======================
# Covariate figures utils (unchanged)
# ======================
def list_figure_files():
    base = os.path.join("assets", "figures")
    if not os.path.isdir(base):
        return []
    exts = {".avif", ".webp", ".jpg", ".jpeg", ".png"}
    files = [f for f in os.listdir(base)
             if os.path.isfile(os.path.join(base, f))
             and os.path.splitext(f)[1].lower() in exts]
    files.sort(key=lambda f: os.path.getsize(os.path.join(base, f)))
    return files


# ======================
# Newest "top_features" finder
# ======================
from typing import Optional, Tuple

def _parse_date_from_name(fname: str) -> Optional[Tuple[int, int, int]]:
    """
    Try to extract a (year, month, day) from a filename.
    Supports patterns like:
      - ..._2024_11.png
      - ..._2024-11-03.webp
      - ...202411.png
      - ...20241103.jpg
    Returns None if no plausible date is found.
    """
    name, _ = os.path.splitext(os.path.basename(fname))
    name_lower = name.lower()

    # Common patterns with separators (underscore or dash)
    # e.g. _2024_11, -2024-11-03, _2024-11-03, etc.
    m = re.findall(r'(?:^|[_-])(20\d{2})[_-]?([01]\d)(?:[_-]?([0-3]\d))?', name_lower)
    if m:
        # Take the last match in case there are multiple tokens
        y, mo, d = m[-1]
        year = int(y)
        month = int(mo)
        day = int(d) if d else 1
        return (year, month, day)

    # Fallback: contiguous 6–8 digits that look like YYYYMM or YYYYMMDD
    m2 = re.findall(r'(20\d{2})([01]\d)([0-3]\d)?', name_lower)
    if m2:
        y, mo, d = m2[-1]
        year = int(y)
        month = int(mo)
        day = int(d) if d else 1
        return (year, month, day)

    return None


def newest_varimp_filename(outcome_stub: Optional[str] = None) -> Optional[str]:
    """
    Return the newest 'top_features...' image filename in assets/figures.

    'Newest' is defined primarily by the date encoded in the filename
    (e.g. top_features_2024_11.png, top_features_2024-11-03.png).
    If no date can be parsed, falls back to filesystem mtime.
    On ties, prefers PNG.

    If outcome_stub is provided, restrict to files that contain it
    (case-insensitive).
    """
    base = os.path.join("assets", "figures")
    if not os.path.isdir(base):
        return None

    exts = {".png", ".webp", ".jpg", ".jpeg"}

    def ok(fname: str) -> bool:
        name, ext = os.path.splitext(fname)
        if ext.lower() not in exts:
            return False
        if not name.lower().startswith("top_features"):
            return False
        if outcome_stub and outcome_stub.lower() not in fname.lower():
            return False
        return os.path.isfile(os.path.join(base, fname))

    files = [f for f in os.listdir(base) if ok(f)]
    if not files:
        return None

    def sort_key(f: str):
        full = os.path.join(base, f)
        mtime = os.path.getmtime(full)
        parsed_date = _parse_date_from_name(f)
        has_date = parsed_date is not None
        # prefer PNG if everything else is equal
        prefer_png = (os.path.splitext(f)[1].lower() == ".png")
        # Sort by:
        #   1) has_date (files with date first)
        #   2) parsed_date (newest date)
        #   3) mtime (newest modification time)
        #   4) prefer_png
        return (has_date, parsed_date or (0, 0, 0), mtime, prefer_png)

    files.sort(key=sort_key, reverse=True)
    return files[0]


def pick_covar_files():
    """
    Return a dict with paths for 'evi', 'prec', 'conflict' by filename keywords.
    Falls back gracefully if a file is missing.
    """
    base = os.path.join("assets", "figures")
    if not os.path.isdir(base):
        return {"evi": None, "prec": None, "conflict": None}

    files = [f for f in os.listdir(base)
             if os.path.isfile(os.path.join(base, f))
             and os.path.splitext(f)[1].lower() in {".avif", ".webp", ".jpg", ".jpeg", ".png"}]

    # Prioritize newest when multiple match
    files_sorted = sorted(files, key=lambda f: os.path.getmtime(os.path.join(base, f)), reverse=True)

    def find_one(patterns):
        for f in files_sorted:
            name = f.lower()
            if any(p in name for p in patterns):
                return f"/assets/figures/{f}"
        return None

    return {
        "evi":      find_one(["evi"]),
        "prec":     find_one(["prec", "rain", "chirps"]),
        "conflict": find_one(["conflict", "acled"]),
    }

def graph_card(
    src,
    title_text,
    bullets=None,
    data_notes=None,
    text_paragraph=None,          # single string
    text_paragraphs=None,         # list/tuple of strings
    max_height="520px"
):
    """
    Render a figure with an explanatory block (no 'What you're seeing' header).
    Prefer `text_paragraphs` for multiple paragraphs; else `text_paragraph`.
    Falls back to bullets if no paragraphs provided.
    """
    placeholder = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="

    if not src:
        return html.Div(
            f"{title_text}: figure not found in ./assets/figures",
            style={"padding": "10px", "fontStyle": "italic", "border": "1px dashed #ddd",
                   "borderRadius": "8px", "background": "#fafafa"}
        )

    fig_block = html.Figure([
        html.Img(
            src=placeholder, **{"data-src": src}, className="lazy",
            style={
                "width": "100%", "height": "auto", "maxHeight": max_height,
                "objectFit": "contain", "border": "1px solid #eee", "borderRadius": "8px"
            }
        ),
        html.Figcaption(
            title_text,
            style={"textAlign": "center", "fontSize": "12px", "color": "#555", "marginTop": "6px"}
        )
    ], style={"margin": 0})

    # Explanatory text: paragraphs > single paragraph > bullets
    explain = None
    if text_paragraphs:
        explain = html.Div(
            [html.P(p, style={"margin": "6px 0"}) for p in text_paragraphs],
            style={"fontSize": "15px", "lineHeight": "1.55", "marginTop": "6px"}
        )
    elif text_paragraph:
        explain = html.Div(
            [html.P(text_paragraph, style={"margin": "6px 0"})],
            style={"fontSize": "15px", "lineHeight": "1.55", "marginTop": "6px"}
        )
    elif bullets:
        explain = html.Div(
            html.Ul([html.Li(b) for b in bullets], style={"margin": "6px 0 0 18px"}),
            style={"fontSize": "15px", "lineHeight": "1.55", "marginTop": "6px"}
        )

    notes_block = (html.Div(
        data_notes,
        style={
            "fontSize": "11px", "opacity": 0.85, "padding": "6px 8px",
            "background": "#f6f8fa", "border": "1px solid #e5e7eb",
            "borderRadius": "6px", "marginTop": "8px"
        }
    ) if data_notes else None)

    content = [fig_block]
    if explain: content.append(explain)
    if notes_block: content.append(notes_block)

    return html.Div(content, style={
        "background": "white", "border": "1px solid #eee", "borderRadius": "10px",
        "padding": "10px 12px", "boxShadow": "0 1px 6px rgba(0,0,0,0.05)"})


def graph_blocks(src, title_text, text_paragraphs=None, max_height="520px"):
    """
    Returns (fig_block, text_block). Text is BIG, justified, and indented; no footnote.
    """
    placeholder = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="

    # Left: figure
    if not src:
        fig_block = html.Div(
            f"{title_text}: figure not found in ./assets/figures",
            style={"padding": "12px", "fontStyle": "italic", "border": "1px dashed #ddd",
                   "borderRadius": "10px", "background": "#fafafa"}
        )
    else:
        fig_block = html.Figure([
            html.Img(
                src=placeholder, **{"data-src": src}, className="lazy",
                style={
                    "width": "100%", "height": "auto", "maxHeight": max_height,
                    "objectFit": "contain", "border": "1px solid #eee", "borderRadius": "10px"
                }
            ),
            html.Figcaption(
                title_text,
                style={"textAlign": "center", "fontSize": "13px", "color": "#444", "marginTop": "8px"}
            )
        ], style={"margin": 0})

    # Right: text (bigger, justified, first-line indent)
    paras = text_paragraphs or []
    text_children = [
        html.P(p, style={"margin": "0 0 16px 0", "textIndent": "1.25em"})
        for p in paras
    ]

    text_block = html.Div(
        text_children,
        style={
            "fontSize": "17px",
            "lineHeight": "1.9",
            "color": "#111827",
            "width": "100%",          # let it use the full right column
            "maxWidth": "unset",
            "textAlign": "justify",   # even edges
            "textJustify": "inter-word",
            "hyphens": "auto"
        }
    )

    return fig_block, text_block

# ======================
# Layout
# ======================
# Set the browser tab title (do this OUTSIDE layout)
app.title = "Child Hunger Early Warning Dashboard"

app.layout = html.Div([
    # =======================
    # Header
    # =======================
    html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "Child Hunger Early Warning Dashboard",
                        style={"margin": 0, "fontWeight": 700, "letterSpacing": "0.3px"},
                    ),
                    html.Div(
                        [
                            html.Span("Kenya • Ward-level analysis", style={"opacity": 0.9}),
                            html.Span("• Pilot counties:", style={"opacity": 0.9, "marginLeft": "4px"}),
                            html.Span("Tana River", className="pill"),
                            html.Span("Marsabit", className="pill"),
                            html.Span("Isiolo", className="pill"),
                        ],
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "alignItems": "center",
                            "marginTop": "6px",
                            "fontSize": "13px",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "column"},
            ),
            html.Div(
                [
                    html.Span("Machine-learning based early warning - ", className="tag"),
                    html.Span("Model retrained monthly using updated data", className="tag"),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center"},
            ),
        ],
        className="app-header",
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "flex-end",
            "padding": "14px 16px",
            "borderRadius": "12px",
            "margin": "8px 12px 6px 12px",
            "background": "linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%)",
            "color": "white",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
        },
    ),

    # =======================
    # Tabs
    # =======================
    dcc.Tabs(id="main-tabs", value="compare-tab", children=[
        # === TAB 1: Four maps + county→ward time series + tables ===
        dcc.Tab(
            label="Ward maps",
            value="compare-tab",
            children=[
                # Reference month + training window note
                html.Div(
                    [
                        html.Label("Reference month:"),
                        dcc.Dropdown(
                            id="month-select-compare",
                            style={"width": "300px"}
                        ),
                        html.Div(
                            id="training-window-note",
                            style={
                                "marginTop": "6px",
                                "fontSize": "12px",
                                "opacity": 0.9,
                                "padding": "6px 8px",
                                "display": "inline-block",
                                "borderRadius": "6px",
                                "background": "#f6f8fa",
                                "border": "1px solid #e5e7eb",
                            },
                        ),
                    ],
                    style={"margin": "12px"},
                ),

                # === Four maps (two rows) + explanatory panel ===
                html.Div([

                    # Row 1: Prevalence maps (side-by-side)
                    html.Div([
                        html.Div([
                            html.H5("Wasting prevalence — Trigger map", style={"margin": "6px 8px"}),
                            dcc.Graph(id="prev-alert-map", config={"displayModeBar": False}, style={"height": "420px"}),
                        ], style={
                            "flex": "1 1 0", "minWidth": 0,
                            "background": "white", "borderRadius": "12px",
                            "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
                            "padding": "8px",
                        }),

                        html.Div([
                            html.H5("Wasting prevalence — Measured values on reference month", style={"margin": "6px 8px"}),
                            dcc.Graph(id="prev-obs-map", config={"displayModeBar": False}, style={"height": "420px"}),
                        ], style={
                            "flex": "1 1 0", "minWidth": 0,
                            "background": "white", "borderRadius": "12px",
                            "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
                            "padding": "8px",
                        }),
                    ], style={
                        "display": "flex",
                        "gap": "12px",
                        "alignItems": "stretch",
                        "justifyContent": "space-between",
                        "flexWrap": "nowrap",
                        "marginBottom": "12px",
                        "background": "rgba(14,165,233,0.06)",  # soft row tint A
                        "padding": "8px",
                        "borderRadius": "8px",
                    }),

                    # Row 2: Risk maps (side-by-side)
                    html.Div([
                        html.Div([
                            html.H5("Wasting risk prevalence — Trigger map", style={"margin": "6px 8px"}),
                            dcc.Graph(id="risk-alert-map", config={"displayModeBar": False}, style={"height": "420px"}),
                        ], style={
                            "flex": "1 1 0", "minWidth": 0,
                            "background": "white", "borderRadius": "12px",
                            "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
                            "padding": "8px",
                        }),

                        html.Div([
                            html.H5("Wasting risk prevalence — Measured values on reference month", style={"margin": "6px 8px"}),
                            dcc.Graph(id="risk-obs-map", config={"displayModeBar": False}, style={"height": "420px"}),
                        ], style={
                            "flex": "1 1 0", "minWidth": 0,
                            "background": "white", "borderRadius": "12px",
                            "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
                            "padding": "8px",
                        }),
                    ], style={
                        "display": "flex",
                        "gap": "12px",
                        "alignItems": "stretch",
                        "justifyContent": "space-between",
                        "flexWrap": "nowrap",
                        "background": "rgba(34,197,94,0.06)",   # soft row tint B
                        "padding": "8px",
                        "borderRadius": "8px",
                    }),

                    # Explanatory panel spanning under all four
                    html.Div([
                        html.Div("Triggering system:", style={"fontWeight": 600, "marginBottom": "4px", "fontSize": "13px"}),

                        # NEW WORDING — anticipatory action anchored to WHO thresholds + 2-month persistence + 3-mo forecast
                        html.P([
                            "The anticipatory action trigger presented in this dashboard is anchored to ",
                            html.B("the WHO guidance on the public health significance of wasting"),
                            ". This way, a ward is triggered for anticipatory action purposes if"
                        ], style={"margin": "0 0 6px 0", "fontSize": "12.5px", "lineHeight": "1.45"}),

                        html.Ul([
                            html.Li([
                                html.B("WHO-referenced threshold check: "),
                                "a ward's observed wasting prevalence falls within the WHO-defined",
                                 " categories indicating serious public health concern (wasting prevalence>=10%)."
                            ]),

                            html.Li([
                                html.B("Three-month forecast confirmation: "),
                                "the model’s 3-month trend must indicate prevalence is ",
                                html.I("still high or increasing"),
                                " over the forecast horizon."
                            ]),
                        ], style={"margin": "0 0 6px 18px", "fontSize": "12.5px", "lineHeight": "1.45"}),

                        html.P(
                            "The same anticipatory logic is applied to wasting and wasting risk. "
                            "Wasting risk serves as an early indicator expected to precede—and translate into—observed wasting.",
                            style={"margin": 0, "fontSize": "12.5px", "lineHeight": "1.45"}
                        ),
                    ], style={
                        "margin": "12px 0 0 0",
                        "padding": "10px 12px",
                        "background": "#fafafa",
                        "border": "1px solid #eee",
                        "borderRadius": "10px",
                    }),
                ], style={"padding": "0 12px"}),

                html.Hr(),

                # === Selectors for ward-level time series
                html.Div(
                    [
                        html.Div([
                            html.Label("County:"),
                            dcc.Dropdown(
                                id="compare-county-select",
                                options=[], value=None,
                                style={"width": "320px"},
                            ),
                        ], style={"margin": "8px 12px", "display": "inline-block"}),

                        html.Div([
                            html.Label("Ward:"),
                            dcc.Dropdown(
                                id="compare-ward-select",
                                options=[], value=None,
                                style={"width": "360px"},
                            ),
                        ], style={"margin": "8px 12px", "display": "inline-block"}),

                        html.Div([
                            html.Label("Time window:"),
                            dcc.RadioItems(
                                id="compare-range-select",
                                options=[
                                    {"label": "6 months", "value": 6},
                                    {"label": "12 months", "value": 12},
                                    {"label": "18 months", "value": 18},
                                    {"label": "24 months", "value": 24},
                                    {"label": "All", "value": -1},
                                ],
                                value=12,
                                inline=True,
                            ),
                        ], style={"margin": "8px 12px", "display": "inline-block"}),
                    ]
                ),

                # === Prevalence plot + 9-month table
                html.Div([
                    html.Div([
                        dcc.Graph(id="prev-ward-ts", config={"displayModeBar": False}),
                    ], style={
                        "width": "68%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingRight": "8px",
                    }),
                    html.Div([
                        html.H6("Last 6 Months — Wasting Prevalence"),
                        make_ci_table("prev-ward-table"),
                    ], style={
                        "width": "31%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    }),
                ], style={"padding": "0 12px"}),

                html.Hr(style={"margin": "12px 0"}),
                # === Risk plot + 9-month table
                html.Div([
                    html.Div([
                        dcc.Graph(id="risk-ward-ts", config={"displayModeBar": False}),
                    ], style={
                        "width": "68%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingRight": "8px",
                    }),
                    # --- Risk mini table ---
                    html.Div([
                        html.H6("Last 6 Months — Wasting Risk Prevalence"),
                        make_ci_table("risk-ward-table"),
                    ], style={
                        "width": "31%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    }),
                ], style={"padding": "0 12px 12px 12px"}),
            ],
        ),

        # === Other tabs (unchanged scaffolding) ===
        dcc.Tab(
            label="All Alert Wards Time Series",
            value="alerts-tab",
            children=[
                html.Div([
                    html.Label("Reference Date (Month):"),
                    dcc.Dropdown(id="month-select-alerts", style={"width": "300px", "margin": "20px"}),
                    html.Div(id="alert-timeseries-plots"),
                ])
            ],
        ),
        dcc.Tab(
            label="Covariate Graphs",
            value="predictors-tab",
            children=[
                html.Div([
                    dcc.Interval(id="predictor-scan", interval=1, n_intervals=0, max_intervals=1),
                    html.Div(id="predictor-grid", style={"padding": "12px"}),
                ])
            ],
        ),
        dcc.Tab(
            label="Variable Importance",
            value="varimp-tab",
            children=[
                html.Div([
                    dcc.Interval(id="varimp-scan", interval=1, n_intervals=0, max_intervals=1),
                    html.Div(
                        id="varimp-grid",
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr",
                            "gap": "12px",
                            "padding": "12px",
                        },
                    ),
                ])
            ],
        ),
    ])
])

# ======================
# Callbacks
# ======================
# ---- Compare tab: month selector for the 4 maps
@app.callback(
    Output("month-select-compare", "options"),
    Output("month-select-compare", "value"),
    Input("main-tabs", "value")
)
def init_compare_month(tab):
    # initialize when entering compare-tab (also works on first load if it's default)
    if tab != "compare-tab":
        fallback = CACHE["wasting"]["last_map_month_str"]
        return [{"label": fallback, "value": fallback}], fallback

    prev_months = set(CACHE["wasting"]["trend_df"]["time_period"].dt.to_period("M").astype(str))
    risk_months = set(CACHE["wasting_risk"]["trend_df"]["time_period"].dt.to_period("M").astype(str))
    common = sorted(prev_months.intersection(risk_months)) or sorted(prev_months) or sorted(risk_months)
    value = common[-1] if common else None
    options = [{"label": m, "value": m} for m in common]
    return options, value

# --- Training window note (36 months ending at reference month - 1) ---
@app.callback(
    Output("training-window-note", "children"),
    Input("month-select-compare", "value")
)
def show_training_window(month_str):
    if not month_str:
        return ""
    end_ref = pd.Period(month_str, freq="M")
    train_end = end_ref - 1
    train_start = train_end - 35  # inclusive span of 36 months
    return f"Model training window (36 months): {train_start.start_time:%b %Y} – {train_end.start_time:%b %Y}"


# ---- Render the 4 maps (driven by the month dropdown)
@app.callback(
    Output("prev-alert-map", "figure"),
    Output("prev-obs-map", "figure"),
    Output("risk-alert-map", "figure"),
    Output("risk-obs-map", "figure"),
    Input("month-select-compare", "value")
)
def render_four_maps(month_str):
    if not month_str:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    month = pd.Period(month_str).to_timestamp()

    prev_trend = CACHE["wasting"]["trend_df"]
    risk_trend = CACHE["wasting_risk"]["trend_df"]

    fig_prev_alert = _build_alert_map_from_df(gdf, counties, prev_trend, month)
    fig_prev_obs   = _build_obs_map_from_df(gdf, counties, prev_trend, month)
    fig_risk_alert = _build_alert_map_from_df(gdf, counties, risk_trend, month)
    fig_risk_obs   = _build_obs_map_from_df(gdf, counties, risk_trend, month)

    return fig_prev_alert, fig_prev_obs, fig_risk_alert, fig_risk_obs


# ---- County selector (intersection across both targets, filtered to allowed set)
@app.callback(
    Output("compare-county-select", "options"),
    Output("compare-county-select", "value"),
    Input("main-tabs", "value")
)
def init_compare_counties(tab_value):
    c_prev = set(CACHE["wasting"]["trend_df"]["County"].dropna().unique())
    c_risk = set(CACHE["wasting_risk"]["trend_df"]["County"].dropna().unique())
    counties_common = sorted((c_prev & c_risk) & ALLOWED_COUNTIES)
    default = counties_common[0] if counties_common else None
    return [{"label": c, "value": c} for c in counties_common], default


# ---- Ward selector depends on county; must exist in BOTH datasets
@app.callback(
    Output("compare-ward-select", "options"),
    Output("compare-ward-select", "value"),
    Input("compare-county-select", "value")
)
def update_compare_wards(county):
    if not county:
        return [], None
    df_prev = CACHE["wasting"]["trend_df"]
    df_risk = CACHE["wasting_risk"]["trend_df"]
    wards_prev = set(df_prev.loc[df_prev["County"] == county, "Ward"].dropna().unique())
    wards_risk = set(df_risk.loc[df_risk["County"] == county, "Ward"].dropna().unique())
    wards = sorted(wards_prev & wards_risk)
    default = wards[0] if wards else None
    return [{"label": w, "value": w} for w in wards], default


# ---- Ward time series (driven by Ward + Time window ONLY; do NOT cap by map month)

@app.callback(
    Output("prev-ward-ts", "figure"),
    Output("risk-ward-ts", "figure"),
    Output("prev-ward-table", "data"),
    Output("risk-ward-table", "data"),
    Input("compare-county-select", "value"),
    Input("compare-ward-select", "value"),
    Input("compare-range-select", "value"),
    State("month-select-compare", "value")
)
def render_compare_ward_timeseries(county, ward, months_val, month_str):
    import pandas as pd
    import plotly.graph_objects as go

    if not ward:
        return go.Figure(), go.Figure(), [], []

    # charts time window
    months_back = None if (months_val in (None, -1)) else int(months_val)
    ref_ts = pd.Period(month_str, freq="M").to_timestamp() if month_str else None

    df_prev = build_ward_monthly_frame(
        CACHE["wasting"]["hb3"],
        CACHE["wasting"]["trend_df"],
        ward,
        county=county,
        end_month=ref_ts,   # optional cap; builder will still extend to latest pred
    )

    df_risk = build_ward_monthly_frame(
        CACHE["wasting_risk"]["hb3"],
        CACHE["wasting_risk"]["trend_df"],
        ward,
        county=county,
        end_month=ref_ts,
    )


    # -------------------------------
    # 1) FIGURES – ONLY HB3
    # -------------------------------
    def _alert_months_for_ward(trend_df, ward, county=None):
        import pandas as pd
        if trend_df is None or trend_df.empty:
            return []

        w = str(ward).strip()

        df = trend_df.copy()
        df["Ward"] = df["Ward"].astype(str).str.strip()

        if county is not None and "County" in df.columns:
            df = df[(df["Ward"] == w) & (df["County"] == county)]
        else:
            df = df[df["Ward"] == w]

        df = df[df["alert_flag"] == True]
        if df.empty:
            return []

        # convert to month-start timestamps
        return pd.to_datetime(df["time_period"]).dt.to_period("M").dt.start_time.tolist()


    alert_months_prev = _alert_months_for_ward(CACHE["wasting"]["trend_df"], ward, county)
    alert_months_risk = _alert_months_for_ward(CACHE["wasting_risk"]["trend_df"], ward, county)

    fig_prev = _first_tab_like_ts(
    None, None, df_prev, ward, "Wasting Prevalence",
    end_month_ts=None,
    months_back=months_back,
    county=None,  # already filtered
    alert_months=alert_months_prev
    )

    fig_risk = _first_tab_like_ts(
        None, None, df_risk, ward, "Wasting Risk Prevalence",
        end_month_ts=None,
        months_back=months_back,
        county=None,
        alert_months=alert_months_risk
    )




    # -------------------------------
    # 2) TABLES – ONLY HB3, SAME CUT
    # -------------------------------

    def latest_ts_for_hb3(df_hb3, w, county=None):
        """Latest time_period for this ward/county in hb3, month-start datetime."""
        if county is not None and "County" in df_hb3.columns:
            sub = df_hb3[
                (df_hb3["Ward"].astype(str).str.strip() == str(w).strip())
                & (df_hb3["County"] == county)
            ].copy()
        else:
            sub = df_hb3[df_hb3["Ward"].astype(str).str.strip() == str(w).strip()].copy()

        if sub.empty or "time_period" not in sub.columns:
            return pd.NaT

        tp = pd.to_datetime(sub["time_period"], errors="coerce")
        tp = tp.dt.to_period("M").dt.start_time
        return tp.max()

    # latest available prediction month across BOTH targets (using hb3 only)
    end_prev = latest_ts_for_hb3(CACHE["wasting"]["hb3"], ward, county=county)
    end_risk = latest_ts_for_hb3(CACHE["wasting_risk"]["hb3"], ward, county=county)

    # global end_ts across the two targets
    #valid_ends = [ts for ts in [end_prev, end_risk] if pd.notna(ts)]
    end_ts = max(
    pd.to_datetime(df_prev["time_period"], errors="coerce").max(),
    pd.to_datetime(df_risk["time_period"], errors="coerce").max(),
    )

    prev_rows = _table_last_n_months_ci(None, None, df_prev, ward, end_ts, n=10, county=None)
    risk_rows = _table_last_n_months_ci(None, None, df_risk, ward, end_ts, n=10, county=None)


    return fig_prev, fig_risk, prev_rows, risk_rows



# ---- Alerts tab 
@app.callback(
    Output("month-select-alerts", "options"),
    Output("month-select-alerts", "value"),
    Input("main-tabs", "value")
)
def init_alerts_month(tab):
    if tab != "alerts-tab":
        fallback = CACHE["wasting"]["last_map_month_str"]
        return [{"label": fallback, "value": fallback}], fallback

    prev_months = set(CACHE["wasting"]["trend_df"]["time_period"].dt.to_period("M").astype(str))
    risk_months = set(CACHE["wasting_risk"]["trend_df"]["time_period"].dt.to_period("M").astype(str))

    # Prefer intersection (months available in both); fallback to union; else to wasting only
    common = sorted(prev_months & risk_months)
    if common:
        months = common
    else:
        union = sorted(prev_months | risk_months)
        months = union if union else sorted(prev_months)

    value = months[-1] if months else None
    return [{"label": m, "value": m} for m in months], value


@app.callback(
    Output("alert-timeseries-plots", "children"),
    Input("month-select-alerts", "value")
)
def display_alert_ward_timeseries(month_str):
    if not month_str:
        return html.P("No date selected.", style={"margin": "20px", "fontStyle": "italic"})

    month = pd.Period(month_str).to_timestamp()

    # Data: trends to detect alerts, hb3s to plot time series
    prev_trend = CACHE["wasting"]["trend_df"]
    risk_trend = CACHE["wasting_risk"]["trend_df"]
    hb3_prev   = CACHE["wasting"]["hb3"]
    hb3_risk   = CACHE["wasting_risk"]["hb3"]

    # Alert sets
    prev_alerts = set(prev_trend[(prev_trend["time_period"] == month) & (prev_trend["alert_flag"] == True)]["Ward"].astype(str))
    risk_alerts = set(risk_trend[(risk_trend["time_period"] == month) & (risk_trend["alert_flag"] == True)]["Ward"].astype(str))
    all_alerts  = sorted(prev_alerts | risk_alerts)
    if not all_alerts:
        return html.P("No wards flagged for alert at this date (wasting or risk).",
                      style={"margin": "20px", "fontStyle": "italic"})

    # Category + styles
    def ward_category(w):
        w = str(w)
        return "both" if (w in prev_alerts and w in risk_alerts) else ("wasting" if w in prev_alerts else "risk")

    CAT_STYLES = {
        "wasting": {"pill_bg": "#fef3c7", "pill_fg": "#92400e", "border": "#f59e0b"},
        "risk":    {"pill_bg": "#cffafe", "pill_fg": "#115e59", "border": "#06b6d4"},
        "both":    {"pill_bg": "#e9d5ff", "pill_fg": "#6b21a8", "border": "#8b5cf6"},
    }
    CAT_LABEL = {"wasting": "WASTING", "risk": "RISK", "both": "BOTH"}

    # Plot styles for series
    SERIES = {
        "wasting": {
            "pred": dict(color="rgb(33,113,181)", dash="dot", width=3),   # blue
            "obs":  dict(color="black", width=3),
            "ci":   "rgba(33,113,181,0.12)",
            "ci_line": "rgba(33,113,181,0.28)"
        },
        "risk": {
            "pred": dict(color="rgb(0,150,136)", dash="dash", width=3),   # teal
            "obs":  dict(color="purple", width=2),
            "ci":   "rgba(0,150,136,0.12)",
            "ci_line": "rgba(0,150,136,0.28)"
        }
    }

    # Sort: BOTH → WASTING → RISK, then County, then Ward
    def sort_key(w):
        cat = ward_category(w)
        rank = {"both": 0, "wasting": 1, "risk": 2}[cat]
        dfw = hb3_prev[hb3_prev["Ward"].astype(str).str.strip() == str(w)]
        county = dfw["County"].dropna().iloc[0] if ("County" in dfw.columns and not dfw.empty and dfw["County"].notna().any()) else ""
        return (rank, str(county), str(w))

    def _prep(df):
        if df.empty:
            return df
        d = df.copy()
        d["time_period"] = pd.to_datetime(d["time_period"], errors="coerce")
        for c in ["observed","pred_3mo","lower_bound_3mo","upper_bound_3mo"]:
            if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
        return d.sort_values("time_period")

    children = [
        html.Div([
            html.Span("Legend:", style={"fontWeight": 600, "marginRight": "8px"}),
            html.Span("WASTING", className="pill",
                      style={"background": CAT_STYLES["wasting"]["pill_bg"],
                              "color": CAT_STYLES["wasting"]["pill_fg"], "marginRight": "8px"}),
            html.Span("RISK", className="pill",
                      style={"background": CAT_STYLES["risk"]["pill_bg"], 
                             "color": CAT_STYLES["risk"]["pill_fg"], "marginRight": "8px"}),
            html.Span("BOTH", className="pill",
                      style={"background": CAT_STYLES["both"]["pill_bg"], "color": CAT_STYLES["both"]["pill_fg"]}),
        ], style={"margin": "8px 20px 4px 20px", "fontSize": "12px"})
    ]

    for ward in sorted(all_alerts, key=sort_key):
        cat = ward_category(ward)
        style = CAT_STYLES[cat]

        # Build aligned month grid: observed from trend_df, pred/CI from hb3
        d_prev = build_ward_monthly_frame(hb3_prev, prev_trend, ward, county=None, end_month=None)
        d_risk = build_ward_monthly_frame(hb3_risk, risk_trend, ward, county=None, end_month=None)

        # prep (same cleaning / sorting)
        d_prev = _prep(d_prev)
        d_risk = _prep(d_risk)


        # --- Build figure ---
        fig = go.Figure()

        # --- helper: add CI fill with deduped legend entries (once per series) ---
        ci_legend_shown = {"wasting": False, "risk": False}
        def _add_ci_fill(fig_, d, color_fill, color_line, label_key):
            m = d["lower_bound_3mo"].notna() & d["upper_bound_3mo"].notna()
            dci = d.loc[m, ["time_period","lower_bound_3mo","upper_bound_3mo"]]
            if dci.empty:
                return
            runs = (dci["time_period"].diff().dt.days.gt(35) | dci["time_period"].diff().isna()).cumsum()
            for _, seg in dci.groupby(runs):
                x_poly = pd.concat([seg["time_period"], seg["time_period"].iloc[::-1]])
                y_poly = pd.concat([seg["lower_bound_3mo"], seg["upper_bound_3mo"].iloc[::-1]])
                fig_.add_trace(go.Scatter(
                    x=x_poly, y=y_poly, mode="lines", fill="toself",
                    line=dict(width=0, color=color_line),
                    fillcolor=color_fill,
                    name=("Wasting 3mo CI" if label_key == "wasting" else "Risk 3mo CI"),
                    hoverinfo="skip",
                    showlegend=(not ci_legend_shown[label_key])
                ))
                ci_legend_shown[label_key] = True

        # helpers: add a series (pred/obs + CI)
        def add_series(d, label_key):
            if d.empty: return
            _add_ci_fill(fig, d, SERIES[label_key]["ci"], SERIES[label_key]["ci_line"], label_key)
            if d["pred_3mo"].notna().any():
                fig.add_trace(go.Scatter(
                    x=d["time_period"], y=d["pred_3mo"],
                    mode="lines+markers",
                    name=("Predicted (3mo) — Wasting" if label_key=="wasting" else "Predicted (3mo) — Risk"),
                    line=SERIES[label_key]["pred"],
                    connectgaps=True
                ))
            if d["observed"].notna().any():
                fig.add_trace(go.Scatter(
                    x=d["time_period"], y=d["observed"],
                    mode="lines+markers",
                    name=("Observed — Wasting" if label_key=="wasting" else "Observed — Risk"),
                    line=SERIES[label_key]["obs"]
                ))

        if cat == "wasting":
            add_series(d_prev, "wasting")
        elif cat == "risk":
            add_series(d_risk, "risk")
        else:  # both
            add_series(d_prev, "wasting")
            add_series(d_risk, "risk")

        # Title with county (prefer prevalence df, else risk df)
        title = f"{ward}"
        for d in (d_prev, d_risk):
            if "County" in d.columns and d["County"].notna().any():
                title += f" ({d['County'].dropna().iloc[0]})"
                break

        # --- Layout: legend BELOW the plot, extra bottom margin, left-aligned title ---
        fig.update_layout(
            title=dict(text=title, x=0, xanchor="left"),
            height=320,
            margin=dict(t=56, r=20, b=96, l=60),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,   # below plotting area
                xanchor="left",
                x=0
            ),
            yaxis=dict(tickformat=".0%")
        )
        fig.update_xaxes(tickformat="%Y-%m")

        # --- Card ---
        children.append(
            html.Div([
                html.Div([
                    html.Span(CAT_LABEL[cat], className="pill",
                              style={"background": style["pill_bg"], "color": style["pill_fg"],
                                     "fontWeight": 700, "padding": "2px 8px", "borderRadius": "999px",
                                     "fontSize": "11px", "marginRight": "8px"}),
                    html.Span("Alerted by ", style={"opacity": 0.85, "fontSize": "12px"}),
                    html.Span(CAT_LABEL[cat].lower(), style={"fontStyle": "italic", "fontSize": "12px"}),
                ], style={"margin": "4px 0 6px 2px"}),

                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ], style={"borderLeft": f"4px solid {style['border']}", "background": "white",
                      "borderRadius": "8px", "padding": "10px 12px", "margin": "8px 20px",
                      "boxShadow": "0 1px 6px rgba(0,0,0,0.06)"}),
        )

    return children


# ---- Predictors & VarImp (unchanged)
@app.callback(
    Output("predictor-grid", "children"),
    Input("predictor-scan", "n_intervals"),
)
def load_predictor_plots(_):
    cov = pick_covar_files()

    # Text (two paragraphs each). You can keep using implicit concatenation for readability.
    prec_paras = [
        ("For each month, this plot shows the precipitation z-score relative to a 20-year baseline for the same "
         "calendar month. The z-score is the number of standard deviations that month’s total accumulated rainfall "
         "differs from its long-term normal value. Negative values mean unusually dry conditions (possible drought "
         "signal); positive values mean wetter-than-normal conditions. Months with anomalies are indicated with a "
         "shaded background. Long and short rain seasons are also indicated for better tracking of conditions during "
         "those months."),
        ("The number of wasted children in the sample is also overlaid on the graph. The graph shows a spike in the "
         "number of wasted children after nearly two consecutive years of drought culminating in 2023, which—as of "
         "2025—has not yet reverted to previous values.")
    ]
    evi_paras = [
        ("EVI tracks vegetation greenness and biomass. For each month, the average EVI z-score relative"
        " to a medium-term 5-year baseline is shown. The z-score shows how the vegetation greenness in that month "
        "compares to its medium-term normal value. Negative values indicate abnormally low EVI signal"
        " for that month. Persistent declines can therefore indicate reduced forage/crop."),
        ("The number of wasted children in the sample is also overlaid on the graph. The graph shows a spike in the "
         "number of wasted children after two consecutive years of lower than average EVI during both the long and shortrain seasons"
          "which—as of 2025—has not yet reverted to previous values.")
    ]
    conflict_paras = [
        ("This series counts reported violent conflict events. As an imperfect measure of conflict intensity"
         "the number of total fatalities are also shown."),
        ("The graph indicates a steady increase in the total number of conflicts over time.")
    ]

    # Build figure/text split blocks
    prec_fig, prec_txt = graph_blocks(
        cov["prec"], "Precipitation — CHIRPS (monthly z-score)",
        text_paragraphs=prec_paras,
        #data_notes="Baseline: 20-year climatology by calendar month (2000–2020); values are standard deviations (unitless)."
    )
    evi_fig, evi_txt = graph_blocks(
        cov["evi"], "EVI — Vegetation index",
        text_paragraphs=evi_paras,
        #data_notes="If smoothed, note the smoothing window when aligning to rainfall or observed prevalence."
    )
    conflict_fig, conflict_txt = graph_blocks(
        cov["conflict"], "Conflict — ACLED events",
        text_paragraphs=conflict_paras,
        #data_notes="Event counts reflect reports; reporting intensity may vary by place and time."
    )

    # Row factory: text to the RIGHT, vertically centered
    def row(fig_block, txt_block, tint):
        return html.Div(
            [
                html.Div(fig_block, style={"flex": "3 1 0", "minWidth": 0}),
                html.Div(
                    txt_block,
                    style={
                        "flex": "2.4 1 0",          # ← give text more width
                        "minWidth": "320px",
                        "paddingLeft": "18px",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center"  # vertical centering
                    }
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "alignItems": "stretch",
                "background": tint,
                "padding": "12px",
                "borderRadius": "12px",
                "marginBottom": "12px",
                "minHeight": "460px",
                "boxShadow": "0 1px 6px rgba(0,0,0,0.06)"
            }
        )

    # Precip first, then EVI, then Conflict
    return [
        row(prec_fig,     prec_txt,     "rgba(14,165,233,0.06)"),
        row(evi_fig,      evi_txt,      "rgba(14,165,233,0.00)"),
        row(conflict_fig, conflict_txt, "rgba(34,197,94,0.06)"),
    ]


VARIMP_TEXT = {
    "wasting_smoothed": [
        ("Bars show the most influential predictor groups in the latest model run, "
         "summarized for easier interpretation. Use this to see which external drivers "
         "— climate, vegetation, conflict, access, etc. — are carrying signal now.")
    ]
}

@app.callback(Output("varimp-grid", "children"), Input("varimp-scan", "n_intervals"))
def load_varimp_plot(_):
    base = os.path.join("assets", "figures")

    # newest image for prevalence
    f_prev = newest_varimp_filename("wasting_smoothed")


    base = os.path.join("assets", "figures")
    placeholder = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="

    def img_block(fname, title_text):
        if not fname:
            return html.Div(
                f"{title_text}: figure not found in ./assets/figures",
                style={"padding": "12px", "fontStyle": "italic", "border": "1px dashed #ddd",
                       "borderRadius": "10px", "background": "#fafafa"}
            )
        version = int(os.path.getmtime(os.path.join(base, fname)))
        real_src = f"/assets/figures/{fname}?v={version}"
        return html.Figure([
            html.Img(
                src=placeholder, **{"data-src": real_src}, className="lazy",
                style={
                    "width": "100%", "height": "auto", "maxHeight": "520px",
                    "objectFit": "contain", "border": "1px solid #eee", "borderRadius": "10px"
                }
            ),
            html.Figcaption(
                title_text,
                style={"textAlign": "center", "fontSize": "13px", "color": "#444", "marginTop": "8px"}
            )
        ], style={"margin": 0})

    def text_block(paragraphs):
        return html.Div(
            [html.P(p, style={"margin": "0 0 16px 0", "textIndent": "1.25em"}) for p in (paragraphs or [])],
            style={
                "fontSize": "17px",
                "lineHeight": "1.9",
                "color": "#111827",
                "textAlign": "justify",
                "textJustify": "inter-word",
                "hyphens": "auto",
                "width": "100%",
            }
        )

    def row(fig_block, txt_block, tint):
        return html.Div(
            [
                html.Div(fig_block, style={"flex": "3 1 0", "minWidth": 0}),
                html.Div(
                    txt_block,
                    style={
                        "flex": "2.4 1 0",
                        "minWidth": "320px",
                        "paddingLeft": "18px",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center"
                    }
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "alignItems": "stretch",
                "background": tint,
                "padding": "12px",
                "borderRadius": "12px",
                "marginBottom": "12px",
                "minHeight": "460px",
                "boxShadow": "0 1px 6px rgba(0,0,0,0.06)"
            }
        )

    # top card: wasting prevalence
    prev_fig = img_block(f_prev, "Top features — Wasting prevalence")
    prev_txt = text_block(VARIMP_TEXT.get("wasting_smoothed"))
    row_prev = row(prev_fig, prev_txt, "rgba(14,165,233,0.06)")


    return [row_prev]


# Run
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
