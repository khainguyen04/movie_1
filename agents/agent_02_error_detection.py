"""
agent_02_error_detection
========================
Data error detection (Lecture 1, slide 23) and definition of the modelling population.

Rules (keep as much data as possible):
  * revenue = 0          -> unknown target, not usable (target is never imputed)
  * status != Released   -> contradictory (revenue for unreleased film) -> drop
  * release date missing -> drop (needed for time-aware features)
  * budget & revenue both < 1000 -> recorded in millions -> multiply by 1e6 (FIX, keep)
  * revenue < 1000 otherwise     -> unreliable target -> drop
  * budget < 1000 otherwise      -> unit error -> set missing, impute in agent 03 (keep)
  * budget = 0 / runtime = 0     -> "0 means unknown" -> set missing (keep)
  * extreme ROI, runtime > 300   -> flag only (keep)
  * gender code 0                -> "unknown"
  * post-release columns (popularity, votes) -> removed (data leakage)

Output: state["movies"], data/interim/movies_checked.csv, tables + data-flow figure
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.io import log, log_data_flow, save_table, get_df, load_json
from utils.parsing import VALID_GENRES, split_pipe
from utils.plotting import set_style, save_fig, MAIN


def _iqr_outliers(s: pd.Series) -> int:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def _plot_data_flow(cfg: dict) -> None:
    path = os.path.join(cfg["paths"]["logs"], "data_flow_log.csv")
    flow = pd.read_csv(path).drop_duplicates("step", keep="last")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(flow) + 1.5))
    ax.barh(flow["step"], flow["rows_after"], color=MAIN)
    for i, (n, rem) in enumerate(zip(flow["rows_after"], flow["removed"])):
        ax.text(n, i, f"  {n:,}" + (f"  (-{rem:,})" if rem > 0 else ""), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of movies")
    ax.set_title("Data flow: from raw file to modelling dataset")
    save_fig(fig, "01_data_quality", "data_flow.png", cfg)


def run(state: dict, cfg: dict) -> dict:
    log("agent_02_error_detection: start")
    set_style()
    interim = cfg["paths"]["interim"]
    df_all = get_df(state, "movies_all", os.path.join(interim, "movies_parsed.csv"))
    checks = state.get("raw_checks")
    if checks is None:
        p = os.path.join(cfg["paths"]["logs"], "raw_checks.json")
        checks = load_json(p) if os.path.exists(p) else {}

    c = cfg["cleaning"]
    thr = c["unit_error_threshold"]
    report = []

    def rec(check, n, action, note=""):
        report.append({"check": check, "n_rows": n, "action": action, "note": note})

    # ---- A. raw-file level errors (already handled in agent 01)
    rec("Malformed rows (shifted columns, non-numeric id)", checks.get("metadata_malformed_rows"),
        "removed", "company names leaked into 'genres'")
    rec("Exact duplicate rows in metadata", checks.get("metadata_exact_duplicates"), "removed")
    rec("Duplicate ids in metadata", checks.get("metadata_duplicate_ids_removed"),
        "removed (kept first)", "differ only in popularity/vote_count")
    rec("Duplicate ids in credits", checks.get("credits_duplicate_ids_removed"),
        "removed (kept most complete crew)")
    rec("Duplicate ids in keywords", checks.get("keywords_duplicate_ids_removed"),
        "removed", "fully identical rows")

    genre_counts = Counter(g for s in df_all["genres"] for g in split_pipe(s))
    invalid = {g: n for g, n in genre_counts.items() if g not in VALID_GENRES}
    rec("Invalid genre labels after cleaning", sum(invalid.values()), "checked",
        ", ".join(list(invalid)[:5]) or "none")

    # ---- B. modelling population
    df = df_all.copy()
    for col in ["budget", "revenue", "runtime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    n = len(df)
    df = df[df["revenue"] > 0].copy()
    rec("Revenue = 0 (unknown box office)", n - len(df), "excluded",
        "target variable cannot be imputed")
    log_data_flow("02a keep movies with known revenue", n, len(df),
                  "revenue = 0 means unknown", cfg, "filter")

    n = len(df)
    bad = df["status"] != "Released"
    rec("Revenue reported but status not 'Released'", int(bad.sum()), "removed",
        str(df.loc[bad, "status"].value_counts().to_dict()))
    if c["drop_status_not_released"]:
        df = df[~bad].copy()
    log_data_flow("02b status conflict", n, len(df), "Rumored/Post Production with revenue", cfg)

    n = len(df)
    df = df[df["release_date"].notna()].copy()
    rec("Missing release date", n - len(df), "removed", "needed for time-aware features")
    log_data_flow("02c missing release date", n, len(df), "no date", cfg)

    # ---- C. unit errors (values recorded in millions)
    b, r = df["budget"], df["revenue"]
    both_small = (r < thr) & (b > 0) & (b < thr)
    ex_fix = df.loc[both_small, ["id", "title", "year", "budget", "revenue"]].copy()
    df.loc[both_small, ["budget", "revenue"]] = df.loc[both_small, ["budget", "revenue"]] * 1e6
    df["unit_fixed"] = both_small.astype(int)
    rec(f"Budget AND revenue < {thr} (recorded in millions)", int(both_small.sum()),
        "fixed x 1,000,000 (kept)")
    log_data_flow("02d fix unit errors (x1e6)", len(df), len(df),
                  "budget & revenue both recorded in millions", cfg, "fix")

    n = len(df)
    rev_bad = df["revenue"] < thr
    ex_drop = df.loc[rev_bad, ["id", "title", "year", "budget", "revenue"]].copy()
    df = df[~rev_bad].copy()
    rec(f"Revenue < {thr} not fixable", n - len(df), "removed", "unreliable target")
    log_data_flow("02e unreliable revenue < 1000", n, len(df), "cannot verify unit", cfg)

    bud_bad = (df["budget"] > 0) & (df["budget"] < thr)
    df["budget_unit_error"] = bud_bad.astype(int)
    df.loc[bud_bad, "budget"] = np.nan
    rec(f"Budget < {thr} with valid revenue", int(bud_bad.sum()),
        "set to missing -> imputed (kept)")

    # ---- D. zeros that mean 'unknown'
    zero_b = df["budget"] == 0
    df.loc[zero_b, "budget"] = np.nan
    rec("Budget = 0 (unknown)", int(zero_b.sum()), "set to missing -> imputed (kept)")
    zero_rt = df["runtime"] == 0
    df.loc[zero_rt, "runtime"] = np.nan
    rec("Runtime = 0 (unknown)", int(zero_rt.sum()), "set to missing -> imputed (kept)")

    # ---- E. flags only
    df["runtime_extreme"] = (df["runtime"] > c["runtime_max"]).astype(int)
    rec(f"Runtime > {c['runtime_max']} min", int(df["runtime_extreme"].sum()), "flagged (kept)")
    roi = df["revenue"] / df["budget"]
    df["roi_outlier"] = ((roi > 1000) | (roi < 0.001)).fillna(False).astype(int)
    rec("Extreme ROI (>1000x or <0.001x)", int(df["roi_outlier"].sum()), "flagged (kept)",
        "e.g. micro-budget hits; log scale reduces influence")

    dup_title = df.duplicated(subset=["title", "year"], keep=False)
    rec("Same title + same year, different id", int(dup_title.sum()), "checked (kept)",
        "different films sharing a title")

    # ---- F. gender code 0 = unknown
    gmap = {1: "female", 2: "male"}
    df["director_gender_cat"] = df["director_gender"].map(gmap).fillna("unknown")
    df["lead_gender_cat"] = df["lead_gender"].map(gmap).fillna("unknown")
    rec("Director gender code 0/missing", int((df["director_gender_cat"] == "unknown").sum()),
        "recoded 'unknown'")
    rec("Lead actor gender code 0/missing", int((df["lead_gender_cat"] == "unknown").sum()),
        "recoded 'unknown'")

    # ---- G. leakage columns (only known after release)
    leak = [col for col in df.columns if col.startswith("POST_")] + \
           [col for col in cfg["features"]["leakage_columns"] if col in df.columns]
    df = df.drop(columns=sorted(set(leak)))
    rec("Post-release columns (popularity, votes)", len(leak), "removed (leakage)",
        ", ".join(sorted(set(leak))))
    rec("belongs_to_collection is retrospective", int(df["is_franchise"].sum()),
        "kept for EDA only", "model uses is_sequel from agent 06")

    # ---- H. outlier counts (IQR rule) on raw vs log scale - informs transformation
    out_rows = []
    for col in ["revenue", "budget", "runtime", "cast_size", "crew_size"]:
        s = df[col]
        out_rows.append({"variable": col, "iqr_outliers_raw": _iqr_outliers(s),
                         "iqr_outliers_log": _iqr_outliers(np.log1p(s)),
                         "skew_raw": round(float(s.skew()), 2),
                         "skew_log": round(float(np.log1p(s).skew()), 2)})

    # ---- save
    save_table(pd.DataFrame(report), "02_data_error_summary.csv", cfg)
    save_table(pd.DataFrame(out_rows), "02_outliers_iqr.csv", cfg)
    save_table(ex_fix, "02_unit_fixed_examples.csv", cfg)
    save_table(ex_drop, "02_unreliable_revenue_removed.csv", cfg)
    df.to_csv(os.path.join(interim, "movies_checked.csv"), index=False)
    _plot_data_flow(cfg)

    log(f"agent_02_error_detection: done -> {len(df):,} movies "
        f"({df['is_franchise'].mean():.1%} in a collection)")
    state["movies"] = df
    return state