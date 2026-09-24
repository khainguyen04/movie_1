"""
agent_04_descriptive
====================
Descriptive statistics (Lecture 1, slides 25-27):
  continuous  -> count, mean, std, min, Q1, median, Q3, max, IQR, CV, skew, kurtosis
  categorical -> count, proportion, mode
Also revenue summaries by franchise, genre, month, decade, language and a check for
constant (non-informative) columns.

Output: outputs/tables/04_*.csv, outputs/tables/04_key_numbers.json
"""

import os

import numpy as np
import pandas as pd

from utils.io import log, save_table, save_json, get_df
from utils.parsing import split_pipe

CONT_VARS = ["revenue", "budget", "runtime", "year", "cast_size", "crew_size", "n_keywords",
             "n_companies", "n_countries", "n_genres", "n_spoken_languages", "n_writers",
             "n_producers", "cast_female_share"]
CAT_VARS = {"primary_genre": None, "original_language": 10, "month": None, "weekday": None,
            "decade": None, "segment": None, "director_gender_cat": None,
            "lead_gender_cat": None, "budget_missing": None, "runtime_imputed": None}


def describe_continuous(df: pd.DataFrame, cols: list, label_suffix: str = "") -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        rows.append({"variable": c + label_suffix, "n": len(s), "mean": s.mean(), "std": s.std(),
                     "min": s.min(), "q1": q1, "median": med, "q3": q3, "max": s.max(),
                     "iqr": q3 - q1, "cv": s.std() / s.mean() if s.mean() else np.nan,
                     "skew": s.skew(), "kurtosis": s.kurt()})
    return pd.DataFrame(rows)


def describe_categorical(s: pd.Series, name: str, top: int = None) -> pd.DataFrame:
    vc = s.astype(str).value_counts()
    if top and len(vc) > top:
        vc = pd.concat([vc.head(top), pd.Series({"Other": vc.iloc[top:].sum()})])
    t = vc.rename_axis("category").reset_index(name="count")
    t["proportion"] = (t["count"] / len(s)).round(4)
    t.insert(0, "variable", name)
    return t


def group_summary(df: pd.DataFrame, by: str, value: str = "revenue") -> pd.DataFrame:
    g = df.groupby(by)[value]
    out = pd.DataFrame({"n": g.size(), "share": g.size() / len(df), "mean": g.mean(),
                        "median": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75),
                        "std": g.std()})
    out["iqr"] = out["q3"] - out["q1"]
    return out.reset_index().sort_values("median", ascending=False)


def run(state: dict, cfg: dict) -> dict:
    log("agent_04_descriptive: start")
    df = get_df(state, "movies", os.path.join(cfg["paths"]["interim"], "movies_clean.csv")).copy()
    df["segment"] = np.where(df["is_franchise"] == 1, "Franchise", "Standalone")
    df["log_revenue"] = np.log(df["revenue"])
    df["log_budget"] = np.log(df["budget"])

    # ---- continuous
    cont = pd.concat([
        describe_continuous(df, CONT_VARS),
        describe_continuous(df[df["budget_missing"] == 0], ["budget"], " (observed only)"),
        describe_continuous(df, ["log_revenue", "log_budget"]),
    ], ignore_index=True)
    save_table(cont.round(3), "04_descriptive_continuous.csv", cfg)

    by_seg = []
    for seg, d in df.groupby("segment"):
        t = describe_continuous(d, ["revenue", "budget", "runtime", "cast_size"])
        t.insert(0, "segment", seg)
        by_seg.append(t)
    save_table(pd.concat(by_seg).round(3), "04_descriptive_by_franchise.csv", cfg)

    # ---- categorical
    cats, modes = [], []
    for col, top in CAT_VARS.items():
        t = describe_categorical(df[col], col, top)
        cats.append(t)
        vc = df[col].astype(str).value_counts()
        modes.append({"variable": col, "n_categories": df[col].nunique(), "mode": vc.index[0],
                      "mode_count": int(vc.iloc[0]), "mode_proportion": round(vc.iloc[0] / len(df), 4)})
    # genres are multi-label: proportion of movies having each genre (sums to > 1)
    glists = df["genres"].map(split_pipe)
    gcount = pd.Series([g for l in glists for g in l]).value_counts()
    g_tab = gcount.rename_axis("category").reset_index(name="count")
    g_tab["proportion"] = (g_tab["count"] / len(df)).round(4)
    g_tab.insert(0, "variable", "genres (multi-label)")
    cats.append(g_tab)
    modes.append({"variable": "genres (multi-label)", "n_categories": len(gcount),
                  "mode": gcount.index[0], "mode_count": int(gcount.iloc[0]),
                  "mode_proportion": round(gcount.iloc[0] / len(df), 4)})
    save_table(pd.concat(cats, ignore_index=True), "04_descriptive_categorical.csv", cfg)
    save_table(pd.DataFrame(modes), "04_categorical_modes.csv", cfg)

    # ---- constant / near-constant columns (no information for the model)
    const = [{"column": c, "n_unique": df[c].nunique(dropna=False),
              "top_value_share": round(df[c].astype(str).value_counts(normalize=True).iloc[0], 4)}
             for c in df.columns]
    const = pd.DataFrame(const)
    const = const[(const["n_unique"] <= 1) | (const["top_value_share"] >= 0.99)]
    save_table(const, "04_constant_columns.csv", cfg)

    # ---- revenue by group
    save_table(group_summary(df, "segment").round(0), "04_revenue_by_franchise.csv", cfg)
    save_table(group_summary(df, "month").round(0), "04_revenue_by_month.csv", cfg)
    save_table(group_summary(df, "decade").round(0), "04_revenue_by_decade.csv", cfg)
    lang = df["original_language"].where(
        df["original_language"].isin(df["original_language"].value_counts().head(10).index), "Other")
    save_table(group_summary(df.assign(language=lang), "language").round(0),
               "04_revenue_by_language.csv", cfg)
    ex = df.assign(genre=glists).explode("genre")
    save_table(group_summary(ex, "genre").round(0), "04_revenue_by_genre.csv", cfg)

    # ---- key numbers for slides
    seg_med = df.groupby("segment")["revenue"].median()
    month_med = df.groupby("month")["revenue"].median()
    thr = cfg["classification"]["blockbuster_threshold"]
    key = {
        "n_movies": len(df), "median_revenue": df["revenue"].median(),
        "mean_revenue": df["revenue"].mean(),
        "median_revenue_franchise": seg_med.get("Franchise"),
        "median_revenue_standalone": seg_med.get("Standalone"),
        "franchise_to_standalone_ratio": seg_med.get("Franchise") / seg_med.get("Standalone"),
        "share_franchise": df["is_franchise"].mean(),
        "share_blockbuster": (df["revenue"] >= thr).mean(),
        "best_month_by_median": int(month_med.idxmax()),
        "worst_month_by_median": int(month_med.idxmin()),
        "revenue_skew": df["revenue"].skew(), "log_revenue_skew": df["log_revenue"].skew(),
        "share_budget_imputed": df["budget_missing"].mean(),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "share_english": (df["original_language"] == "en").mean(),
    }
    save_json(key, os.path.join(cfg["paths"]["tables"], "04_key_numbers.json"))
    log(f"agent_04_descriptive: done -> median revenue ${key['median_revenue']/1e6:.1f}M, "
        f"franchise/standalone median ratio {key['franchise_to_standalone_ratio']:.1f}x")
    state["movies"] = df.drop(columns=["segment", "log_revenue", "log_budget"])
    return state