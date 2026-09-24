"""
agent_06_feature_engineering
============================
Build features that are KNOWN BEFORE RELEASE.

Key design decision: 'belongs_to_collection' is assigned retrospectively (Toy Story 1995 is
tagged as franchise although nobody knew in 1995). Using it directly = data leakage.
-> is_sequel = the film has an EARLIER film in the same collection.

Time-aware history (only films released strictly earlier):
  director / top-3 cast / franchise -> number of prior films and mean log revenue of prior
  films with known revenue. No history -> filled with the overall median + a 0/1 flag.
This is a leakage-safe form of the target encoding from Lecture 1 (slide 35).

Output: data/processed/features_full.csv, outputs/logs/feature_lists.json, segment table/figures
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.io import log, save_table, save_json, get_df
from utils.plotting import set_style, slide_style, save_fig, money_axis, money_fmt, MAIN, PALETTE
from utils.features_shared import (GENRE_COLS, genre_dummies, timing_features, basic_features,
                                   history_features)

NUMERIC = ["log_budget", "runtime", "year", "log_cast_size", "log_crew_size", "n_keywords",
           "n_companies", "n_countries", "n_genres", "n_spoken_languages", "cast_female_share",
           "director_prior_films_log", "director_prior_mean_log_rev", "cast_prior_films_log",
           "cast_prior_mean_log_rev", "cast_prior_max_log_rev", "collection_prior_films",
           "collection_prior_mean_log_rev"]
BINARY = ["is_sequel", "budget_missing", "runtime_imputed", "director_has_history",
          "cast_has_history", "collection_has_rev_history", "lang_en", "us_production",
          "is_friday"] + GENRE_COLS
CATEGORICAL = ["release_season", "original_language", "director_gender_cat", "lead_gender_cat",
               "lead_company"]
SEGMENTS = ["Standalone", "Franchise starter", "Sequel"]


def run(state: dict, cfg: dict) -> dict:
    log("agent_06_feature_engineering: start")
    set_style()
    interim = cfg["paths"]["interim"]
    df = get_df(state, "movies", os.path.join(interim, "movies_clean.csv")).copy()
    all_df = get_df(state, "movies_all", os.path.join(interim, "movies_parsed.csv"))
    thr = cfg["classification"]["blockbuster_threshold"]

    # ---- targets
    df["log_revenue"] = np.log(df["revenue"])
    df["blockbuster"] = (df["revenue"] >= thr).astype(int)
    df["roi_hit"] = np.where(df["budget_missing"] == 0,
                             (df["revenue"] >= 2.5 * df["budget"]).astype(float), np.nan)

    # ---- deterministic features
    for part in (basic_features(df), timing_features(df), genre_dummies(df["genres"])):
        df = pd.concat([df.drop(columns=part.columns, errors="ignore"), part], axis=1)

    # ---- time-aware history features
    log("   computing director / cast / franchise history (strictly earlier films) ...")
    hist = history_features(all_df, df)
    df = df.merge(hist, on="id", how="left")

    med = df["log_revenue"].median()
    for c in ["director_prior_films", "director_prior_rev_films", "cast_prior_films",
              "cast_prior_rev_films", "collection_prior_films", "collection_prior_rev_films"]:
        df[c] = df[c].fillna(0)
    df["director_has_history"] = (df["director_prior_rev_films"] > 0).astype(int)
    df["cast_has_history"] = (df["cast_prior_rev_films"] > 0).astype(int)
    df["collection_has_rev_history"] = (df["collection_prior_rev_films"] > 0).astype(int)
    for c in ["director_prior_mean_log_rev", "cast_prior_mean_log_rev", "cast_prior_max_log_rev",
              "collection_prior_mean_log_rev"]:
        df[c] = df[c].fillna(med)
    df["director_prior_films_log"] = np.log1p(df["director_prior_films"])
    df["cast_prior_films_log"] = np.log1p(df["cast_prior_films"])

    df["is_sequel"] = (df["collection_prior_films"] > 0).astype(int)
    df["franchise_segment"] = np.select(
        [df["is_sequel"] == 1, df["is_franchise"] == 1], ["Sequel", "Franchise starter"],
        default="Standalone")

    missing = df[NUMERIC + BINARY + CATEGORICAL].isna().sum()
    assert missing.sum() == 0, f"NaN in features: {missing[missing > 0].to_dict()}"

    # ---- segment summary (standalone / franchise starter / sequel)
    seg = df.groupby("franchise_segment").agg(
        n=("id", "size"), median_revenue=("revenue", "median"), mean_revenue=("revenue", "mean"),
        median_budget=("budget", "median"), blockbuster_rate=("blockbuster", "mean")
    ).reindex(SEGMENTS).reset_index()
    seg["share"] = seg["n"] / len(df)
    save_table(seg.round(3), "06_segment_summary.csv", cfg)

    cov = pd.DataFrame([{
        "feature": f, "share_of_films_with_history": round(float(df[f].mean()), 4)}
        for f in ["director_has_history", "cast_has_history", "collection_has_rev_history",
                  "is_sequel"]])
    save_table(cov, "06_history_coverage.csv", cfg)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df.assign(log10_revenue=np.log10(df["revenue"])), x="franchise_segment",
                y="log10_revenue", order=SEGMENTS, color=MAIN, fliersize=1, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("log10(revenue USD)")
    ax.set_title("Revenue by segment: starters vs sequels vs standalone")
    save_fig(fig, "04_bivariate", "box_revenue_by_segment.png", cfg)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(seg["franchise_segment"], seg["median_revenue"],
           color=[PALETTE["Standalone"], PALETTE["Franchise starter"], PALETTE["Sequel"]])
    for i, (v, n) in enumerate(zip(seg["median_revenue"], seg["n"])):
        ax.text(i, v, f"{money_fmt(v)}\n(n={n:,})", ha="center", va="bottom", fontsize=11)
    money_axis(ax, "y")
    slide_style(ax, "Sequels earn the most - the franchise effect is real")
    save_fig(fig, "07_slides", "slide_segment_median_revenue.png", cfg)

    # ---- save
    lists = {"target_regression": "log_revenue", "target_classification": "blockbuster",
             "numeric": NUMERIC, "binary": BINARY, "categorical": CATEGORICAL,
             "eda_only": ["is_franchise", "franchise_segment", "revenue", "budget", "roi_hit",
                          "decade", "is_summer", "is_holiday"]}
    save_json(lists, os.path.join(cfg["paths"]["logs"], "feature_lists.json"))
    df.to_csv(os.path.join(cfg["paths"]["processed"], "features_full.csv"), index=False)

    log(f"agent_06_feature_engineering: done -> {len(df):,} films, "
        f"{len(NUMERIC)} numeric + {len(BINARY)} binary + {len(CATEGORICAL)} categorical features; "
        f"sequels {df['is_sequel'].mean():.1%}, blockbusters {df['blockbuster'].mean():.1%}")
    state["features"] = df
    state["feature_lists"] = lists
    return state