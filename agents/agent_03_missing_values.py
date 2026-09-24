"""
agent_03_missing_values
=======================
Missing value analysis + imputation (Lecture 1, slide 24):
  1) replace with a meaningful value  -> 'Unknown' genre/company/director/cast, 0 keywords
  2) replace with mean / median / mode -> runtime (median by genre), language (mode)
  3) model-based                       -> budget: linear regression on log(budget) from
                                          other pre-release variables (never uses revenue)
Every imputed variable gets a 0/1 flag so the models can learn from missingness.

Output: state["movies"], data/interim/movies_clean.csv, tables + figures
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

from utils.io import log, save_table, get_df
from utils.parsing import VALID_GENRES, split_pipe, is_missing_text, primary_genre
from utils.plotting import set_style, save_fig, PALETTE, money_axis

TEXT_COLS = {  # column -> meaningful replacement
    "genres": "Unknown", "production_companies": "Unknown", "production_countries": "Unknown",
    "spoken_languages": "Unknown", "directors": "Unknown", "top5_cast": "Unknown",
    "writers": "Unknown", "keywords": "", "composer": "Unknown", "collection_name": "",
}


def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["budget", "runtime", "original_language", "genres", "production_companies",
                "production_countries", "directors", "top5_cast", "keywords",
                "cast_female_share"]:
        m = is_missing_text(df[col]) if df[col].dtype == object or \
            pd.api.types.is_string_dtype(df[col]) else df[col].isna()
        rows.append({"variable": col, "n_missing": int(m.sum()),
                     "pct_missing": round(float(m.mean()) * 100, 2)})
    for col in ["director_gender_cat", "lead_gender_cat"]:
        m = df[col] == "unknown"
        rows.append({"variable": col + " (unknown)", "n_missing": int(m.sum()),
                     "pct_missing": round(float(m.mean()) * 100, 2)})
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def _budget_design(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-release predictors for budget imputation (no revenue -> no target leakage)."""
    glists = df["genres"].map(split_pipe)
    X = pd.DataFrame(index=df.index)
    X["year"] = df["year"]
    X["year_sq"] = (df["year"] - 2000) ** 2
    X["runtime"] = df["runtime"]
    X["log_cast_size"] = np.log1p(df["cast_size"].fillna(0))
    X["log_crew_size"] = np.log1p(df["crew_size"].fillna(0))
    X["n_companies"] = df["n_companies"].fillna(0)
    X["n_countries"] = df["n_countries"].fillna(0)
    X["n_keywords"] = df["n_keywords"].fillna(0)
    X["lang_en"] = (df["original_language"] == "en").astype(int)
    X["us_production"] = df["production_countries"].fillna("").str.contains(r"\bUS\b").astype(int)
    for g in VALID_GENRES:
        X["g_" + g] = glists.map(lambda l, g=g: int(g in l))
    return X


def _median_group_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline: median log(budget) by primary genre x decade (fallback genre, then overall)."""
    tr = train.assign(lb=np.log(train["budget"]))
    by_gd = tr.groupby(["primary_genre", "decade"])["lb"].median()
    by_g = tr.groupby("primary_genre")["lb"].median()
    overall = tr["lb"].median()
    keys = list(zip(test["primary_genre"], test["decade"]))
    pred = [by_gd.get(k, by_g.get(k[0], overall)) for k in keys]
    return np.asarray(pred, dtype=float)


def _impute_budget(df: pd.DataFrame, cfg: dict) -> tuple:
    seed = cfg["seed"]
    obs = df["budget"].notna()
    X = _budget_design(df)
    y = np.log(df.loc[obs, "budget"])
    kf = KFold(5, shuffle=True, random_state=seed)

    reg = LinearRegression()
    r2_model = cross_val_score(reg, X[obs], y, cv=kf, scoring="r2")

    r2_med = []
    d_obs = df[obs]
    for tr_idx, te_idx in kf.split(d_obs):
        tr, te = d_obs.iloc[tr_idx], d_obs.iloc[te_idx]
        r2_med.append(r2_score(np.log(te["budget"]), _median_group_predict(tr, te)))

    method = cfg["cleaning"]["impute_budget_method"]
    if method == "model":
        reg.fit(X[obs], y)
        pred_log = reg.predict(X[~obs])
    else:
        pred_log = _median_group_predict(df[obs], df[~obs])
    lo, hi = y.min(), y.max()
    pred = np.exp(np.clip(pred_log, lo, hi))

    comparison = pd.DataFrame([
        {"method": "linear regression on log(budget)", "cv5_r2_mean": r2_model.mean(),
         "cv5_r2_std": r2_model.std(), "used": method == "model"},
        {"method": "median by primary genre x decade", "cv5_r2_mean": np.mean(r2_med),
         "cv5_r2_std": np.std(r2_med), "used": method != "model"},
    ])
    return pred, comparison


def _plot_missing(before: pd.DataFrame, after: pd.DataFrame, cfg: dict) -> None:
    m = before.merge(after, on="variable", suffixes=("_before", "_after"))
    m = m[m["pct_missing_before"] > 0]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(m) + 1.5))
    ax.barh(m["variable"], m["pct_missing_before"], color=PALETTE["Imputed"], label="Before")
    ax.barh(m["variable"], m["pct_missing_after"], color=PALETTE["Observed"], label="After")
    for i, v in enumerate(m["pct_missing_before"]):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of movies missing")
    ax.set_title("Missing values before vs after imputation")
    ax.legend()
    save_fig(fig, "01_data_quality", "missing_before_after.png", cfg)


def _plot_budget_imputation(df: pd.DataFrame, cfg: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for flag, label in [(0, "Observed"), (1, "Imputed")]:
        sns.kdeplot(np.log10(df.loc[df["budget_missing"] == flag, "budget"]), ax=ax,
                    fill=True, alpha=0.35, color=PALETTE[label],
                    label=f"{label} (n={int((df['budget_missing'] == flag).sum()):,})")
    ax.set_xlabel("Budget (log10 USD)")
    ax.set_title("Budget distribution: observed vs model-imputed")
    ax.legend()
    save_fig(fig, "01_data_quality", "budget_imputation_check.png", cfg)


def run(state: dict, cfg: dict) -> dict:
    log("agent_03_missing_values: start")
    set_style()
    interim = cfg["paths"]["interim"]
    df = get_df(state, "movies", os.path.join(interim, "movies_checked.csv")).copy()
    n_start = len(df)

    before = _missing_report(df)
    plan = []

    # 1) meaningful values for text columns
    for col, val in TEXT_COLS.items():
        if col in df.columns:
            m = is_missing_text(df[col])
            if m.any():
                df.loc[m, col] = val
                plan.append({"variable": col, "n_imputed": int(m.sum()),
                             "method": f"meaningful value '{val}'" if val else "empty = none"})
    df["n_keywords"] = df["n_keywords"].fillna(0)
    df["primary_genre"] = df["genres"].map(primary_genre)
    df["decade"] = (df["year"] // 10 * 10).astype(int)

    # 2) mode for language
    m = df["original_language"].isna()
    if m.any():
        mode = df["original_language"].mode()[0]
        df.loc[m, "original_language"] = mode
        plan.append({"variable": "original_language", "n_imputed": int(m.sum()),
                     "method": f"mode ('{mode}')"})

    # 2) median by primary genre for runtime
    df["runtime_imputed"] = df["runtime"].isna().astype(int)
    med = df.groupby("primary_genre")["runtime"].median()
    df["runtime"] = df["runtime"].fillna(df["primary_genre"].map(med)).fillna(df["runtime"].median())
    plan.append({"variable": "runtime", "n_imputed": int(df["runtime_imputed"].sum()),
                 "method": "median runtime of the same primary genre"})

    # 2) median for cast_female_share (flag keeps the information)
    df["cast_gender_unknown"] = df["cast_female_share"].isna().astype(int)
    df["cast_female_share"] = df["cast_female_share"].fillna(df["cast_female_share"].median())
    plan.append({"variable": "cast_female_share", "n_imputed": int(df["cast_gender_unknown"].sum()),
                 "method": "median + flag cast_gender_unknown"})

    # 3) model-based for budget
    df["budget_missing"] = df["budget"].isna().astype(int)
    pred, comparison = _impute_budget(df, cfg)
    df.loc[df["budget_missing"] == 1, "budget"] = pred
    plan.append({"variable": "budget", "n_imputed": int(df["budget_missing"].sum()),
                 "method": f"{cfg['cleaning']['impute_budget_method']}-based on log(budget) "
                           "+ flag budget_missing"})

    # sanity checks
    key_cols = ["budget", "revenue", "runtime", "year", "month", "original_language", "genres"]
    assert df[key_cols].isna().sum().sum() == 0, "missing values remain in key columns"
    assert len(df) == n_start, "imputation must not drop rows"

    after = _missing_report(df)
    save_table(before, "03_missing_before.csv", cfg)
    save_table(after, "03_missing_after.csv", cfg)
    save_table(pd.DataFrame(plan), "03_imputation_plan.csv", cfg)
    save_table(comparison, "03_budget_imputation_cv.csv", cfg)
    _plot_missing(before, after, cfg)
    _plot_budget_imputation(df, cfg)

    df.to_csv(os.path.join(interim, "movies_clean.csv"), index=False)
    log(f"agent_03_missing_values: done -> {len(df):,} movies, "
        f"budget imputed for {df['budget_missing'].sum():,} "
        f"({df['budget_missing'].mean():.1%}); CV R2 budget model = "
        f"{comparison.loc[0, 'cv5_r2_mean']:.3f} vs median {comparison.loc[1, 'cv5_r2_mean']:.3f}")
    state["movies"] = df
    return state