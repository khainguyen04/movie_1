"""
agent_07_dependence
===================
Measures of dependence (Lecture 1, slides 36-40):
  * Pearson correlation (linear) + p-values, heatmap
  * Mutual information (any dependence) for log_revenue and for blockbuster
  * VIF + highly correlated predictor pairs (multicollinearity check before regression)

Output: outputs/tables/07_*.csv, figures in 05_dependence and 07_slides
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.io import log, save_table, get_df, load_json
from utils.plotting import set_style, slide_style, save_fig, MAIN, ACCENT
from utils.features_shared import FEATURE_LABELS

HEATMAP_VARS = ["log_revenue", "log_budget", "runtime", "year", "log_cast_size", "log_crew_size",
                "n_keywords", "n_companies", "n_genres", "director_prior_films_log",
                "director_prior_mean_log_rev", "cast_prior_films_log", "cast_prior_mean_log_rev",
                "collection_prior_mean_log_rev", "is_sequel", "budget_missing"]


def run(state: dict, cfg: dict) -> dict:
    log("agent_07_dependence: start")
    set_style()
    seed = cfg["seed"]
    df = get_df(state, "features", os.path.join(cfg["paths"]["processed"], "features_full.csv"))
    fl = state.get("feature_lists") or load_json(os.path.join(cfg["paths"]["logs"],
                                                              "feature_lists.json"))
    num_bin = [c for c in fl["numeric"] + fl["binary"] if df[c].std() > 0]
    y = df["log_revenue"]

    # ---- Pearson correlation with the target
    rows = []
    for c in num_bin:
        r, p = pearsonr(df[c], y)
        rows.append({"feature": c, "label": FEATURE_LABELS.get(c, c), "pearson_r": r,
                     "abs_r": abs(r), "p_value": p})
    pear = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    save_table(pear.round(4), "07_pearson_with_log_revenue.csv", cfg)

    corr = df[HEATMAP_VARS].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Pearson correlation matrix (key features and log revenue)")
    save_fig(fig, "05_dependence", "pearson_heatmap.png", cfg)

    # ---- highly correlated predictor pairs
    pc = df[num_bin].corr()
    pairs = [{"feature_a": a, "feature_b": b, "pearson_r": pc.loc[a, b]}
             for i, a in enumerate(num_bin) for b in num_bin[i + 1:] if abs(pc.loc[a, b]) >= 0.7]
    save_table(pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson_r"]).round(3),
               "07_high_correlation_pairs.csv", cfg)

    # ---- Mutual information (numeric + binary + label-encoded categoricals)
    X = df[num_bin].astype(float).copy()
    discrete = [c in fl["binary"] for c in num_bin]
    for c in fl["categorical"]:
        X[c] = pd.factorize(df[c])[0]
        discrete.append(True)
    mi_reg = mutual_info_regression(X, y, discrete_features=discrete, random_state=seed)
    mi_clf = mutual_info_classif(X, df["blockbuster"], discrete_features=discrete,
                                 random_state=seed)
    mi = pd.DataFrame({"feature": X.columns,
                       "label": [FEATURE_LABELS.get(c, c) for c in X.columns],
                       "mi_log_revenue": mi_reg, "mi_blockbuster": mi_clf}
                      ).sort_values("mi_log_revenue", ascending=False)
    save_table(mi.round(4), "07_mutual_information.csv", cfg)

    top = mi.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["label"], top["mi_log_revenue"], color=MAIN)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual information with log revenue")
    ax.set_title("Top 20 features by mutual information")
    save_fig(fig, "05_dependence", "mutual_information_top20.png", cfg)

    comp = pear.merge(mi[["feature", "mi_log_revenue"]], on="feature")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(comp["abs_r"], comp["mi_log_revenue"], color=MAIN)
    for _, r_ in comp.nlargest(10, "mi_log_revenue").iterrows():
        ax.annotate(r_["label"], (r_["abs_r"], r_["mi_log_revenue"]), fontsize=8,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("|Pearson r| (linear dependence)")
    ax.set_ylabel("Mutual information (any dependence)")
    ax.set_title("Linear vs general dependence with log revenue")
    save_fig(fig, "05_dependence", "pearson_vs_mi.png", cfg)

    # ---- VIF (multicollinearity)
    Xv = sm.add_constant(df[num_bin].astype(float))
    vif = pd.DataFrame({"feature": num_bin,
                        "VIF": [variance_inflation_factor(Xv.values, i + 1)
                                for i in range(len(num_bin))]})
    vif["flag"] = np.select([vif["VIF"] >= 10, vif["VIF"] >= 5], ["high (>=10)", "moderate (>=5)"],
                            default="ok")
    save_table(vif.sort_values("VIF", ascending=False).round(2), "07_vif.csv", cfg)

    # ---- combined ranking + slide figure
    rank = mi.merge(pear[["feature", "pearson_r", "p_value"]], on="feature", how="left") \
             .merge(vif, on="feature", how="left")
    save_table(rank.round(4), "07_feature_ranking.csv", cfg)

    s = mi.head(8)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(s["label"], s["mi_log_revenue"], color=[ACCENT] * 3 + [MAIN] * 5)
    ax.invert_yaxis()
    ax.set_xticks([])
    slide_style(ax, "What drives box office? Strongest pre-release signals")
    save_fig(fig, "07_slides", "slide_top_drivers.png", cfg)

    log(f"agent_07_dependence: done -> top MI feature: {mi.iloc[0]['label']}; "
        f"{int((vif['VIF'] >= 10).sum())} features with VIF >= 10; "
        f"{len(pairs)} predictor pairs with |r| >= 0.7")
    return state