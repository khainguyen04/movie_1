"""
agent_08_imbalance_split
========================
1) Quantify class imbalance: blockbuster (~19%), sequel (~11%), franchise (~20%).
2) Stratified 80/20 train/test split on is_sequel x blockbuster (Lecture 2, slides 47-52),
   so both rare groups keep the same share in train and test.
3) Check the split is representative (shares, means, KS test on log revenue).

Output: data/processed/train.csv, test.csv + imbalance tables/figures
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from utils.io import log, save_table, get_df
from utils.plotting import set_style, save_fig, MAIN, ACCENT

STRATEGY = [
    {"issue": "Blockbusters are only ~19% of films (about 1 : 4)",
     "mitigation": "Stratified train/test split and stratified K-fold keep the same share everywhere",
     "agent": "08, 12"},
    {"issue": "A model can look accurate by predicting 'not blockbuster' for everyone",
     "mitigation": "Logistic regression with class_weight='balanced'", "agent": "11"},
    {"issue": "Accuracy and specificity are misleading when negatives dominate",
     "mitigation": "Precision-recall curve and Average Precision in addition to ROC/AUC",
     "agent": "11"},
    {"issue": "Missing a blockbuster costs more than a false alarm",
     "mitigation": "Threshold from the loss matrix cost_FP/(cost_FP+cost_FN) and CV cost minimisation",
     "agent": "11"},
    {"issue": "Sequels are only ~11% of films",
     "mitigation": "Stratify on is_sequel, interaction log_budget x is_sequel, errors reported per segment",
     "agent": "08, 10, 12"},
]


def run(state: dict, cfg: dict) -> dict:
    log("agent_08_imbalance_split: start")
    set_style()
    p = cfg["paths"]["processed"]
    df = get_df(state, "features", os.path.join(p, "features_full.csv"))
    sp, seed = cfg["split"], cfg["seed"]

    # ---- imbalance
    rows = []
    for col, label in [("blockbuster", "Blockbuster (revenue >= threshold)"),
                       ("is_sequel", "Sequel"), ("is_franchise", "Belongs to a franchise")]:
        pos = int(df[col].sum())
        neg = len(df) - pos
        rows.append({"variable": label, "positives": pos, "negatives": neg,
                     "positive_share": pos / len(df),
                     "imbalance_ratio_neg_to_pos": neg / pos if pos else np.nan})
    save_table(pd.DataFrame(rows).round(4), "08_class_imbalance.csv", cfg)
    seg = df.groupby("franchise_segment").agg(n=("id", "size"),
                                              blockbuster_rate=("blockbuster", "mean")).reset_index()
    save_table(seg.round(4), "08_blockbuster_rate_by_segment.csv", cfg)
    save_table(pd.DataFrame(STRATEGY), "08_imbalance_strategy.csv", cfg)

    # ---- stratified split
    strat_cols = [c for c in sp["stratify_on"] if c in df.columns]
    key = df[strat_cols].astype(str).agg("_".join, axis=1)
    small = key.value_counts()[lambda s: s < 2].index
    key = key.where(~key.isin(small), "rare")
    train, test = train_test_split(df, test_size=sp["test_size"], random_state=seed, stratify=key)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    # ---- representativeness check
    bal = []
    for col in ["blockbuster", "is_sequel", "is_franchise", "budget_missing", "director_has_history"]:
        bal.append({"statistic": f"share {col}", "full": df[col].mean(),
                    "train": train[col].mean(), "test": test[col].mean()})
    for col in ["log_revenue", "log_budget", "year", "runtime"]:
        bal.append({"statistic": f"mean {col}", "full": df[col].mean(),
                    "train": train[col].mean(), "test": test[col].mean()})
    save_table(pd.DataFrame(bal).round(4), "08_split_balance.csv", cfg)
    ks = ks_2samp(train["log_revenue"], test["log_revenue"])
    save_table(pd.DataFrame([{"test": "KS two-sample, log_revenue train vs test",
                              "statistic": ks.statistic, "p_value": ks.pvalue,
                              "conclusion": "same distribution (p >= 0.05)" if ks.pvalue >= 0.05
                              else "distributions differ"}]).round(4),
               "08_split_ks_test.csv", cfg)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = ["blockbuster", "is_sequel", "is_franchise"]
    x = np.arange(len(cols))
    for i, (name, d) in enumerate([("Full", df), ("Train", train), ("Test", test)]):
        vals = [d[c].mean() for c in cols]
        ax.bar(x + (i - 1) * 0.27, vals, width=0.27, label=f"{name} (n={len(d):,})",
               color=[MAIN, ACCENT, "#8C8C8C"][i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Blockbuster", "Sequel", "Franchise"])
    ax.set_ylabel("Share of films")
    ax.set_title("Stratified split keeps rare classes at the same share")
    ax.legend()
    save_fig(fig, "06_models", "split_class_balance.png", cfg)

    train.to_csv(os.path.join(p, "train.csv"), index=False)
    test.to_csv(os.path.join(p, "test.csv"), index=False)
    log(f"agent_08_imbalance_split: done -> train {len(train):,} / test {len(test):,}; "
        f"blockbuster share train {train['blockbuster'].mean():.1%} vs test "
        f"{test['blockbuster'].mean():.1%}; KS p = {ks.pvalue:.3f}")
    state["train"], state["test"] = train, test
    return state