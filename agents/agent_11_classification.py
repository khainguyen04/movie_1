"""
agent_11_classification
=======================
Logistic regression: P(revenue >= $100M) (Lecture 2, slides 27-46), with imbalance handling:
  * statsmodels Logit (unweighted) -> odds ratios exp(beta) + p-values (interpretation)
  * sklearn LogisticRegression unweighted vs class_weight='balanced'
  * thresholds: 0.5 | loss-matrix formula cost_FP/(cost_FP+cost_FN) | CV cost-optimal
  * metrics: ROC/AUC, precision-recall/AP, sensitivity, specificity, precision
Final classifier = (model, threshold) with the lowest 5-fold CV expected cost on TRAIN.

Output: tables 11_*, figures, updates outputs/logs/model_selection.json
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from utils.io import log, save_table, save_json, load_json, load_model_data
from utils.metrics import classification_metrics, expected_cost
from utils.plotting import set_style, slide_style, save_fig, MAIN, ACCENT
from utils.features_shared import pretty_name

warnings.filterwarnings("ignore")
UNW, BAL = "Logistic (unweighted)", "Logistic (class_weight=balanced)"


def run(state: dict, cfg: dict) -> dict:
    log("agent_11_classification: start")
    set_style()
    logs = cfg["paths"]["logs"]
    Xtr, Xte, tr, te = load_model_data(state, cfg)
    sel = state.get("model_selection") or load_json(os.path.join(logs, "model_selection.json"))
    cc = cfg["classification"]
    c_fn, c_fp = cc["loss_false_negative"], cc["loss_false_positive"]
    seed = cfg["seed"]

    cols = sel["regression_columns"]["M3_OLS_interaction"]
    ytr, yte = tr["blockbuster"].values, te["blockbuster"].values

    # ---- statsmodels Logit for interpretation (odds ratios)
    Xc = sm.add_constant(Xtr[cols], has_constant="add")
    try:
        logit = sm.Logit(ytr, Xc).fit(disp=0, maxiter=500)
    except Exception:
        logit = sm.Logit(ytr, Xc).fit(method="bfgs", disp=0, maxiter=3000)
    ci = logit.conf_int()
    orr = pd.DataFrame({"term": logit.params.index, "coef": logit.params.values,
                        "odds_ratio": np.exp(logit.params.values),
                        "or_ci_low": np.exp(ci[0].values), "or_ci_high": np.exp(ci[1].values),
                        "p_value": logit.pvalues.values})
    orr["label"] = orr["term"].map(pretty_name)
    orr["interpretation"] = np.where(
        orr["odds_ratio"] >= 1, "odds x" + orr["odds_ratio"].round(2).astype(str),
        "odds x" + orr["odds_ratio"].round(2).astype(str) + " (lower)")
    save_table(orr.round(5), "11_odds_ratios.csv", cfg)
    with open(os.path.join(cfg["paths"]["tables"], "11_logit_summary.txt"), "w") as fh:
        fh.write(logit.summary().as_text())

    # ---- sklearn models, CV probabilities, thresholds
    models = {UNW: LogisticRegression(C=1e4, max_iter=5000),
              BAL: LogisticRegression(C=1e4, class_weight=cc.get("class_weight", "balanced"),
                                      max_iter=5000)}
    skf = StratifiedKFold(cfg["split"]["cv_folds"], shuffle=True, random_state=seed)
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    t_formula = c_fp / (c_fp + c_fn)
    oof, prob, cost_curve, t_opt = {}, {}, {}, {}
    for name, m in models.items():
        oof[name] = cross_val_predict(m, Xtr[cols], ytr, cv=skf, method="predict_proba")[:, 1]
        cost_curve[name] = np.array([expected_cost(ytr, oof[name], t, c_fn, c_fp) for t in grid])
        t_opt[name] = float(grid[cost_curve[name].argmin()])
        m.fit(Xtr[cols], ytr)
        prob[name] = m.predict_proba(Xte[cols])[:, 1]

    rows = []
    for name in models:
        for rule, t in [("default 0.5", 0.5), ("loss-matrix formula", t_formula),
                        ("CV cost-optimal", t_opt[name])]:
            rows.append({"model": name, "threshold_rule": rule,
                         "cv_expected_cost": expected_cost(ytr, oof[name], t, c_fn, c_fp),
                         "test_expected_cost": expected_cost(yte, prob[name], t, c_fn, c_fp),
                         **classification_metrics(yte, prob[name], t)})
    res = pd.DataFrame(rows)
    save_table(res.round(4), "11_classification_metrics.csv", cfg)
    final = res.loc[res["cv_expected_cost"].idxmin()]

    # ---- figures
    prev = yte.mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, col in [(UNW, MAIN), (BAL, ACCENT)]:
        fpr, tpr, _ = roc_curve(yte, prob[name])
        auc = res[(res["model"] == name)]["AUC"].iloc[0]
        axes[0].plot(fpr, tpr, color=col, label=f"{name} (AUC {auc:.3f})")
        pr, rc, _ = precision_recall_curve(yte, prob[name])
        ap = res[(res["model"] == name)]["AP"].iloc[0]
        axes[1].plot(rc, pr, color=col, label=f"{name} (AP {ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("1 - specificity (false positive rate)")
    axes[0].set_ylabel("Sensitivity (true positive rate)")
    axes[0].set_title("ROC curve (test)")
    axes[1].axhline(prev, color="grey", ls="--", label=f"No-skill baseline ({prev:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve (test) - suited to imbalanced data")
    for ax in axes:
        ax.legend(fontsize=8)
    save_fig(fig, "06_models", "roc_pr_curves.png", cfg)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, col in [(UNW, MAIN), (BAL, ACCENT)]:
        ax.plot(grid, cost_curve[name], color=col, label=name)
        ax.scatter([t_opt[name]], [cost_curve[name].min()], color=col, zorder=3)
    ax.axvline(t_formula, color="grey", ls="--", label=f"Loss-matrix formula ({t_formula:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Expected cost per film (FN={c_fn}, FP={c_fp})")
    ax.set_title("Choosing the threshold with the loss matrix (5-fold CV, train)")
    ax.legend(fontsize=8)
    save_fig(fig, "06_models", "threshold_cost_curve.png", cfg)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (name, t, title) in zip(axes, [(UNW, 0.5, "Unweighted, threshold 0.5"),
                                           (final["model"], final["threshold"],
                                            f"Final: {final['threshold_rule']} ({final['threshold']:.2f})")]):
        cm = confusion_matrix(yte, (prob[name] >= t).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred: no", "Pred: blockbuster"],
                    yticklabels=["Actual: no", "Actual: blockbuster"])
        ax.set_title(title, fontsize=10)
    save_fig(fig, "06_models", "confusion_matrices.png", cfg)

    o = orr[(orr["term"] != "const") & (orr["p_value"] < 0.05)]
    o = o.reindex(np.log(o["odds_ratio"]).abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(o["odds_ratio"], range(len(o)),
                xerr=[o["odds_ratio"] - o["or_ci_low"], o["or_ci_high"] - o["odds_ratio"]],
                fmt="o", color=MAIN, ecolor="#999999", capsize=3)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels(o["label"], fontsize=8)
    ax.set_xlabel("Odds ratio (log scale, 95% CI)")
    ax.set_title("Blockbuster odds ratios (significant terms)")
    save_fig(fig, "06_models", "odds_ratios.png", cfg)

    bars = [("Unweighted\nthreshold 0.5", res[(res.model == UNW) & (res.threshold_rule == "default 0.5")]),
            ("Balanced weights\nthreshold 0.5", res[(res.model == BAL) & (res.threshold_rule == "default 0.5")]),
            ("Final model\n(cost-based)", res.loc[[final.name]])]
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [b[1]["sensitivity_recall"].iloc[0] for b in bars]
    precs = [b[1]["precision"].iloc[0] for b in bars]
    ax.bar([b[0] for b in bars], vals, color=[MAIN, MAIN, ACCENT])
    for i, (v, pr_) in enumerate(zip(vals, precs)):
        ax.text(i, v, f"{v:.0%} caught\n(precision {pr_:.0%})", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1)
    slide_style(ax, "Handling imbalance: share of real blockbusters the model catches")
    save_fig(fig, "07_slides", "slide_blockbuster_recall.png", cfg)

    # ---- save selection
    sel["classification"] = {"columns": cols, "model": final["model"],
                             "class_weight": "balanced" if final["model"] == BAL else None,
                             "threshold": float(final["threshold"]),
                             "threshold_rule": final["threshold_rule"],
                             "loss_matrix": {"FN": c_fn, "FP": c_fp}}
    save_json(sel, os.path.join(logs, "model_selection.json"))
    log(f"agent_11_classification: done -> AUC {final['AUC']:.3f}, AP {final['AP']:.3f} "
        f"(baseline {prev:.2f}); final = {final['model']} @ {final['threshold']:.2f} "
        f"({final['threshold_rule']}), recall {final['sensitivity_recall']:.0%}, "
        f"precision {final['precision']:.0%}")
    state["model_selection"] = sel
    return state