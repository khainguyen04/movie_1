"""
agent_10_regression
===================
Regression models for log(revenue) (Lecture 2) and revenue (GLM, Lecture 3):
  M1 OLS budget only (baseline)
  M2 OLS all pre-release features
  M3 M2 + interaction log_budget x is_sequel  (does budget pay off differently for sequels?)
  M4 backward elimination on M3 by p-value (> 0.05 removed; main effects of kept interactions
     and log_budget are protected)                                         (Lecture 3, slide 42)
  M5 GLM Gamma with log link on revenue, same features as M4               (Lecture 3)
Comparison: AIC/BIC (slide 46) - OLS AIC converted to the revenue scale (lognormal Jacobian)
so OLS and GLM are comparable - plus test-set RMSE/MAE (Lecture 2, slides 19-21).

Output: tables 10_*, figures, outputs/logs/model_selection.json
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.io import log, save_table, save_json, load_json, load_model_data
from utils.metrics import (regression_metrics, fit_ols, predict_ols, fit_glm_gamma,
                           predict_glm_log, drop_rare_columns)
from utils.plotting import set_style, slide_style, save_fig, PALETTE, MAIN, ACCENT
from utils.features_shared import pretty_name

M1, M2, M3, M4, M5 = ("M1_OLS_budget_only", "M2_OLS_full", "M3_OLS_interaction",
                      "M4_OLS_selected", "M5_GLM_Gamma_log")


def backward_eliminate(y, X, alpha: float, protected: list):
    cols, steps = list(X.columns), []
    while True:
        res = fit_ols(y, X[cols])
        pv = res.pvalues.drop("const")
        kept_inter = [c for c in cols if ":" in c]
        prot = set(protected) | {part for c in kept_inter for part in c.split(":")}
        cand = pv[[c for c in pv.index if c not in prot]]
        if cand.empty or cand.max() <= alpha:
            return cols, pd.DataFrame(steps), res
        worst = cand.idxmax()
        steps.append({"step": len(steps) + 1, "removed": worst, "p_value": cand.max(),
                      "AIC_before": res.aic, "n_features_after": len(cols) - 1})
        cols.remove(worst)


def run(state: dict, cfg: dict) -> dict:
    log("agent_10_regression: start")
    set_style()
    p, logs = cfg["paths"]["processed"], cfg["paths"]["logs"]
    Xtr, Xte, tr, te = load_model_data(state, cfg)
    fl = state.get("feature_lists") or load_json(os.path.join(logs, "feature_lists.json"))
    prep = state.get("preprocessor") or joblib.load(os.path.join(p, "preprocessor.joblib"))
    mc = cfg.get("models", {})
    alpha, min_count = mc.get("pvalue_threshold", 0.05), mc.get("min_binary_count", 10)

    ytr, yte = tr["log_revenue"].values, te["log_revenue"].values
    ytr_m = tr["revenue"].values / 1e6
    sum_log_y = float(np.sum(np.log(ytr_m)))   # Jacobian: AIC(revenue) = AIC(log) + 2*sum(log y)

    usable = drop_rare_columns(Xtr, min_count)
    base = [c for c in usable if ":" not in c]
    inter = [c for c in usable if ":" in c]
    sets = {M1: ["log_budget"], M2: base, M3: base + inter}
    fits = {name: fit_ols(ytr, Xtr[cols]) for name, cols in sets.items()}

    log(f"   backward elimination from {len(sets[M3])} features ...")
    sel_cols, steps, res_sel = backward_eliminate(ytr, Xtr[sets[M3]], alpha, ["log_budget"])
    sets[M4], fits[M4] = sel_cols, res_sel
    glm = fit_glm_gamma(ytr_m, Xtr[sel_cols])
    sets[M5] = sel_cols

    # ---- comparison table
    rows, preds, smear = [], {}, {}
    for name in [M1, M2, M3, M4]:
        res = fits[name]
        smear[name] = float(np.mean(np.exp(res.resid)))
        preds[name] = predict_ols(res, Xte[sets[name]])
        m = regression_metrics(yte, preds[name], smear[name])
        rows.append({"model": name, "n_params": int(res.df_model + 1),
                     "AIC_log_scale": res.aic, "BIC_log_scale": res.bic,
                     "AIC_revenue_scale": res.aic + 2 * sum_log_y,
                     "BIC_revenue_scale": res.bic + 2 * sum_log_y,
                     "train_R2_adj": res.rsquared_adj, "smearing": smear[name],
                     **{f"test_{k}": v for k, v in m.items()}})
    preds[M5] = predict_glm_log(glm, Xte[sel_cols])
    m = regression_metrics(yte, preds[M5], 1.0)
    rows.append({"model": M5, "n_params": int(glm.df_model + 1),
                 "AIC_log_scale": np.nan, "BIC_log_scale": np.nan,
                 "AIC_revenue_scale": glm.aic, "BIC_revenue_scale": getattr(glm, "bic_llf", glm.bic),
                 "train_R2_adj": 1 - glm.deviance / glm.null_deviance, "smearing": 1.0,
                 **{f"test_{k}": v for k, v in m.items()}})
    comp = pd.DataFrame(rows)
    save_table(comp.round(4), "10_model_comparison.csv", cfg)
    save_table(steps.round(4) if len(steps) else pd.DataFrame(columns=["step", "removed"]),
               "10_backward_elimination_steps.csv", cfg)

    cand = comp[comp["model"].isin([M4, M5])].set_index("model")
    best = cand["AIC_revenue_scale"].idxmin()

    # ---- coefficient table of the selected OLS (+ robust HC3 errors)
    res_hc3 = fit_ols(ytr, Xtr[sel_cols], cov_type="HC3")
    ci = res_sel.conf_int()
    num = set(fl["numeric"])

    def unit(t):
        if t == "const":
            return ""
        if ":" in t:
            return "extra effect per +1 SD for sequels"
        if "=" in t:
            return "vs reference category"
        return "per +1 SD" if t in num else "0 -> 1"

    coef = pd.DataFrame({
        "term": res_sel.params.index, "coef": res_sel.params.values, "se": res_sel.bse.values,
        "t": res_sel.tvalues.values, "p_value": res_sel.pvalues.values,
        "ci_low": ci[0].values, "ci_high": ci[1].values,
        "se_HC3": res_hc3.bse.values, "p_value_HC3": res_hc3.pvalues.values})
    coef["effect_pct"] = (np.exp(coef["coef"]) - 1) * 100
    coef["unit"] = coef["term"].map(unit)
    coef["label"] = coef["term"].map(pretty_name)
    save_table(coef.round(5), "10_coefficients_selected_ols.csv", cfg)

    glm_coef = pd.DataFrame({"term": glm.params.index, "coef": glm.params.values,
                             "p_value": glm.pvalues.values,
                             "effect_pct": (np.exp(glm.params.values) - 1) * 100})
    glm_coef["label"] = glm_coef["term"].map(pretty_name)
    save_table(glm_coef.round(5), "10_coefficients_glm_gamma.csv", cfg)

    # budget elasticity: coefficients are per SD of log budget -> divide by SD
    sd_lb = float(prep.stds_["log_budget"])
    b = res_sel.params
    elasticity = {"standalone_or_starter": float(b["log_budget"] / sd_lb)}
    if "log_budget:is_sequel" in b.index:
        elasticity["sequel"] = float((b["log_budget"] + b["log_budget:is_sequel"]) / sd_lb)

    # ---- sensitivity: refit on films with an OBSERVED budget only
    mtr, mte = (tr["budget_missing"] == 0).values, (te["budget_missing"] == 0).values
    cols_s = [c for c in sel_cols if c != "budget_missing" and Xtr.loc[mtr, c].std() > 0]
    res_s = fit_ols(ytr[mtr], Xtr.loc[mtr, cols_s])
    sens = pd.DataFrame([
        {"model": "Selected OLS, all films (budget imputed where missing)",
         "n_train": int(len(ytr)), "coef_log_budget": b["log_budget"],
         "coef_is_sequel": b.get("is_sequel", np.nan), "train_R2_adj": res_sel.rsquared_adj,
         "test_RMSE_log_observed_budget_films":
             regression_metrics(yte[mte], preds[M4][mte])["RMSE_log"]},
        {"model": "Same features, observed-budget films only",
         "n_train": int(mtr.sum()), "coef_log_budget": res_s.params["log_budget"],
         "coef_is_sequel": res_s.params.get("is_sequel", np.nan),
         "train_R2_adj": res_s.rsquared_adj,
         "test_RMSE_log_observed_budget_films":
             regression_metrics(yte[mte], predict_ols(res_s, Xte.loc[mte, cols_s]))["RMSE_log"]},
    ])
    save_table(sens.round(4), "10_sensitivity_observed_budget.csv", cfg)

    with open(os.path.join(cfg["paths"]["tables"], "10_ols_selected_summary.txt"), "w") as fh:
        fh.write(res_sel.summary().as_text())
    with open(os.path.join(cfg["paths"]["tables"], "10_glm_gamma_summary.txt"), "w") as fh:
        fh.write(glm.summary().as_text())

    # ---- figures
    c = coef[coef["term"] != "const"].assign(abs_t=lambda d: d["t"].abs()).nlargest(20, "abs_t")
    c = c.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.errorbar(c["coef"], range(len(c)), xerr=[c["coef"] - c["ci_low"], c["ci_high"] - c["coef"]],
                fmt="o", color=MAIN, ecolor="#999999", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c["label"], fontsize=8)
    ax.set_xlabel("Coefficient on log revenue (95% CI)")
    ax.set_title("Selected OLS: 20 strongest effects")
    save_fig(fig, "06_models", "coefficients_selected_ols.png", cfg)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for seg, col in [("Standalone", PALETTE["Standalone"]),
                     ("Franchise starter", PALETTE["Franchise starter"]),
                     ("Sequel", PALETTE["Sequel"])]:
        mk = (te["franchise_segment"] == seg).values
        ax.scatter(preds[best][mk] / np.log(10), yte[mk] / np.log(10), s=8, alpha=0.4,
                   color=col, label=seg)
    lims = [min(yte.min(), preds[best].min()) / np.log(10), max(yte.max(), preds[best].max()) / np.log(10)]
    ax.plot(lims, lims, color="black", lw=1)
    ax.set_xlabel("Predicted log10(revenue)")
    ax.set_ylabel("Actual log10(revenue)")
    ax.set_title(f"Test set: predicted vs actual ({best})")
    ax.legend()
    save_fig(fig, "06_models", "predicted_vs_actual_test.png", cfg)

    eff = coef[(coef["p_value"] < 0.05) & coef["unit"].isin(["0 -> 1", "vs reference category"])]
    eff = eff.reindex(eff["effect_pct"].abs().sort_values(ascending=False).index).head(8)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(eff["label"], eff["effect_pct"],
            color=[ACCENT if v > 0 else MAIN for v in eff["effect_pct"]])
    for i, v in enumerate(eff["effect_pct"]):
        ax.text(v, i, f" {v:+.0f}% ", va="center", ha="left" if v > 0 else "right", fontsize=11)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    slide_style(ax, "How each factor shifts expected revenue (all else equal)")
    save_fig(fig, "07_slides", "slide_effects_percent.png", cfg)

    # ---- save selection
    sel_info = {"regression_columns": sets, "best_regression": best,
                "selected_columns": sel_cols, "smearing_selected_ols": smear[M4],
                "budget_elasticity": elasticity, "pvalue_threshold": alpha,
                "usable_columns": usable}
    save_json(sel_info, os.path.join(logs, "model_selection.json"))

    r = comp.set_index("model")
    log(f"agent_10_regression: done -> selected {len(sel_cols)} features; best by AIC = {best}; "
        f"test R2(log): M1 {r.loc[M1, 'test_R2_log']:.3f}, M4 {r.loc[M4, 'test_R2_log']:.3f}, "
        f"M5 {r.loc[M5, 'test_R2_log']:.3f}; budget elasticity {elasticity}")
    state["model_selection"] = sel_info
    return state