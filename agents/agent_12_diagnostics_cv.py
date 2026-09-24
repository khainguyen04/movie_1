"""
agent_12_diagnostics_cv
=======================
Model diagnostics (Lecture 2, slides 11-14, 19-26; Lecture 3, slides 39-53):
  OLS assumptions: independence (Durbin-Watson, ordered by release date), normality of
  residuals (Q-Q, KS test, Jarque-Bera), linearity (Rainbow test, observed vs predicted),
  constant variance (Breusch-Pagan, residual plot), multicollinearity (VIF), influence (Cook's D)
  Mitigation for heteroscedasticity: log transform (already applied) + robust HC3 errors.
  GLM Gamma: deviance residual plots.
  5-fold stratified CV for M1-M5 and both logistic models; LOOCV for OLS (closed form).
  Accuracy by segment (standalone / franchise starter / sequel).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from utils.io import log, save_table, load_json, load_model_data
from utils.metrics import (regression_metrics, fit_ols, predict_ols, fit_glm_gamma,
                           predict_glm_log)
from utils.plotting import set_style, slide_style, save_fig, MAIN, ACCENT, PALETTE

warnings.filterwarnings("ignore")
SEGMENTS = ["Standalone", "Franchise starter", "Sequel"]


def _loocv(y, X):
    """Exact LOOCV for OLS without refitting: e_i / (1 - h_ii)."""
    Xc = sm.add_constant(X, has_constant="add").values.astype(float)
    inv = np.linalg.pinv(Xc.T @ Xc)
    h = np.einsum("ij,jk,ik->i", Xc, inv, Xc)
    e = y - Xc @ (inv @ Xc.T @ y)
    loo = e / (1 - h)
    return float(np.sqrt(np.mean(loo ** 2))), float(np.mean(np.abs(loo)))


def run(state: dict, cfg: dict) -> dict:
    log("agent_12_diagnostics_cv: start")
    set_style()
    logs, seed = cfg["paths"]["logs"], cfg["seed"]
    Xtr, Xte, tr, te = load_model_data(state, cfg)
    sel = state.get("model_selection") or load_json(os.path.join(logs, "model_selection.json"))
    sets, cols, best = sel["regression_columns"], sel["selected_columns"], sel["best_regression"]
    ytr, yte = tr["log_revenue"].values, te["log_revenue"].values
    ytr_m = tr["revenue"].values / 1e6

    # ---- OLS assumption tests (selected model)
    res = fit_ols(ytr, Xtr[cols])
    resid, fitted = np.asarray(res.resid), np.asarray(res.fittedvalues)
    order = np.argsort(pd.to_datetime(tr["release_date"]).values)
    dw = durbin_watson(resid[order])
    z = resid / resid.std(ddof=1)
    ks = stats.kstest(z, "norm")
    jb, jb_p, skew, kurt = jarque_bera(resid)
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, res.model.exog)
    rb_f, rb_p = linear_rainbow(res)
    exog = res.model.exog
    vif_max = max(variance_inflation_factor(exog, i) for i in range(1, exog.shape[1]))
    res_hc3 = fit_ols(ytr, Xtr[cols], cov_type="HC3")
    n_sig = int((res.pvalues.drop("const") < 0.05).sum())
    n_sig_hc3 = int((res_hc3.pvalues.drop("const") < 0.05).sum())

    tests = pd.DataFrame([
        {"assumption": "1. Independence", "test": "Durbin-Watson (ordered by release date)",
         "statistic": dw, "p_value": np.nan,
         "conclusion": "OK (1.5-2.5)" if 1.5 <= dw <= 2.5 else "possible autocorrelation",
         "mitigation": "films are separate products; time trend captured by year"},
        {"assumption": "2. Normality of residuals", "test": "Kolmogorov-Smirnov vs N(0,1)",
         "statistic": ks.statistic, "p_value": ks.pvalue,
         "conclusion": "normal" if ks.pvalue >= 0.05 else "deviation from normal (check Q-Q)",
         "mitigation": "log transform; large n -> CLT makes coefficient inference robust"},
        {"assumption": "2. Normality of residuals", "test": f"Jarque-Bera (skew {skew:.2f}, kurtosis {kurt:.2f})",
         "statistic": jb, "p_value": jb_p,
         "conclusion": "normal" if jb_p >= 0.05 else "heavy tails / skew",
         "mitigation": "compare with GLM Gamma (M5)"},
        {"assumption": "3. Linearity", "test": "Rainbow test", "statistic": rb_f, "p_value": rb_p,
         "conclusion": "linear" if rb_p >= 0.05 else "some non-linearity",
         "mitigation": "log budget, interaction term; see observed vs predicted plot"},
        {"assumption": "4. Constant variance", "test": "Breusch-Pagan", "statistic": bp_lm,
         "p_value": bp_p,
         "conclusion": "homoscedastic" if bp_p >= 0.05 else "heteroscedastic",
         "mitigation": f"robust HC3 errors: {n_sig} significant terms (OLS) vs {n_sig_hc3} (HC3)"},
        {"assumption": "Multicollinearity", "test": "max VIF in selected model",
         "statistic": vif_max, "p_value": np.nan,
         "conclusion": "OK (<10)" if vif_max < 10 else "high collinearity",
         "mitigation": "backward elimination removed redundant predictors"},
    ])
    save_table(tests.round(4), "12_assumption_tests.csv", cfg)

    # ---- diagnostic plots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    lw = lowess(resid, fitted, frac=0.3)
    axes[0, 0].scatter(fitted, resid, s=5, alpha=0.3, color=MAIN)
    axes[0, 0].plot(lw[:, 0], lw[:, 1], color=ACCENT, lw=2)
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set(title="Residuals vs fitted (linearity, constant variance)",
                   xlabel="Fitted log revenue", ylabel="Residual")
    sm.qqplot(z, line="45", ax=axes[0, 1], markersize=2, alpha=0.4)
    axes[0, 1].set_title(f"Q-Q plot of standardised residuals (KS p = {ks.pvalue:.3f})")
    axes[1, 0].hist(z, bins=60, density=True, color=MAIN, alpha=0.7)
    xs = np.linspace(z.min(), z.max(), 200)
    axes[1, 0].plot(xs, stats.norm.pdf(xs), color=ACCENT, lw=2)
    axes[1, 0].set_title("Histogram of standardised residuals vs N(0,1)")
    sl = np.sqrt(np.abs(z))
    lw2 = lowess(sl, fitted, frac=0.3)
    axes[1, 1].scatter(fitted, sl, s=5, alpha=0.3, color=MAIN)
    axes[1, 1].plot(lw2[:, 0], lw2[:, 1], color=ACCENT, lw=2)
    axes[1, 1].set(title=f"Scale-location (Breusch-Pagan p = {bp_p:.3g})",
                   xlabel="Fitted log revenue", ylabel="sqrt(|standardised residual|)")
    save_fig(fig, "06_models", "ols_diagnostics_2x2.png", cfg)

    infl = OLSInfluence(res)
    cooks = infl.cooks_distance[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(cooks)), cooks, s=4, color=MAIN)
    ax.axhline(4 / len(cooks), color=ACCENT, ls="--", label="4/n rule of thumb")
    ax.set(title="Cook's distance (influential films)", xlabel="Film index", ylabel="Cook's D")
    ax.legend()
    save_fig(fig, "06_models", "cooks_distance.png", cfg)
    top_inf = tr.assign(cooks_d=cooks).nlargest(10, "cooks_d")[
        ["title", "year", "revenue", "budget", "budget_missing", "roi_outlier", "cooks_d"]]
    save_table(top_inf.round(5), "12_top_influential_films.csv", cfg)

    # ---- GLM Gamma diagnostics
    glm = fit_glm_gamma(ytr_m, Xtr[cols])
    dres = np.asarray(glm.resid_deviance)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(np.log(np.asarray(glm.fittedvalues)), dres, s=5, alpha=0.3, color=MAIN)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(title="GLM Gamma: deviance residuals vs log fitted",
                xlabel="log fitted revenue (USD M)", ylabel="Deviance residual")
    sm.qqplot(dres, line="s", ax=axes[1], markersize=2, alpha=0.4)
    axes[1].set_title("GLM Gamma: Q-Q plot of deviance residuals")
    save_fig(fig, "06_models", "glm_gamma_diagnostics.png", cfg)

    # ---- 5-fold stratified CV (regression)
    strat = (tr["is_sequel"].astype(str) + "_" + tr["blockbuster"].astype(str)).values
    skf = StratifiedKFold(cfg["split"]["cv_folds"], shuffle=True, random_state=seed)
    rows = []
    for name, mcols in sets.items():
        for fold, (a, b) in enumerate(skf.split(Xtr, strat), start=1):
            if name.startswith("M5"):
                g = fit_glm_gamma(ytr_m[a], Xtr.iloc[a][mcols])
                pred, smear = predict_glm_log(g, Xtr.iloc[b][mcols]), 1.0
            else:
                r = fit_ols(ytr[a], Xtr.iloc[a][mcols])
                pred, smear = predict_ols(r, Xtr.iloc[b][mcols]), float(np.mean(np.exp(r.resid)))
            rows.append({"model": name, "fold": fold, **regression_metrics(ytr[b], pred, smear)})
    cv = pd.DataFrame(rows)
    save_table(cv.round(4), "12_cv_regression_folds.csv", cfg)
    cv_sum = cv.drop(columns="fold").groupby("model").agg(["mean", "std"])
    cv_sum.columns = [f"{a}_{b}" for a, b in cv_sum.columns]
    save_table(cv_sum.round(4).reset_index(), "12_cv_regression_summary.csv", cfg)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=cv, x="model", y="RMSE_log", color=MAIN, ax=ax)
    sns.stripplot(data=cv, x="model", y="RMSE_log", color=ACCENT, size=6, ax=ax)
    ax.set(title="5-fold cross-validation: RMSE on log revenue", xlabel="")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "06_models", "cv_rmse_by_model.png", cfg)

    # ---- LOOCV (OLS, closed form)
    loo = [{"model": n, "LOOCV_RMSE_log": _loocv(ytr, Xtr[c])[0], "LOOCV_MAE_log": _loocv(ytr, Xtr[c])[1]}
           for n, c in sets.items() if not n.startswith("M5")]
    save_table(pd.DataFrame(loo).round(4), "12_loocv_ols.csv", cfg)

    # ---- CV for logistic models
    ccols = sel["classification"]["columns"]
    crow = []
    for name, m in [("Logistic (unweighted)", LogisticRegression(C=1e4, max_iter=5000)),
                    ("Logistic (class_weight=balanced)",
                     LogisticRegression(C=1e4, class_weight="balanced", max_iter=5000))]:
        sc = cross_validate(m, Xtr[ccols], tr["blockbuster"].values, cv=skf,
                            scoring=["roc_auc", "average_precision", "recall", "precision"])
        crow.append({"model": name, **{f"{k.replace('test_', '')}_mean": v.mean()
                                       for k, v in sc.items() if k.startswith("test_")},
                     **{f"{k.replace('test_', '')}_std": v.std()
                        for k, v in sc.items() if k.startswith("test_")}})
    save_table(pd.DataFrame(crow).round(4), "12_cv_classification.csv", cfg)

    # ---- accuracy by segment on the test set (best regression model)
    if best.startswith("M5"):
        pred_te, smear = predict_glm_log(glm, Xte[cols]), 1.0
    else:
        pred_te, smear = predict_ols(res, Xte[cols]), float(np.mean(np.exp(resid)))
    seg_rows = [{"segment": "All films", "n": len(yte), **regression_metrics(yte, pred_te, smear)}]
    for seg in SEGMENTS:
        mk = (te["franchise_segment"] == seg).values
        if mk.sum() > 5:
            seg_rows.append({"segment": seg, "n": int(mk.sum()),
                             **regression_metrics(yte[mk], pred_te[mk], smear)})
    seg_df = pd.DataFrame(seg_rows)
    save_table(seg_df.round(4), "12_test_accuracy_by_segment.csv", cfg)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [MAIN] + [PALETTE[s] for s in seg_df["segment"][1:]]
    ax.bar(seg_df["segment"], seg_df["within_2x_pct"], color=colors)
    for i, (v, n) in enumerate(zip(seg_df["within_2x_pct"], seg_df["n"])):
        ax.text(i, v, f"{v:.0f}%\n(n={n:,})", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of films predicted within 2x of actual")
    slide_style(ax, "Forecast accuracy by segment (unseen test films)")
    save_fig(fig, "07_slides", "slide_accuracy_by_segment.png", cfg)

    log(f"agent_12_diagnostics_cv: done -> DW {dw:.2f}, KS p {ks.pvalue:.3g}, BP p {bp_p:.3g}, "
        f"max VIF {vif_max:.1f}; CV RMSE_log best "
        f"{cv_sum['RMSE_log_mean'].idxmin()} = {cv_sum['RMSE_log_mean'].min():.3f}; "
        f"test within-2x: {seg_df.loc[0, 'within_2x_pct']:.0f}%")
    return state