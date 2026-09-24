"""
agent_13_reporter
=================
Collect results for the deliverables:
  * outputs/tables/13_results_summary.md  - every key table in markdown (paste into the report)
  * report/figures/Figure_XX_*.png + figure_list.md with captions
  * presentation/figures/*.png (slide figures)
  * report/sections/01_executive_summary_draft.md and presentation/slide_outline_draft.md
    pre-filled with the actual numbers
"""

import os
import glob
import shutil

import numpy as np
import pandas as pd

from utils.io import log, load_json

REPORT_FIGURES = [
    ("01_data_quality/data_flow.png", "Data flow from the raw file to the modelling dataset"),
    ("01_data_quality/missing_before_after.png", "Missing values before and after imputation"),
    ("01_data_quality/budget_imputation_check.png", "Observed vs model-imputed budget distribution"),
    ("03_distributions/revenue_raw_vs_log.png", "Revenue distribution before and after log transform"),
    ("03_distributions/qq_revenue_raw_vs_log.png", "Q-Q plots: raw vs log revenue"),
    ("04_bivariate/scatter_budget_revenue_by_franchise.png", "Budget vs revenue by franchise status"),
    ("04_bivariate/box_revenue_by_segment.png", "Revenue by segment (standalone, starter, sequel)"),
    ("04_bivariate/box_revenue_by_genre.png", "Revenue by genre"),
    ("05_dependence/pearson_heatmap.png", "Pearson correlation matrix"),
    ("05_dependence/mutual_information_top20.png", "Top 20 features by mutual information"),
    ("06_models/coefficients_selected_ols.png", "Selected OLS coefficients with 95% CI"),
    ("06_models/ols_diagnostics_2x2.png", "OLS residual diagnostics"),
    ("06_models/cv_rmse_by_model.png", "5-fold cross-validation RMSE by model"),
    ("06_models/predicted_vs_actual_test.png", "Predicted vs actual revenue on the test set"),
    ("06_models/roc_pr_curves.png", "ROC and precision-recall curves (blockbuster model)"),
    ("06_models/confusion_matrices.png", "Confusion matrices before/after imbalance handling"),
]


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:,.{digits}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def run(state: dict, cfg: dict) -> dict:
    log("agent_13_reporter: start")
    T, F, L = cfg["paths"]["tables"], cfg["paths"]["figures"], cfg["paths"]["logs"]

    def read(name):
        path = os.path.join(T, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    def jread(path):
        return load_json(path) if os.path.exists(path) else {}

    kn = jread(os.path.join(T, "04_key_numbers.json"))
    sel = jread(os.path.join(L, "model_selection.json"))

    # ---- results summary markdown
    parts = ["# Results summary (auto-generated)\n"]
    flow_p = os.path.join(L, "data_flow_log.csv")
    blocks = [
        ("Data flow", pd.read_csv(flow_p).drop_duplicates("step", keep="last")
         if os.path.exists(flow_p) else None),
        ("Data errors and handling", read("02_data_error_summary.csv")),
        ("Imputation plan", read("03_imputation_plan.csv")),
        ("Budget imputation - 5-fold CV", read("03_budget_imputation_cv.csv")),
        ("Descriptive statistics (continuous)", read("04_descriptive_continuous.csv")),
        ("Categorical modes", read("04_categorical_modes.csv")),
        ("Segment summary", read("06_segment_summary.csv")),
        ("Class imbalance", read("08_class_imbalance.csv")),
        ("Top 10 features by mutual information",
         None if read("07_mutual_information.csv") is None else read("07_mutual_information.csv").head(10)),
        ("Model comparison", read("10_model_comparison.csv")),
        ("Sensitivity (observed budgets only)", read("10_sensitivity_observed_budget.csv")),
        ("Assumption tests", read("12_assumption_tests.csv")),
        ("Cross-validation (regression)", read("12_cv_regression_summary.csv")),
        ("Cross-validation (classification)", read("12_cv_classification.csv")),
        ("Classification metrics (test)", read("11_classification_metrics.csv")),
        ("Accuracy by segment (test)", read("12_test_accuracy_by_segment.csv")),
    ]
    coef = read("10_coefficients_selected_ols.csv")
    if coef is not None:
        c = coef[(coef["term"] != "const") & (coef["p_value"] < 0.05)]
        c = c.reindex(c["t"].abs().sort_values(ascending=False).index).head(15)
        blocks.insert(10, ("Selected OLS - 15 strongest significant terms",
                           c[["label", "coef", "p_value", "p_value_HC3", "effect_pct", "unit"]]))
    for title, df in blocks:
        if df is not None:
            parts += [f"\n## {title}\n", md_table(df), ""]
    if sel:
        parts += ["\n## Budget elasticity\n", str(sel.get("budget_elasticity")), ""]
    with open(os.path.join(T, "13_results_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))

    # ---- copy figures
    rep_dir, pres_dir = "report/figures", "presentation/figures"
    os.makedirs(rep_dir, exist_ok=True)
    os.makedirs(pres_dir, exist_ok=True)
    fig_list = ["# Figure list\n"]
    n = 0
    for rel, caption in REPORT_FIGURES:
        src = os.path.join(F, rel)
        if os.path.exists(src):
            n += 1
            dst = f"Figure_{n:02d}_{os.path.basename(rel)}"
            shutil.copy(src, os.path.join(rep_dir, dst))
            fig_list.append(f"- **Figure {n}.** {caption} (`{dst}`)")
    with open(os.path.join(rep_dir, "figure_list.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(fig_list))
    slides = sorted(glob.glob(os.path.join(F, "07_slides", "*.png")))
    for s in slides:
        shutil.copy(s, pres_dir)

    # ---- executive summary draft with real numbers
    comp, cls, seg = read("10_model_comparison.csv"), read("11_classification_metrics.csv"), \
        read("12_test_accuracy_by_segment.csv")
    try:
        best = comp[comp["model"] == sel["best_regression"]].iloc[0]
        m1 = comp[comp["model"] == "M1_OLS_budget_only"].iloc[0]
        ci = sel["classification"]
        fin = cls[(cls["model"] == ci["model"]) & (cls["threshold_rule"] == ci["threshold_rule"])].iloc[0]
        el = sel.get("budget_elasticity", {})
        within = seg.loc[seg["segment"] == "All films", "within_2x_pct"].iloc[0]
        summary = f"""# Executive summary (DRAFT - numbers filled automatically, edit wording)

**Aim.** Help a film distributor forecast box-office revenue *before release* for both standalone
and franchise films, using only information known in advance.

**Data.** {kn.get('n_movies', 'N'):,} films with known revenue from The Movies Dataset (Kaggle),
{kn.get('year_range', ['', ''])[0]}-{kn.get('year_range', ['', ''])[1]}. After careful cleaning we kept
98% of films with reported revenue; missing budgets ({kn.get('share_budget_imputed', 0):.0%}) were
imputed with a regression model and flagged.

**Key findings.**
- Franchise films earn about {kn.get('franchise_to_standalone_ratio', 0):.1f}x more than standalone
  films (median revenue).
- Budget is the strongest single driver: +1% budget is associated with about
  {el.get('standalone_or_starter', 0):.2f}% more revenue for original films
  {f"and {el['sequel']:.2f}% for sequels" if 'sequel' in el else ''}.
- Track record of director, lead cast and earlier franchise films adds predictive power
  beyond budget (test R² on log revenue {m1['test_R2_log']:.2f} with budget only vs
  {best['test_R2_log']:.2f} with the full model).

**Model performance on unseen films.** {within:.0f}% of forecasts fall within 2x of actual
revenue. The blockbuster classifier (revenue >= $100M, only {kn.get('share_blockbuster', 0):.0%} of films)
reaches AUC {fin['AUC']:.2f} and catches {fin['sensitivity_recall']:.0%} of real blockbusters
(precision {fin['precision']:.0%}) after handling class imbalance.

**Recommendation.** Use the forecast as a planning range, not a point estimate; combine it with
marketing and competition information that the model does not see.
"""
        with open("report/sections/01_executive_summary_draft.md", "w", encoding="utf-8") as fh:
            fh.write(summary)
    except Exception as e:
        log(f"   executive summary draft skipped ({e})")

    outline = """# Slide outline draft (10-15 min, non-technical decision makers)

1. Title + the business question (why forecast before release?)
2. The data & how we cleaned it -> data_flow.png (report fig) / key message: we kept 98% of films
3. Box office is highly skewed -> slide_revenue_by_budget_band.png
4. Franchise effect -> slide_segment_median_revenue.png, slide_franchise_vs_standalone.png
5. Timing and genre -> slide_revenue_by_month.png, slide_top_genres.png
6. What drives revenue -> slide_top_drivers.png
7. The model in one picture -> slide_effects_percent.png
8. Class imbalance and how we handled it -> slide_class_imbalance.png, slide_blockbuster_recall.png
9. How accurate is it? -> slide_accuracy_by_segment.png
10. Live demo of the forecasting app (Streamlit)
11. Limitations & data ethics (incl. Indigenous data sovereignty reflection)
12. Recommendations & next steps
"""
    with open("presentation/slide_outline_draft.md", "w", encoding="utf-8") as fh:
        fh.write(outline)

    log(f"agent_13_reporter: done -> {n} report figures, {len(slides)} slide figures, "
        f"13_results_summary.md + executive summary draft")
    return state