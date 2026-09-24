import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
for _p in (str(ROOT), str(APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json

import pandas as pd
import streamlit as st

from components.ui import setup_page, hero, kpi_row, section, note, fmt_money, require_artifacts, footer
from components import charts
from services.predictor import get_predictor

setup_page("About the model", "📘")
require_artifacts()
P = get_predictor()
card = P.card
rm, cm = card["regression"]["test_metrics"], card["classification"]["test_metrics"]
hero("About the model", "What it does, how it was built, how accurate it is – and when not to rely on it.", "📘")

kpi_row([("Training films", f"{card['data']['n_train']:,}", f"+ {card['data']['n_test']:,} held-out test films", "brand"),
         ("Within 2× of actual", f"{rm['within_2x_pct']:.0f}%", "test films"),
         ("R² (log revenue)", f"{rm['R2_log']:.2f}", "share of variation explained"),
         ("Blockbuster AUC", f"{cm['AUC']:.2f}", f"average precision {cm['AP']:.2f}")])

section("What it does")
st.markdown(f"""
- **Revenue forecast** – a {'linear regression on log revenue' if card['regression']['kind'] == 'ols' else 'Gamma GLM with log link'}
  using {card['regression']['n_features']} pre-release features (budget, genre, runtime, release timing, language,
  studio, director and cast track record, franchise history).
- **Blockbuster chance** – a logistic regression for revenue ≥ {fmt_money(card['classification']['blockbuster_threshold_usd']).replace('$', chr(92) + '$')},
  with class-imbalance handling (`{card['classification']['model']}`, threshold rule: *{card['classification']['threshold_rule']}*).
- Only information available **before release** is used – popularity, votes and ratings are excluded to avoid data leakage.
""")

section("How it was built")
st.markdown("""
1. **Data quality** – malformed and duplicate records removed, unit errors fixed, 98% of films with known revenue kept.
2. **Missing values** – meaningful replacements, medians, and a regression model for missing budgets (flagged).
3. **Features** – log transforms, standardisation, one-hot encoding, and time-aware track records
   (only films released *earlier* count).
4. **Models** – OLS variants, backward elimination by p-value, Gamma GLM, compared by AIC/BIC and cross-validation.
5. **Checks** – residual diagnostics, Kolmogorov-Smirnov test, robust errors, 5-fold and leave-one-out CV.
""")

section("Accuracy on unseen films")
acc = pd.DataFrame([
    {"Metric": "Median absolute % error", "Value": f"{rm['MedianAPE_pct']:.0f}%"},
    {"Metric": "Forecasts within 2× of actual", "Value": f"{rm['within_2x_pct']:.0f}%"},
    {"Metric": "RMSE (log revenue)", "Value": f"{rm['RMSE_log']:.2f}"},
    {"Metric": "Blockbusters caught (recall)", "Value": f"{cm['sensitivity_recall']:.0%}"},
    {"Metric": "Precision of blockbuster calls", "Value": f"{cm['precision']:.0%}"},
])
st.dataframe(acc, hide_index=True, use_container_width=True)

if card.get("top_effects"):
    section("Strongest effects")
    eff = card["top_effects"][:10]
    st.plotly_chart(charts.effects_bar([f"{e['label']} ({e['unit']})" for e in eff],
                                       [e["effect_pct"] for e in eff]), use_container_width=True)

section("Limitations & responsible use")
st.markdown("\n".join(f"- {l}" for l in card["limitations"]))
note("Use forecasts as one input to distribution decisions alongside market knowledge, marketing plans "
     "and competitive release calendars.")

figs = sorted((ROOT / "outputs" / "figures" / "07_slides").glob("*.png"))
if figs:
    section("Key charts from the analysis")
    cols = st.columns(2)
    for i, f in enumerate(figs):
        cols[i % 2].image(str(f), use_container_width=True)

st.download_button("⬇️ Download model card (JSON)", json.dumps(card, indent=2), file_name="model_card.json",
                   mime="application/json")
footer()