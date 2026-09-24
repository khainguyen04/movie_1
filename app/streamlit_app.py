import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
for _p in (str(ROOT), str(APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from components.ui import setup_page, hero, kpi_row, section, step_card, note, footer, fmt_money
from components import charts
from services.predictor import artifacts_ready, get_predictor

setup_page("Home", "🎬")
hero("Box Office Forecaster",
     "Estimate a film's box office <b>before release</b> – for originals and franchise sequels – "
     "from budget, genre, talent, franchise history and release timing.")

if not artifacts_ready():
    note("⚙️ The model has not been built yet. Run <code>python run_pipeline.py</code> in the project "
         "folder, then refresh this page.")
    footer()
    st.stop()

P = get_predictor()
card = P.card
rm, cm = card["regression"]["test_metrics"], card["classification"]["test_metrics"]
el = card["regression"].get("budget_elasticity", {})

kpi_row([
    ("Films analysed", f"{card['data']['n_films']:,}",
     f"{card['data']['year_range'][0]}–{card['data']['year_range'][1]} · known box office", "brand"),
    ("Forecasts within 2×", f"{rm['within_2x_pct']:.0f}%", "of unseen test films", "good"),
    ("Blockbuster detection", f"AUC {cm['AUC']:.2f}", f"catches {cm['sensitivity_recall']:.0%} of $100M+ hits"),
    ("Budget elasticity", f"{el.get('standalone_or_starter', 0):.2f}",
     "+1% budget → % change in revenue (originals)"),
])

section("How it works")
c1, c2, c3 = st.columns(3)
c1.markdown(step_card(1, "Describe the film", "Budget, genres, release date, director, lead cast, "
                      "franchise status and studio – only what is known before release."), unsafe_allow_html=True)
c2.markdown(step_card(2, "Get the forecast", "Expected box office with a realistic range, the chance of a "
                      "$100M+ hit and a breakdown of what drives the number."), unsafe_allow_html=True)
c3.markdown(step_card(3, "Test scenarios", "Change budget, release month or franchise strategy and "
                      "compare outcomes side by side before committing."), unsafe_allow_html=True)

section("Start here")
try:
    if hasattr(st, "page_link"):
        c1, c2, c3, c4 = st.columns(4)
        c1.page_link("pages/1_Predict_Revenue.py", label="Predict revenue", icon="🎯")
        c2.page_link("pages/2_Blockbuster_Probability.py", label="Blockbuster chance", icon="🏆")
        c3.page_link("pages/3_Scenario_Compare.py", label="Compare scenarios", icon="🔀")
        c4.page_link("pages/4_About_Model.py", label="About the model", icon="📘")
    else:
        note("Use the pages in the left sidebar: Predict Revenue → Blockbuster Probability → Scenario Compare.")
except Exception:
    note("Use the pages in the left sidebar: Predict Revenue → Blockbuster Probability → Scenario Compare.")

if card.get("top_effects"):
    section("What moves box office", "Strongest effects in the model, holding everything else equal.")
    eff = [e for e in card["top_effects"] if e["unit"] in ("0 -> 1", "vs reference category")][:8]
    if eff:
        st.plotly_chart(charts.effects_bar([e["label"] for e in eff], [e["effect_pct"] for e in eff]),
                        use_container_width=True)
footer()