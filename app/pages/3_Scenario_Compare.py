import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
for _p in (str(ROOT), str(APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import streamlit as st

from components.ui import (setup_page, hero, kpi_row, section, note, fmt_money, require_artifacts,
                           footer, film_summary)
from components.inputs import movie_form
from components import charts
from services.predictor import get_predictor, friday_in_month
from utils.features_shared import FRANCHISE_MODES

setup_page("Compare scenarios", "🔀")
require_artifacts()
P = get_predictor()
hero("Scenario planner", "Change budget, release month, runtime or franchise strategy and see the impact instantly.", "🔀")

base = st.session_state.get("movie_inputs")
if base is None:
    note("Start by describing the film (Scenario A).")
    base = movie_form(P, key="scn")
    if base is None:
        footer()
        st.stop()
    st.session_state["movie_inputs"] = base

st.markdown("**Scenario A (current plan):** " + film_summary(base), unsafe_allow_html=True)
months = P.options["months"]
cur_d = pd.Timestamp(base["release_date"])
base_seq = base.get("franchise_mode", "").startswith("Sequel")

section("Build scenario B")
c1, c2, c3, c4 = st.columns(4)
b_budget = c1.slider("Budget (USD M)", 1, 400, int(max(1, round(base["budget"] / 1e6))))
b_month = c2.selectbox("Release month", list(range(1, 13)), index=cur_d.month - 1,
                       format_func=lambda m: months[m - 1])
b_runtime = c3.slider("Runtime (min)", 70, 200, int(min(200, max(70, base["runtime"]))))
b_seq = c4.toggle("Sequel / franchise film", value=base_seq)
b_prev = c4.number_input("Earlier films' avg box office (USD M)", 0, 3000, 300, disabled=not b_seq or base_seq)

B = dict(base, budget=b_budget * 1e6, runtime=b_runtime)
if b_month != cur_d.month:
    B["release_date"] = friday_in_month(cur_d.year, b_month)
if b_seq and not base_seq:
    B.update(franchise_mode=FRANCHISE_MODES[2], collection=None,
             manual_franchise={"n_prior": 1, "mean_revenue": b_prev * 1e6 if b_prev > 0 else None})
elif not b_seq:
    B.update(franchise_mode=FRANCHISE_MODES[0], collection=None, manual_franchise=None)

ra, rb = P.predict(base), P.predict(B)
delta = rb["revenue"] / ra["revenue"] - 1
kpi_row([("Scenario A", fmt_money(ra["revenue"]), f"{ra['prob']:.0%} chance of $100M+", "brand"),
         ("Scenario B", fmt_money(rb["revenue"]), f"{rb['prob']:.0%} chance of $100M+"),
         ("Change", f"{delta:+.0%}", f"{fmt_money(rb['revenue'] - ra['revenue'])} difference",
          "good" if delta > 0 else "bad" if delta < 0 else ""),
         ("Revenue ÷ budget (B)", f"{rb['revenue'] / B['budget']:.1f}×", "≈2.5× to break even",
          "good" if rb["revenue"] / B["budget"] >= 2.5 else "warn")])

c1, c2 = st.columns([1, 1.4])
with c1:
    st.plotly_chart(charts.scenario_bars(["Scenario A", "Scenario B"], [ra["revenue"], rb["revenue"]],
                                         [ra["prob"], rb["prob"]]), use_container_width=True)
with c2:
    budgets = np.geomspace(2e6, 400e6, 30)
    sa, sb = P.sweep(base, "budget", budgets), P.sweep(B, "budget", budgets)
    st.plotly_chart(charts.budget_curve(budgets, sa["revenue"], (base["budget"], ra["revenue"]),
                                        sb["revenue"], (B["budget"], rb["revenue"])), use_container_width=True)

section("Best release month for scenario A", "Each bar re-runs the forecast with a Friday release in that month.")
sm = P.sweep(base, "month", range(1, 13))
st.plotly_chart(charts.month_bars(months, list(sm["revenue"]), cur_d.month), use_container_width=True)
best = int(sm["revenue"].idxmax())
up = sm["revenue"].iloc[best] / sm["revenue"].iloc[cur_d.month - 1] - 1
note(f"📅 Best month for this film: <b>{months[best]}</b> "
     f"({'+' if up >= 0 else ''}{up:.0%} vs {months[cur_d.month - 1]}). Timing effects reflect historical "
     f"seasonality, not this year's competing releases.")
footer()