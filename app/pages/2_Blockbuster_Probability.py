import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
for _p in (str(ROOT), str(APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import streamlit as st

from components.ui import (setup_page, hero, kpi_row, badge, section, note, fmt_money,
                           require_artifacts, footer, film_summary)
from components.inputs import movie_form
from components import charts
from services.predictor import get_predictor

setup_page("Blockbuster chance", "🏆")
require_artifacts()
P = get_predictor()
cc = P.card["classification"]
hero("Blockbuster chance", f"Probability that the film reaches {fmt_money(cc['blockbuster_threshold_usd'])}+ "
     "at the box office – and what would tip the balance.", "🏆")

inp = st.session_state.get("movie_inputs")
if inp is None:
    note("No film yet – fill in the form below (or start on the <b>Predict Revenue</b> page).")
    inp = movie_form(P, key="bb")
    if inp is None:
        footer()
        st.stop()
    st.session_state["movie_inputs"] = inp
else:
    st.markdown(film_summary(inp), unsafe_allow_html=True)
    with st.expander("✏️ Edit film details"):
        new = movie_form(P, key="bb_edit", defaults=inp)
        if new is not None:
            inp = new
            st.session_state["movie_inputs"] = new

res = P.predict(inp)
tm = cc["test_metrics"]
c1, c2 = st.columns([1, 1.25])
with c1:
    st.plotly_chart(charts.gauge(res["prob"], res["threshold_display"]), use_container_width=True)
with c2:
    section("Verdict")
    st.markdown(badge("🏆 Treat as a potential blockbuster", "good") if res["is_blockbuster"]
                else badge("🎞️ Plan as a mid-size release", "brand"), unsafe_allow_html=True)
    thr_txt = fmt_money(cc['blockbuster_threshold_usd']).replace("$", "&#36;")
    fc_txt = fmt_money(res['revenue']).replace("$", "&#36;")
    st.markdown(f"Estimated chance of passing **{thr_txt}**: **{res['prob']:.0%}** (forecast {fc_txt}).")
    note(f"Only about <b>1 in 5</b> films in our data reached $100M, so the model was trained to pay "
         f"extra attention to these rare hits. Missing a blockbuster is treated as "
         f"<b>{cc['loss_matrix']['FN']}× more costly</b> than a false alarm, which is why the decision line "
         f"sits below 50%.")
    kpi_row([("Hits caught", f"{tm['sensitivity_recall']:.0%}", "of real $100M+ films (test)"),
             ("Precision", f"{tm['precision']:.0%}", "of 'blockbuster' calls were right"),
             ("AUC", f"{tm['AUC']:.2f}", "ranking quality (1 = perfect)")])

section("What budget would tip the balance?")
budgets = np.geomspace(2e6, 400e6, 40)
sw = P.sweep(inp, "budget", budgets)
st.plotly_chart(charts.prob_curve(budgets, sw["prob"], res["threshold_display"],
                                  current=(inp["budget"], res["prob"])), use_container_width=True)
hit = sw[sw["is_blockbuster"]]
if res["is_blockbuster"]:
    note("✅ At the current budget the model already classifies the film as a potential blockbuster.")
elif len(hit):
    note(f"📈 With everything else unchanged, the model would flip to <b>blockbuster</b> at a budget of about "
         f"<b>{fmt_money(hit['budget'].iloc[0])}</b>. Budget alone is rarely enough – talent and franchise "
         f"strength move the odds too.")
else:
    note("Even at $400M the model does not expect a $100M+ result for this profile – genre, talent or "
         "franchise factors are holding it back.")
footer()