import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
for _p in (str(ROOT), str(APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json

import numpy as np
import pandas as pd
import streamlit as st

from components.ui import (setup_page, hero, kpi_row, badge, section, note, fmt_money,
                           require_artifacts, footer, film_summary)
from components.inputs import movie_form
from components import charts
from services.predictor import get_predictor

setup_page("Predict revenue", "🎯")
require_artifacts()
P = get_predictor()
hero("Box office forecast", "Describe the film as known before release and get an instant, explained forecast.", "🎯")

new = movie_form(P, key="predict", defaults=st.session_state.get("movie_inputs"))
if new is not None:
    st.session_state["movie_inputs"] = new
inp = st.session_state.get("movie_inputs")
if inp is None:
    note("👆 A sample franchise sequel is pre-filled – press <b>Forecast box office</b> to try it, "
         "then change anything you like.")
    footer()
    st.stop()

res = P.predict(inp)
roi = res["revenue"] / inp["budget"]
section(f"Forecast for “{inp.get('title') or 'Untitled film'}”")
st.markdown(film_summary(inp), unsafe_allow_html=True)
for w in P.warnings(inp):
    st.warning(w, icon="⚠️")

kpi_row([
    ("Predicted box office", fmt_money(res["revenue"]),
     f"Likely range {fmt_money(res['low'])} – {fmt_money(res['high'])}", "brand"),
    ("Revenue ÷ budget", f"{roi:.1f}×", "≈2.5× needed to break even",
     "good" if roi >= 2.5 else ("warn" if roi >= 1 else "bad")),
    ("Chance of $100M+", f"{res['prob']:.0%}",
     "Model says: blockbuster" if res["is_blockbuster"] else "Model says: not a blockbuster",
     "good" if res["is_blockbuster"] else ""),
    ("Segment", res["segment"], "franchise follow-up" if res["segment"] == "Sequel" else "new story / starter"),
])
if res["is_blockbuster"]:
    verdict = badge("🏆 Blockbuster potential", "good")
elif roi >= 2.5:
    verdict = badge("✅ Likely profitable", "good")
elif roi >= 1:
    verdict = badge("⚖️ Recovers budget, thin margin", "warn")
else:
    verdict = badge("⚠️ Commercial risk", "bad")
st.markdown(verdict, unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["📊 Why this forecast", "🎬 Comparable films", "📐 Uncertainty", "🧾 Inputs & export"])
with t1:
    st.plotly_chart(charts.waterfall(res["baseline"], [(g, d) for g, d, _ in res["steps"]], res["revenue"]),
                    use_container_width=True)
    impact = pd.DataFrame([{"Factor group": g, "Effect vs average film": f"{pct:+.0f}%",
                            "Change": ("+" if d >= 0 else "−") + fmt_money(abs(d))}
                           for g, d, pct in res["steps"]])
    st.dataframe(impact, hide_index=True, use_container_width=True)
    st.caption("Start from an average film in the data; each factor group pushes the forecast up (green) "
               "or down (red). Effects multiply and are shown from largest to smallest.")
with t2:
    comp = P.comparable(inp)
    show = pd.DataFrame({
        "Film": comp["title"], "Year": comp["year"].astype(int), "Genre": comp["primary_genre"],
        "Segment": comp["franchise_segment"], "Budget": comp["budget"].map(fmt_money),
        "Box office": comp["revenue"].map(fmt_money),
        "Multiple": (comp["revenue"] / comp["budget"]).map(lambda v: f"{v:.1f}×")})
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption("Historical films with the same main genre, a similar budget and the same franchise status.")
with t3:
    st.plotly_chart(charts.range_bar(res["low"], res["revenue"], res["high"]), use_container_width=True)
    note(f"Box office is volatile: for films like this, 8 in 10 land between <b>{fmt_money(res['low'])}</b> "
         f"and <b>{fmt_money(res['high'])}</b>. Plan with the range, not only the single number.")
with t4:
    h = res["history"]
    track = pd.DataFrame([
        {"Item": "Director – earlier films", "Value": f"{h['director_prior_films']:.0f}"},
        {"Item": "Director – average past box office",
         "Value": fmt_money(np.exp(h["director_prior_mean_log_rev"])) if h["director_prior_rev_films"] else "no record"},
        {"Item": "Lead cast – average earlier films", "Value": f"{h['cast_prior_films']:.1f}"},
        {"Item": "Franchise – earlier films", "Value": f"{h['collection_prior_films']:.0f}"},
        {"Item": "Franchise – average past box office",
         "Value": fmt_money(np.exp(h["collection_prior_mean_log_rev"])) if h["collection_prior_rev_films"] else "–"},
    ])
    st.dataframe(track, hide_index=True, use_container_width=True)
    export = {"inputs": inp, "forecast": {k: (float(v) if isinstance(v, (int, float, np.floating, np.bool_)) else v)
                                          for k, v in res.items() if k in ("revenue", "low", "high", "prob", "is_blockbuster")}}
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download forecast (JSON)", json.dumps(export, indent=2, default=str),
                       file_name="forecast.json", mime="application/json", use_container_width=True)
    c2.download_button("⬇️ Download forecast (CSV)",
                       pd.DataFrame([{**{k: v for k, v in inp.items() if not isinstance(v, (list, dict))},
                                      **export["forecast"]}]).to_csv(index=False),
                       file_name="forecast.csv", mime="text/csv", use_container_width=True)
footer()