"""The film input form (reused by every page)."""

from datetime import date

import streamlit as st

from utils.features_shared import FRANCHISE_MODES

NEW_DIRECTOR = "— New director / not in database —"
NO_COLL = "— Select a franchise —"


def _idx(options, value, default=0):
    try:
        return options.index(value)
    except (ValueError, TypeError):
        return default


def movie_form(P, key: str, defaults: dict = None):
    opts, sd = P.options, P.schema["defaults"]
    d = {**P.sample, **(defaults or {})}
    lang_codes = [l["code"] for l in opts["languages"]]
    lang_name = {l["code"]: l["name"] for l in opts["languages"]}
    actors = set(P.actor_names)
    manual = d.get("manual_franchise") or {}

    with st.form(f"movie_form_{key}"):
        st.markdown('<div class="form-section">🎞️ The film</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 1, 1])
        title = c1.text_input("Working title", value=d.get("title", ""), placeholder="e.g. Galaxy Raiders II")
        budget_m = c2.number_input("Budget (USD millions)", min_value=0.1, max_value=600.0,
                                   value=float(d["budget"]) / 1e6, step=5.0, format="%.1f",
                                   help="Production budget, excluding marketing.")
        runtime = c3.number_input("Runtime (minutes)", min_value=60, max_value=240,
                                  value=int(round(float(d["runtime"]))), step=5)
        c1, c2, c3 = st.columns([1, 2, 1])
        release = c1.date_input("Planned release date", value=date.fromisoformat(str(d["release_date"])[:10]),
                                min_value=date(1990, 1, 1), max_value=date(2035, 12, 31))
        genres = c2.multiselect("Genres (up to 4)", opts["genres"],
                                default=[g for g in d.get("genres", []) if g in opts["genres"]],
                                max_selections=4)
        lang = c3.selectbox("Original language", lang_codes, index=_idx(lang_codes, d.get("original_language")),
                            format_func=lambda c: lang_name.get(c, c))

        st.markdown('<div class="form-section">🎬 Talent</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        dir_opts = [NEW_DIRECTOR] + P.director_names
        director = c1.selectbox("Director (type to search)", dir_opts, index=_idx(dir_opts, d.get("director")),
                                help="The model uses the director's number of earlier films and their box office.")
        dir_gender = c2.selectbox("Director gender", opts["genders"],
                                  index=_idx(opts["genders"], d.get("director_gender_cat"), 2))
        c1, c2 = st.columns([3, 1])
        cast = c1.multiselect("Lead cast – top 3 billed (type to search)", P.actor_names,
                              default=[a for a in d.get("cast", []) if a in actors], max_selections=3)
        lead_gender = c2.selectbox("Lead actor gender", opts["genders"],
                                   index=_idx(opts["genders"], d.get("lead_gender_cat"), 2))

        st.markdown('<div class="form-section">🔁 Franchise</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        mode = c1.radio("Franchise status", FRANCHISE_MODES,
                        index=_idx(FRANCHISE_MODES, d.get("franchise_mode")))
        coll_opts = [NO_COLL] + P.collection_names
        coll = c2.selectbox("Franchise in database (for option 2)", coll_opts,
                            index=_idx(coll_opts, d.get("collection")))
        c3, c4 = c2.columns(2)
        n_prior = c3.number_input("Earlier films (for option 3)", min_value=1, max_value=30,
                                  value=int(manual.get("n_prior", 1)))
        mean_rev_m = c4.number_input("Their average box office, USD M (option 3)", min_value=0.0,
                                     max_value=3000.0, value=float((manual.get("mean_revenue") or 3e8) / 1e6),
                                     step=10.0)

        st.markdown('<div class="form-section">🏢 Production</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        companies = opts["companies"]
        company = c1.selectbox("Lead studio", companies, index=_idx(companies, d.get("lead_company")),
                               format_func=lambda c: "Other / independent" if c == "Other" else c)
        n_companies = c2.number_input("Production companies", 1, 20, int(d.get("n_companies", 2)))
        n_countries = c3.number_input("Production countries", 1, 10, int(d.get("n_countries", 1)))
        us = c4.checkbox("US production", value=bool(d.get("us_production", True)))

        with st.expander("⚙️ Advanced details (pre-filled with a typical film)"):
            c1, c2, c3, c4, c5 = st.columns(5)
            n_spoken = c1.number_input("Spoken languages", 1, 10, int(d.get("n_spoken_languages", 1)))
            n_kw = c2.number_input("Keywords / themes", 0, 60, int(d.get("n_keywords", sd["n_keywords"])))
            cast_size = c3.number_input("Credited cast", 1, 200, int(d.get("cast_size", sd["cast_size"])))
            crew_size = c4.number_input("Credited crew", 1, 400, int(d.get("crew_size", sd["crew_size"])))
            female = c5.slider("Female share of cast", 0.0, 1.0,
                               float(d.get("cast_female_share", sd["cast_female_share"])), 0.05)

        submitted = st.form_submit_button("🎯  Forecast box office", type="primary",
                                          use_container_width=True)

    if not submitted:
        return None
    if not genres:
        st.warning("Please select at least one genre.")
        return None
    collection, manual_f = None, None
    if mode == FRANCHISE_MODES[1]:
        if coll == NO_COLL:
            st.warning("Pick a franchise from the list, or choose 'Sequel – enter details'.")
            return None
        collection = coll
    elif mode == FRANCHISE_MODES[2]:
        manual_f = {"n_prior": int(n_prior), "mean_revenue": mean_rev_m * 1e6 if mean_rev_m > 0 else None}
    return {
        "title": title, "budget": budget_m * 1e6, "runtime": int(runtime),
        "release_date": release.isoformat(), "genres": genres, "original_language": lang,
        "director": None if director == NEW_DIRECTOR else director, "director_gender_cat": dir_gender,
        "cast": cast, "lead_gender_cat": lead_gender, "franchise_mode": mode,
        "collection": collection, "manual_franchise": manual_f, "lead_company": company,
        "n_companies": int(n_companies), "n_countries": int(n_countries), "us_production": bool(us),
        "n_spoken_languages": int(n_spoken), "n_keywords": int(n_kw), "cast_size": int(cast_size),
        "crew_size": int(crew_size), "cast_female_share": float(female),
        "budget_missing": 0, "runtime_imputed": 0,
    }