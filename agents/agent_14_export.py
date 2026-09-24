"""
agent_14_export
===============
Export everything the UI needs (artifacts/):
  models/revenue_model.joblib     Pipeline: MoviePreprocessor -> ColumnSelector -> RevenueRegressor
  models/blockbuster_model.joblib Pipeline: MoviePreprocessor -> ColumnSelector -> LogisticRegression
  metadata/*.json                 feature schema, dropdown options, model card, sample input
  lookups/*.csv                   director / actor / franchise track records, reference films
The deployed models are fitted on the TRAINING set, i.e. exactly the models evaluated on the test
set in the report. A round-trip test reloads the files and predicts the sample film.
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils.io import log, save_json, load_json, get_df, ensure_dirs
from utils.metrics import regression_metrics, classification_metrics
from utils.parsing import VALID_GENRES
from utils.features_shared import (MoviePreprocessor, ColumnSelector, RevenueRegressor, MONTHS,
                                   FRANCHISE_MODES, build_lookups, lookups_to_dict,
                                   history_from_lookups, build_feature_row)

LANG_NAMES = {"en": "English", "fr": "French", "hi": "Hindi", "ru": "Russian", "es": "Spanish",
              "ja": "Japanese", "it": "Italian", "de": "German", "ko": "Korean", "zh": "Mandarin",
              "cn": "Cantonese", "ta": "Tamil", "te": "Telugu", "sv": "Swedish", "da": "Danish",
              "pt": "Portuguese", "Other": "Other language"}
LIMITATIONS = [
    "Revenue is nominal (not inflation-adjusted); release year partly controls for this.",
    "Trained on films up to 2017; streaming-era behaviour is not represented.",
    "86% of training films are English-language; other markets are less reliable.",
    "About a quarter of budgets were imputed; a sensitivity check on observed budgets is reported.",
    "Marketing spend, competition on the release weekend and reviews are not in the data.",
    "Forecasts are associations learned from history, not causal effects.",
]


def _prep(fl: dict, cfg: dict) -> MoviePreprocessor:
    fc = cfg["features"]
    return MoviePreprocessor(
        numeric=fl["numeric"], binary=fl["binary"], categorical=fl["categorical"],
        top_k={"original_language": fc["language_top_k"], "lead_company": fc["company_top_k"]},
        interactions=[list(i) for i in fc.get("interactions", [["log_budget", "is_sequel"]])])


def run(state: dict, cfg: dict) -> dict:
    log("agent_14_export: start")
    ensure_dirs(cfg)
    P, L, A, ex = cfg["paths"]["processed"], cfg["paths"]["logs"], cfg["paths"]["artifacts"], cfg["export"]
    train = get_df(state, "train", os.path.join(P, "train.csv"))
    test = get_df(state, "test", os.path.join(P, "test.csv"))
    feats = get_df(state, "features", os.path.join(P, "features_full.csv"))
    all_df = get_df(state, "movies_all", os.path.join(cfg["paths"]["interim"], "movies_parsed.csv"))
    fl = state.get("feature_lists") or load_json(os.path.join(L, "feature_lists.json"))
    sel = load_json(os.path.join(L, "model_selection.json"))
    ci = sel["classification"]

    # ---- regression pipeline
    kind = "glm" if sel["best_regression"].startswith("M5") else "ols"
    reg = Pipeline([("prep", _prep(fl, cfg)), ("select", ColumnSelector(sel["selected_columns"])),
                    ("model", RevenueRegressor(kind=kind))]).fit(train, train["revenue"])
    lp = reg[-1].predict_log(reg[:-1].transform(test))
    resid = test["log_revenue"].values - lp
    q = {f"q{k}": float(np.quantile(resid, k / 100)) for k in (10, 25, 75, 90)}
    reg_m = regression_metrics(test["log_revenue"], lp, reg[-1].smearing_)

    # ---- classification pipeline
    clf = Pipeline([("prep", _prep(fl, cfg)), ("select", ColumnSelector(ci["columns"])),
                    ("model", LogisticRegression(C=1e4, class_weight=ci["class_weight"],
                                                 max_iter=5000))]).fit(train, train["blockbuster"])
    clf_m = classification_metrics(test["blockbuster"], clf.predict_proba(test)[:, 1],
                                   ci["threshold"])
    joblib.dump(reg, ex["revenue_model"])
    joblib.dump(clf, ex["blockbuster_model"])

    # ---- lookups + reference films
    lk = build_lookups(all_df, feats)
    for k, df in lk.items():
        df.to_csv(os.path.join(A, "lookups", f"{k}_stats.csv"), index=False)
    ref_cols = ["id", "title", "year", "budget", "revenue", "primary_genre", "genres",
                "franchise_segment", "budget_missing", "directors"]
    feats[ref_cols].to_csv(os.path.join(A, "lookups", "reference_films.csv"), index=False)

    # ---- schema & options
    med = lambda c: float(feats[c].median())
    obs = feats[feats["budget_missing"] == 0]
    prep_fit = reg.named_steps["prep"]
    schema = {
        "numeric": fl["numeric"], "binary": fl["binary"], "categorical": fl["categorical"],
        "fill_log_rev": float(feats["log_revenue"].median()),
        "defaults": {c: med(c) for c in ["runtime", "n_keywords", "n_companies", "n_countries",
                                         "n_spoken_languages", "cast_size", "crew_size",
                                         "cast_female_share"]} | {"budget": float(obs["budget"].median())},
        "ranges": {"budget": [float(obs["budget"].quantile(0.01)), float(obs["budget"].quantile(0.99))],
                   "runtime": [float(feats["runtime"].quantile(0.01)), float(feats["runtime"].quantile(0.99))],
                   "year": [int(feats["year"].min()), int(feats["year"].max())]},
    }
    save_json(schema, ex["feature_schema"])
    langs = [l for l in prep_fit.levels_["original_language"] if l != "Unknown"]
    comps = [c for c in prep_fit.levels_["lead_company"] if c != "Unknown"]
    options = {
        "genres": VALID_GENRES, "months": MONTHS, "franchise_modes": FRANCHISE_MODES,
        "languages": [{"code": l, "name": LANG_NAMES.get(l, l)} for l in langs] +
                     [{"code": "Other", "name": "Other language"}],
        "companies": comps + ["Other"], "genders": ["male", "female", "unknown"],
    }
    save_json(options, ex["input_options"])

    # ---- model card
    coef_p = os.path.join(cfg["paths"]["tables"], "10_coefficients_selected_ols.csv")
    top = []
    if os.path.exists(coef_p):
        c = pd.read_csv(coef_p)
        c = c[(c["term"] != "const") & (c["p_value"] < 0.05)]
        c = c.reindex(c["effect_pct"].abs().sort_values(ascending=False).index).head(10)
        top = c[["label", "effect_pct", "unit", "p_value"]].to_dict("records")
    card = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__},
        "intended_use": "Pre-release box-office planning for film distribution; decision support only.",
        "data": {"source": "The Movies Dataset (Kaggle, Banik 2017)", "n_films": len(feats),
                 "n_train": len(train), "n_test": len(test),
                 "year_range": schema["ranges"]["year"],
                 "median_revenue": float(feats["revenue"].median()),
                 "share_sequel": float(feats["is_sequel"].mean()),
                 "share_blockbuster": float(feats["blockbuster"].mean())},
        "regression": {"model": sel["best_regression"], "kind": kind,
                       "n_features": len(sel["selected_columns"]), "test_metrics": reg_m,
                       "interval_log_quantiles": q, "smearing": reg[-1].smearing_,
                       "budget_elasticity": sel.get("budget_elasticity", {})},
        "classification": {"model": ci["model"], "class_weight": ci["class_weight"],
                           "threshold": ci["threshold"], "threshold_rule": ci["threshold_rule"],
                           "loss_matrix": ci["loss_matrix"], "test_metrics": clf_m,
                           "blockbuster_threshold_usd": cfg["classification"]["blockbuster_threshold"],
                           "train_neg_pos_ratio": float((train["blockbuster"] == 0).sum()
                                                        / max((train["blockbuster"] == 1).sum(), 1))},
        "top_effects": top, "limitations": LIMITATIONS,
    }
    save_json(card, ex["model_card"])

    # ---- sample input (a realistic franchise sequel from the data)
    d_top = lk["director"].sort_values(["rev_films", "mean_log_rev"], ascending=False).iloc[0]["name"]
    a_top = list(lk["actor"].sort_values(["rev_films", "mean_log_rev"], ascending=False)["name"].head(3))
    c_top = lk["collection"].sort_values(["rev_films", "mean_log_rev"], ascending=False).iloc[0]["name"]
    sample = {
        "title": "Sample: summer franchise sequel", "budget": 150_000_000.0, "runtime": 130,
        "release_date": "2018-07-13", "genres": ["Action", "Adventure", "Science Fiction"],
        "original_language": "en", "lead_company": comps[0] if comps else "Other",
        "us_production": True, "n_companies": 3, "n_countries": 1, "n_spoken_languages": 1,
        "n_keywords": schema["defaults"]["n_keywords"], "cast_size": schema["defaults"]["cast_size"],
        "crew_size": schema["defaults"]["crew_size"],
        "cast_female_share": schema["defaults"]["cast_female_share"],
        "director": d_top, "director_gender_cat": "male", "cast": a_top, "lead_gender_cat": "male",
        "franchise_mode": FRANCHISE_MODES[1], "collection": c_top, "manual_franchise": None,
        "budget_missing": 0, "runtime_imputed": 0,
    }
    save_json(sample, os.path.join(A, "metadata", "sample_input.json"))

    # ---- round-trip test
    reg2, clf2 = joblib.load(ex["revenue_model"]), joblib.load(ex["blockbuster_model"])
    h = history_from_lookups(sample["director"], sample["cast"], sample["collection"],
                             lookups_to_dict(lk), schema["fill_log_rev"])
    row = build_feature_row(sample, h)
    pred, prob = float(reg2.predict(row)[0]), float(clf2.predict_proba(row)[0, 1])
    assert np.isfinite(pred) and pred > 0 and 0 <= prob <= 1, "round-trip prediction failed"

    log(f"agent_14_export: done -> {kind.upper()} revenue model (test R2 log "
        f"{reg_m['R2_log']:.3f}, within 2x {reg_m['within_2x_pct']:.0f}%), blockbuster AUC "
        f"{clf_m['AUC']:.3f}; lookups: {len(lk['director']):,} directors, "
        f"{len(lk['actor']):,} actors, {len(lk['collection']):,} franchises; "
        f"sample film -> ${pred / 1e6:,.0f}M, P(blockbuster) {prob:.0%}")
    return state