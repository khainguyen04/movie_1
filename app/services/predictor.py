"""Load artifacts once and turn UI inputs into forecasts, explanations and scenarios."""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.features_shared import (build_feature_row, history_from_lookups, lookups_to_dict,
                                   contribution_group)

ART = ROOT / "artifacts"
REQUIRED = ["models/revenue_model.joblib", "models/blockbuster_model.joblib",
            "metadata/feature_schema.json", "metadata/input_options.json",
            "metadata/model_card.json", "metadata/sample_input.json",
            "lookups/director_stats.csv", "lookups/actor_stats.csv",
            "lookups/collection_stats.csv", "lookups/reference_films.csv"]


def artifacts_ready() -> bool:
    return all((ART / f).exists() for f in REQUIRED)


def friday_in_month(year: int, month: int) -> str:
    d = date(year, month, 8)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d.isoformat()


def _json(rel):
    return json.loads((ART / rel).read_text(encoding="utf-8"))


class Predictor:
    def __init__(self):
        self.reg = joblib.load(ART / REQUIRED[0])
        self.clf = joblib.load(ART / REQUIRED[1])
        self.schema, self.options = _json("metadata/feature_schema.json"), _json("metadata/input_options.json")
        self.card, self.sample = _json("metadata/model_card.json"), _json("metadata/sample_input.json")
        dfs = {k: pd.read_csv(ART / f"lookups/{k}_stats.csv") for k in ["director", "actor", "collection"]}
        self.lookups = lookups_to_dict(dfs)
        self.director_names = sorted(self.lookups["director"])
        self.actor_names = sorted(self.lookups["actor"])
        self.collection_names = sorted(self.lookups["collection"])
        self.ref = pd.read_csv(ART / "lookups/reference_films.csv")
        c = self.card["classification"]
        self.threshold = float(c["threshold"])
        self.balanced = c.get("class_weight") == "balanced"
        self.neg_pos = float(c.get("train_neg_pos_ratio", 1.0))
        self.q = self.card["regression"]["interval_log_quantiles"]
        self.fill = float(self.schema["fill_log_rev"])

    # -------------------------------------------------------------- core
    def calibrate(self, p):
        """Undo the prior shift of class_weight='balanced' so probabilities are realistic."""
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if not self.balanced:
            return p
        logit = np.log(p / (1 - p)) - np.log(self.neg_pos)
        return 1 / (1 + np.exp(-logit))

    def history(self, inp):
        return history_from_lookups(inp.get("director"), inp.get("cast"), inp.get("collection"),
                                    self.lookups, self.fill, inp.get("manual_franchise"))

    def _rows(self, inputs_list):
        return pd.concat([build_feature_row(i, self.history(i)) for i in inputs_list], ignore_index=True)

    def predict_batch(self, inputs_list) -> pd.DataFrame:
        rows = self._rows(inputs_list)
        m = self.reg[-1]
        lp = m.predict_log(self.reg[:-1].transform(rows))
        prob = self.clf.predict_proba(rows)[:, 1]
        return pd.DataFrame({"revenue": np.exp(lp) * m.smearing_, "log_pred": lp,
                             "low": np.exp(lp + self.q["q10"]), "high": np.exp(lp + self.q["q90"]),
                             "prob_raw": prob, "prob": self.calibrate(prob),
                             "is_blockbuster": prob >= self.threshold,
                             "is_sequel": rows["is_sequel"].values})

    def predict(self, inp: dict) -> dict:
        out = self.predict_batch([inp]).iloc[0].to_dict()
        rows = self._rows([inp])
        m = self.reg[-1]
        contrib = m.contributions(self.reg[:-1].transform(rows)).iloc[0]
        groups = contrib.groupby(contrib.index.map(contribution_group)).sum()
        groups = groups.reindex(groups.abs().sort_values(ascending=False).index)
        baseline = float(np.exp(m.baseline_log()) * m.smearing_)
        steps, cur = [], baseline
        for g, v in groups.items():
            new = cur * np.exp(v)
            steps.append((g, new - cur, (np.exp(v) - 1) * 100))
            cur = new
        out.update({"baseline": baseline, "steps": steps, "history": self.history(inp),
                    "threshold_display": float(self.calibrate(self.threshold)),
                    "segment": "Sequel" if out["is_sequel"] else "Original"})
        return out

    # -------------------------------------------------------------- helpers
    def sweep(self, inp: dict, field: str, values) -> pd.DataFrame:
        year = pd.Timestamp(inp["release_date"]).year
        lst = []
        for v in values:
            i = dict(inp)
            if field == "month":
                i["release_date"] = friday_in_month(year, int(v))
            else:
                i[field] = v
            lst.append(i)
        df = self.predict_batch(lst)
        df.insert(0, field, list(values))
        return df

    def comparable(self, inp: dict, n: int = 8) -> pd.DataFrame:
        ref = self.ref.copy()
        g = inp["genres"][0] if inp.get("genres") else None
        if g:
            ref = ref[ref["genres"].fillna("").str.contains(g, regex=False)]
        seq = inp.get("franchise_mode", "").startswith("Sequel")
        year = pd.Timestamp(inp["release_date"]).year
        dist = (np.abs(np.log(ref["budget"]) - np.log(inp["budget"]))
                + 0.5 * ((ref["franchise_segment"] == "Sequel") != seq)
                + 0.03 * np.abs(ref["year"] - min(year, 2017)))
        return ref.assign(distance=dist).nsmallest(n, "distance")

    def warnings(self, inp: dict) -> list:
        w, r = [], self.schema["ranges"]
        if not r["budget"][0] <= inp["budget"] <= r["budget"][1]:
            w.append("Budget is outside the range of most training films – treat the forecast with extra caution.")
        if pd.Timestamp(inp["release_date"]).year > r["year"][1]:
            w.append(f"Training data ends in {r['year'][1]}; later release years are extrapolated from the trend.")
        if inp.get("original_language") != "en":
            w.append("Most training films are English-language; forecasts for other languages are less reliable.")
        return w


@st.cache_resource(show_spinner="Loading the forecasting model …")
def get_predictor() -> Predictor:
    return Predictor()