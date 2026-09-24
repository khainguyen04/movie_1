"""Exported artifacts load, predict sensibly and keep their test-set performance."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score, roc_auc_score

from utils.features_shared import build_feature_row, history_from_lookups, lookups_to_dict

ART = ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "models/revenue_model.joblib").exists(),
                                reason="run the pipeline (agent 14) first")


@pytest.fixture(scope="module")
def art():
    load = lambda p: json.loads((ART / p).read_text(encoding="utf-8"))
    lk = lookups_to_dict({k: pd.read_csv(ART / f"lookups/{k}_stats.csv")
                          for k in ["director", "actor", "collection"]})
    return {"reg": joblib.load(ART / "models/revenue_model.joblib"),
            "clf": joblib.load(ART / "models/blockbuster_model.joblib"),
            "schema": load("metadata/feature_schema.json"), "sample": load("metadata/sample_input.json"),
            "lk": lk}


def test_sample_prediction(art):
    s = art["sample"]
    h = history_from_lookups(s["director"], s["cast"], s["collection"], art["lk"], art["schema"]["fill_log_rev"])
    row = build_feature_row(s, h)
    rev = float(art["reg"].predict(row)[0])
    prob = float(art["clf"].predict_proba(row)[0, 1])
    assert np.isfinite(rev) and 1e5 < rev < 1e10
    assert 0.0 <= prob <= 1.0


def test_budget_increases_forecast(art):
    s = dict(art["sample"])
    h = history_from_lookups(s["director"], s["cast"], s["collection"], art["lk"], art["schema"]["fill_log_rev"])
    low = art["reg"].predict(build_feature_row({**s, "budget": 20e6}, h))[0]
    high = art["reg"].predict(build_feature_row({**s, "budget": 200e6}, h))[0]
    assert high > low


def test_test_set_performance(art):
    test = pd.read_csv(ROOT / "data/processed/test.csv", low_memory=False)
    pred = art["reg"].predict(test)
    assert r2_score(test["log_revenue"], np.log(pred)) > 0.3
    assert roc_auc_score(test["blockbuster"], art["clf"].predict_proba(test)[:, 1]) > 0.7   