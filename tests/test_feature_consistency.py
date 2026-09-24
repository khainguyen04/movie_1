"""The UI must build exactly the same features the model was trained on."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from utils.features_shared import build_feature_row, HISTORY_KEYS, SEASON_OF_MONTH
from utils.parsing import split_pipe

FEAT = ROOT / "data/processed/features_full.csv"
LISTS = ROOT / "outputs/logs/feature_lists.json"
pytestmark = pytest.mark.skipif(not (FEAT.exists() and LISTS.exists()), reason="run agents 01-06 first")


def _raw(r) -> dict:
    return {"budget": r["budget"], "runtime": r["runtime"], "release_date": r["release_date"],
            "genres": [g for g in split_pipe(r["genres"]) if g != "Unknown"],
            "original_language": r["original_language"], "lead_company": r["lead_company"],
            "us_production": r["us_production"], "n_companies": r["n_companies"],
            "n_countries": r["n_countries"], "n_spoken_languages": r["n_spoken_languages"],
            "n_keywords": r["n_keywords"], "cast_size": r["cast_size"], "crew_size": r["crew_size"],
            "cast_female_share": r["cast_female_share"], "director_gender_cat": r["director_gender_cat"],
            "lead_gender_cat": r["lead_gender_cat"], "budget_missing": r["budget_missing"],
            "runtime_imputed": r["runtime_imputed"]}


def test_ui_rows_match_training_features():
    df = pd.read_csv(FEAT, low_memory=False).sample(300, random_state=0)
    fl = json.loads(LISTS.read_text())
    cats = set(fl["categorical"])
    mismatches = {}
    for _, r in df.iterrows():
        row = build_feature_row(_raw(r), {k: r[k] for k in HISTORY_KEYS}).iloc[0]
        for c in fl["numeric"] + fl["binary"] + fl["categorical"]:
            ok = str(row[c]) == str(r[c]) if c in cats else np.isclose(float(row[c]), float(r[c]), atol=1e-6)
            if not ok:
                mismatches[c] = mismatches.get(c, 0) + 1
    assert not mismatches, f"UI features differ from training features: {mismatches}"


def test_every_month_has_a_season():
    assert set(SEASON_OF_MONTH) == set(range(1, 13))