"""
agent_09_encode_scale
=====================
Fit MoviePreprocessor on TRAIN only (no information from the test set leaks in):
  standardisation of numeric features, one-hot of categoricals (top-k language/company),
  interaction log_budget x is_sequel. Apply the same fitted object to test.

Output: data/processed/X_train.csv, X_test.csv, preprocessor.joblib + encoding tables
"""

import os

import joblib
import pandas as pd

from utils.io import log, save_table, get_df, load_json
from utils.features_shared import MoviePreprocessor


def run(state: dict, cfg: dict) -> dict:
    log("agent_09_encode_scale: start")
    p = cfg["paths"]["processed"]
    train = get_df(state, "train", os.path.join(p, "train.csv"))
    test = get_df(state, "test", os.path.join(p, "test.csv"))
    fl = state.get("feature_lists") or load_json(os.path.join(cfg["paths"]["logs"],
                                                              "feature_lists.json"))
    fc = cfg["features"]
    inter = [list(i) for i in fc.get("interactions", [["log_budget", "is_sequel"]])]

    prep = MoviePreprocessor(
        numeric=fl["numeric"], binary=fl["binary"], categorical=fl["categorical"],
        top_k={"original_language": fc["language_top_k"], "lead_company": fc["company_top_k"]},
        interactions=inter)
    prep.fit(train)
    X_train = prep.transform(train).reset_index(drop=True)
    X_test = prep.transform(test).reset_index(drop=True)

    # evidence for the report: train mean ~0 / sd ~1 after standardisation
    chk = pd.DataFrame({
        "feature": fl["numeric"],
        "train_mean_raw": prep.means_.values, "train_sd_raw": prep.stds_.values,
        "train_mean_scaled": X_train[fl["numeric"]].mean().values,
        "train_sd_scaled": X_train[fl["numeric"]].std(ddof=0).values,
        "test_mean_scaled": X_test[fl["numeric"]].mean().values,
    })
    save_table(chk.round(4), "09_standardisation_check.csv", cfg)
    enc = pd.DataFrame([{"variable": c, "reference_level": lv[0], "n_levels_kept": len(lv),
                         "has_other": prep.has_other_[c], "levels": " | ".join(lv)}
                        for c, lv in prep.levels_.items()])
    save_table(enc, "09_encoding_levels.csv", cfg)

    X_train.to_csv(os.path.join(p, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(p, "X_test.csv"), index=False)
    joblib.dump(prep, os.path.join(p, "preprocessor.joblib"))
    log(f"agent_09_encode_scale: done -> {X_train.shape[1]} model columns "
        f"({len(fl['numeric'])} scaled numeric, {len(fl['binary'])} binary, "
        f"{sum(1 for c in X_train.columns if '=' in c)} dummies, {len(inter)} interaction)")
    state.update({"X_train": X_train, "X_test": X_test, "preprocessor": prep})
    return state