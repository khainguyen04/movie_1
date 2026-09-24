"""Shared I/O helpers: logging, folders, tables, JSON, data-flow log, state loading."""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

FIG_SUBDIRS = ["01_data_quality", "02_descriptive", "03_distributions", "04_bivariate",
               "05_dependence", "06_models", "07_slides"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def ensure_dirs(cfg: dict) -> None:
    p = cfg["paths"]
    for key in ["raw", "interim", "processed", "figures", "tables", "logs", "artifacts"]:
        os.makedirs(p[key], exist_ok=True)
    for sub in FIG_SUBDIRS:
        os.makedirs(os.path.join(p["figures"], sub), exist_ok=True)
    for sub in ["models", "metadata", "lookups"]:
        os.makedirs(os.path.join(p["artifacts"], sub), exist_ok=True)


def save_table(df: pd.DataFrame, name: str, cfg: dict, index: bool = False) -> str:
    path = os.path.join(cfg["paths"]["tables"], name)
    df.to_csv(path, index=index)
    return path


def _to_py(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def save_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_to_py, ensure_ascii=False)
    return path


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reset_data_flow(cfg: dict) -> None:
    path = os.path.join(cfg["paths"]["logs"], "data_flow_log.csv")
    if os.path.exists(path):
        os.remove(path)


def log_data_flow(step: str, n_before: int, n_after: int, reason: str,
                  cfg: dict, action: str = "drop") -> None:
    """Append one line to outputs/logs/data_flow_log.csv (evidence of how few rows were removed)."""
    path = os.path.join(cfg["paths"]["logs"], "data_flow_log.csv")
    row = pd.DataFrame([{"step": step, "action": action, "rows_before": n_before,
                         "rows_after": n_after, "removed": n_before - n_after, "reason": reason}])
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def get_df(state: dict, key: str, path: str) -> pd.DataFrame:
    """Return state[key]; if missing (agent run alone), load it from disk."""
    if key in state and state[key] is not None:
        return state[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{key}' not in state and {path} not found - run previous agents first.")
    log(f"loading {key} from {path}")
    return pd.read_csv(path, low_memory=False)

def load_model_data(state: dict, cfg: dict):
    """X_train, X_test, train, test (from state or data/processed/*.csv)."""
    p = cfg["paths"]["processed"]
    return tuple(get_df(state, k, os.path.join(p, f"{k}.csv"))
                 for k in ["X_train", "X_test", "train", "test"])