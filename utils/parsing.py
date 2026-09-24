"""Parsing helpers for the python-literal JSON strings used in The Movies Dataset."""

import ast

import numpy as np
import pandas as pd

VALID_GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
                "Family", "Fantasy", "Foreign", "History", "Horror", "Music", "Mystery",
                "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"]


def parse_literal(x):
    """'[{'id': 1, 'name': 'Drama'}]' -> python list. Returns None if empty, 'PARSE_ERROR' if broken."""
    if isinstance(x, (list, dict)):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return "PARSE_ERROR"


def names_from(obj, key: str = "name") -> list:
    if isinstance(obj, list):
        return [d.get(key) for d in obj if isinstance(d, dict) and d.get(key) is not None]
    return []


def split_pipe(s) -> list:
    """'Drama|Comedy' -> ['Drama', 'Comedy'];  NaN/'' -> []"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s)
    return [v for v in s.split("|") if v] if s else []


def is_missing_text(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def primary_genre(s) -> str:
    lst = split_pipe(s)
    return lst[0] if lst else "Unknown"