"""
Feature logic shared by the pipeline (agent_06, agent_14) AND the UI (app/services).
One implementation -> no train/serve skew.

History features are time-aware: for each film only films released STRICTLY EARLIER
are used, so nothing unknown before release leaks into the features.
"""

import re

import numpy as np
import pandas as pd

from utils.parsing import VALID_GENRES, split_pipe

SEASON_OF_MONTH = {1: "Jan-Apr", 2: "Jan-Apr", 3: "Jan-Apr", 4: "Jan-Apr",
                   5: "Summer (May-Aug)", 6: "Summer (May-Aug)", 7: "Summer (May-Aug)",
                   8: "Summer (May-Aug)", 9: "Sep-Oct", 10: "Sep-Oct",
                   11: "Holiday (Nov-Dec)", 12: "Holiday (Nov-Dec)"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def genre_col(g: str) -> str:
    return "g_" + re.sub(r"\W+", "_", g)


GENRE_COLS = [genre_col(g) for g in VALID_GENRES]

FEATURE_LABELS = {  # plain-English names for slides and the UI
    "log_budget": "Production budget", "runtime": "Runtime", "year": "Release year",
    "log_cast_size": "Cast size", "log_crew_size": "Crew size", "n_keywords": "Number of keywords",
    "n_companies": "Number of production companies", "n_countries": "Number of countries",
    "n_genres": "Number of genres", "n_spoken_languages": "Number of spoken languages",
    "cast_female_share": "Share of female cast",
    "director_prior_films_log": "Director experience (prior films)",
    "director_prior_mean_log_rev": "Director track record (past revenue)",
    "cast_prior_films_log": "Lead cast experience", "cast_prior_mean_log_rev": "Lead cast track record",
    "cast_prior_max_log_rev": "Biggest past hit of lead cast",
    "collection_prior_films": "Number of earlier franchise films",
    "collection_prior_mean_log_rev": "Earlier franchise films' revenue",
    "is_sequel": "Sequel / franchise follow-up", "budget_missing": "Budget not reported",
    "runtime_imputed": "Runtime not reported", "director_has_history": "Director has track record",
    "cast_has_history": "Lead cast has track record", "collection_has_rev_history": "Franchise has revenue history",
    "lang_en": "English-language film", "us_production": "US production", "is_friday": "Friday release",
    "release_season": "Release season", "original_language": "Original language",
    "director_gender_cat": "Director gender", "lead_gender_cat": "Lead actor gender",
    "lead_company": "Lead production company",
}
FEATURE_LABELS.update({genre_col(g): f"Genre: {g}" for g in VALID_GENRES})


def first_name(s):
    lst = split_pipe(s)
    return lst[0] if lst and lst[0] != "Unknown" else None


# ----------------------------------------------------------------------------- simple features
def genre_dummies(genres: pd.Series) -> pd.DataFrame:
    lists = genres.map(split_pipe)
    return pd.DataFrame({genre_col(g): lists.map(lambda l, g=g: int(g in l)) for g in VALID_GENRES},
                        index=genres.index)


def timing_features(df: pd.DataFrame) -> pd.DataFrame:
    m = df["month"].astype(int)
    return pd.DataFrame({
        "release_season": m.map(SEASON_OF_MONTH),
        "is_summer": m.between(5, 8).astype(int),
        "is_holiday": m.isin([11, 12]).astype(int),
        "is_friday": (df["weekday"] == "Friday").astype(int),
        "decade": (df["year"] // 10 * 10).astype(int),
    }, index=df.index)


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "log_budget": np.log(df["budget"]),
        "log_cast_size": np.log1p(df["cast_size"]),
        "log_crew_size": np.log1p(df["crew_size"]),
        "lang_en": (df["original_language"] == "en").astype(int),
        "us_production": df["production_countries"].fillna("").astype(str)
                           .str.split("|").map(lambda l: int("US" in l)),
        "lead_company": df["production_companies"].map(lambda s: (split_pipe(s) or ["Unknown"])[0]),
    }, index=df.index)


# ----------------------------------------------------------------------------- history features
def prior_stats(events: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    events: one row per (film, person/collection) with columns id, date, log_rev (NaN if unknown).
    Returns per row: prior_n (earlier films), prior_k (earlier films with known revenue),
    prior_mean (mean log revenue of those earlier films). Same-date films are NOT counted.
    """
    ev = events.dropna(subset=[key])
    g = (ev.groupby([key, "date"])
           .agg(n=("id", "size"), s=("log_rev", "sum"), k=("log_rev", "count"))
           .reset_index().sort_values([key, "date"]))
    for col in ["n", "s", "k"]:
        g["prior_" + col] = g.groupby(key)[col].cumsum() - g[col]
    g["prior_mean"] = g["prior_s"] / g["prior_k"].where(g["prior_k"] > 0)
    return ev[["id", key, "date"]].merge(g[[key, "date", "prior_n", "prior_k", "prior_mean"]],
                                         on=[key, "date"], how="left")


def _events(all_df: pd.DataFrame, log_rev_by_id: pd.Series) -> pd.DataFrame:
    ev = all_df[["id", "release_date", "directors", "top5_cast", "collection_id"]].copy()
    ev["date"] = pd.to_datetime(ev["release_date"], errors="coerce")
    ev = ev.dropna(subset=["date"])
    ev["log_rev"] = ev["id"].map(log_rev_by_id)
    return ev


def history_features(all_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Director, top-3 cast and franchise history for every film in clean_df."""
    log_rev = clean_df.drop_duplicates("id").set_index("id")["log_revenue"]
    ev = _events(all_df, log_rev)

    ev["director"] = ev["directors"].map(first_name)
    d = prior_stats(ev, "director").rename(columns={
        "prior_n": "director_prior_films", "prior_k": "director_prior_rev_films",
        "prior_mean": "director_prior_mean_log_rev"})
    d = d[["id", "director_prior_films", "director_prior_rev_films", "director_prior_mean_log_rev"]]

    cast = ev[["id", "date", "log_rev", "top5_cast"]].copy()
    cast["actor"] = cast["top5_cast"].map(lambda s: [a for a in split_pipe(s)[:3] if a != "Unknown"])
    cast = cast.explode("actor").dropna(subset=["actor"])
    cs = prior_stats(cast, "actor")
    c = cs.groupby("id").agg(cast_prior_films=("prior_n", "mean"),
                             cast_prior_rev_films=("prior_k", "sum"),
                             cast_prior_mean_log_rev=("prior_mean", "mean"),
                             cast_prior_max_log_rev=("prior_mean", "max")).reset_index()

    col = prior_stats(ev, "collection_id").rename(columns={
        "prior_n": "collection_prior_films", "prior_k": "collection_prior_rev_films",
        "prior_mean": "collection_prior_mean_log_rev"})
    col = col[["id", "collection_prior_films", "collection_prior_rev_films",
               "collection_prior_mean_log_rev"]]

    return (clean_df[["id"]].merge(d, on="id", how="left")
            .merge(c, on="id", how="left").merge(col, on="id", how="left"))


# ----------------------------------------------------------------------------- preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


def pretty_name(term: str) -> str:
    """Model term -> plain English (for tables, slides and the UI)."""
    if term == "const":
        return "Intercept"
    if ":" in term:
        a, b = term.split(":", 1)
        return f"{pretty_name(a)} x {pretty_name(b)}"
    if "=" in term:
        a, b = term.split("=", 1)
        return f"{FEATURE_LABELS.get(a, a)}: {b}"
    return FEATURE_LABELS.get(term, term)


class MoviePreprocessor(BaseEstimator, TransformerMixin):
    """
    Fitted on TRAIN only, then applied unchanged to test data and to the UI input.
      numeric     -> standardised (x - mean_train) / sd_train       (Lecture 1, slide 31)
      binary      -> passed through as 0/1
      categorical -> one-hot, most frequent level = reference (dropped), optional top-k with
                     'Other' (Lecture 1, slide 35); unseen levels -> 'Other' or reference
      interactions-> product of two transformed columns, e.g. log_budget:is_sequel
    """

    def __init__(self, numeric=None, binary=None, categorical=None, top_k=None,
                 interactions=None):
        self.numeric = numeric
        self.binary = binary
        self.categorical = categorical
        self.top_k = top_k
        self.interactions = interactions

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        num = list(self.numeric or [])
        self.means_ = X[num].astype(float).mean()
        sd = X[num].astype(float).std(ddof=0)
        self.stds_ = sd.where(sd > 0, 1.0)
        self.levels_, self.has_other_ = {}, {}
        top_k = self.top_k or {}
        for c in (self.categorical or []):
            vc = X[c].astype(str).value_counts()
            k = top_k.get(c)
            self.levels_[c] = list(vc.index[:k]) if k else list(vc.index)
            self.has_other_[c] = bool(k) and len(vc) > k
        self.feature_names_out_ = list(self._transform(X.head(5)).columns)
        return self

    def _transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X)
        num, binr = list(self.numeric or []), list(self.binary or [])
        parts = [(X[num].astype(float) - self.means_) / self.stds_, X[binr].astype(float)]
        cat_cols = {}
        for c in (self.categorical or []):
            keep = self.levels_[c]
            v = X[c].astype(str)
            v = v.where(v.isin(keep), "Other")
            for lev in keep[1:] + (["Other"] if self.has_other_[c] else []):
                cat_cols[f"{c}={lev}"] = (v == lev).astype(float)
        parts.append(pd.DataFrame(cat_cols, index=X.index))
        out = pd.concat(parts, axis=1)
        for a, b in (self.interactions or []):
            out[f"{a}:{b}"] = out[a] * out[b]
        return out

    def transform(self, X) -> pd.DataFrame:
        return self._transform(X)[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

# ----------------------------------------------------------------------------- deployment helpers
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, GammaRegressor

FRANCHISE_MODES = ["Original film", "Sequel – franchise in database", "Sequel – enter details"]
HISTORY_KEYS = ["director_prior_films", "director_prior_rev_films", "director_prior_mean_log_rev",
                "cast_prior_films", "cast_prior_rev_films", "cast_prior_mean_log_rev",
                "cast_prior_max_log_rev", "collection_prior_films", "collection_prior_rev_films",
                "collection_prior_mean_log_rev"]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep the model's columns in a fixed order (missing ones -> 0). Stateless."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = list(self.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(X).reindex(columns=list(self.columns), fill_value=0.0)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)

    def __sklearn_is_fitted__(self):
        return True


class RevenueRegressor(BaseEstimator, RegressorMixin):
    """
    kind='ols': LinearRegression on log(revenue), back-transform with Duan smearing.
    kind='glm': GammaRegressor (log link) on revenue in USD millions.
    predict() returns USD. Both are linear on the log scale -> contributions() explains a forecast.
    """

    def __init__(self, kind: str = "ols"):
        self.kind = kind

    def fit(self, X, y):
        X = pd.DataFrame(X).astype(float)
        y = np.asarray(y, dtype=float)
        self.columns_ = list(X.columns)
        self.x_mean_ = X.mean().values
        if self.kind == "glm":
            m = GammaRegressor(alpha=0.0, max_iter=10000).fit(X.values, y / 1e6)
            self.coef_, self.intercept_ = m.coef_, float(m.intercept_ + np.log(1e6))
            self.smearing_ = 1.0
        else:
            m = LinearRegression().fit(X.values, np.log(y))
            self.coef_, self.intercept_ = m.coef_, float(m.intercept_)
            resid = np.log(y) - (self.intercept_ + X.values @ self.coef_)
            self.smearing_ = float(np.mean(np.exp(resid)))
        return self

    def predict_log(self, X):
        X = pd.DataFrame(X)[self.columns_].astype(float)
        return self.intercept_ + X.values @ self.coef_

    def predict(self, X):
        return np.exp(self.predict_log(X)) * self.smearing_

    def baseline_log(self) -> float:
        """Log prediction for an 'average film' (all columns at their training mean)."""
        return float(self.intercept_ + self.x_mean_ @ self.coef_)

    def contributions(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X)[self.columns_].astype(float)
        return pd.DataFrame((X.values - self.x_mean_) * self.coef_, columns=self.columns_,
                            index=X.index)


def contribution_group(col: str) -> str:
    """Group model columns into business-friendly factor groups for the UI."""
    base = col.split("=")[0]
    if base in ("log_budget", "budget_missing"):
        return "Budget"
    if col.startswith(("collection_", "is_sequel", "log_budget:is_sequel")):
        return "Franchise"
    if base.startswith("director_"):
        return "Director"
    if base.startswith("cast_") or base in ("lead_gender_cat", "log_cast_size"):
        return "Cast"
    if base.startswith("g_") or base == "n_genres":
        return "Genre"
    if base in ("release_season", "is_friday", "year"):
        return "Release timing"
    if base in ("original_language", "lang_en", "lead_company", "us_production", "n_companies",
                "n_countries", "n_spoken_languages"):
        return "Language & production"
    return "Other details"


def derive_history_features(h: dict) -> dict:
    """Same derivations as agent_06 (flags + log counts) from raw history values."""
    return {
        "director_prior_films_log": float(np.log1p(h["director_prior_films"])),
        "director_prior_mean_log_rev": float(h["director_prior_mean_log_rev"]),
        "director_has_history": int(h["director_prior_rev_films"] > 0),
        "cast_prior_films_log": float(np.log1p(h["cast_prior_films"])),
        "cast_prior_mean_log_rev": float(h["cast_prior_mean_log_rev"]),
        "cast_prior_max_log_rev": float(h["cast_prior_max_log_rev"]),
        "cast_has_history": int(h["cast_prior_rev_films"] > 0),
        "collection_prior_films": float(h["collection_prior_films"]),
        "collection_prior_mean_log_rev": float(h["collection_prior_mean_log_rev"]),
        "collection_has_rev_history": int(h["collection_prior_rev_films"] > 0),
        "is_sequel": int(h["collection_prior_films"] > 0),
    }


def history_from_lookups(director, cast, collection, lookups: dict, fill: float,
                         manual: dict = None) -> dict:
    """For a NEW film every film in the data is 'earlier' -> use full-history lookups."""
    h = {}
    d = lookups["director"].get(director) if director else None
    h["director_prior_films"] = float(d["films"]) if d else 0.0
    h["director_prior_rev_films"] = float(d["rev_films"]) if d else 0.0
    h["director_prior_mean_log_rev"] = float(d["mean_log_rev"]) if d and d["rev_films"] > 0 else fill

    stats_ = [lookups["actor"].get(a) for a in (cast or [])[:3]]
    films = [s["films"] if s else 0 for s in stats_]
    means = [s["mean_log_rev"] for s in stats_ if s and s["rev_films"] > 0]
    h["cast_prior_films"] = float(np.mean(films)) if films else 0.0
    h["cast_prior_rev_films"] = float(sum(s["rev_films"] for s in stats_ if s))
    h["cast_prior_mean_log_rev"] = float(np.mean(means)) if means else fill
    h["cast_prior_max_log_rev"] = float(np.max(means)) if means else fill

    if manual and manual.get("n_prior", 0) > 0:
        mr = manual.get("mean_revenue")
        h["collection_prior_films"] = float(manual["n_prior"])
        h["collection_prior_rev_films"] = float(manual["n_prior"]) if mr else 0.0
        h["collection_prior_mean_log_rev"] = float(np.log(mr)) if mr else fill
    else:
        c = lookups["collection"].get(collection) if collection else None
        h["collection_prior_films"] = float(c["films"]) if c else 0.0
        h["collection_prior_rev_films"] = float(c["rev_films"]) if c else 0.0
        h["collection_prior_mean_log_rev"] = (float(c["mean_log_rev"])
                                              if c and c["rev_films"] > 0 else fill)
    return h


def build_feature_row(inp: dict, hist: dict) -> pd.DataFrame:
    """UI input + history -> one raw row with every column MoviePreprocessor expects."""
    date = pd.Timestamp(inp["release_date"])
    genres = list(inp.get("genres") or [])
    row = {
        "log_budget": float(np.log(float(inp["budget"]))),
        "runtime": float(inp["runtime"]), "year": int(date.year),
        "log_cast_size": float(np.log1p(inp["cast_size"])),
        "log_crew_size": float(np.log1p(inp["crew_size"])),
        "n_keywords": float(inp["n_keywords"]), "n_companies": float(inp["n_companies"]),
        "n_countries": float(inp["n_countries"]), "n_genres": len(genres),
        "n_spoken_languages": float(inp["n_spoken_languages"]),
        "cast_female_share": float(inp["cast_female_share"]),
        "budget_missing": int(inp.get("budget_missing", 0)),
        "runtime_imputed": int(inp.get("runtime_imputed", 0)),
        "lang_en": int(inp["original_language"] == "en"),
        "us_production": int(bool(inp["us_production"])),
        "is_friday": int(date.day_name() == "Friday"),
        "release_season": SEASON_OF_MONTH[int(date.month)],
        "original_language": str(inp["original_language"]),
        "director_gender_cat": inp.get("director_gender_cat", "unknown"),
        "lead_gender_cat": inp.get("lead_gender_cat", "unknown"),
        "lead_company": inp.get("lead_company", "Other"),
    }
    row.update(derive_history_features(hist))
    for g in VALID_GENRES:
        row[genre_col(g)] = int(g in genres)
    return pd.DataFrame([row])


def build_lookups(all_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Full-history stats per director / top-3 actor / franchise (for the UI)."""
    log_rev = clean_df.drop_duplicates("id").set_index("id")["log_revenue"]
    ev = _events(all_df, log_rev)

    def agg(frame, key):
        g = (frame.dropna(subset=[key]).groupby(key)
             .agg(films=("id", "size"), rev_films=("log_rev", "count"),
                  mean_log_rev=("log_rev", "mean"), last_date=("date", "max")).reset_index())
        g["last_year"] = g["last_date"].dt.year
        return g.drop(columns="last_date")

    ev["director"] = ev["directors"].map(first_name)
    director = agg(ev, "director").rename(columns={"director": "name"})

    cast = ev[["id", "date", "log_rev", "top5_cast"]].copy()
    cast["actor"] = cast["top5_cast"].map(lambda s: [a for a in split_pipe(s)[:3] if a != "Unknown"])
    cast = cast.explode("actor").dropna(subset=["actor"])
    actor = agg(cast, "actor").rename(columns={"actor": "name"})
    actor = actor[(actor["films"] >= 2) | (actor["rev_films"] >= 1)]

    coll = agg(ev, "collection_id")
    names = (all_df.dropna(subset=["collection_id"]).drop_duplicates("collection_id")
             .set_index("collection_id")["collection_name"])
    coll["name"] = coll["collection_id"].map(names)
    coll = (coll.dropna(subset=["name"]).sort_values("films", ascending=False)
            .drop_duplicates("name"))
    return {"director": director, "actor": actor, "collection": coll}


def lookups_to_dict(dfs: dict) -> dict:
    return {k: df.set_index("name")[["films", "rev_films", "mean_log_rev"]].to_dict("index")
            for k, df in dfs.items()}    