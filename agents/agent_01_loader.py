"""
agent_01_loader
===============
Load raw CSVs (movies_metadata, credits, keywords), remove malformed rows, dedupe ids,
parse JSON-string columns, build one row per movie and merge.
Fallback: data/inspect/movies_slim.csv (same schema) if raw files are missing.

Output: state["movies_all"] (all ~45k movies), state["raw_checks"]
        data/interim/movies_parsed.csv, outputs/logs/raw_checks.json
"""

import os

import numpy as np
import pandas as pd

from utils.io import log, ensure_dirs, reset_data_flow, log_data_flow, save_json, load_json
from utils.parsing import parse_literal, names_from

JSON_COLS = {"genres": "name", "production_companies": "name",
             "production_countries": "iso_3166_1", "spoken_languages": "iso_639_1"}


# ----------------------------------------------------------------------------- metadata
def _load_metadata(path: str, checks: dict) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    checks["metadata_raw_rows"] = len(df)
    checks["metadata_exact_duplicates"] = int(df.duplicated().sum())

    id_num = pd.to_numeric(df["id"], errors="coerce")
    checks["metadata_malformed_rows"] = int(id_num.isna().sum())
    df = df.loc[id_num.notna()].copy()
    df["id"] = id_num.loc[id_num.notna()].astype(int)

    checks["metadata_duplicate_ids_removed"] = int(df["id"].duplicated().sum())
    df = df.drop_duplicates("id", keep="first").reset_index(drop=True)

    lists = {c: df[c].map(parse_literal).map(lambda x, k=k: names_from(x, k))
             for c, k in JSON_COLS.items()}
    coll = df["belongs_to_collection"].map(parse_literal)
    rd = pd.to_datetime(df["release_date"], errors="coerce")

    def num(c):
        return pd.to_numeric(df[c], errors="coerce")

    return pd.DataFrame({
        "id": df["id"], "imdb_id": df["imdb_id"], "title": df["title"],
        "release_date": rd.dt.strftime("%Y-%m-%d"), "year": rd.dt.year,
        "month": rd.dt.month, "weekday": rd.dt.day_name(),
        "status": df["status"], "adult": df["adult"], "video": df["video"],
        "original_language": df["original_language"],
        "budget": num("budget"), "revenue": num("revenue"), "runtime": num("runtime"),
        "genres": lists["genres"].map("|".join), "n_genres": lists["genres"].map(len),
        "collection_id": coll.map(lambda x: x.get("id") if isinstance(x, dict) else np.nan),
        "collection_name": coll.map(lambda x: x.get("name") if isinstance(x, dict) else None),
        "production_companies": lists["production_companies"].map(lambda l: "|".join(l[:3])),
        "n_companies": lists["production_companies"].map(len),
        "production_countries": lists["production_countries"].map("|".join),
        "n_countries": lists["production_countries"].map(len),
        "spoken_languages": lists["spoken_languages"].map("|".join),
        "n_spoken_languages": lists["spoken_languages"].map(len),
        "has_homepage": df["homepage"].notna().astype(int),
        "has_tagline": df["tagline"].notna().astype(int),
        "overview_len": df["overview"].fillna("").astype(str).str.len(),
        "POST_popularity": num("popularity"),
        "POST_vote_average": num("vote_average"),
        "POST_vote_count": num("vote_count"),
    }).assign(is_franchise=lambda d: d["collection_id"].notna().astype(int))


# ----------------------------------------------------------------------------- credits
def _credit_features(mid, cast, crew) -> dict:
    cast = cast if isinstance(cast, list) else []
    crew = [d for d in (crew if isinstance(crew, list) else []) if isinstance(d, dict)]
    cast = sorted([d for d in cast if isinstance(d, dict)], key=lambda d: d.get("order", 999))
    directors = [d for d in crew if d.get("job") == "Director"]
    genders = [d.get("gender") for d in cast]
    known_g = [g for g in genders if g in (1, 2)]
    return {
        "id": mid,
        "cast_size": len(cast),
        "crew_size": len(crew),
        "directors": "|".join(d.get("name", "") for d in directors),
        "n_directors": len(directors),
        "director_gender": directors[0].get("gender") if directors else np.nan,
        "top5_cast": "|".join(d.get("name", "") for d in cast[:5]),
        "lead_gender": genders[0] if genders else np.nan,
        "cast_female_share": (sum(g == 1 for g in known_g) / len(known_g)) if known_g else np.nan,
        "writers": "|".join(d.get("name", "") for d in crew
                            if d.get("department") == "Writing")[:300],
        "n_writers": sum(d.get("department") == "Writing" for d in crew),
        "n_producers": sum(d.get("job") in ("Producer", "Executive Producer") for d in crew),
        "composer": next((d.get("name") for d in crew
                          if d.get("job") == "Original Music Composer"), None),
    }


def _load_credits(path: str, checks: dict) -> pd.DataFrame:
    c = pd.read_csv(path)
    c["id"] = pd.to_numeric(c["id"], errors="coerce")
    c = c.dropna(subset=["id"])
    c["id"] = c["id"].astype(int)
    checks["credits_duplicate_ids_removed"] = int(c["id"].duplicated().sum())
    # duplicates differ only in crew -> keep the most complete (longest) crew record
    c["_crew_len"] = c["crew"].astype(str).str.len()
    c = c.sort_values("_crew_len", ascending=False).drop_duplicates("id", keep="first")
    log(f"   parsing cast/crew for {len(c):,} movies ...")
    rows = [_credit_features(mid, parse_literal(ca), parse_literal(cr))
            for mid, ca, cr in zip(c["id"], c["cast"], c["crew"])]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- keywords
def _load_keywords(path: str, checks: dict) -> pd.DataFrame:
    k = pd.read_csv(path)
    k["id"] = pd.to_numeric(k["id"], errors="coerce")
    k = k.dropna(subset=["id"])
    k["id"] = k["id"].astype(int)
    checks["keywords_duplicate_ids_removed"] = int(k["id"].duplicated().sum())
    k = k.drop_duplicates("id", keep="first")
    lists = k["keywords"].map(parse_literal).map(names_from)
    return pd.DataFrame({"id": k["id"].values, "n_keywords": lists.map(len).values,
                         "keywords": lists.map(lambda l: "|".join(l[:15])).values})


# ----------------------------------------------------------------------------- fallback
def _dup_removed(file_info: dict):
    d = file_info.get("duplicates", {})
    if "duplicate_id_rows" in d:
        return d["duplicate_id_rows"] - d["duplicate_ids_unique"]
    return None


def _load_fallback(cfg: dict, checks: dict) -> pd.DataFrame:
    path = cfg["raw_files"]["slim_fallback"]
    if not os.path.exists(path):
        raise FileNotFoundError("No raw files in data/raw and no fallback " + path)
    log(f"raw files not found -> using fallback {path}")
    df = pd.read_csv(path, low_memory=False)
    summ = os.path.join(cfg["paths"]["inspect"], "inspect_summary.json")
    if os.path.exists(summ):
        files = load_json(summ)["files"]
        md = files.get("movies_metadata.csv", {})
        checks.update({
            "metadata_raw_rows": md.get("rows"),
            "metadata_exact_duplicates": md.get("exact_duplicate_rows_after_first"),
            "metadata_malformed_rows": md.get("malformed_rows_non_numeric_id"),
            "metadata_duplicate_ids_removed": _dup_removed(md),
            "credits_duplicate_ids_removed": _dup_removed(files.get("credits.csv", {})),
            "keywords_duplicate_ids_removed": _dup_removed(files.get("keywords.csv", {})),
        })
    return df


# ----------------------------------------------------------------------------- run
def run(state: dict, cfg: dict) -> dict:
    log("agent_01_loader: start")
    ensure_dirs(cfg)
    reset_data_flow(cfg)
    raw, rf = cfg["paths"]["raw"], cfg["raw_files"]
    p_meta, p_cred, p_kw = (os.path.join(raw, rf[k]) for k in ["metadata", "credits", "keywords"])
    checks = {}

    if os.path.exists(p_meta):
        checks["source"] = "raw"
        df = _load_metadata(p_meta, checks)
        log(f"   metadata: {len(df):,} unique movies")
        if os.path.exists(p_cred):
            cred = _load_credits(p_cred, checks)
            df = df.merge(cred, on="id", how="left", indicator="_c")
            df["in_credits"] = (df.pop("_c") == "both").astype(int)
        if os.path.exists(p_kw):
            kw = _load_keywords(p_kw, checks)
            df = df.merge(kw, on="id", how="left", indicator="_k")
            df["in_keywords"] = (df.pop("_k") == "both").astype(int)
    else:
        checks["source"] = "fallback_movies_slim"
        df = _load_fallback(cfg, checks)

    n_raw = checks.get("metadata_raw_rows") or len(df)
    log_data_flow("01 load & dedupe metadata", n_raw, len(df),
                  "3 malformed (shifted-column) rows + duplicate ids removed", cfg)

    df.to_csv(os.path.join(cfg["paths"]["interim"], "movies_parsed.csv"), index=False)
    save_json(checks, os.path.join(cfg["paths"]["logs"], "raw_checks.json"))
    log(f"agent_01_loader: done -> {len(df):,} movies, {df.shape[1]} columns")

    state["movies_all"] = df
    state["raw_checks"] = checks
    return state