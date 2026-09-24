"""
agent_05_visualisation
======================
Distribution plots, bivariate plots (Lecture 1, slides 28-30) and simple slide figures.
  03_distributions : raw vs log histograms, Q-Q plot of log revenue (normality, slide 34)
  04_bivariate     : scatter (continuous), boxplots (categorical), binary-response plot
  02_descriptive   : category counts
  07_slides        : one-message figures for non-technical decision makers
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils.io import log, get_df
from utils.parsing import split_pipe
from utils.plotting import (set_style, slide_style, save_fig, money_axis, money_fmt,
                            PALETTE, MAIN, ACCENT)
from utils.features_shared import MONTHS


def _prep(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    d = df.copy()
    d["log10_revenue"] = np.log10(d["revenue"])
    d["log10_budget"] = np.log10(d["budget"])
    d["segment"] = np.where(d["is_franchise"] == 1, "Franchise", "Standalone")
    d["blockbuster"] = (d["revenue"] >= thr).astype(int)
    d["month_name"] = d["month"].astype(int).map(lambda m: MONTHS[m - 1])
    top_lang = d["original_language"].value_counts().head(8).index
    d["language"] = d["original_language"].where(d["original_language"].isin(top_lang), "Other")
    return d


def _hist_raw_log(s: pd.Series, title: str, fname: str, cfg: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(s, bins=60, color=MAIN)
    money_axis(axes[0], "x")
    axes[0].set_title(f"{title} (raw) - skewness {s.skew():.2f}")
    axes[0].set_ylabel("Number of movies")
    ls = np.log10(s)
    axes[1].hist(ls, bins=60, color=ACCENT)
    axes[1].set_title(f"log10({title}) - skewness {ls.skew():.2f}")
    axes[1].set_xlabel("log10(USD)")
    save_fig(fig, "03_distributions", fname, cfg)


def _box(d, x, y, order, title, fname, cfg, horizontal=False, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        sns.boxplot(data=d, x=y, y=x, order=order, color=MAIN, fliersize=1, ax=ax)
        ax.set_xlabel("log10(revenue USD)")
        ax.set_ylabel("")
    else:
        sns.boxplot(data=d, x=x, y=y, order=order, color=MAIN, fliersize=1, ax=ax)
        ax.set_ylabel("log10(revenue USD)")
        ax.set_xlabel("")
    ax.set_title(title)
    save_fig(fig, "04_bivariate", fname, cfg)


def _slide_bar(labels, values, title, fname, cfg, highlight=None, money=True,
               horizontal=False, fmt=None, figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = [ACCENT if (highlight and l in highlight) else MAIN for l in labels]
    fmt = fmt or (money_fmt if money else (lambda v: f"{v:.0%}"))
    if horizontal:
        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        for i, v in enumerate(values):
            ax.text(v, i, " " + fmt(v), va="center", fontsize=11)
        if money:
            money_axis(ax, "x")
    else:
        ax.bar(labels, values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=11)
        if money:
            money_axis(ax, "y")
    slide_style(ax, title)
    save_fig(fig, "07_slides", fname, cfg)


def run(state: dict, cfg: dict) -> dict:
    log("agent_05_visualisation: start")
    set_style()
    thr = cfg["classification"]["blockbuster_threshold"]
    df = get_df(state, "movies", os.path.join(cfg["paths"]["interim"], "movies_clean.csv"))
    d = _prep(df, thr)
    obs = d[d["budget_missing"] == 0]

    # ---- 03 distributions
    _hist_raw_log(d["revenue"], "Revenue", "revenue_raw_vs_log.png", cfg)
    _hist_raw_log(obs["budget"], "Budget (observed)", "budget_raw_vs_log.png", cfg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(d["runtime"], bins=50, color=MAIN)
    axes[0].set_title(f"Runtime (minutes) - skewness {d['runtime'].skew():.2f}")
    axes[1].hist(d["year"], bins=range(int(d["year"].min()), int(d["year"].max()) + 2), color=MAIN)
    axes[1].set_title("Movies with known revenue by release year")
    save_fig(fig, "03_distributions", "runtime_year_hist.png", cfg)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    stats.probplot(d["revenue"], dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q plot: raw revenue")
    stats.probplot(np.log(d["revenue"]), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot: log(revenue)")
    save_fig(fig, "03_distributions", "qq_revenue_raw_vs_log.png", cfg)

    # ---- 02 category counts
    gl = d["genres"].map(split_pipe).explode()
    gc = gl.value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(gc.index, gc.values, color=MAIN)
    axes[0].invert_yaxis()
    axes[0].set_title("Movies per genre (multi-label)")
    mc = d["month_name"].value_counts().reindex(MONTHS)
    axes[1].bar(mc.index, mc.values, color=MAIN)
    axes[1].set_title("Movies per release month")
    save_fig(fig, "02_descriptive", "category_counts_genre_month.png", cfg)

    # ---- 04 bivariate: scatter
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for seg in ["Standalone", "Franchise"]:
        s = obs[obs["segment"] == seg]
        ax.scatter(s["log10_budget"], s["log10_revenue"], s=7, alpha=0.35,
                   color=PALETTE[seg], label=f"{seg} (n={len(s):,})")
        b1, b0 = np.polyfit(s["log10_budget"], s["log10_revenue"], 1)
        xs = np.linspace(s["log10_budget"].min(), s["log10_budget"].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=PALETTE[seg], lw=2.5, label=f"{seg} fit: slope {b1:.2f}")
    r = np.corrcoef(obs["log10_budget"], obs["log10_revenue"])[0, 1]
    ax.set_xlabel("log10(budget USD)")
    ax.set_ylabel("log10(revenue USD)")
    ax.set_title(f"Budget vs revenue (observed budgets), Pearson r = {r:.2f}")
    ax.legend()
    save_fig(fig, "04_bivariate", "scatter_budget_revenue_by_franchise.png", cfg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.regplot(data=d, x="runtime", y="log10_revenue", lowess=True, ax=axes[0],
                scatter_kws={"s": 5, "alpha": 0.25, "color": MAIN}, line_kws={"color": ACCENT})
    axes[0].set_title("Runtime vs log10 revenue (LOWESS)")
    sns.regplot(data=d, x="year", y="log10_revenue", lowess=True, ax=axes[1],
                scatter_kws={"s": 5, "alpha": 0.25, "color": MAIN}, line_kws={"color": ACCENT})
    axes[1].set_title("Release year vs log10 revenue (LOWESS)")
    save_fig(fig, "04_bivariate", "scatter_runtime_year.png", cfg)

    # ---- 04 bivariate: boxplots
    ex = d.assign(genre=d["genres"].map(split_pipe)).explode("genre")
    order = ex.groupby("genre")["log10_revenue"].median().sort_values(ascending=False).index
    _box(ex, "genre", "log10_revenue", order, "Revenue by genre (multi-label)",
         "box_revenue_by_genre.png", cfg, horizontal=True, figsize=(9, 7))
    _box(d, "month_name", "log10_revenue", MONTHS, "Revenue by release month",
         "box_revenue_by_month.png", cfg)
    _box(d, "segment", "log10_revenue", ["Standalone", "Franchise"],
         "Revenue: standalone vs franchise", "box_revenue_by_franchise.png", cfg, figsize=(6, 5))
    _box(d, "decade", "log10_revenue", sorted(d["decade"].unique()), "Revenue by decade",
         "box_revenue_by_decade.png", cfg)
    lang_order = d.groupby("language")["log10_revenue"].median().sort_values(ascending=False).index
    _box(d, "language", "log10_revenue", lang_order, "Revenue by original language",
         "box_revenue_by_language.png", cfg)
    _box(d, "director_gender_cat", "log10_revenue", ["male", "female", "unknown"],
         "Revenue by director gender", "box_revenue_by_director_gender.png", cfg, figsize=(6, 5))

    # ---- binary response (blockbuster) vs budget decile
    obs_dec = obs.assign(decile=pd.qcut(obs["budget"], 10, labels=False) + 1)
    rate = obs_dec.groupby("decile")["blockbuster"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(rate.index.astype(str), rate.values, color=MAIN)
    ax.set_xlabel("Budget decile (1 = lowest 10% budgets)")
    ax.set_ylabel(f"Share with revenue >= {money_fmt(thr)}")
    ax.set_title("Binary response: blockbuster rate rises with budget")
    save_fig(fig, "04_bivariate", "blockbuster_rate_by_budget_decile.png", cfg)

    # ---- 07 slides (one message each)
    seg_med = d.groupby("segment")["revenue"].median().reindex(["Standalone", "Franchise"])
    _slide_bar(list(seg_med.index), list(seg_med.values),
               "Franchise films earn several times more (median revenue)",
               "slide_franchise_vs_standalone.png", cfg, highlight=["Franchise"], figsize=(7, 5))

    mm = d.groupby("month_name")["revenue"].median().reindex(MONTHS)
    top3 = list(mm.sort_values(ascending=False).head(3).index)
    _slide_bar(MONTHS, list(mm.values), "Release timing matters: median revenue by month",
               "slide_revenue_by_month.png", cfg, highlight=top3, figsize=(11, 5))

    bands = pd.cut(obs["budget"], [0, 1e7, 4e7, 1e8, np.inf],
                   labels=["< $10M", "$10-40M", "$40-100M", "> $100M"])
    bm = obs.groupby(bands, observed=False)["revenue"].median()
    _slide_bar([str(b) for b in bm.index], list(bm.values),
               "Bigger budgets, bigger box office (median revenue)",
               "slide_revenue_by_budget_band.png", cfg, highlight=["> $100M"])

    gm = ex.groupby("genre").agg(n=("revenue", "size"), med=("revenue", "median"))
    gm = gm[gm["n"] >= 100].sort_values("med", ascending=False).head(8)
    _slide_bar(list(gm.index), list(gm["med"].values), "Top genres by median revenue",
               "slide_top_genres.png", cfg, highlight=list(gm.index[:3]), horizontal=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (col, labels, title) in zip(axes, [
            ("blockbuster", ["Not blockbuster", "Blockbuster"], f"Revenue >= {money_fmt(thr)}"),
            ("is_franchise", ["Standalone", "Franchise"], "Franchise membership")]):
        share = d[col].value_counts(normalize=True).reindex([0, 1]).fillna(0).values
        ax.bar(labels, share, color=[MAIN, ACCENT])
        for i, v in enumerate(share):
            ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=13)
        ax.set_ylim(0, 1)
        slide_style(ax, title)
    fig.suptitle("Class imbalance: only about 1 in 5 films is a blockbuster or a franchise film",
                 fontsize=13, x=0.02, ha="left")
    save_fig(fig, "07_slides", "slide_class_imbalance.png", cfg)

    n_figs = sum(len(files) for _, _, files in os.walk(cfg["paths"]["figures"]))
    log(f"agent_05_visualisation: done -> {n_figs} figures in outputs/figures")
    return state