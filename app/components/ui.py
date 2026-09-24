"""Layout helpers: page setup, hero banner, KPI cards, badges, notes, footer."""

import html
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "app" / "assets" / "style.css"


def fmt_money(x) -> str:
    if x is None or x != x:
        return "–"
    a = abs(x)
    if a >= 1e9:
        return f"${x / 1e9:.2f}B"
    if a >= 1e6:
        return f"${x / 1e6:.1f}M"
    if a >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"


def _esc(s) -> str:
    """Avoid Streamlit/Markdown treating $...$ as math while keeping HTML structure intact."""
    return html.escape(str(s)).replace("$", "&#36;")


def setup_page(title: str, icon: str = "🎬") -> None:
    st.set_page_config(page_title=f"{title} · Box Office Forecaster", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    if CSS.exists():
        st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="side-brand">🎬 Box Office<br><span>Forecaster</span></div>',
                    unsafe_allow_html=True)
        st.caption("Pre-release revenue forecasting for film distribution planning.")


def hero(title: str, subtitle: str, emoji: str = "🎬") -> None:
    st.markdown(f'<div class="hero"><div class="hero-emoji">{_esc(emoji)}</div><div>'
                f'<h1>{_esc(title)}</h1><p>{subtitle}</p></div></div>',
                unsafe_allow_html=True)


def kpi_html(label, value, sub="", tone="") -> str:
    return (f'<div class="kpi {tone}"><div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(value)}</div><div class="sub">{_esc(sub)}</div></div>')


def kpi_row(items) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        label, value, sub, tone = (list(it) + ["", ""])[:4]
        col.markdown(kpi_html(label, value, sub, tone), unsafe_allow_html=True)


def badge(text: str, tone: str = "brand") -> str:
    return f'<span class="badge {tone}">{_esc(text)}</span>'


def section(title: str, caption: str = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{_esc(text)}</div>', unsafe_allow_html=True)


def step_card(num, title, text) -> str:
    return (f'<div class="step"><div class="num">{num}</div><div class="step-title">{_esc(title)}</div>'
            f'<div class="step-text">{_esc(text)}</div></div>')


def film_summary(inp: dict) -> str:
    d = pd.Timestamp(inp["release_date"])
    title = _esc(inp.get("title") or "Untitled film")
    genres = _esc(", ".join(inp.get("genres", [])[:3]))
    kind = "Sequel" if inp.get("franchise_mode", "").startswith("Sequel") else "Original"
    return (f'<div class="film-chip">🎬 {title} &nbsp;·&nbsp; {fmt_money(inp["budget"]).replace("$", "&#36;")} &nbsp;·&nbsp; '
            f'{genres} &nbsp;·&nbsp; {kind} &nbsp;·&nbsp; {d:%b %Y}</div>')


def require_artifacts() -> None:
    from services.predictor import artifacts_ready
    if not artifacts_ready():
        st.error("Model files were not found in `artifacts/`. Run the full pipeline first:")
        st.code("python run_pipeline.py", language="bash")
        st.stop()


def footer() -> None:
    st.markdown('<div class="footer">Data: The Movies Dataset (Kaggle, Banik 2017) · '
                'Forecasts are statistical estimates for planning, not guarantees.</div>',
                unsafe_allow_html=True)