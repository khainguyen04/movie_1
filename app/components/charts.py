"""Plotly charts with a consistent cinematic theme."""

import numpy as np
import plotly.graph_objects as go

from components.ui import fmt_money

BRAND, PINK, GOOD, BAD, MUTED, INK = "#6D28D9", "#DB2777", "#059669", "#DC2626", "#94A3B8", "#0F172A"


def _layout(fig, height=380, title=None):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=55 if title else 20, b=10),
                      title=dict(text=title, x=0, font=dict(size=16, color=INK)) if title else None,
                      font=dict(family="Inter, Segoe UI, sans-serif", color=INK),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge(prob: float, threshold: float, title: str = "Chance of $100M+"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"size": 46, "color": INK}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": BRAND, "thickness": 0.3},
               "steps": [{"range": [0, threshold * 100], "color": "#F1F5F9"},
                         {"range": [threshold * 100, 100], "color": "#FCE7F3"}],
               "threshold": {"line": {"color": PINK, "width": 4}, "thickness": 0.85,
                             "value": threshold * 100}}))
    return _layout(fig, 300, title)


def waterfall(baseline: float, steps: list, final: float):
    x = ["Average film"] + [s[0] for s in steps] + ["This film"]
    y = [baseline] + [s[1] for s in steps] + [0]
    text = [fmt_money(baseline)] + [("+" if d >= 0 else "−") + fmt_money(abs(d)) for _, d in steps] \
        + [fmt_money(final)]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute"] + ["relative"] * len(steps) + ["total"],
        x=x, y=y, text=text, textposition="outside",
        increasing={"marker": {"color": GOOD}}, decreasing={"marker": {"color": BAD}},
        totals={"marker": {"color": BRAND}}, connector={"line": {"color": MUTED, "dash": "dot"}}))
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return _layout(fig, 430, "From an average film to this forecast")


def range_bar(low: float, mid: float, high: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[low, high], y=[0, 0], mode="lines",
                             line=dict(color="#C4B5FD", width=22), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[low, mid, high], y=[0, 0, 0], mode="markers+text",
                             marker=dict(size=[14, 24, 14], color=[BRAND, PINK, BRAND]),
                             text=[f"Low<br>{fmt_money(low)}", f"Forecast<br>{fmt_money(mid)}",
                                   f"High<br>{fmt_money(high)}"],
                             textposition="top center", hoverinfo="skip"))
    fig.update_yaxes(visible=False, range=[-1, 1.6])
    fig.update_xaxes(type="log", tickprefix="$", tickformat="~s")
    fig.update_layout(showlegend=False)
    return _layout(fig, 230, "Likely range (8 in 10 similar films land here)")


def budget_curve(budgets, rev_a, current_a, rev_b=None, current_b=None, mult=2.5):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=budgets, y=np.asarray(budgets) * mult, name=f"Break-even ({mult}× budget)",
                             line=dict(color=MUTED, dash="dash")))
    fig.add_trace(go.Scatter(x=budgets, y=rev_a, name="Scenario A", line=dict(color=BRAND, width=3)))
    fig.add_trace(go.Scatter(x=[current_a[0]], y=[current_a[1]], mode="markers", name="A today",
                             marker=dict(size=13, color=BRAND, line=dict(color="white", width=2))))
    if rev_b is not None:
        fig.add_trace(go.Scatter(x=budgets, y=rev_b, name="Scenario B", line=dict(color=PINK, width=3)))
        fig.add_trace(go.Scatter(x=[current_b[0]], y=[current_b[1]], mode="markers", name="B today",
                                 marker=dict(size=13, color=PINK, line=dict(color="white", width=2))))
    fig.update_xaxes(type="log", tickprefix="$", tickformat="~s", title="Production budget")
    fig.update_yaxes(type="log", tickprefix="$", tickformat="~s", title="Forecast box office")
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return _layout(fig, 420, "How the forecast responds to budget")


def prob_curve(budgets, probs, threshold, current=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=budgets, y=np.asarray(probs) * 100, line=dict(color=BRAND, width=3),
                             name="Chance of $100M+"))
    fig.add_hline(y=threshold * 100, line=dict(color=PINK, dash="dash"),
                  annotation_text="Decision threshold", annotation_position="top left")
    if current:
        fig.add_trace(go.Scatter(x=[current[0]], y=[current[1] * 100], mode="markers", name="This film",
                                 marker=dict(size=13, color=PINK, line=dict(color="white", width=2))))
    fig.update_xaxes(type="log", tickprefix="$", tickformat="~s", title="Production budget")
    fig.update_yaxes(title="Probability (%)", range=[0, 100])
    fig.update_layout(showlegend=False)
    return _layout(fig, 380, "Blockbuster chance vs budget")


def month_bars(months, revenues, current_month: int):
    best = int(np.argmax(revenues))
    colors = [PINK if i == best else (BRAND if i == current_month - 1 else "#C7D2FE")
              for i in range(12)]
    fig = go.Figure(go.Bar(x=months, y=revenues, marker_color=colors,
                           text=[fmt_money(v) for v in revenues], textposition="outside"))
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return _layout(fig, 380, "Forecast by release month (pink = best, purple = current)")


def scenario_bars(names, revenues, probs):
    fig = go.Figure(go.Bar(x=names, y=revenues, marker_color=[BRAND, PINK][:len(names)],
                           text=[f"{fmt_money(r)}<br>{p:.0%} chance of $100M+" for r, p in zip(revenues, probs)],
                           textposition="outside"))
    fig.update_yaxes(tickprefix="$", tickformat="~s", range=[0, max(revenues) * 1.3])
    return _layout(fig, 380, "Scenario comparison")


def effects_bar(labels, pct):
    order = np.argsort(np.abs(pct))
    labels, pct = [labels[i] for i in order], [pct[i] for i in order]
    fig = go.Figure(go.Bar(y=labels, x=pct, orientation="h",
                           marker_color=[GOOD if v > 0 else BAD for v in pct],
                           text=[f"{v:+.0f}%" for v in pct], textposition="outside"))
    fig.update_xaxes(ticksuffix="%", title="Effect on expected revenue (all else equal)")
    return _layout(fig, 60 + 38 * len(labels), "What moves box office")