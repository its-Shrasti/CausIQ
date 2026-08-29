"""
Chart layer.

Palette is the validated default from the data-viz reference instance, checked
with the six-check validator before any chart was written:

    categorical  #2a78d6 blue / #eb6834 orange / #1baf7a aqua
                 -> adjacent CVD dE 9.2, normal-vision dE 27.6  (PASS)
    diverging    #2a78d6 blue <-> #e34948 red, gray midpoint
                 -> CVD dE 21.6, normal-vision dE 32.3          (PASS)

Aqua sits below 3:1 against the light surface, so it never carries meaning
alone -- wherever it appears the mark is directly labelled.

Rules followed throughout: one y-axis per chart (never dual), categorical hues
assigned in fixed order and never cycled, colour attached to the entity rather
than its rank, thin marks with recessive gridlines, a legend whenever two or
more series are present, and hover enabled on every plot.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# --- palette ----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#ebeae6"

SERIES_1 = "#2a78d6"      # blue
SERIES_2 = "#eb6834"      # orange
SERIES_3 = "#1baf7a"      # aqua  (labelled wherever used)
NEG = "#e34948"           # diverging red pole
POS = "#2a78d6"           # diverging blue pole
NEUTRAL = "#c9c7c0"

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif",
            size=13, color=INK)


def _base(fig: go.Figure, height: int = 320, ylab: str = "",
          xlab: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=FONT,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color=INK_2)),
        xaxis=dict(showgrid=False, linecolor=GRID, ticks="outside",
                   tickcolor=GRID, tickfont=dict(color=INK_2, size=11),
                   title=dict(text=xlab, font=dict(color=INK_2, size=11))),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                   linecolor="rgba(0,0,0,0)", tickfont=dict(color=INK_2, size=11),
                   title=dict(text=ylab, font=dict(color=INK_2, size=11))),
    )
    return fig


def _fmt(v: float, unit: str) -> str:
    if unit == "INR":
        return f"₹{v/1e7:,.2f} Cr"
    if unit in ("ratio",):
        return f"{v*100:.3f}%"
    return f"{v:,.2f}"


# -----------------------------------------------------------------------------
# DETECT — actual vs expected with the prediction interval
# -----------------------------------------------------------------------------

def baseline_chart(detection: dict[str, Any], unit: str = "INR") -> go.Figure:
    s = detection.get("series") or {}
    if not s:
        return _base(go.Figure(), 260)

    fig = go.Figure()

    # Prediction interval as a band. Drawn first so marks sit above it.
    fig.add_trace(go.Scatter(
        x=s["window_dates"] + s["window_dates"][::-1],
        y=s["window_pi_high"] + s["window_pi_low"][::-1],
        fill="toself", fillcolor="rgba(42,120,214,0.10)",
        line=dict(width=0), hoverinfo="skip",
        name="95% prediction interval", showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=s["history_dates"], y=s["history_values"],
        mode="lines", name="Actual (history)",
        line=dict(color=INK_MUTED, width=2),
        hovertemplate="%{x}<br>%{y:,.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=s["window_dates"], y=s["window_expected"],
        mode="lines+markers", name="Expected",
        line=dict(color=SERIES_1, width=2, dash="dot"),
        marker=dict(size=8, color=SERIES_1),
        hovertemplate="%{x}<br>expected %{y:,.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=s["window_dates"], y=s["window_actual"],
        mode="lines+markers", name="Actual",
        line=dict(color=SERIES_2, width=2),
        marker=dict(size=9, color=SERIES_2,
                    line=dict(width=2, color=SURFACE)),
        hovertemplate="%{x}<br>actual %{y:,.4f}<extra></extra>",
    ))

    if s["window_dates"]:
        fig.add_vline(x=s["window_dates"][0], line=dict(color=INK_MUTED,
                      width=1, dash="dash"))
    return _base(fig, 320, ylab=detection.get("kpi_label", ""))


# -----------------------------------------------------------------------------
# EXPLAIN — contribution waterfall (polarity -> diverging)
# -----------------------------------------------------------------------------

def contribution_waterfall(drivers: list[dict[str, Any]],
                           attribution: dict[str, Any]) -> go.Figure:
    """What hurt, what helped, and what is still unexplained.

    Polarity is the job, so this is a diverging encoding: blue for
    contributions that helped, red for those that hurt, neutral gray for the
    totals and the unexplained remainder. The remainder is drawn as a bar of
    its own rather than folded away, because a decomposition that always
    reaches 100% is hiding something.
    """
    confirmed = [d for d in drivers if d["verdict"] == "CONFIRMED"]
    confirmed.sort(key=lambda d: d.get("revenue_impact_pct", 0))

    # Short labels: the full contract names ("Checkout experience / release
    # health") wrap into three lines at this width and collide with each other.
    SHORT = {
        "checkout_experience": "Checkout<br>release",
        "in_stock_rate": "Stockout",
        "competitor_price": "Competitor<br>price",
        "marketing_spend": "Marketing<br>cut",
        "discount_depth": "Accessories<br>promo",
        "seasonality": "Seasonality",
    }

    labels: list[str] = []
    measures: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    texts: list[str] = []

    for d in confirmed:
        v = d.get("revenue_impact_pct", 0.0)
        labels.append(SHORT.get(d["driver"], d["driver_label"]))
        measures.append("relative")
        values.append(v)
        colors.append(POS if v > 0 else NEG)
        texts.append(f"{v:+.2f}%")

    unexp = attribution.get("unexplained_pct", 0.0)
    unconf = attribution.get("unconfirmed_pct", 0.0)
    if abs(unconf) > 0.01:
        labels.append("Unconfirmed")
        measures.append("relative")
        values.append(unconf)
        colors.append(NEUTRAL)
        texts.append(f"{unconf:+.2f}%")
    labels.append("Unexplained")
    measures.append("relative")
    values.append(unexp)
    colors.append(NEUTRAL)
    texts.append(f"{unexp:+.2f}%")

    labels.append("Actual")
    measures.append("total")
    values.append(0.0)
    colors.append(NEUTRAL)
    texts.append(f"{attribution.get('total_movement_pct', 0):+.2f}%")

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=labels, y=values,
        text=texts, textposition="outside",
        textfont=dict(size=11, color=INK_2),
        connector=dict(line=dict(color=GRID, width=1)),
        decreasing=dict(marker=dict(color=NEG,
                                    line=dict(color=SURFACE, width=2))),
        increasing=dict(marker=dict(color=POS,
                                    line=dict(color=SURFACE, width=2))),
        totals=dict(marker=dict(color=NEUTRAL,
                                line=dict(color=SURFACE, width=2))),
        hovertemplate="%{x}<br>%{y:+.2f}%% of revenue<extra></extra>",
    ))
    fig = _base(fig, 380, ylab="% of expected revenue")
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10, color=INK_2))
    return fig


# -----------------------------------------------------------------------------
# CHALLENGE — parallel trends
# -----------------------------------------------------------------------------

def parallel_trends(plot: dict[str, Any]) -> go.Figure:
    if not plot or not plot.get("dates"):
        return _base(go.Figure(), 260)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot["dates"], y=plot["control"], mode="lines",
        name=f"Control ({plot.get('control_label', '')})",
        line=dict(color=SERIES_1, width=2),
        hovertemplate="%{x}<br>control %{y:+.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=plot["dates"], y=plot["treated"], mode="lines",
        name=f"Treated ({plot.get('treated_label', '')})",
        line=dict(color=SERIES_2, width=2),
        hovertemplate="%{x}<br>treated %{y:+.2f}%<extra></extra>",
    ))
    fig.add_vline(x=plot["treatment_date"],
                  line=dict(color=STATUS["critical"], width=2, dash="dash"))
    fig.add_annotation(x=plot["treatment_date"], y=1, yref="paper",
                       text="release", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=STATUS["critical"]))
    fig = _base(fig, 300, ylab="% vs own pre-period mean")
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    return fig


# -----------------------------------------------------------------------------
# KNOW — confidence components (single series -> sequential, no legend)
# -----------------------------------------------------------------------------

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def confidence_bars(components: list[dict[str, Any]]) -> go.Figure:
    comps = sorted(components, key=lambda c: c["score"])
    names = [c["name"].replace("_", " ") for c in comps]
    scores = [c["score"] for c in comps]
    # Sequential ramp by magnitude: darker = stronger evidence.
    colors = [SEQ[min(int(s * len(SEQ)), len(SEQ) - 1)] for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color=SURFACE, width=2)),
        text=[f"{s:.2f}  ×{c['weight']:.2f}" for s, c in zip(scores, comps)],
        textposition="outside", textfont=dict(size=11, color=INK_2),
        hovertemplate="%{y}<br>score %{x:.3f}<extra></extra>",
        cliponaxis=False,
    ))
    fig = _base(fig, 260, xlab="component score (0–1)")
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_xaxes(range=[0, 1.25], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig


# -----------------------------------------------------------------------------
# Telemetry — where the time went
# -----------------------------------------------------------------------------

def telemetry_split(summary: dict[str, Any]) -> go.Figure:
    det = summary.get("deterministic_ms", 0.0)
    llm = summary.get("llm_ms", 0.0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["latency"], x=[det], orientation="h", name="Deterministic",
        marker=dict(color=SERIES_1, line=dict(color=SURFACE, width=2)),
        hovertemplate="deterministic %{x:.0f} ms<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=["latency"], x=[llm], orientation="h", name="LLM",
        marker=dict(color=SERIES_2, line=dict(color=SURFACE, width=2)),
        hovertemplate="LLM %{x:.0f} ms<extra></extra>",
    ))
    fig = _base(fig, 150, xlab="milliseconds")
    fig.update_layout(barmode="stack", hovermode="closest")
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return fig
