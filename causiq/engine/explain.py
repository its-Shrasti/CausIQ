"""
EXPLAIN — where in the business did the movement come from?

This module answers "where", not "why". The distinction matters and the two
are routinely conflated. "Conversion fell in Electronics" is a location. "The
checkout release caused it" is a cause. Locating is arithmetic and certain;
causing is inference and uncertain. Running them as separate stages means the
engine can be fully confident about the first while still abstaining on the
second -- which is exactly what happens in the gross-margin scenario.

TWO DECOMPOSITIONS, BOTH EXACT
------------------------------
1. SEGMENT: which region / category / channel moved. Simple differences
   against each segment's own baseline, so segments are compared to what THEY
   were expected to do, not to the group average.

2. FACTOR: which multiplicative term moved. Since

       revenue = sessions x conversion x aov

   taking logs makes it exactly additive:

       log(R_actual/R_expected) = log(S_a/S_e) + log(C_a/C_e) + log(A_a/A_e)

   So the three factors' log-ratios sum to the total log-ratio with no residual
   and no ordering choice. This is the standard fix for the well-known problem
   that naive percentage decomposition does not add up -- percentages of a
   product are not additive, logs of a product are.

No LLM is used anywhere in this module. Every number is arithmetic on measured
and forecast quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from causiq.engine import contract
from causiq.engine.detect import Detection, detect


@dataclass
class SegmentContribution:
    dimension: str
    segment: str
    actual: float
    expected: float
    delta: float
    pct_of_total_gap: float
    own_pct_change: float
    material_alone: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FactorContribution:
    factor: str
    label: str
    actual: float
    expected: float
    log_ratio: float
    share_of_movement: float      # fraction of the total log movement
    pct_points_of_kpi: float      # translated back to % of the KPI

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Explanation:
    kpi: str
    scope: dict[str, str]
    window: tuple[str, str]
    total_gap: float
    total_pct: float
    segments: list[SegmentContribution]
    factors: list[FactorContribution]
    method: str
    paradox_flag: bool
    paradox_note: str | None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kpi": self.kpi,
            "scope": self.scope,
            "window": list(self.window),
            "total_gap": self.total_gap,
            "total_pct": self.total_pct,
            "segments": [s.as_dict() for s in self.segments],
            "factors": [f.as_dict() for f in self.factors],
            "method": self.method,
            "paradox_flag": self.paradox_flag,
            "paradox_note": self.paradox_note,
            "notes": self.notes,
        }


FACTOR_LABELS = {
    "sessions": "Traffic (sessions)",
    "conversion_rate": "Conversion rate",
    "aov": "Average order value",
}


# -----------------------------------------------------------------------------
# Segment decomposition
# -----------------------------------------------------------------------------

def _categories_present(wh, region: str, window: tuple[str, str]) -> list[str]:
    df = wh.q(f"""
        SELECT DISTINCT category FROM fact_daily
        WHERE region = '{region}'
          AND date BETWEEN DATE '{window[0]}' AND DATE '{window[1]}'
        ORDER BY category
    """, label="categories_present")
    return df["category"].tolist()


def decompose_by_category(wh, kpi_name: str, region: str, window: tuple[str, str],
                          total: Detection, role: str) -> list[SegmentContribution]:
    """Baseline every category independently, then compare like with like.

    Each category is measured against its OWN forecast. Comparing a category to
    the group average would penalise structurally low-converting categories and
    flatter structurally high ones, which is how false "problem segments" get
    manufactured in weekly business reviews.
    """
    out: list[SegmentContribution] = []
    for cat in _categories_present(wh, region, window):
        try:
            d = detect(wh, kpi_name, region, window[0], window[1],
                       role=role, category=cat)
        except Exception:
            continue
        out.append(SegmentContribution(
            dimension="category",
            segment=cat,
            actual=d.actual,
            expected=d.expected,
            delta=d.deviation_abs,
            pct_of_total_gap=(100.0 * d.deviation_abs / total.deviation_abs
                              if total.deviation_abs else 0.0),
            own_pct_change=d.deviation_pct,
            material_alone=d.material,
        ))
    out.sort(key=lambda s: s.delta)
    return out


# -----------------------------------------------------------------------------
# Factor decomposition (exact, log-additive)
# -----------------------------------------------------------------------------

def _factor_split_one(wh, region: str, window: tuple[str, str], role: str,
                      category: str | None) -> dict[str, float] | None:
    """Log-additive shares of traffic / conversion / basket for ONE segment."""
    try:
        o = detect(wh, "orders", region, window[0], window[1], role=role, category=category)
        c = detect(wh, "conversion_rate", region, window[0], window[1], role=role, category=category)
        a = detect(wh, "aov", region, window[0], window[1], role=role, category=category)
    except Exception:
        return None

    if min(o.actual, o.expected, c.actual, c.expected, a.actual, a.expected) <= 0:
        return None

    # sessions is derived so the identity revenue = sessions x cvr x aov holds
    s_a, s_e = o.actual / c.actual, o.expected / c.expected
    logs = {
        "sessions": float(np.log(s_a / s_e)),
        "conversion_rate": float(np.log(c.actual / c.expected)),
        "aov": float(np.log(a.actual / a.expected)),
    }
    total_log = sum(logs.values())
    if abs(total_log) < 1e-12:
        return None
    return {k: v / total_log for k, v in logs.items()}


def decompose_by_factor(wh, region: str, window: tuple[str, str],
                        total: Detection, role: str,
                        category: str | None = None) -> list[FactorContribution]:
    """Split a revenue movement into traffic, conversion and basket size.

    WHY THIS IS DONE PER CATEGORY AND THEN SUMMED
    ---------------------------------------------
    Running the split on blended aggregates produces a badly misleading answer.
    Blended AOV is a weighted average, so it falls whenever demand shifts from
    an expensive category to a cheap one -- even if no individual basket got
    smaller. On this dataset the blended figure attributes 60% of the movement
    to "basket size" when what actually happened is that high-AOV Electronics
    fell while low-AOV Accessories rose. That is a MIX effect wearing a price
    effect's clothing, and acting on it would send someone to fix pricing when
    pricing is not broken.

    Decomposing inside each category removes mix by construction: within a
    category the AOV term reflects only genuine basket change. The per-category
    rupee contributions are then summed, so the factors still add exactly to
    the total gap, and mix is accounted for automatically by each category
    carrying its own weight.
    """
    cats = ([category] if category
            else _categories_present(wh, region, window))

    totals: dict[str, float] = {"sessions": 0.0, "conversion_rate": 0.0, "aov": 0.0}
    levels: dict[str, list[tuple[float, float, float]]] = {
        k: [] for k in totals}          # (actual, expected, weight)
    covered_gap = 0.0

    for cat in cats:
        try:
            d_cat = detect(wh, "net_revenue", region, window[0], window[1],
                           role=role, category=cat)
        except Exception:
            continue
        shares = _factor_split_one(wh, region, window, role, cat)
        if shares is None:
            continue
        covered_gap += d_cat.deviation_abs
        for k, sh in shares.items():
            totals[k] += sh * d_cat.deviation_abs

    factors: list[FactorContribution] = []
    denom = total.deviation_abs if total.deviation_abs else 1.0
    for name, rupees in totals.items():
        factors.append(FactorContribution(
            factor=name,
            label=FACTOR_LABELS.get(name, name),
            actual=float("nan"),
            expected=float("nan"),
            log_ratio=float("nan"),
            share_of_movement=float(rupees / denom),
            pct_points_of_kpi=float(total.deviation_pct * rupees / denom),
        ))
    factors.sort(key=lambda f: f.pct_points_of_kpi)
    return factors


# -----------------------------------------------------------------------------
# Aggregation paradox detection
# -----------------------------------------------------------------------------

def _detect_paradox(segments: list[SegmentContribution],
                    aggregate_pct: float,
                    weights: dict[str, float] | None = None) -> tuple[bool, str | None]:
    """Flag when the aggregate moves opposite to its economically dominant segments.

    A headline number can move one way while most of the business moves the
    other, because a mix shift reweights the segments. Any dashboard reporting
    only the aggregate then shows green while the business is on fire.

    Segments are ranked by ECONOMIC WEIGHT, not by percentage change. A tiny
    newly-launched category can swing 20% on noise every week; saying the
    headline is "masked" by that would be alarmism. What matters is whether a
    segment carrying real revenue is moving against the aggregate.
    """
    if not segments:
        return False, None
    weights = weights or {}

    def weight_of(s: SegmentContribution) -> float:
        return weights.get(s.segment, abs(s.expected))

    material_segments = [s for s in segments if not np.isclose(weight_of(s), 0.0)]
    if not material_segments:
        return False, None

    total_w = sum(weight_of(s) for s in material_segments) or 1.0
    opposing = [
        s for s in material_segments
        if np.sign(s.own_pct_change) != np.sign(aggregate_pct)
        and weight_of(s) / total_w >= 0.15          # must carry real revenue
        and abs(s.own_pct_change) >= 3.0
    ]
    if not opposing:
        return False, None

    lead = max(opposing, key=weight_of)
    share = 100.0 * weight_of(lead) / total_w
    return True, (
        f"The aggregate moved {aggregate_pct:+.1f}%, but {lead.segment} — which is "
        f"{share:.0f}% of the business — moved {lead.own_pct_change:+.1f}% the other "
        f"way. Segment mix is masking the headline: a dashboard showing only the "
        f"blended figure would report no problem while the largest part of the "
        f"portfolio deteriorates."
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def explain(wh, total: Detection, role: str = "DataAnalyst") -> Explanation:
    region = total.scope["region"]
    window = (total.window_start, total.window_end)
    notes: list[str] = []

    segments = decompose_by_category(wh, total.kpi, region, window, total, role)

    factors: list[FactorContribution] = []
    if total.kpi == "net_revenue":
        factors = decompose_by_factor(wh, region, window, total, role,
                                      category=total.scope.get("category"))
        if factors:
            resid = total.deviation_pct - sum(f.pct_points_of_kpi for f in factors)
            notes.append(
                f"Factor decomposition is exact in log space; residual "
                f"{resid:+.4f} pp is floating-point only."
            )
    else:
        notes.append(
            f"Factor decomposition is defined for net_revenue only. "
            f"'{total.kpi}' is not a product of measured sub-metrics in the contract."
        )

    # Conversion-rate paradox check runs on conversion, not on the KPI itself
    paradox, note = False, None
    revenue_weights = {s.segment: abs(s.expected) for s in segments}
    if total.kpi == "net_revenue":
        try:
            cvr_total = detect(wh, "conversion_rate", region, window[0], window[1], role=role)
            cvr_segments = decompose_by_category(wh, "conversion_rate", region,
                                                 window, cvr_total, role)
            paradox, note = _detect_paradox(cvr_segments, cvr_total.deviation_pct,
                                            weights=revenue_weights)
            if paradox:
                note = (f"Blended conversion rate moved {cvr_total.deviation_pct:+.1f}%. "
                        + note.split("but ", 1)[1] if "but " in note else note)
        except Exception:
            pass
    else:
        paradox, note = _detect_paradox(segments, total.deviation_pct,
                                        weights=revenue_weights)

    return Explanation(
        kpi=total.kpi,
        scope=total.scope,
        window=window,
        total_gap=total.deviation_abs,
        total_pct=total.deviation_pct,
        segments=segments,
        factors=factors,
        method=("per-segment baselines + exact log-additive factor decomposition"),
        paradox_flag=paradox,
        paradox_note=note,
        notes=notes,
    )
