"""
KNOW — how much should anyone trust this, and when should the engine refuse?

Confidence here is a DETERMINISTIC function of five measurable properties of
the evidence. It is not a number an LLM produces, and it is not a softmax
probability from a classifier. Each component is separately inspectable, so a
low score can always be explained as "which part was weak", and the weights
live in the contract where a governance reviewer can change them without
touching code.

    freshness       were the sources this conclusion needs inside their SLA?
    coverage        what share of required rows actually arrived?
    power           how large is the effect relative to its standard error?
    refutation      did the causal claim survive falsification?
    corroboration   does the unstructured record agree?

ABSTENTION IS A FEATURE, NOT A FAILURE
--------------------------------------
Most analytics systems always produce an answer, because producing one is what
they are for. That is precisely how confident nonsense reaches a decision
meeting. This engine refuses when any of the contract's abstention rules fire,
and refusing well means three things:

  1. Say WHICH rule fired and by how much.
  2. Present the surviving hypotheses WITHOUT ranking them, so the reader is
     not nudged toward whichever happens to be marginally ahead.
  3. Say what evidence would resolve it, ordered by how much uncertainty each
     item would remove per unit of effort -- Value of Information.

An abstention that names the missing evidence is more useful to an analyst
than a hedged answer, because it converts "I don't know" into a work item.

No LLM is involved in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causiq.engine import contract


# -----------------------------------------------------------------------------
# Components
# -----------------------------------------------------------------------------

@dataclass
class ConfidenceComponent:
    name: str
    score: float           # 0..1
    weight: float
    detail: str

    @property
    def contribution(self) -> float:
        return self.score * self.weight

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 4),
                "weight": self.weight, "contribution": round(self.contribution, 4),
                "detail": self.detail}


@dataclass
class ConfidenceResult:
    score: float
    tier: str
    action: str
    components: list[ConfidenceComponent]
    abstained: bool
    triggered_rules: list[str]
    contradictions: list[str]
    evidence_gaps: list[dict[str, Any]]
    next_best_evidence: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "tier": self.tier,
            "action": self.action,
            "components": [c.as_dict() for c in self.components],
            "abstained": self.abstained,
            "triggered_rules": self.triggered_rules,
            "contradictions": self.contradictions,
            "evidence_gaps": self.evidence_gaps,
            "next_best_evidence": self.next_best_evidence,
        }


# -----------------------------------------------------------------------------
# Component scoring
# -----------------------------------------------------------------------------

def _freshness_score(wh, required_sources: list[str]) -> ConfidenceComponent:
    """Penalise by how far past SLA a source is, not merely whether it breached.

    A feed four hours late and one ten days late are not the same risk, and a
    binary in-SLA flag treats them identically.
    """
    ratios, details = [], []
    for s in required_sources:
        f = wh.freshness.get(s)
        if not f:
            continue
        ratio = f.age_hours / f.sla_hours if f.sla_hours else 1.0
        ratios.append(min(ratio, 4.0))
        if ratio > 1.0:
            details.append(f"{s} is {f.age_hours:.0f}h old against a "
                           f"{f.sla_hours:.0f}h SLA ({ratio:.1f}x)")
    if not ratios:
        return ConfidenceComponent("source_freshness", 1.0, 0.0, "no sources declared")

    worst = max(ratios)
    score = 1.0 if worst <= 1.0 else float(max(0.0, 1.0 - (worst - 1.0) / 1.5))
    return ConfidenceComponent(
        "source_freshness", score, 0.0,
        "; ".join(details) if details else "all required sources within SLA",
    )


def _coverage_score(coverage: dict[str, float]) -> ConfidenceComponent:
    if not coverage:
        return ConfidenceComponent("data_coverage", 1.0, 0.0, "no coverage measured")
    worst_col = min(coverage, key=lambda k: coverage[k])
    worst = coverage[worst_col]
    gaps = [f"{k} at {v:.1%}" for k, v in sorted(coverage.items(), key=lambda x: x[1])
            if v < 0.99]
    return ConfidenceComponent(
        "data_coverage", float(worst), 0.0,
        ("; ".join(gaps) if gaps else "all required columns fully populated"),
    )


def _power_score(estimates: list) -> ConfidenceComponent:
    """Effect size relative to its own standard error, averaged over drivers."""
    ratios = []
    for e in estimates:
        if e.std_error and np.isfinite(e.std_error) and e.std_error > 0:
            ratios.append(abs(e.effect_abs) / e.std_error)
    if not ratios:
        return ConfidenceComponent("statistical_power", 0.0, 0.0,
                                   "no estimate could be fitted")
    mean_t = float(np.mean(ratios))
    # t of 2 is the conventional bar; saturate around 5 where extra precision
    # stops changing any decision.
    score = float(np.clip((mean_t - 1.0) / 4.0, 0.0, 1.0))
    return ConfidenceComponent(
        "statistical_power", score, 0.0,
        f"mean |effect|/SE across {len(ratios)} driver(s) = {mean_t:.2f}",
    )


def _refutation_score(estimates: list) -> ConfidenceComponent:
    total = passed = 0
    unrefuted = 0
    for e in estimates:
        if e.refutations:
            total += len(e.refutations)
            passed += sum(1 for r in e.refutations if r.passed)
        else:
            unrefuted += 1
    if total == 0:
        # No falsification was possible: observational estimates only. That is
        # weaker evidence and is scored as such rather than as a free pass.
        return ConfidenceComponent(
            "refutation_survival", 0.55, 0.0,
            f"{unrefuted} estimate(s) observational; no falsification test available",
        )
    frac = passed / total
    return ConfidenceComponent(
        "refutation_survival", float(frac), 0.0,
        f"{passed}/{total} refutation tests passed",
    )


def _corroboration_score(docs_by_driver: dict[str, list]) -> ConfidenceComponent:
    if not docs_by_driver:
        return ConfidenceComponent("context_corroboration", 0.5, 0.0,
                                   "no context retrieved")
    with_docs = sum(1 for v in docs_by_driver.values() if v)
    frac = with_docs / len(docs_by_driver)
    cites = [d.doc_id for v in docs_by_driver.values() for d in v[:1]]
    return ConfidenceComponent(
        "context_corroboration", float(frac), 0.0,
        f"{with_docs}/{len(docs_by_driver)} drivers corroborated"
        + (f" ({', '.join(cites)})" if cites else ""),
    )


# -----------------------------------------------------------------------------
# Contradiction detection
# -----------------------------------------------------------------------------

def _find_contradictions(estimates: list, detection) -> list[str]:
    """Flag evidence that points in incompatible directions."""
    out: list[str] = []

    confirmed = [e for e in estimates if e.verdict == "CONFIRMED"]
    if confirmed and detection is not None:
        same_sign = [e for e in confirmed
                     if np.sign(e.effect_pct) == np.sign(detection.deviation_pct)]
        opposite = [e for e in confirmed if e not in same_sign]
        if opposite and not same_sign:
            out.append(
                f"Every confirmed driver moves {detection.kpi} in the OPPOSITE "
                f"direction to the observed change. Something material is "
                f"missing from the driver set."
            )

    # Two drivers of similar magnitude and opposite sign: the net is a small
    # difference between two large, separately uncertain quantities.
    for i, a in enumerate(confirmed):
        for b in confirmed[i + 1:]:
            if np.sign(a.effect_pct) != np.sign(b.effect_pct):
                if min(abs(a.effect_pct), abs(b.effect_pct)) > 0.6 * max(
                        abs(a.effect_pct), abs(b.effect_pct)):
                    out.append(
                        f"{a.driver_label} ({a.effect_pct:+.1f}%) and "
                        f"{b.driver_label} ({b.effect_pct:+.1f}%) are comparable in "
                        f"size and opposite in sign; the net effect is a small "
                        f"difference between two uncertain quantities."
                    )
    return out


# -----------------------------------------------------------------------------
# Value of Information
# -----------------------------------------------------------------------------

def _value_of_information(components: list[ConfidenceComponent],
                          wh, required_sources: list[str],
                          coverage: dict[str, float]) -> list[dict[str, Any]]:
    """Rank the evidence that would most reduce uncertainty per unit of effort.

    Uplift is estimated as the weighted headroom on each component: closing a
    gap on a heavily weighted, currently-weak component is worth more than
    perfecting one that is already strong. Ranking by uplift ALONE would keep
    recommending the most expensive possible action, so effort is divided out.
    """
    items: list[dict[str, Any]] = []
    by_name = {c.name: c for c in components}

    for s in required_sources:
        f = wh.freshness.get(s)
        if f and not f.within_sla:
            c = by_name.get("source_freshness")
            uplift = (1.0 - c.score) * c.weight if c else 0.0
            items.append({
                "action": f"Re-run the {s} load and re-analyse",
                "why": (f"{s} is {f.age_hours:.0f}h old against a "
                        f"{f.sla_hours:.0f}h SLA; conclusions leaning on it are "
                        f"based on stale state"),
                "owner": "Data Platform",
                "effort_hours": 2,
                "expected_confidence_uplift": round(uplift, 3),
                "value_per_hour": round(uplift / 2, 4),
            })

    for col, cov in sorted(coverage.items(), key=lambda x: x[1]):
        if cov < 0.95:
            c = by_name.get("data_coverage")
            uplift = (1.0 - c.score) * c.weight if c else 0.0
            items.append({
                "action": f"Backfill the missing {col} rows",
                "why": f"only {cov:.1%} of required {col} values are present in scope",
                "owner": "Data Engineering",
                "effort_hours": 4,
                "expected_confidence_uplift": round(uplift, 3),
                "value_per_hour": round(uplift / 4, 4),
            })

    power = by_name.get("statistical_power")
    if power and power.score < 0.6:
        uplift = (1.0 - power.score) * power.weight
        items.append({
            "action": "Extend the observation window by 7 days and re-estimate",
            "why": (f"effects are small relative to their standard errors "
                    f"({power.detail}); more observations would separate signal "
                    f"from noise"),
            "owner": "Analytics",
            "effort_hours": 168,
            "expected_confidence_uplift": round(uplift, 3),
            "value_per_hour": round(uplift / 168, 5),
        })

    refute = by_name.get("refutation_survival")
    if refute and refute.score < 0.7:
        uplift = (1.0 - refute.score) * refute.weight
        items.append({
            "action": "Run a controlled holdout or staged rollout on the leading driver",
            "why": "no natural experiment exists, so the causal claim rests on "
                   "an untestable adjustment-set assumption",
            "owner": "Analytics + Product",
            "effort_hours": 40,
            "expected_confidence_uplift": round(uplift, 3),
            "value_per_hour": round(uplift / 40, 5),
        })

    items.sort(key=lambda x: x["value_per_hour"], reverse=True)
    return items


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def score(wh, detection, estimates: list, required_sources: list[str],
          coverage: dict[str, float],
          docs_by_driver: dict[str, list] | None = None) -> ConfidenceResult:
    cfg = contract.confidence_config()
    weights = cfg["weights"]

    components = [
        _freshness_score(wh, required_sources),
        _coverage_score(coverage),
        _power_score(estimates),
        _refutation_score(estimates),
        _corroboration_score(docs_by_driver or {}),
    ]
    for c in components:
        c.weight = float(weights.get(c.name, 0.0))

    total = float(sum(c.contribution for c in components))

    tier, action = "low", "abstain"
    for name in ("high", "medium", "low"):
        t = cfg["tiers"][name]
        if total >= float(t["min"]):
            tier, action = name, t["action"]
            break

    contradictions = _find_contradictions(estimates, detection)

    # --- abstention rules, evaluated literally from the contract ------------
    triggered: list[str] = []
    by_name = {c.name: c for c in components}

    if total < 0.60:
        triggered.append(f"confidence {total:.2f} is below the 0.60 gate")

    confirmed = sorted([e for e in estimates if e.verdict == "CONFIRMED"],
                       key=lambda e: -abs(e.effect_pct))
    if len(confirmed) >= 2:
        a, b = abs(confirmed[0].effect_pct), abs(confirmed[1].effect_pct)
        if a > 0 and (a - b) / a < 0.10:
            triggered.append(
                f"two leading hypotheses are within 10% of each other "
                f"({confirmed[0].driver_label} {confirmed[0].effect_pct:+.1f}% vs "
                f"{confirmed[1].driver_label} {confirmed[1].effect_pct:+.1f}%); "
                f"the evidence does not separate them"
            )

    breached = [s for s in required_sources
                if s in wh.freshness and not wh.freshness[s].within_sla]
    if breached:
        triggered.append(
            f"load-bearing source(s) past SLA: {', '.join(breached)}"
        )

    if by_name["data_coverage"].score < 0.85:
        triggered.append(
            f"required data coverage {by_name['data_coverage'].score:.1%} "
            f"is below the 85% floor"
        )

    if contradictions:
        triggered.append("contradictory evidence detected")

    abstained = bool(triggered)
    if abstained:
        action = "abstain"
        tier = "low"

    gaps = [
        {"component": c.name, "score": round(c.score, 3),
         "weight": c.weight, "detail": c.detail}
        for c in components if c.score < 0.9
    ]

    return ConfidenceResult(
        score=total,
        tier=tier,
        action=action,
        components=components,
        abstained=abstained,
        triggered_rules=triggered,
        contradictions=contradictions,
        evidence_gaps=gaps,
        next_best_evidence=_value_of_information(components, wh,
                                                 required_sources, coverage),
    )
