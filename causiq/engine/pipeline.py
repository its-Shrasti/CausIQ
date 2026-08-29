"""
Orchestration — run the full analysis and emit the LOCKED EVIDENCE PACKAGE.

This is the boundary the whole design rests on. Everything above this line is
deterministic: SQL, statistics, causal inference, business rules. The output is
a frozen JSON structure containing every number that may ever reach a user.

Everything BELOW this line -- narrative generation, persona phrasing -- may
only read that structure. It cannot recompute, adjust or supplement a figure,
and a downstream guard re-checks that every number appearing in generated prose
exists here. The brief's requirement that "the LLM should not be treated as the
source of quantitative truth" is therefore enforced structurally, not by asking
a model nicely in a system prompt.

Entitlements are applied BEFORE the package is built, so a restricted role's
package never contains the values it is not allowed to see. There is no prompt
injection that can reveal them because they were never in the context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from causiq.engine import (challenge, confidence, contract, detect, explain,
                           reconcile, retrieve)
from causiq.telemetry.tracker import Telemetry


# -----------------------------------------------------------------------------
# Scenario registry — the four cases the prototype must demonstrate
# -----------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "revenue_drop": {
        "title": "EU net revenue down 8% — multi-driver",
        "kpi": "net_revenue",
        "region": "EU",
        "window": ("2026-08-17", "2026-08-23"),
        "category": None,
        "drivers": ["checkout_experience", "in_stock_rate", "competitor_price",
                    "discount_depth", "marketing_spend"],
        "required_sources": ["wh_sales", "web_events", "ctx_docs"],
        "coverage_columns": ["sessions", "discount_inr"],
        "demonstrates": "Multiple interacting drivers with a mediation trap.",
    },
    "margin_abstain": {
        "title": "EU gross margin down 2.4pp — insufficient evidence",
        "kpi": "gross_margin_pct",
        "region": "EU",
        "window": ("2026-08-20", "2026-08-26"),
        "category": None,
        "drivers": ["discount_depth", "in_stock_rate"],
        "required_sources": ["wh_sales", "ops_inventory", "ctx_docs"],
        "coverage_columns": ["discount_inr", "cogs_inr", "in_stock_rate"],
        "demonstrates": "Abstention with ranked next-best evidence.",
    },
    "wearables_sparse": {
        "title": "EU Wearables conversion — six weeks of history",
        "kpi": "conversion_rate",
        "region": "EU",
        "window": ("2026-08-17", "2026-08-23"),
        "category": "Wearables",
        "drivers": ["competitor_price"],
        "required_sources": ["wh_sales", "web_events"],
        "coverage_columns": ["sessions"],
        "demonstrates": "Method switching under sparse history.",
    },
}


# -----------------------------------------------------------------------------
# Evidence package
# -----------------------------------------------------------------------------

@dataclass
class EvidencePackage:
    scenario: str
    title: str
    persona: str
    role: str
    generated_at: str

    detection: dict[str, Any]
    explanation: dict[str, Any] | None
    attribution: dict[str, Any]
    drivers: list[dict[str, Any]]
    mediation: dict[str, Any]
    context: dict[str, list[dict[str, Any]]]
    confidence: dict[str, Any]
    freshness: list[dict[str, Any]]
    lineage: dict[str, Any]
    access: dict[str, Any]
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)

    # ---- the allow-list the narrative guard checks against ---------------
    def allowed_numbers(self) -> set[float]:
        """Every numeric value present anywhere in the package.

        The guard extracts numbers from generated prose and requires each to
        appear here. That is what makes it impossible for the model to invent
        a figure and have it reach a user.
        """
        found: set[float] = set()

        def walk(o: Any) -> None:
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
            elif isinstance(o, bool):
                return
            elif isinstance(o, (int, float)) and np.isfinite(o):
                found.add(float(o))

        walk(self.as_dict())
        return found


# -----------------------------------------------------------------------------
# Revenue attribution helpers
# -----------------------------------------------------------------------------

def _category_revenue_share(wh, region: str, w0: str, w1: str,
                            role: str) -> dict[str, float]:
    """Each category's share of expected revenue in the analysis window."""
    df = wh.q(f"""
        SELECT category, SUM(net_revenue_inr) AS rev
        FROM fact_daily
        WHERE region = '{region}' AND date BETWEEN DATE '{w0}' AND DATE '{w1}'
        GROUP BY category
    """, label="category_revenue_share")
    total = float(df["rev"].sum()) or 1.0
    return {r["category"]: float(r["rev"]) / total for _, r in df.iterrows()}


def _exposure_fraction(wh, driver_name: str, window: tuple[str, str]) -> float:
    """Fraction of the analysis window during which the driver was live.

    A stockout that ran five days of a seven-day week did five days of damage,
    not seven. Ignoring duration systematically overstates short events and
    understates persistent ones.
    """
    spec = contract.driver(driver_name)
    exp = spec.get("exposure") or {}
    w0, w1 = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    days = (w1 - w0).days + 1

    if exp.get("kind") == "event_window":
        rows = wh.q(f"""
            SELECT MIN(CAST(event_start AS DATE)) AS lo,
                   MAX(CAST(event_end   AS DATE)) AS hi
            FROM raw_ctx_docs WHERE doc_type = '{exp["doc_type"]}'
        """, label=f"exposure_fraction[{driver_name}]")
        lo, hi = pd.Timestamp(rows["lo"].iloc[0]), pd.Timestamp(rows["hi"].iloc[0])
        overlap = (min(hi, w1) - max(lo, w0)).days + 1
        return float(max(0, min(overlap, days))) / days

    ident = contract.identification(driver_name) or {}
    if ident.get("treatment_date"):
        t = pd.Timestamp(ident["treatment_date"])
        if t <= w0:
            return 1.0
        overlap = (w1 - t).days + 1
        return float(max(0, min(overlap, days))) / days
    return 1.0


def _to_revenue_pct(wh, estimate, session_estimate, cat_share: dict[str, float],
                    window: tuple[str, str], aov_estimate=None) -> float:
    """Convert causal effects on conversion and sessions into % of revenue.

    revenue = sessions x conversion x aov, so a proportional move in either
    pathway passes through proportionally, weighted by how much of the revenue
    base is exposed and for how long.
    """
    spec = contract.driver(estimate.driver)
    cats = spec.get("applies_to_categories")
    share = sum(cat_share.get(c, 0.0) for c in cats) if cats else 1.0

    cvr_eff = estimate.effect_pct if estimate.outcome == "conversion_rate" else 0.0
    ses_eff = session_estimate.effect_pct if session_estimate is not None else 0.0
    if estimate.outcome == "sessions":
        ses_eff, cvr_eff = estimate.effect_pct, 0.0

    # An inconclusive effect contributes nothing: reporting a point estimate the
    # engine just declined to confirm would smuggle it back in through the
    # ranking.
    if estimate.verdict not in ("CONFIRMED",):
        cvr_eff = 0.0
    if session_estimate is not None and session_estimate.verdict != "CONFIRMED":
        ses_eff = 0.0

    aov_eff = 0.0
    if aov_estimate is not None and aov_estimate.verdict == "CONFIRMED":
        aov_eff = aov_estimate.effect_pct

    frac = _exposure_fraction(wh, estimate.driver, window)
    return float((cvr_eff + ses_eff + aov_eff) * share * frac)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run(wh, scenario_key: str, persona_key: str = "analyst",
        telemetry: Telemetry | None = None,
        driver_prior: dict[str, float] | None = None) -> EvidencePackage:
    sc = SCENARIOS[scenario_key]
    tel = telemetry or Telemetry()
    persona = contract.persona(persona_key)
    role = persona["role"]
    ent = contract.entitlement(role)

    region, (w0, w1) = sc["region"], sc["window"]

    # --- 1. DETECT -------------------------------------------------------
    with tel.step("detect", "DETECT", "Holt-Winters baseline + materiality gate"):
        det = detect.detect(wh, sc["kpi"], region, w0, w1,
                            role=role, category=sc["category"])

    # --- 2. EXPLAIN ------------------------------------------------------
    exp = None
    if det.material:
        with tel.step("explain", "EXPLAIN",
                      "per-segment baselines + log-additive factor split"):
            exp = explain.explain(wh, det, role=role)

    # --- 3. CHALLENGE ----------------------------------------------------
    estimates: list = []
    session_effects: dict[str, Any] = {}
    aov_effects: dict[str, Any] = {}
    mediation: dict[str, Any] = {"sessions_is_mediator": False, "evidence": []}

    if det.material and sc["drivers"]:
        # Drivers with a real control group go to difference-in-differences.
        for drv in sc["drivers"]:
            ident = contract.identification(drv)
            if ident and ident.get("method") == "difference_in_differences":
                with tel.step(f"challenge:did:{drv}", "CHALLENGE",
                              "difference-in-differences + 3 refutation tests"):
                    try:
                        estimates.append(challenge.estimate_did(wh, drv))
                    except Exception as ex:
                        tel.steps and None
                        print(f"  [warn] DiD failed for {drv}: {ex}")

        # The rest are estimated jointly so overlapping events cannot absorb
        # one another.
        joint_drivers = [d for d in sc["drivers"]
                         if not (contract.identification(d) or {}).get("method")
                         == "difference_in_differences"]
        if joint_drivers and sc["kpi"] != "gross_margin_pct":
            with tel.step("challenge:joint", "CHALLENGE",
                          "joint event study, two-way fixed effects"):
                try:
                    got = challenge.estimate_joint_event_study(
                        wh, region, "conversion_rate", "2026-07-15", "2026-08-26",
                        joint_drivers)
                    estimates.extend(got.values())
                except Exception as ex:
                    print(f"  [warn] joint event study failed: {ex}")

            # --- basket pathway: revenue = sessions x conversion x AOV -----
            # A promotion that lifts conversion while discounting the basket
            # affects revenue through both. Counting only the conversion lift
            # credits it with roughly twice its real contribution.
            with tel.step("challenge:aov", "CHALLENGE",
                          "joint event study on basket size"):
                try:
                    aov_effects.update(challenge.estimate_joint_event_study(
                        wh, region, "aov", "2026-07-15", "2026-08-26",
                        joint_drivers))
                except Exception as ex:
                    print(f"  [warn] aov event study failed: {ex}")

            # --- mediation test: is the traffic decline exogenous? ---------
            with tel.step("challenge:mediation", "CHALLENGE",
                          "session-pathway test against the contract DAG"):
                try:
                    ses = challenge.estimate_joint_event_study(
                        wh, region, "sessions", "2026-07-15", "2026-08-26",
                        joint_drivers)
                    session_effects.update(ses)
                    med = [e for k, e in ses.items()
                           if e.verdict == "CONFIRMED" and k != "marketing_spend"]
                    mediation = {
                        "sessions_is_mediator": bool(med),
                        "evidence": [
                            {"driver": e.driver, "driver_label": e.driver_label,
                             "effect_on_sessions_pct": round(e.effect_pct, 3),
                             "p_value": round(e.p_value, 5)}
                            for e in med
                        ],
                        "implication": (
                            "Part of the session decline is an EFFECT of these "
                            "drivers, not an independent cause. Attributing the "
                            "full traffic drop to the marketing budget would "
                            "double count, and buying traffic back would push "
                            "visitors into a funnel that is still broken."
                            if med else
                            "No confirmed driver acts on sessions, so the traffic "
                            "movement can be treated as exogenous."
                        ),
                    }
                except Exception as ex:
                    print(f"  [warn] mediation test failed: {ex}")

    # --- 4. RETRIEVE -----------------------------------------------------
    docs_by_driver: dict[str, list] = {}
    with tel.step("retrieve", "EXPLAIN", "BM25 over operational documents"):
        idx = retrieve.build_index(wh)
        for drv in sc["drivers"]:
            docs_by_driver[drv] = retrieve.corroborate(
                idx, drv, region, ("2026-08-01", "2026-08-26"))

    # --- 5. SCORE + GATE -------------------------------------------------
    with tel.step("confidence", "KNOW",
                  "deterministic weighted score + contract abstention rules"):
        cov = reconcile.coverage_in_scope(wh, sc["coverage_columns"], region, w0, w1)
        conf = confidence.score(wh, det, estimates, sc["required_sources"],
                                cov, docs_by_driver)

    # --- 6. TRANSLATE TO REVENUE, THEN RANK ------------------------------
    # A driver's effect on conversion is not its importance. The Accessories
    # promotion lifts conversion 29%, but Accessories is 6% of revenue and the
    # promotion ran part of the week, so its revenue impact is around +1%.
    # The checkout release moves conversion only 3.6% -- across the ENTIRE
    # business, every day. Ranking on effect size alone puts the promotion
    # first and buries the actual problem.
    #
    # Since revenue = sessions x conversion x aov, a proportional change in
    # either pathway flows through to revenue in proportion, scaled by the
    # revenue weight of the categories affected and the fraction of the window
    # the driver was actually live.
    with tel.step("attribute", "EXPLAIN",
                  "translate causal effects into revenue contribution"):
        cat_share = _category_revenue_share(wh, region, w0, w1, role)
        for e in estimates:
            e.revenue_impact_pct = _to_revenue_pct(
                wh, e, session_effects.get(e.driver), cat_share, (w0, w1),
                aov_estimate=aov_effects.get(e.driver))

    def rank_key(e) -> float:
        prior = (driver_prior or {}).get(e.driver, 1.0)
        return -abs(getattr(e, "revenue_impact_pct", 0.0) or 0.0) * prior

    ranked = sorted([e for e in estimates if e.verdict != "REJECTED"], key=rank_key)
    rejected = [e for e in estimates if e.verdict == "REJECTED"]

    driver_rows: list[dict[str, Any]] = []
    for e in ranked + rejected:
        row = e.as_dict()
        row["revenue_impact_pct"] = round(getattr(e, "revenue_impact_pct", 0.0) or 0.0, 3)
        row["revenue_impact_inr"] = round(
            (row["revenue_impact_pct"] / 100.0) * det.expected, 0)
        se = session_effects.get(e.driver)
        row["effect_on_sessions_pct"] = round(se.effect_pct, 3) if se else None
        av = aov_effects.get(e.driver)
        row["effect_on_aov_pct"] = round(av.effect_pct, 3) if av else None
        row["prior_weight"] = round((driver_prior or {}).get(e.driver, 1.0), 3)
        row["citations"] = [d.citation() for d in docs_by_driver.get(e.driver, [])]
        lv = contract.levers_for(e.driver)
        row["available_levers"] = [l["id"] for l in lv]
        driver_rows.append(row)

    # --- 7. ATTRIBUTION SUMMARY ------------------------------------------
    # Three buckets, never two. Most tools report only "explained" and let the
    # reader assume the rest is noise. Separating CONFIRMED from UNCONFIRMED
    # from genuinely UNEXPLAINED is the difference between "we know" and "we
    # have a hunch" -- and the residual is a standing invitation to look for
    # the driver nobody has thought of yet.
    confirmed_pct = sum(r["revenue_impact_pct"] for r in driver_rows
                        if r["verdict"] == "CONFIRMED")
    unconfirmed_pct = 0.0
    for e in estimates:
        if e.verdict == "CONFIRMED":
            continue
        se = session_effects.get(e.driver)
        cats = contract.driver(e.driver).get("applies_to_categories")
        share = sum(cat_share.get(c, 0.0) for c in cats) if cats else 1.0
        raw = (e.effect_pct if e.outcome == "conversion_rate" else 0.0)
        raw += (se.effect_pct if se is not None and se.verdict != "CONFIRMED" else 0.0)
        unconfirmed_pct += raw * share * _exposure_fraction(wh, e.driver, (w0, w1))

    total_pct = det.deviation_pct
    attribution = {
        "total_movement_pct": round(total_pct, 3),
        "confirmed_pct": round(confirmed_pct, 3),
        "unconfirmed_pct": round(unconfirmed_pct, 3),
        "unexplained_pct": round(total_pct - confirmed_pct - unconfirmed_pct, 3),
        "confirmed_share_of_movement": (
            round(100 * confirmed_pct / total_pct, 1) if total_pct else 0.0),
        "note": (
            "Confirmed contributions come from drivers that reached significance "
            "AND survived falsification. Unconfirmed are point estimates the "
            "evidence could not establish; they are shown for completeness and "
            "excluded from ranking and recommendations. The unexplained remainder "
            "is left visible rather than distributed across the known drivers to "
            "make the decomposition appear complete."
        ),
    }

    return EvidencePackage(
        attribution=attribution,
        scenario=scenario_key,
        title=sc["title"],
        persona=persona_key,
        role=role,
        generated_at=str(reconcile.NOW),
        detection=det.as_dict(),
        explanation=exp.as_dict() if exp else None,
        drivers=driver_rows,
        mediation=mediation,
        context={k: [d.as_dict() for d in v] for k, v in docs_by_driver.items()},
        confidence=conf.as_dict(),
        freshness=[f.as_dict() for f in wh.freshness.values()],
        lineage={
            "kpi_definition": contract.kpi(sc["kpi"]).definition,
            "formula": contract.kpi(sc["kpi"]).formula,
            "columns": contract.kpi(sc["kpi"]).lineage,
            "sources": sc["required_sources"],
            "contract_version": contract.load_contract()["contract_version"],
            "sql_executed": len(wh.sql_log),
        },
        access={
            "role": role,
            "row_filter": ent.get("row_filter"),
            "denied_columns": ent.get("denied_columns") or [],
            "denied_kpis": ent.get("denied_kpis") or [],
            "insight_depth": ent.get("insight_depth"),
            "out_of_scope_policy": ent.get("out_of_scope_policy"),
        },
        telemetry=tel.summary(),
    )
