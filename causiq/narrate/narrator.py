"""
NARRATE + ACT — turn a locked evidence package into persona-specific output.

Two things happen here and it matters that they are separate:

RECOMMENDATION is deterministic. Actions are RETRIEVED from the governed lever
library, filtered by whether the contract permits them (a lever blocked by the
mediation finding never appears), and their expected impact is computed from
the causal estimate. An LLM does not decide what a business should do.

NARRATION is generative, and it is the only generative step in the system. The
model receives a locked evidence JSON and a persona instruction, and returns
prose. It cannot compute, adjust or add a figure, and the numeric guard checks
that it did not.

The persona layer is a rendering choice, not an analytical one. All three
personas read from the SAME evidence, computed once. What differs is which
fields are selected, at what depth, and in what register -- never the numbers.
Recomputing per persona would let two people leave the same meeting with two
different versions of the truth, which is the failure mode a semantic layer is
supposed to prevent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from causiq.engine import contract
from causiq.narrate import guard as guard_mod
from causiq.narrate.provider import BaseProvider, get_provider
from causiq.telemetry.tracker import Telemetry


# -----------------------------------------------------------------------------
# Action selection — deterministic
# -----------------------------------------------------------------------------

@dataclass
class Recommendation:
    lever_id: str
    driver: str
    driver_label: str
    lever: str
    action: str
    expected_impact_pct: float
    expected_impact_inr: float
    margin_note: str
    cost_inr: float
    risk: str
    risk_notes: str
    owner: str
    decision_rights: str
    constraints: list[str]
    reversible: bool | None
    confidence: float
    monitoring: dict[str, Any]
    blocked: bool = False
    blocked_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _fill(template: str, **kw) -> str:
    out = " ".join(str(template).split())
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def recommend(evidence, top_n: int = 3) -> list[Recommendation]:
    """Retrieve permitted actions for the confirmed drivers, best first.

    Expected impact is capped at the causally attributed loss. A lever cannot
    recover more than the driver took, and letting an optimistic elasticity
    imply otherwise is how business cases get built on arithmetic nobody
    checked.
    """
    out: list[Recommendation] = []
    mediator = bool(evidence.mediation.get("sessions_is_mediator"))

    for row in evidence.drivers:
        if row["verdict"] != "CONFIRMED":
            continue
        if row.get("revenue_impact_pct", 0) >= 0:
            continue                      # only fix what is hurting

        for lever in contract.levers_for(row["driver"]):
            blocked, reason = False, ""
            for cond in (lever.get("blocked_when") or []):
                if cond.strip() == "sessions_is_mediator == true" and mediator:
                    blocked = True
                    reason = (
                        "Blocked by the contract: sessions are a mediator here, "
                        "so part of the traffic decline is an effect of other "
                        "drivers. Buying traffic back would push visitors into a "
                        "funnel that is still broken and destroy return on spend."
                    )

            el = lever.get("elasticity") or {}
            attributed = abs(row.get("revenue_impact_pct", 0.0))
            recov = float(el.get("recovery_fraction_of_attributed_loss", 0.0))
            if recov:
                impact_pct = attributed * recov
            elif el.get("own_price_elasticity"):
                impact_pct = attributed * 0.55
            elif el.get("spend_to_sessions_elasticity"):
                impact_pct = attributed * 0.60
            else:
                impact_pct = attributed * 0.50

            expected_inr = (impact_pct / 100.0) * evidence.detection["expected"]
            cost = float((lever.get("cost") or {}).get("one_off_inr") or 0.0)
            recurring = float((lever.get("cost") or {}).get(
                "recurring_inr_per_month") or 0.0)

            out.append(Recommendation(
                lever_id=lever["id"],
                driver=row["driver"],
                driver_label=row["driver_label"],
                lever=lever["lever"],
                action=_fill(lever["action_template"],
                             release_name="v4.2", region=evidence.detection
                             ["scope"].get("region", "the region"),
                             pct=8, n=3, days=14, days_live=5,
                             category="Electronics", error_class="payment validation"),
                expected_impact_pct=round(impact_pct, 3),
                expected_impact_inr=round(expected_inr, 0),
                margin_note=str((lever.get("cost") or {}).get("margin_cost", "")),
                cost_inr=cost + recurring,
                risk=lever.get("risk", "unknown"),
                risk_notes=lever.get("risk_notes", "").strip(),
                owner=lever.get("owner_role") or "unassigned",
                decision_rights=lever.get("decision_rights", ""),
                constraints=lever.get("constraints") or [],
                reversible=lever.get("reversible"),
                confidence=round(float(evidence.confidence["score"]), 3),
                monitoring=lever.get("monitoring") or {},
                blocked=blocked,
                blocked_reason=reason,
            ))

    # Rank by recoverable rupees per rupee spent, not by raw impact. A lever
    # recovering slightly less for a fraction of the cost is the better call,
    # and ranking on impact alone hides that.
    def score(r: Recommendation) -> float:
        if r.blocked:
            return -1e9
        return r.expected_impact_inr / max(r.cost_inr, 1e4)

    out.sort(key=score, reverse=True)
    return out[:top_n] + [r for r in out[top_n:] if r.blocked]


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

SYSTEM_RULES = """You are CausIQ's narration layer for an enterprise KPI engine.

ABSOLUTE CONSTRAINTS
1. Every number you write MUST appear in the evidence block below. Never
   compute, round differently, infer, extrapolate or estimate a figure.
2. If a quantity is not in the evidence, describe it in words or omit it.
   Do not approximate.
3. Never assert a cause the evidence marks INCONCLUSIVE or REJECTED. Those may
   be mentioned only as unconfirmed.
4. If the evidence says abstained, do NOT provide a cause or a recommendation.
   State what is missing and what would resolve it.
5. Write plain professional English. No preamble, no headings, no bullet
   characters unless the persona asks for a list.

An automated guard checks every figure you produce against the evidence. Output
containing an unverifiable number is discarded."""


def numbers_in(obj: Any) -> set[float]:
    """Every finite number reachable inside a payload."""
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
        elif isinstance(o, (int, float)):
            try:
                f = float(o)
                if f == f and abs(f) != float("inf"):
                    found.add(f)
            except Exception:
                pass

    walk(obj)
    return found


def build_prompt(evidence, persona_key: str,
                 recommendations: list[Recommendation]) -> tuple[str, dict[str, Any]]:
    p = contract.persona(persona_key)
    det = evidence.detection
    conf = evidence.confidence

    # Only the fields this persona is permitted to see are serialised. A denied
    # value is not "hidden from the prompt" -- it is absent from the payload, so
    # no instruction to the model can surface it.
    payload: dict[str, Any] = {
        "persona": persona_key,
        "audience": f"{p['display_name']}, {p['title']}",
        "abstained": conf["abstained"],
        "material": det["material"],
        "headline": {
            "kpi_label": det["kpi_label"],
            "region": det["scope"].get("region"),
            "window": [det["window_start"], det["window_end"]],
            "actual": det["actual"],
            "expected": det["expected"],
            "deviation_pct": det["deviation_pct"],
            "deviation_abs": det["deviation_abs"],
            "baseline_method": det["baseline_method"],
            "history_weeks": det["history_weeks"],
        },
        "confidence": {"score": conf["score"], "tier": conf["tier"]},
        "attribution": evidence.attribution,
    }

    # Depth is an ENTITLEMENT property, not a persona preference. Reading it
    # from the role means a persona cannot be configured into seeing more than
    # its role permits, and the two can never drift apart.
    depth = contract.entitlement(evidence.role).get("insight_depth", "executive")
    if depth == "technical":
        payload["headline"].update({
            "z_score": det["z_score"], "p_value": det["p_value"],
            "pi_low": det["pi_low"], "pi_high": det["pi_high"],
        })

    if conf["abstained"]:
        payload["triggered_rules"] = conf["triggered_rules"]
        payload["next_best_evidence"] = conf["next_best_evidence"][:3]
        payload["contradictions"] = conf["contradictions"]
    else:
        keep = ["driver", "driver_label", "revenue_impact_pct", "verdict",
                "method", "effect_pct"]
        if depth == "technical":
            keep += ["p_value", "std_error", "ci_low", "ci_high", "n_obs",
                     "citations", "effect_on_sessions_pct"]
        payload["drivers"] = [
            {k: d.get(k) for k in keep}
            for d in evidence.drivers
            if d["verdict"] == "CONFIRMED" or depth == "technical"
        ][:6]
        payload["mediation"] = evidence.mediation
        live = [r for r in recommendations if not r.blocked]
        if live:
            r = live[0]
            payload["recommended_action"] = {
                "action": r.action, "owner": r.owner,
                "expected_impact_pct": r.expected_impact_pct,
                "expected_impact_inr": r.expected_impact_inr,
                "risk": r.risk, "decision_rights": r.decision_rights,
                "monitoring": r.monitoring.get("metrics", []),
            }
        blocked = [r for r in recommendations if r.blocked]
        if blocked:
            payload["blocked_actions"] = [
                {"action": b.action, "reason": b.blocked_reason} for b in blocked]

    if evidence.explanation and evidence.explanation.get("paradox_flag"):
        payload["aggregation_warning"] = evidence.explanation["paradox_note"]

    instr = (
        f"Write for {p['display_name']}, {p['title']}.\n"
        f"Tone: {p['tone']}\n"
        f"Maximum {p['max_words']} words.\n"
        f"Include: {', '.join(p['include'])}.\n"
        f"Exclude entirely: {', '.join(p['exclude']) or 'nothing'}."
    )

    prompt = (f"{SYSTEM_RULES}\n\n<evidence>\n"
              f"{json.dumps(payload, indent=2, default=str)}\n</evidence>\n\n"
              f"{instr}\n\nNarrative:")
    return prompt, payload


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

@dataclass
class Narrative:
    persona: str
    display_name: str
    channel: str
    text: str
    guard: dict[str, Any]
    guard_passed: bool
    provider: str
    model: str
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    fell_back: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def narrate(evidence, persona_key: str, telemetry: Telemetry,
            provider: BaseProvider | None = None,
            mode: str = "auto") -> Narrative:
    p = contract.persona(persona_key)
    prov = provider or get_provider(mode)

    with telemetry.step("recommend", "SIMULATE",
                        "lever library retrieval + impact capped at attributed loss"):
        recs = recommend(evidence)

    prompt, payload = build_prompt(evidence, persona_key, recs)

    with telemetry.step(f"narrate:{persona_key}", "DECIDE",
                        "LLM narrative over locked evidence",
                        uses_llm=True, model=prov.model) as rec:
        resp = prov.generate(prompt, temperature=0.2,
                             max_tokens=int(p["max_words"] * 2))
        rec.prompt_tokens = resp.prompt_tokens
        rec.completion_tokens = resp.completion_tokens
        rec.cached_tokens = resp.cached_tokens
        rec.model = resp.model
        # Mark the step by what ACTUALLY ran. In offline mode no model is
        # invoked, so counting it as a model call would overstate LLM usage in
        # the very breakdown that is supposed to prove how little of this system
        # depends on one. Token counts are still recorded -- they are what a real
        # call would have consumed, which keeps the cost projection meaningful.
        rec.uses_llm = resp.model != "none"
        rec.note = (f"provider={resp.provider} cached={resp.from_cache}"
                    + ("" if rec.uses_llm else
                       " | no model invoked; tokens are the projected cost of a real call"))

    with telemetry.step("numeric_guard", "DECIDE",
                        "regex extraction + evidence allow-list check"):
        # Guard against the PAYLOAD, not the whole evidence package.
        #
        # The model can only have seen what was put in the prompt, so any figure
        # it writes must come from there. Checking against the full package --
        # 336 values across every scale -- made the allow-list so dense that a
        # fabricated "14.7%" landed within tolerance of an unrelated internal
        # value and passed. A guard that accepts anything plausible is not a
        # guard. Scoping to the ~20 numbers actually shown makes a collision
        # unlikely and the check meaningful.
        allowed = numbers_in(payload)
        for r in recs:
            allowed |= {r.expected_impact_pct, r.expected_impact_inr, r.cost_inr}
        from causiq.narrate.provider import OfflineProvider, _render_offline
        fallback = _render_offline(prompt)
        text, gres = guard_mod.enforce(resp.text, allowed, fallback)

    return Narrative(
        persona=persona_key,
        display_name=p["display_name"],
        channel=p["channel"],
        text=text,
        guard=gres.as_dict(),
        guard_passed=gres.passed,
        provider=resp.provider,
        model=resp.model,
        recommendations=[r.as_dict() for r in recs],
        fell_back=not gres.passed,
    )
