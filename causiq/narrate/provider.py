"""
LLM providers — the ONLY place in CausIQ that calls a language model.

Three providers behind one interface:

  GeminiProvider   real calls to Gemini Flash, used when GEMINI_API_KEY is set
  OfflineProvider  deterministic template rendering, no network, no key
  HallucinationProvider
                   deliberately fabricates a figure, so the numeric guard can
                   be demonstrated catching one on camera

The offline provider is not a stub for a missing feature. A demo that depends
on a live API call is a demo that breaks in front of judges, and a rate-limited
free tier makes that likely rather than unlucky. Every response is cached to
disk by prompt hash, so a recorded run is reproducible and costs nothing to
replay.

Token accounting is recorded for all three. The offline provider reports the
tokens a real call WOULD have consumed, estimated from the prompt it built, so
the cost model stays meaningful when demoing without a key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_ms: float
    from_cache: bool
    provider: str


def _estimate_tokens(text: str) -> int:
    """~4 characters per token. Good enough for cost projection."""
    return max(1, len(text) // 4)


def _key(prompt: str, model: str, temperature: float) -> str:
    h = hashlib.sha256(f"{model}|{temperature}|{prompt}".encode()).hexdigest()
    return h[:32]


class BaseProvider:
    name = "base"
    model = "none"

    def _cache_get(self, k: str) -> dict[str, Any] | None:
        p = CACHE_DIR / f"{k}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _cache_put(self, k: str, payload: dict[str, Any]) -> None:
        (CACHE_DIR / f"{k}.json").write_text(json.dumps(payload, indent=2))

    def generate(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 600, use_cache: bool = True) -> LLMResponse:
        raise NotImplementedError


class GeminiProvider(BaseProvider):
    """Gemini Flash. Chosen for the cost/latency profile, not the leaderboard.

    The narrative task is constrained rewriting of a fixed evidence structure --
    no reasoning, no arithmetic, no retrieval. A frontier model would cost
    roughly 12x more per insight for output nobody could distinguish. Model
    choice IS an architecture decision and it belongs in the cost story.
    """
    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self._client = None

    def _ensure(self):
        if self._client is None:
            import google.generativeai as genai
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            genai.configure(api_key=key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def generate(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 600, use_cache: bool = True) -> LLMResponse:
        k = _key(prompt, self.model, temperature)
        if use_cache:
            hit = self._cache_get(k)
            if hit:
                return LLMResponse(
                    text=hit["text"], model=self.model,
                    prompt_tokens=hit["prompt_tokens"],
                    completion_tokens=hit["completion_tokens"],
                    cached_tokens=hit["prompt_tokens"],   # full prompt cache hit
                    latency_ms=0.4, from_cache=True, provider=self.name,
                )

        client = self._ensure()
        t0 = time.perf_counter()
        resp = client.generate_content(
            prompt,
            generation_config={"temperature": temperature,
                               "max_output_tokens": max_tokens},
        )
        dt = (time.perf_counter() - t0) * 1000.0
        text = (resp.text or "").strip()

        pt = ct = 0
        try:
            um = resp.usage_metadata
            pt, ct = int(um.prompt_token_count), int(um.candidates_token_count)
        except Exception:
            pt, ct = _estimate_tokens(prompt), _estimate_tokens(text)

        self._cache_put(k, {"text": text, "prompt_tokens": pt,
                            "completion_tokens": ct, "model": self.model})
        return LLMResponse(text=text, model=self.model, prompt_tokens=pt,
                           completion_tokens=ct, cached_tokens=0,
                           latency_ms=dt, from_cache=False, provider=self.name)


class OfflineProvider(BaseProvider):
    """Deterministic rendering from the evidence, with no model involved.

    The prompt carries a JSON evidence block; this provider reads it and fills
    a persona template. Output is plainer than a model's, and every figure is
    lifted verbatim from the evidence -- so it passes the numeric guard by
    construction, which is a useful control when testing the guard itself.
    """
    name = "offline"
    model = "none"

    def generate(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 600, use_cache: bool = True) -> LLMResponse:
        t0 = time.perf_counter()
        text = _render_offline(prompt)
        dt = (time.perf_counter() - t0) * 1000.0
        return LLMResponse(
            text=text, model="none",
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(text),
            cached_tokens=0, latency_ms=dt, from_cache=False, provider=self.name,
        )


class HallucinationProvider(OfflineProvider):
    """Demo-only. Injects a figure that appears in no evidence field.

    Exists so the numeric guard can be shown REJECTING something during the
    walkthrough. A guard that has never been seen to fire is indistinguishable
    from no guard at all.
    """
    name = "hallucinating"

    def generate(self, prompt: str, **kw) -> LLMResponse:
        r = super().generate(prompt, **kw)
        r.text += (" Revenue is expected to recover by 14.7% within nine days "
                   "based on similar past incidents.")
        r.provider = self.name
        return r


# -----------------------------------------------------------------------------
# Offline rendering
# -----------------------------------------------------------------------------

def _extract_block(prompt: str, tag: str) -> dict[str, Any]:
    """Pull the JSON payload out of <tag>...</tag>.

    Deliberately anchored on the closing tag rather than matching braces: a
    non-greedy brace match stops at the FIRST closing brace, which on a nested
    payload silently yields a truncated object. That failed quietly -- the
    narrative rendered with default values, so every p-value printed as 1.0000
    and looked like a statistics bug rather than a parsing one.
    """
    m = re.search(rf"<{tag}>\s*(\{{.*\}})\s*</{tag}>", prompt, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}


def _render_offline(prompt: str) -> str:
    ev = _extract_block(prompt, "evidence")
    if not ev:
        return "No evidence was supplied, so no narrative can be produced."

    persona = ev.get("persona", "analyst")
    d = ev.get("headline", {})
    drivers = ev.get("drivers", [])
    conf = ev.get("confidence", {})

    if ev.get("abstained"):
        rules = ev.get("triggered_rules", [])
        nxt = ev.get("next_best_evidence", [])
        out = [
            f"{d.get('kpi_label', 'The metric')} moved "
            f"{d.get('deviation_pct', 0):+.2f}% against expectation in "
            f"{d.get('region', 'the region')}, and the movement is real.",
            "",
            f"CausIQ is not attributing a cause. Confidence scored "
            f"{conf.get('score', 0):.2f} against a required 0.60, and "
            f"{len(rules)} abstention rule(s) fired:",
        ]
        out += [f"  - {r}" for r in rules]
        if nxt:
            out += ["", "Most useful evidence to obtain next:"]
            out += [f"  - {n['action']} ({n['owner']}, ~{n['effort_hours']}h)"
                    for n in nxt[:3]]
        return "\n".join(out)

    if not ev.get("material", True):
        return (
            f"{d.get('kpi_label', 'The metric')} moved "
            f"{d.get('deviation_pct', 0):+.2f}% in {d.get('region', 'the region')}, "
            f"but this is not distinguishable from normal variation for a series "
            f"with only {d.get('history_weeks', 0):.1f} weeks of history "
            f"(p = {d.get('p_value', 1):.3f}). No action is recommended. "
            f"CausIQ will re-assess as history accumulates."
        )

    lead = drivers[0] if drivers else {}
    if persona == "cmo":
        act = ev.get("recommended_action", {})
        return (
            f"{d.get('kpi_label')} came in {d.get('deviation_pct', 0):+.2f}% "
            f"against plan in {d.get('region')}, a shortfall of "
            f"Rs {abs(d.get('deviation_abs', 0)) / 1e7:.2f} Cr. "
            f"The largest identified cause is {lead.get('driver_label', 'unknown')}, "
            f"accounting for {lead.get('revenue_impact_pct', 0):+.2f}%. "
            f"Recommended action: {act.get('action', 'none available')} "
            f"Owner: {act.get('owner', 'unassigned')}. "
            f"Confidence: {conf.get('tier', 'unknown')}."
        )

    if persona == "regional_manager":
        act = ev.get("recommended_action", {})
        oos = ev.get("out_of_scope_drivers", [])
        lines = [
            f"{d.get('kpi_label')} in your region moved "
            f"{d.get('deviation_pct', 0):+.2f}% against expectation.",
            f"Leading cause within your scope: {lead.get('driver_label', 'none')} "
            f"at {lead.get('revenue_impact_pct', 0):+.2f}%.",
            f"Action: {act.get('action', 'none available')}",
            f"Owner: {act.get('owner', 'unassigned')}",
        ]
        if oos:
            lines.append(
                "Note: one or more material drivers lie outside your data scope "
                "and have been routed to the owning team."
            )
        return "\n".join(lines)

    # analyst
    lines = [
        f"{d.get('kpi_label')} moved {d.get('deviation_pct', 0):+.2f}% versus a "
        f"{d.get('baseline_method', 'baseline')} expectation "
        f"(z = {d.get('z_score', 0):+.2f}, p = {d.get('p_value', 1):.4f}).",
        "",
        "Ranked drivers by revenue contribution:",
    ]
    for i, dr in enumerate(drivers, 1):
        lines.append(
            f"  {i}. {dr.get('driver_label')}: "
            f"{dr.get('revenue_impact_pct', 0):+.2f}% of revenue "
            f"[{dr.get('method')}, p = {dr.get('p_value', 1):.4f}, "
            f"{dr.get('verdict')}]"
        )
    att = ev.get("attribution", {})
    if att:
        lines += [
            "",
            f"Confirmed drivers account for {att.get('confirmed_pct', 0):+.2f}% of "
            f"the {att.get('total_movement_pct', 0):+.2f}% movement. "
            f"{att.get('unexplained_pct', 0):+.2f}% remains unexplained and is "
            f"reported as such.",
        ]
    return "\n".join(lines)


def get_provider(mode: str = "auto") -> BaseProvider:
    if mode == "offline":
        return OfflineProvider()
    if mode == "hallucinate":
        return HallucinationProvider()
    if mode == "gemini" or (mode == "auto" and os.environ.get("GEMINI_API_KEY")):
        try:
            return GeminiProvider()
        except Exception:
            return OfflineProvider()
    return OfflineProvider()
