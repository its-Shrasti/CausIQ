"""
Runtime telemetry — latency, model calls, tokens and cost per insight.

The brief asks for this explicitly, and it is the part teams usually leave
until last and then estimate. Measuring from the first line of engine code
instead means the numbers are real, and it forces the architectural question
that actually matters: which steps need a language model at all?

Every step records whether it used an LLM. The resulting split is not a
marketing claim, it is a measurement -- and on this workload it comes out
overwhelmingly deterministic, which is the point.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# Published rates, INR per 1M tokens. Gemini Flash free tier bills at zero, but
# paid rates are carried so the cost projection reflects what production would
# actually cost rather than what a hackathon demo costs.
MODEL_RATES_INR = {
    "gemini-2.0-flash":      {"in": 8.3,   "out": 33.2,  "free_tier": True},
    "gemini-2.0-flash-lite": {"in": 6.2,   "out": 24.9,  "free_tier": True},
    "gemini-1.5-pro":        {"in": 104.0, "out": 415.0, "free_tier": False},
    "none":                  {"in": 0.0,   "out": 0.0,   "free_tier": True},
}


@dataclass
class StepRecord:
    name: str
    stage: str
    uses_llm: bool
    method: str
    latency_ms: float
    model: str = "none"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_inr: float = 0.0
    rows_scanned: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Telemetry:
    steps: list[StepRecord] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def step(self, name: str, stage: str, method: str, uses_llm: bool = False,
             model: str = "none", note: str = ""):
        rec = StepRecord(name=name, stage=stage, uses_llm=uses_llm,
                         method=method, latency_ms=0.0, model=model, note=note)
        t = time.perf_counter()
        try:
            yield rec
        finally:
            rec.latency_ms = (time.perf_counter() - t) * 1000.0
            if rec.uses_llm:
                rec.cost_inr = self.cost_of(rec.model, rec.prompt_tokens,
                                            rec.completion_tokens, rec.cached_tokens)
            self.steps.append(rec)

    @staticmethod
    def cost_of(model: str, prompt: int, completion: int, cached: int = 0) -> float:
        r = MODEL_RATES_INR.get(model, MODEL_RATES_INR["none"])
        # Cached prompt tokens bill at 25% on Gemini.
        billable_in = (prompt - cached) + cached * 0.25
        return (billable_in * r["in"] + completion * r["out"]) / 1_000_000.0

    # --- summaries --------------------------------------------------------
    def total_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    def llm_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps if s.uses_llm)

    def deterministic_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps if not s.uses_llm)

    def model_calls(self) -> int:
        return sum(1 for s in self.steps if s.uses_llm)

    def tokens(self) -> dict[str, int]:
        return {
            "prompt": sum(s.prompt_tokens for s in self.steps),
            "completion": sum(s.completion_tokens for s in self.steps),
            "cached": sum(s.cached_tokens for s in self.steps),
        }

    def cost_inr(self) -> float:
        return sum(s.cost_inr for s in self.steps)

    def summary(self) -> dict[str, Any]:
        tot = self.total_ms() or 1.0
        tk = self.tokens()
        return {
            "total_latency_ms": round(self.total_ms(), 1),
            "deterministic_ms": round(self.deterministic_ms(), 1),
            "llm_ms": round(self.llm_ms(), 1),
            "deterministic_share_pct": round(100 * self.deterministic_ms() / tot, 1),
            "llm_share_pct": round(100 * self.llm_ms() / tot, 1),
            "steps_total": len(self.steps),
            "steps_using_llm": self.model_calls(),
            "steps_deterministic": len(self.steps) - self.model_calls(),
            "model_calls": self.model_calls(),
            "prompt_tokens": tk["prompt"],
            "completion_tokens": tk["completion"],
            "cached_tokens": tk["cached"],
            "cache_hit_rate_pct": round(
                100 * tk["cached"] / tk["prompt"], 1) if tk["prompt"] else 0.0,
            "cost_per_insight_inr": round(self.cost_inr(), 4),
            "projected_monthly_inr_at_500_per_day": round(
                self.cost_inr() * 500 * 30, 2),
        }

    def table(self) -> list[dict[str, Any]]:
        return [s.as_dict() for s in self.steps]
