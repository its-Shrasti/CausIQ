"""
NUMERIC GUARD — every figure in generated prose must exist in the evidence.

The brief is explicit that an LLM must not be the source of quantitative truth.
Prompt instructions cannot enforce that: a model told "use only these numbers"
will still occasionally produce a plausible one, and plausible is exactly what
makes it dangerous. So the check happens after generation, in code.

HOW IT WORKS
------------
1. Extract every number from the narrative, normalising currency and percent
   forms so "Rs 8.14 Cr", "8.14 Cr" and "81,400,000" are recognised as the
   same quantity.
2. Compare each against the evidence package's allow-list, at several scales
   (raw, crore, lakh, percent-of-unit) with a relative tolerance for rounding.
3. Anything unmatched is a VIOLATION. The narrative is rejected and either
   regenerated with the offending figures named, or replaced by the
   deterministic offline rendering.

WHAT IS DELIBERATELY ALLOWED
----------------------------
Ordinals and small counts ("the top 3 drivers", "5 days"), years, and list
numbering are not quantitative claims about the business. Treating them as
violations would make the guard fire constantly and get switched off, which is
how safety mechanisms actually die. The exclusion list is narrow and explicit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Currency and magnitude suffixes used in Indian business writing
_MULTIPLIERS = {
    "cr": 1e7, "crore": 1e7, "crores": 1e7,
    "lakh": 1e5, "lakhs": 1e5, "l": 1e5,
    "k": 1e3, "m": 1e6, "mn": 1e6, "bn": 1e9,
}

_NUM = re.compile(
    r"(?<![\w.])"
    r"(?:Rs\.?\s*|₹\s*)?"
    r"(-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s*"
    r"(%|pp|percentage points|cr|crores?|lakhs?|k|m|mn|bn)?"
    r"(?![\w])",
    re.I,
)

# Numbers that are not business quantities
_SAFE_EXACT = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
               12.0, 24.0, 100.0}
_YEAR_RANGE = (1990.0, 2100.0)


@dataclass
class Violation:
    literal: str
    parsed_value: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GuardResult:
    passed: bool
    checked: int
    violations: list[Violation] = field(default_factory=list)
    matched: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "numbers_checked": self.checked,
            "numbers_matched": self.matched,
            "violations": [v.as_dict() for v in self.violations],
        }

    def summary(self) -> str:
        if self.passed:
            return (f"PASS — {self.matched}/{self.checked} figures traced to "
                    f"the evidence package")
        names = ", ".join(v.literal for v in self.violations)
        return (f"REJECTED — {len(self.violations)} figure(s) not present in "
                f"the evidence: {names}")


def _parse(literal: str, suffix: str | None) -> tuple[float, float] | None:
    """Return (value, absolute tolerance implied by how it was written).

    Tolerance comes from DISPLAY PRECISION, not a flat relative epsilon. A
    figure written "-0.70" was rounded from somewhere in [-0.705, -0.695], so
    that is exactly the band the guard should accept. A flat relative tolerance
    gets this wrong at both ends: too tight for small magnitudes (-0.705 shown
    as -0.70 is a 0.7% relative error and would be rejected) and far too loose
    for large ones, where it opens a window wide enough for a fabricated figure
    to slip through by coincidence.
    """
    clean = literal.replace(",", "")
    try:
        v = float(clean)
    except ValueError:
        return None

    decimals = len(clean.split(".")[1]) if "." in clean else 0
    tol = 0.5 * (10 ** -decimals)

    if suffix:
        s = suffix.lower().strip()
        if s in _MULTIPLIERS:
            v *= _MULTIPLIERS[s]
            tol *= _MULTIPLIERS[s]
    return v, tol * 1.001          # a hair of slack for float representation


def _candidate_scales(v: float) -> list[tuple[float, float]]:
    """Scales a figure might be stored at, each with its tolerance multiplier.

    "Rs 8.14 Cr" in prose and 81_400_000.0 in the evidence are the same claim,
    as are "-8.59%" and either -8.59 or -0.0859. Reconciling these is necessary
    or the guard rejects correct narratives -- and a guard with false positives
    gets switched off.

    The list is deliberately SHORT. Every extra scale widens the target and
    raises the chance that an invented figure coincidentally lands on some
    unrelated internal value. Only percent-vs-fraction and crore/lakh
    conversions are genuinely ambiguous in this domain, so only those are tried.
    """
    return [
        (v, 1.0),
        (v / 100.0, 0.01),      # "8.59%" stored as 0.0859
        (v * 100.0, 100.0),     # 0.0859 stored, written as "8.59"
    ]


def check(text: str, allowed: set[float], rel_tol: float = 0.005
          ) -> GuardResult:
    """Verify every business figure in `text` appears in `allowed`."""
    allowed_list = sorted(allowed)
    violations: list[Violation] = []
    checked = matched = 0

    for m in _NUM.finditer(text):
        raw, suffix = m.group(1), m.group(2)
        parsed = _parse(raw, suffix)
        if parsed is None:
            continue
        val, abs_tol = parsed

        # Skip non-quantitative numbers
        bare = abs(float(raw.replace(",", "")))
        if suffix is None and (bare in _SAFE_EXACT
                               or _YEAR_RANGE[0] <= bare <= _YEAR_RANGE[1]):
            continue

        checked += 1
        ok = False
        for cand, scale in _candidate_scales(val):
            tol = max(abs_tol * scale, rel_tol * abs(cand))
            for a in allowed_list:
                if abs(a - cand) <= tol:
                    ok = True
                    break
            if ok:
                break

        if ok:
            matched += 1
        else:
            violations.append(Violation(
                literal=m.group(0).strip(),
                parsed_value=val,
                reason=("no field in the evidence package holds this value at "
                        "any recognised scale"),
            ))

    return GuardResult(passed=not violations, checked=checked,
                       violations=violations, matched=matched)


def enforce(text: str, allowed: set[float], fallback: str,
            max_attempts: int = 1, regenerate=None) -> tuple[str, GuardResult]:
    """Run the guard; regenerate or fall back if it fails.

    Failure is never allowed to reach a user as-is. Either the model produces
    something that survives the check, or the deterministic rendering is served
    instead. Silently passing an unverified narrative through would defeat the
    entire mechanism.
    """
    result = check(text, allowed)
    attempts = 0
    while not result.passed and regenerate and attempts < max_attempts:
        attempts += 1
        offending = ", ".join(v.literal for v in result.violations)
        text = regenerate(offending)
        result = check(text, allowed)

    if not result.passed:
        return fallback, result
    return text, result
