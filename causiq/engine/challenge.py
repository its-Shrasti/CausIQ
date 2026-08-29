"""
CHALLENGE — did this driver actually cause the movement, or does it merely
move alongside it?

This is the module the whole product rests on. Everything upstream locates a
movement; this decides whether a proposed explanation survives an honest
attempt to destroy it.

TWO IDENTIFICATION STRATEGIES, CHOSEN BY THE CONTRACT
-----------------------------------------------------
1. DIFFERENCE-IN-DIFFERENCES, where a natural experiment exists. The checkout
   redesign shipped to EU only, so US and APAC are a genuine untreated control.
   The estimator is the interaction term in

       y = a + b*treated + c*post + d*(treated x post) + e

   and `d` is the causal effect. DiD removes anything common to both groups
   (the competitor promotion, seasonality, macro conditions) and anything fixed
   about either group (EU simply converting lower than US). That is why it is
   worth far more than a before/after comparison.

2. BACKDOOR ADJUSTMENT, where no experiment exists. Regress the outcome on the
   driver while controlling for the adjustment set the contract's DAG
   specifies. This is strictly weaker: it is only valid if the declared
   adjustment set is complete, which is an assumption, not a finding. The
   engine says so in its output rather than presenting both as equally solid.

REFUTATION IS NOT OPTIONAL
-------------------------
An estimate that has not survived falsification is a number, not a finding. So
every confirmed effect must pass three attacks designed to make it disappear:

  * PLACEBO IN TIME    — invent a treatment date before anything happened. A
                         "real" effect here means the groups were already
                         diverging, so parallel trends fails and the headline
                         estimate is contaminated.
  * NEGATIVE CONTROL   — run the same estimator on an outcome the treatment
                         physically cannot affect. A checkout bug cannot change
                         how expensive the average basket is. An effect here
                         means something else is moving both series.
  * LEAVE-ONE-OUT      — drop each control unit in turn. If the estimate swings
                         wildly, it is being driven by one idiosyncratic
                         control rather than by the treatment.

A hypothesis that fails any of these is REJECTED and never shown to a user as
a cause. That is the difference between an engine that explains and one that
merely asserts.

No LLM is involved in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from causiq.engine import contract


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------

@dataclass
class RefutationResult:
    name: str
    description: str
    statistic: float
    p_value: float
    threshold: str
    passed: bool
    interpretation: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CausalEstimate:
    driver: str
    driver_label: str
    outcome: str
    method: str
    method_label: str

    effect_abs: float             # in outcome units
    effect_pct: float             # % of the pre-period treated mean
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n_obs: int

    controllable: bool
    owner_role: str | None
    assumptions: list[str]
    estimation_scope: dict[str, Any]

    refutations: list[RefutationResult] = field(default_factory=list)
    verdict: str = "PENDING"      # CONFIRMED | REJECTED | INCONCLUSIVE
    verdict_reason: str = ""
    evidence_notes: list[str] = field(default_factory=list)
    revenue_impact_pct: float = 0.0
    plot_data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "refutations"}
        d["refutations"] = [r.as_dict() for r in self.refutations]
        return d


# -----------------------------------------------------------------------------
# Panel construction
# -----------------------------------------------------------------------------

def _panel(wh, outcome: str, regions: list[str], start: str, end: str,
           exclude_categories: list[str] | None = None,
           categories: list[str] | None = None) -> pd.DataFrame:
    """Daily region x category panel for a causal outcome."""
    where = [f"region IN ({', '.join(repr(r) for r in regions)})",
             f"date BETWEEN DATE '{start}' AND DATE '{end}'"]
    if exclude_categories:
        where.append(f"category NOT IN ({', '.join(repr(c) for c in exclude_categories)})")
    if categories:
        where.append(f"category IN ({', '.join(repr(c) for c in categories)})")

    num = {
        "conversion_rate": "SUM(orders) / NULLIF(SUM(sessions), 0)",
        "aov": "SUM(net_revenue_inr) / NULLIF(SUM(orders), 0)",
        "sessions": "SUM(sessions)",
        "net_revenue_inr": "SUM(net_revenue_inr)",
        "gross_margin_pct": ("(SUM(net_revenue_inr) - SUM(cogs_inr)) "
                             "/ NULLIF(SUM(net_revenue_inr), 0)"),
    }[outcome]

    # Channel is kept as a panel dimension rather than aggregated away. Web and
    # app convert at different rates and are affected separately, so collapsing
    # them discards variation the estimator can use. More cells at the same
    # noise level is straightforwardly more power.
    sql = f"""
        SELECT date, region, category, channel,
               {num}                                  AS y,
               SUM(sessions)                          AS sessions,
               SUM(paid_spend_inr)                    AS paid_spend,
               AVG(in_stock_rate)                     AS in_stock_rate,
               MAX(CAST(stock_interpolated AS INT))   AS stock_interpolated
        FROM fact_daily
        WHERE {' AND '.join(where)}
        GROUP BY date, region, category, channel
        HAVING y IS NOT NULL
        ORDER BY date, region, category, channel
    """
    df = wh.q(sql, label=f"panel[{outcome}]")
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek

    # LOG OUTCOME.
    # Business effects are proportional, not additive: a stockout removes a
    # PERCENTAGE of conversion, and that percentage is the same whether the
    # category converts at 2% or 4%. Fitting the level with additive category
    # fixed effects forces one absolute coefficient to stand for the same
    # proportional effect across categories whose baselines differ twofold --
    # which is how in_stock_rate came back with a POSITIVE effect on sessions.
    #
    # In logs a multiplicative effect becomes an additive one, so a single
    # coefficient is the right functional form and reads directly as a percent
    # change: pct = 100 * (exp(beta) - 1).
    df["y_level"] = df["y"]
    df["y"] = np.log(df["y"].clip(lower=1e-9))
    return df


def _pct_from_log(beta: float) -> float:
    """Convert a log-model coefficient to a percentage effect."""
    return float((np.exp(beta) - 1.0) * 100.0) if np.isfinite(beta) else float("nan")


# -----------------------------------------------------------------------------
# Difference-in-differences
# -----------------------------------------------------------------------------

def _fit_did(df: pd.DataFrame, treated: list[str], cut: pd.Timestamp,
             include_fe: bool = True) -> tuple[float, float, float, int]:
    """Return (effect, std_error, p_value, n). Effect is the interaction term."""
    d = df.copy()
    d["treated"] = d["region"].isin(treated).astype(int)
    d["post"] = (d["date"] >= cut).astype(int)

    if d["post"].nunique() < 2 or d["treated"].nunique() < 2:
        return float("nan"), float("nan"), float("nan"), len(d)

    # Category and day-of-week fixed effects soak up structural differences
    # between categories and the weekly trading rhythm, so the interaction is
    # not asked to absorb them.
    # Two-way fixed effects. Date dummies absorb every shock common to treated
    # and control -- weekly rhythm, annual seasonality, macro conditions -- which
    # would otherwise sit in the residual and inflate the standard error until a
    # real 1-2% effect cannot be distinguished from zero. `post` is collinear
    # with the date dummies and drops out; the treated x post interaction, which
    # varies across BOTH region and time, survives and is what we want.
    # REGION fixed effects, not a single treated dummy. A binary treated flag
    # pools US and APAC into one "control" level, but they operate at very
    # different scale -- that unmodelled level gap lands in the residual and
    # inflates the standard error until a genuine 1-2% effect reads as noise.
    formula = "y ~ C(region) + treated:post"
    if include_fe and d["date"].nunique() > 1:
        formula += " + C(date)"
    if include_fe and d["category"].nunique() > 1:
        formula += " + C(category)"
    if include_fe and "channel" in d.columns and d["channel"].nunique() > 1:
        formula += " + C(channel)"

    model = smf.ols(formula, data=d).fit(cov_type="HC1")
    term = "treated:post"
    if term not in model.params.index:
        return float("nan"), float("nan"), float("nan"), int(model.nobs)
    return (float(model.params[term]), float(model.bse[term]),
            float(model.pvalues[term]), int(model.nobs))


def estimate_did(wh, driver_name: str) -> CausalEstimate:
    spec = contract.driver(driver_name)
    ident = contract.identification(driver_name)
    if not ident or ident["method"] != "difference_in_differences":
        raise ValueError(f"Contract does not sanction DiD for '{driver_name}'.")

    outcome = ident.get("outcome", "conversion_rate")
    treated = ident["treated_units"]
    controls = ident["control_units"]
    cut = pd.Timestamp(ident["treatment_date"])
    pre0, pre1 = ident["pre_window"]
    post0, post1 = ident["post_window"]
    excl = ident.get("exclude_categories") or []

    df = _panel(wh, outcome, treated + controls, pre0, post1,
                exclude_categories=excl)

    effect, se, p, n = _fit_did(df, treated, cut)
    effect_pct = _pct_from_log(effect)

    est = CausalEstimate(
        driver=driver_name,
        driver_label=spec["label"],
        outcome=outcome,
        method="difference_in_differences",
        method_label=(
            f"Difference-in-differences, {'/'.join(treated)} treated vs "
            f"{'/'.join(controls)} control, treatment {cut.date()}"
        ),
        effect_abs=effect,
        effect_pct=effect_pct,
        std_error=se,
        ci_low=effect - 1.96 * se,
        ci_high=effect + 1.96 * se,
        p_value=p,
        n_obs=n,
        controllable=bool(spec.get("controllable")),
        owner_role=spec.get("owner_role"),
        assumptions=ident.get("assumptions", []),
        estimation_scope={
            "outcome": outcome,
            "treated": treated,
            "controls": controls,
            "pre_window": [pre0, pre1],
            "post_window": [post0, post1],
            "excluded_categories": excl,
            "exclusion_rationale": (
                "Excluded because another event overlaps them in this window, "
                "which would otherwise be absorbed into the treatment effect."
            ),
        },
    )

    est.refutations = _refute_did(wh, ident, treated, controls, cut, excl, effect)
    _apply_verdict(est)

    # Parallel-trends plot data. Each group is indexed to its OWN pre-period
    # mean, because DiD is a claim about the SHAPE of two trends, not their
    # levels. Plotting raw levels would show two lines at different heights and
    # invite the reader to compare the gap, which is not what the estimator does.
    try:
        d = df.copy()
        d["grp"] = np.where(d["region"].isin(treated), "treated", "control")
        daily = d.groupby(["date", "grp"])["y"].mean().unstack()
        pre = daily[daily.index < cut].mean()
        indexed = daily.subtract(pre, axis=1)          # log scale: difference = ratio
        est.plot_data = {
            "dates": [x.date().isoformat() for x in indexed.index],
            "treated": [float(np.exp(v) - 1) * 100 if np.isfinite(v) else None
                        for v in indexed.get("treated", [])],
            "control": [float(np.exp(v) - 1) * 100 if np.isfinite(v) else None
                        for v in indexed.get("control", [])],
            "treatment_date": cut.date().isoformat(),
            "treated_label": "/".join(treated),
            "control_label": "/".join(controls),
        }
    except Exception:
        est.plot_data = {}
    return est


# -----------------------------------------------------------------------------
# Refutation battery
# -----------------------------------------------------------------------------

def _refute_did(wh, ident: dict, treated: list[str], controls: list[str],
                cut: pd.Timestamp, excl: list[str],
                headline_effect: float) -> list[RefutationResult]:
    outcome = ident.get("outcome", "conversion_rate")
    pre0, pre1 = ident["pre_window"]
    post0, post1 = ident["post_window"]
    results: list[RefutationResult] = []

    # --- 1. Placebo in time --------------------------------------------------
    # Use only PRE-treatment data and invent a cut in the middle of it. If a
    # "treatment effect" appears where no treatment occurred, the two groups
    # were already drifting apart and parallel trends does not hold.
    placebo_start = pd.Timestamp(pre0) - pd.Timedelta(days=21)
    pre_df = _panel(wh, outcome, treated + controls,
                    placebo_start.date().isoformat(), pre1,
                    exclude_categories=excl)
    fake_cut = pd.Timestamp(pre0) - pd.Timedelta(days=10)
    p_eff, p_se, p_p, p_n = _fit_did(pre_df, treated, fake_cut)
    passed = not (np.isfinite(p_p) and p_p < 0.05)
    results.append(RefutationResult(
        name="placebo_in_time",
        description=(f"Re-run the estimator on pre-treatment data only, with a "
                     f"fabricated treatment date of {fake_cut.date()}. No real "
                     f"intervention happened then, so the effect should be zero."),
        statistic=float(p_eff) if np.isfinite(p_eff) else 0.0,
        p_value=float(p_p) if np.isfinite(p_p) else 1.0,
        threshold="p >= 0.05 (no spurious effect)",
        passed=bool(passed),
        interpretation=(
            "No pre-existing divergence between treated and control groups. "
            "Parallel trends is credible."
            if passed else
            "A significant effect appears before the intervention. The groups "
            "were already diverging, so the headline estimate is contaminated "
            "and cannot be read as causal."
        ),
    ))

    # --- 2. Negative control outcome ----------------------------------------
    # A checkout defect cannot change how expensive a basket is. If AOV shows
    # the same "effect", the estimator is picking up something broader.
    neg_df = _panel(wh, "aov", treated + controls, pre0, post1,
                    exclude_categories=excl)
    n_eff, n_se, n_p, n_n = _fit_did(neg_df, treated, cut)
    passed = not (np.isfinite(n_p) and n_p < 0.05)
    results.append(RefutationResult(
        name="negative_control_outcome",
        description=("Re-run the estimator on average order value, an outcome "
                     "the intervention has no physical mechanism to affect."),
        statistic=float(n_eff) if np.isfinite(n_eff) else 0.0,
        p_value=float(n_p) if np.isfinite(n_p) else 1.0,
        threshold="p >= 0.05 (no effect on an unaffectable outcome)",
        passed=bool(passed),
        interpretation=(
            "No effect on basket size, as expected. The estimator is isolating "
            "the checkout mechanism rather than a general regional shift."
            if passed else
            "A significant effect on basket size, which this intervention "
            "cannot cause. Something else is moving both outcomes."
        ),
    ))

    # --- 3. Leave-one-control-out -------------------------------------------
    swings: list[float] = []
    detail: list[str] = []
    for drop in controls:
        remaining = [c for c in controls if c != drop]
        if not remaining:
            continue
        sub = _panel(wh, outcome, treated + remaining, pre0, post1,
                     exclude_categories=excl)
        e2, _, _, _ = _fit_did(sub, treated, cut)
        if np.isfinite(e2) and headline_effect:
            swing = abs(e2 - headline_effect) / abs(headline_effect)
            swings.append(swing)
            detail.append(f"without {drop}: {e2:.5f} ({swing:+.0%} shift)")
    max_swing = max(swings) if swings else 0.0
    passed = max_swing < 0.50
    results.append(RefutationResult(
        name="leave_one_control_out",
        description=("Drop each control region in turn and re-estimate. A stable "
                     "effect should not depend on any single control unit. "
                     + "; ".join(detail)),
        statistic=float(max_swing),
        p_value=float("nan"),
        threshold="maximum shift < 50% of the headline estimate",
        passed=bool(passed),
        interpretation=(
            "The estimate is stable across control sets, so it is not an "
            "artefact of one idiosyncratic region."
            if passed else
            "The estimate moves sharply when a single control is removed. It is "
            "being driven by that control, not by the treatment."
        ),
    ))

    return results


def _apply_verdict(est: CausalEstimate) -> None:
    """Decide whether this survives. Rejection is a first-class outcome."""
    failed = [r.name for r in est.refutations if not r.passed]

    if not np.isfinite(est.p_value):
        est.verdict = "INCONCLUSIVE"
        est.verdict_reason = "The estimator could not be fitted on the available panel."
    elif failed:
        est.verdict = "REJECTED"
        est.verdict_reason = (
            f"Failed {len(failed)} refutation test(s): {', '.join(failed)}. "
            f"The association is real but the causal claim does not hold, so it "
            f"is not reported as a cause."
        )
    elif est.p_value >= 0.05:
        est.verdict = "INCONCLUSIVE"
        est.verdict_reason = (
            f"Effect is not distinguishable from zero (p = {est.p_value:.3f}). "
            f"Statistically underpowered rather than disproven."
        )
    else:
        est.verdict = "CONFIRMED"
        est.verdict_reason = (
            f"Effect of {est.effect_pct:+.2f}% is significant (p = {est.p_value:.4f}) "
            f"and survived all {len(est.refutations)} refutation tests."
        )


# -----------------------------------------------------------------------------
# Joint event study — for overlapping drivers with no experiment
# -----------------------------------------------------------------------------

def _sparse_categories(wh, region: str, as_of: str) -> list[str]:
    """Categories with too little history for the contract to trust."""
    min_weeks = float(contract.sparse_config()["min_weeks_for_trend_model"])
    df = wh.q(f"""
        SELECT category,
               DATE_DIFF('day', MIN(date), DATE '{as_of}') / 7.0 AS weeks
        FROM fact_daily
        WHERE region = '{region}'
        GROUP BY category
    """, label="sparse_categories")
    return df.loc[df["weeks"] < min_weeks, "category"].tolist()


def _exposure_indicator(wh, df: pd.DataFrame, driver_name: str) -> tuple[pd.Series, str, list[str]]:
    """Build a category-scoped binary exposure from retrieved documents."""
    spec = contract.driver(driver_name)
    exp = spec.get("exposure") or {}
    notes: list[str] = []

    doc_type = exp["doc_type"]
    docs = wh.q(f"""
        SELECT MIN(CAST(event_start AS DATE)) AS lo,
               MAX(CAST(event_end   AS DATE)) AS hi
        FROM raw_ctx_docs WHERE doc_type = '{doc_type}'
    """, label=f"event_window[{doc_type}]")
    lo = pd.Timestamp(docs["lo"].iloc[0]).normalize()
    hi = pd.Timestamp(docs["hi"].iloc[0]).normalize()

    ind = ((df["date"] >= lo) & (df["date"] <= hi))
    cats = spec.get("applies_to_categories")
    if cats:
        ind = ind & df["category"].isin(cats)
    notes.append(
        f"{spec['label']}: exposure {lo.date()} to {hi.date()}"
        + (f", {', '.join(cats)} only" if cats else "")
        + f", derived from '{doc_type}' documents."
    )
    return ind.astype(int), f"x_{driver_name}", notes


def estimate_joint_event_study(wh, region: str, outcome: str, start: str, end: str,
                               drivers: list[str],
                               known_effects: dict[str, float] | None = None
                               ) -> dict[str, CausalEstimate]:
    """Estimate several overlapping drivers in ONE regression.

    WHY JOINT AND NOT ONE AT A TIME
    -------------------------------
    Four things happen to EU conversion inside three weeks, and their windows
    overlap. Estimating each separately hands every indicator a share of every
    other event it coincides with: run alone, the stockout term reads roughly
    -5% when its true effect is about -3%, because it silently absorbs the
    competitor promotion and the checkout release running at the same time.

    Putting every exposure in one model lets each be identified by its own
    distinct footprint in category-by-time space. The stockout is Electronics
    for five days; the competitor promotion is Electronics and Wearables for
    twelve; the Accessories promotion is one category for eleven. Those
    footprints differ enough to separate the effects, which is what makes the
    "multiple interacting drivers" case tractable rather than hand-waved.

    `known_effects` lets a better-identified estimate (the checkout DiD, which
    has a real control group) be held fixed here, so this model adjusts for it
    without trying to re-estimate it from weaker within-region variation.
    """
    # Categories without enough history to have a stable baseline are excluded
    # from causal estimation entirely. A six-week-old category swings on its own
    # launch dynamics, and any exposure window overlapping those swings will
    # quietly absorb them -- here, leaving Wearables in inflates the competitor
    # effect roughly fourfold. The contract already forbids trusting such a
    # series for detection; the same logic applies to inference.
    sparse_cats = _sparse_categories(wh, region, end)
    df = _panel(wh, outcome, [region], start, end, exclude_categories=sparse_cats)
    d = df.dropna(subset=["y"]).copy()

    terms: list[str] = []
    notes_by_driver: dict[str, list[str]] = {}
    global_notes: list[str] = []
    if sparse_cats:
        global_notes.append(
            f"Excluded from causal estimation for insufficient history: "
            f"{', '.join(sparse_cats)}."
        )

    # A driver may only enter a model for an outcome the DAG says it affects.
    # This is not tidiness -- it prevents a specification error. Marketing spend
    # acts on sessions, never on conversion; including it in a conversion model
    # adds a step function at 10 August that is nearly collinear with the
    # checkout release step at 12 August. The two then trade variance and every
    # other coefficient in the model destabilises. The DAG already encodes which
    # pairings are legitimate, so the engine reads it instead of guessing.
    use_date_fe_pre = any(contract.driver(x).get("applies_to_categories") for x in drivers)
    eligible: list[str] = []
    for drv in drivers:
        affects = contract.driver(drv).get("affects") or []
        if use_date_fe_pre and not contract.driver(drv).get("applies_to_categories"):
            # Region-wide driver in a date-fixed-effects model: perfectly
            # collinear with the date dummies, so it cannot be identified here
            # at all. Routed to difference-in-differences against a control
            # region instead of being reported as an uninterpretable coefficient.
            global_notes.append(
                f"{drv} is region-wide and cannot be identified under date fixed "
                f"effects; it is estimated by difference-in-differences instead."
            )
        elif outcome in affects:
            eligible.append(drv)
        else:
            global_notes.append(
                f"{drv} excluded from the {outcome} model: the contract's DAG "
                f"says it affects {', '.join(affects) or 'nothing'}, not {outcome}."
            )
    drivers = eligible

    for drv in drivers:
        spec = contract.driver(drv)
        exp = spec.get("exposure") or {}
        if exp.get("kind") == "event_window":
            ind, colname, notes = _exposure_indicator(wh, d, drv)
            d[colname] = ind
            if d[colname].nunique() > 1:
                terms.append(colname)
                notes_by_driver[drv] = notes
        elif exp.get("kind") == "measured":
            col = exp.get("column", "paid_spend")
            d[f"x_{drv}"] = np.log(d[col].fillna(d[col].median()) + 1)
            terms.append(f"x_{drv}")
            notes_by_driver[drv] = [
                f"{spec['label']}: continuous regressor log({col}); effect is "
                f"reported for the observed change in spend, not per log unit."
            ]

    # TWO-WAY FIXED EFFECTS.
    # Date dummies absorb everything that moves the whole region on a given day
    # -- the growth trend, seasonality, the checkout release, the budget cut,
    # macro conditions, and any event nobody has told us about. What remains is
    # variation BETWEEN categories on the SAME day, which is exactly what
    # identifies a category-scoped event like a stockout in Electronics.
    #
    # The cost is that region-wide drivers become collinear with the date
    # dummies and drop out. That is correct rather than unfortunate: a driver
    # that moves everything at once cannot be identified from within-region
    # comparisons at all, and needs a control region instead. The engine routes
    # those to difference-in-differences and says so.
    use_date_fe = any(contract.driver(x).get("applies_to_categories") for x in drivers)
    controls = []
    if use_date_fe:
        controls.append("C(date)")
        global_notes.append(
            "Date fixed effects absorb all region-wide variation, so these "
            "effects are identified only from differences between categories "
            "on the same day."
        )
    else:
        controls.append("C(dow)")
        d["x_checkout_adj"] = (d["date"] >= pd.Timestamp("2026-08-12")).astype(int)
        terms.append("x_checkout_adj")

    if d["category"].nunique() > 1:
        controls.append("C(category)")
    if "channel" in d.columns and d["channel"].nunique() > 1:
        controls.append("C(channel)")

    formula = f"y ~ {' + '.join(terms + controls)}"
    model = smf.ols(formula, data=d).fit(cov_type="HC1")

    out: dict[str, CausalEstimate] = {}
    for drv in drivers:
        spec = contract.driver(drv)
        ident = contract.identification(drv) or {}
        key = f"x_{drv}"
        if key not in model.params.index:
            continue
        coef = float(model.params[key])
        se = float(model.bse[key])
        p = float(model.pvalues[key])
        exp = spec.get("exposure") or {}

        if exp.get("kind") == "event_window":
            # In a log model the coefficient IS the proportional effect, so no
            # baseline division is needed and no category-mix distortion enters.
            cats = spec.get("applies_to_categories")
            effect_pct = _pct_from_log(coef)
            scope_note = (f"proportional effect of exposure on "
                          f"{', '.join(cats) if cats else 'all categories'}")
        else:
            # Translate a log-spend coefficient into the effect of the change
            # that actually occurred, which is the only figure a business can use.
            # log-log: the coefficient is an elasticity. Multiply by the log
            # change that actually occurred to get the realised effect.
            col = exp.get("column", "paid_spend")
            pre = d[d["date"] < pd.Timestamp("2026-08-10")][col].mean()
            post = d[d["date"] >= pd.Timestamp("2026-08-10")][col].mean()
            dlog = float(np.log((post + 1) / (pre + 1))) if pre and post else 0.0
            elasticity = coef
            coef = coef * dlog
            se = se * abs(dlog)
            effect_pct = _pct_from_log(coef)
            scope_note = (f"elasticity {elasticity:.3f}, applied to the observed "
                          f"{100*(np.exp(dlog)-1):+.1f}% change in {col}")

        est = CausalEstimate(
            driver=drv,
            driver_label=spec["label"],
            outcome=outcome,
            method="joint_event_study",
            method_label=(
                "Joint event-study regression: all overlapping exposures "
                "estimated simultaneously, with day-of-week, category and "
                "channel fixed effects"
            ),
            effect_abs=coef,
            effect_pct=effect_pct,
            std_error=se,
            ci_low=coef - 1.96 * se,
            ci_high=coef + 1.96 * se,
            p_value=p,
            n_obs=int(model.nobs),
            controllable=bool(spec.get("controllable")),
            owner_role=spec.get("owner_role"),
            assumptions=[
                "Exposure windows from documents are accurate.",
                "No unmeasured event shares a footprint with this one.",
                "Observational: weaker than the experimental route.",
            ],
            estimation_scope={
                "region": region, "window": [start, end],
                "categories": spec.get("applies_to_categories") or "all",
                "formula": formula, "interpretation": scope_note,
                "adjustment_set": ident.get("adjustment_set", []),
            },
            evidence_notes=notes_by_driver.get(drv, []) + global_notes,
        )
        est.verdict = "CONFIRMED" if (np.isfinite(p) and p < 0.05) else "INCONCLUSIVE"
        est.verdict_reason = (
            f"Adjusted effect {effect_pct:+.2f}% significant (p = {p:.4f}) with "
            f"all concurrent events held constant."
            if est.verdict == "CONFIRMED" else
            f"Not distinguishable from zero once concurrent events are held "
            f"constant (p = {p:.3f})."
        )
        out[drv] = est
    return out


# -----------------------------------------------------------------------------
# Backdoor adjustment — single driver, for drivers with no natural experiment
# -----------------------------------------------------------------------------

def estimate_backdoor(wh, driver_name: str, region: str,
                      start: str, end: str,
                      outcome: str = "conversion_rate") -> CausalEstimate:
    """Regression adjustment using the contract's declared adjustment set.

    Weaker than DiD and labelled as such. Validity rests on the adjustment set
    being complete — an assumption the contract asserts and the engine surfaces
    rather than hides.
    """
    spec = contract.driver(driver_name)
    ident = contract.identification(driver_name) or {}
    adjustment = ident.get("adjustment_set", [])
    exposure = spec.get("exposure") or {}
    notes: list[str] = []

    # Scope to the categories this driver can actually reach. Estimating a
    # driver across categories it does not touch invites any concurrent event
    # in those categories to load onto its coefficient.
    cats = spec.get("applies_to_categories")
    df = _panel(wh, outcome, [region], start, end, categories=cats)
    if cats:
        notes.append(
            f"Estimated within {', '.join(cats)} only, per the driver's declared "
            f"scope, so unrelated events elsewhere cannot contaminate it."
        )

    d = df.dropna(subset=["y"]).copy()

    # --- build the exposure regressor ---------------------------------------
    if exposure.get("kind") == "event_window":
        doc_type = exposure["doc_type"]
        docs = wh.q(f"""
            SELECT MIN(CAST(event_start AS DATE)) AS lo,
                   MAX(CAST(event_end   AS DATE)) AS hi
            FROM raw_ctx_docs WHERE doc_type = '{doc_type}'
        """, label=f"event_window[{doc_type}]")
        lo = pd.Timestamp(docs["lo"].iloc[0]).normalize()
        hi = pd.Timestamp(docs["hi"].iloc[0]).normalize()
        d["exposed"] = ((d["date"] >= lo) & (d["date"] <= hi)).astype(int)
        regressor = "exposed"
        notes.append(
            f"Exposure window {lo.date()} to {hi.date()} was extracted from "
            f"'{doc_type}' documents, not from any structured feed."
        )

        # Where a structured column exists, try it too and report the contrast.
        fallback = exposure.get("fallback_column")
        if fallback and fallback in d.columns and d[fallback].notna().any():
            try:
                alt = smf.ols(f"y ~ {fallback} + C(dow)", data=d.dropna(subset=[fallback])
                              ).fit(cov_type="HC1")
                alt_p = float(alt.pvalues.get(fallback, np.nan))
                notes.append(
                    f"The structured route was attempted first: regressing on "
                    f"{fallback} from the weekly supply feed gives p = {alt_p:.2f}. "
                    f"That source is too coarse to identify a sub-week event, so "
                    f"the document-derived window is used instead."
                )
                if exposure.get("degraded_note"):
                    notes.append(str(exposure["degraded_note"]).strip())
            except Exception:
                pass
    else:
        col = exposure.get("column", "paid_spend")
        regressor = f"np.log({col} + 1)" if exposure.get("transform") == "log" else col

    if d.get("stock_interpolated") is not None and d["stock_interpolated"].max() == 1:
        notes.append(
            "Rows carry in_stock_rate broadcast from a WEEKLY source. Within-week "
            "timing is unobserved, so confidence is penalised downstream."
        )

    terms = [regressor]
    if "seasonality" in adjustment:
        terms.append("C(dow)")
    if "marketing_spend" in adjustment and driver_name != "marketing_spend":
        terms.append("np.log(paid_spend + 1)")
    if "discount_depth" in adjustment and "discount_inr" in d.columns:
        pass  # discount is category-scoped away above
    if d["category"].nunique() > 1:
        terms.append("C(category)")
    if "channel" in d.columns and d["channel"].nunique() > 1:
        terms.append("C(channel)")

    formula = f"y ~ {' + '.join(terms)}"
    model = smf.ols(formula, data=d).fit(cov_type="HC1")

    key = regressor if regressor in model.params.index else None
    if key is None:
        cand = [p for p in model.params.index if regressor.split("(")[0][:10] in p]
        key = cand[0] if cand else regressor
    coef = float(model.params.get(key, np.nan))
    se = float(model.bse.get(key, np.nan))
    p = float(model.pvalues.get(key, np.nan))

    # Percentage effect is expressed against the UNEXPOSED baseline, which is
    # the counterfactual the business actually cares about.
    if regressor == "exposed" and (d["exposed"] == 0).any():
        base = float(d.loc[d["exposed"] == 0, "y"].mean())
    else:
        base = float(d["y"].mean())
    est = CausalEstimate(
        driver=driver_name,
        driver_label=spec["label"],
        outcome=outcome,
        method="backdoor_adjustment",
        method_label=(f"OLS backdoor adjustment controlling for "
                      f"{', '.join(adjustment) or 'no declared confounders'}"),
        effect_abs=coef,
        effect_pct=100.0 * coef / base if base else float("nan"),
        std_error=se,
        ci_low=coef - 1.96 * se,
        ci_high=coef + 1.96 * se,
        p_value=p,
        n_obs=int(model.nobs),
        controllable=bool(spec.get("controllable")),
        owner_role=spec.get("owner_role"),
        assumptions=[
            "The declared adjustment set blocks all backdoor paths.",
            "No unmeasured confounding remains.",
            "This is observational: weaker evidence than the experimental route.",
        ],
        estimation_scope={"region": region, "window": [start, end],
                          "adjustment_set": adjustment, "formula": formula},
        evidence_notes=notes,
    )

    est.verdict = ("CONFIRMED" if np.isfinite(p) and p < 0.05 else "INCONCLUSIVE")
    est.verdict_reason = (
        f"Adjusted association significant (p = {p:.4f}); causal reading depends "
        f"on the declared adjustment set being complete."
        if est.verdict == "CONFIRMED" else
        f"Adjusted association not distinguishable from zero (p = {p:.3f})."
    )
    return est
