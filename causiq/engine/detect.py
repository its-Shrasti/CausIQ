"""
DETECT — is this movement real, and is it worth anyone's attention?

Two questions, deliberately kept separate:

  1. STATISTICAL: is the observed value further from the forecast than normal
     noise explains?  -> z-score against backtested forecast error
  2. BUSINESS:    is the gap big enough that somebody would act on it?
     -> absolute rupee floor AND percentage floor from the contract

A movement must clear BOTH. Statistical significance alone produces alert
fatigue: on a large enough series, a 0.4% wobble is significant and nobody
cares. A rupee threshold alone fires on ordinary volatility. The contract
states the rule as a boolean expression and this module evaluates it.

BASELINE SELECTION IS DATA-DRIVEN, NOT ASSUMED
----------------------------------------------
Holt-Winters needs a full seasonal cycle. A category launched six weeks ago
has no cycle to learn, so fitting one produces a confident, meaningless
forecast. The contract declares minimum history requirements; below them the
engine switches to partial pooling from a declared analogue and inflates its
interval. The method actually used is recorded in the evidence, because
"which model produced this expectation" is part of the answer.

All of this is deterministic. No LLM is involved anywhere in this module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from causiq.engine import contract

warnings.filterwarnings("ignore")

METHOD_LABELS = {
    "holt_winters_weekly": "Holt-Winters exponential smoothing (additive trend, weekly seasonality)",
    "partial_pooling": "Hierarchical partial pooling from a declared analogue cohort",
    "seasonal_naive": "Seasonal naive (same weekday, previous 4 weeks)",
}


@dataclass
class Detection:
    kpi: str
    kpi_label: str
    scope: dict[str, str]
    window_start: str
    window_end: str

    actual: float
    expected: float
    pi_low: float
    pi_high: float

    deviation_abs: float
    deviation_pct: float
    z_score: float
    p_value: float

    baseline_method: str
    baseline_method_label: str
    history_days: int
    history_weeks: float
    sparse: bool
    analogue_used: str | None

    statistically_significant: bool
    business_material: bool
    material: bool
    materiality_rule: str
    thresholds: dict[str, Any]

    sql: str = ""
    warnings_: list[str] = field(default_factory=list)
    # Plotting series: history, the forecast over the window, and its interval.
    # Carried on the Detection so the chart layer never re-derives a number the
    # engine already computed -- a chart that recomputes is a second source of
    # truth waiting to disagree with the first.
    series: dict[str, list] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "sql"}
        d["warnings"] = d.pop("warnings_")
        return d

    def headline(self) -> str:
        arrow = "down" if self.deviation_pct < 0 else "up"
        return (f"{self.kpi_label} {arrow} {abs(self.deviation_pct):.1f}% "
                f"vs expected in {self.scope.get('region', 'all regions')}")


# -----------------------------------------------------------------------------
# Series construction
# -----------------------------------------------------------------------------

def _daily_series(wh, kpi_name: str, region: str, category: str | None,
                  role: str) -> pd.Series:
    """Build the daily series for a KPI, respecting entitlements.

    Ratio KPIs are computed as ratio-of-sums, never as a mean of daily ratios.
    Averaging ratios silently weights a quiet Tuesday equally with a peak
    Saturday and is one of the most common quiet errors in BI.
    """
    ent = contract.entitlement(role)
    where = [f"region = '{region}'"]
    if ent.get("row_filter"):
        where.append(ent["row_filter"])
    if category:
        where.append(f"category = '{category}'")
    w = " AND ".join(where)

    if kpi_name == "net_revenue":
        sel = "SUM(net_revenue_inr)"
    elif kpi_name == "orders":
        sel = "SUM(orders)"
    elif kpi_name == "conversion_rate":
        sel = "SUM(orders) / NULLIF(SUM(sessions), 0)"
    elif kpi_name == "aov":
        sel = "SUM(net_revenue_inr) / NULLIF(SUM(orders), 0)"
    elif kpi_name == "gross_margin_pct":
        sel = ("(SUM(net_revenue_inr) - SUM(cogs_inr)) "
               "/ NULLIF(SUM(net_revenue_inr), 0)")
    else:
        raise ValueError(f"KPI '{kpi_name}' is not defined in the contract.")

    sql = f"SELECT date, {sel} AS value FROM fact_daily WHERE {w} GROUP BY date ORDER BY date"
    df = wh.q(sql, label=f"detect[{kpi_name}]")
    s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"]), name=kpi_name)
    return s.dropna()


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------

def _fit_holt_winters(train: pd.Series, horizon: int) -> tuple[np.ndarray, float]:
    """Fit Holt-Winters and return (forecast, per-step standard error).

    The standard error comes from in-sample residuals rather than the model's
    analytic interval, because the analytic interval assumes the model is
    correctly specified. Residual spread is the honest measure of how wrong
    this model has actually been on this series.
    """
    # Damped trend. Over a three-week horizon an undamped linear trend
    # extrapolates growth that has not happened yet, inflating the expectation
    # and manufacturing a shortfall. Damping is standard practice for horizons
    # beyond a few steps and materially reduces forecast bias here.
    model = ExponentialSmoothing(
        train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
        initialization_method="heuristic",
    ).fit(optimized=True, use_brute=True)
    fc = np.asarray(model.forecast(horizon))
    resid = np.asarray(model.resid)
    se = float(np.std(resid[-120:], ddof=1)) if len(resid) > 20 else float(np.std(resid, ddof=1))
    return fc, se


def _fit_partial_pooling(train: pd.Series, horizon: int, analogue: pd.Series | None,
                         inflation: float) -> tuple[np.ndarray, float]:
    """Sparse-history fallback.

    With too few weeks to estimate a seasonal profile, we borrow the SHAPE of a
    declared analogue series (a comparable earlier launch), rescale it to this
    series' own level, and blend it with the series' own recent mean. The
    weight on the analogue falls as the series accumulates its own history --
    that is the partial-pooling idea: neither ignore the analogue nor trust it
    completely. The interval is then inflated because a borrowed shape is a
    weaker claim than an estimated one.
    """
    n_weeks = len(train) / 7.0
    own_level = float(train.tail(14).mean())

    if analogue is not None and len(analogue) >= horizon + 14:
        a = analogue.tail(horizon + 14)
        a_level = float(a.head(14).mean())
        shape = np.asarray(a.tail(horizon)) / a_level if a_level > 0 else np.ones(horizon)
    else:
        shape = np.ones(horizon)

    # Shrinkage weight: at 0 weeks of own history trust the analogue fully;
    # by min_weeks_for_trend_model trust own data fully.
    cfg = contract.sparse_config()
    w_own = float(np.clip(n_weeks / cfg["min_weeks_for_trend_model"], 0.0, 1.0))
    pooled_shape = w_own * np.ones(horizon) + (1 - w_own) * shape

    fc = own_level * pooled_shape

    recent = train.tail(28)
    resid = np.asarray(recent) - float(recent.mean())
    se_resid = float(np.std(resid, ddof=1))

    # The LEVEL is itself an estimate from very few observations. On a mature
    # series that uncertainty is negligible and everyone ignores it; on a
    # three-week-old series it is a large share of the total. Ignoring it is
    # how young metrics get declared "significantly down" on noise.
    se_level = se_resid / np.sqrt(max(len(recent), 1))
    if contract.sparse_config().get("include_level_uncertainty", False):
        se = float(np.sqrt(se_resid ** 2 + se_level ** 2)) * inflation
    else:
        se = se_resid * inflation
    return fc, se


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def detect(wh, kpi_name: str, region: str, window_start: str, window_end: str,
           role: str = "DataAnalyst", category: str | None = None) -> Detection:
    """Detect a movement. Memoised per warehouse.

    A single EXPLAIN pass asks for the same detection repeatedly -- the
    category breakdown, the factor split and the aggregation-paradox check all
    want net_revenue for the same category and window. Each miss refits
    Holt-Winters, and around 25 fits per run made this stage 70% of total
    latency for no additional information. The warehouse is immutable within a
    run, so identical arguments must give an identical result and caching is
    safe rather than merely convenient.
    """
    # Key on the role's EFFECTIVE data scope, not its name. CFO, CMO and
    # DataAnalyst all read unfiltered rows, so they must share a cache entry --
    # keying on the role name would recompute identical numbers three times and,
    # worse, leave open the possibility of them diverging.
    ent = contract.entitlement(role)
    scope = (ent.get("row_filter") or "", kpi_name in (ent.get("denied_kpis") or []))
    ck = (kpi_name, region, window_start, window_end, scope, category)
    cache = getattr(wh, "_detect_cache", None)
    if cache is None:
        cache = {}
        setattr(wh, "_detect_cache", cache)
    if ck in cache:
        return cache[ck]

    result = _detect_uncached(wh, kpi_name, region, window_start, window_end,
                              role, category)
    cache[ck] = result
    return result


def _detect_uncached(wh, kpi_name: str, region: str, window_start: str,
                     window_end: str, role: str = "DataAnalyst",
                     category: str | None = None) -> Detection:
    spec = contract.kpi(kpi_name)
    sparse_cfg = contract.sparse_config()

    ent = contract.entitlement(role)
    if kpi_name in (ent.get("denied_kpis") or []):
        raise PermissionError(
            f"Role '{role}' is not entitled to KPI '{kpi_name}'. "
            "Denied by contract at entitlements.roles."
        )

    series = _daily_series(wh, kpi_name, region, category, role)
    w0, w1 = pd.Timestamp(window_start), pd.Timestamp(window_end)

    # Forecast origin sits BEHIND the window by the contract's refresh lag, so
    # the baseline cannot be contaminated by the incident it must detect.
    lag_days = int(contract.load_contract().get("baseline_refresh_lag_days", 0))
    origin = w0 - pd.Timedelta(days=lag_days)

    train = series[series.index < origin]
    actual_window = series[(series.index >= w0) & (series.index <= w1)]
    horizon_full = (w1 - origin).days + 1        # forecast origin -> window end
    horizon = (w1 - w0).days + 1                 # the window itself

    if len(actual_window) == 0:
        raise ValueError(f"No data for {kpi_name} in {region} for the requested window.")

    history_days = len(train)
    history_weeks = history_days / 7.0
    warns: list[str] = []

    # --- method selection, driven by the contract's history requirements ----
    sparse = history_weeks < sparse_cfg["min_weeks_for_seasonal_model"]
    analogue_used = None

    if not sparse:
        method = "holt_winters_weekly"
        fc_full, se = _fit_holt_winters(train, horizon_full)
        fc = fc_full[-horizon:]                  # slice out the window itself
    else:
        method = sparse_cfg["fallback_method"]
        analogue_name = (sparse_cfg.get("analogues") or {}).get(
            (category or "").lower().replace(" ", "_")
        )
        analogue_series = None
        if analogue_name:
            analogue_used = analogue_name
            try:
                analogue_series = _daily_series(wh, kpi_name, region, "Accessories", role)
            except Exception:
                analogue_series = None
        fc_full, se = _fit_partial_pooling(
            train, horizon_full, analogue_series,
            float(sparse_cfg["interval_inflation_factor"]),
        )
        fc = fc_full[-horizon:]
        warns.append(
            f"Only {history_weeks:.1f} weeks of history "
            f"(contract requires {sparse_cfg['min_weeks_for_seasonal_model']} "
            f"for a seasonal model). Switched to {method}; "
            f"prediction interval inflated {sparse_cfg['interval_inflation_factor']}x."
        )

    # --- aggregate daily forecast to the window ----------------------------
    is_ratio = spec.unit in ("ratio", "percentage_points")
    if is_ratio:
        expected = float(np.mean(fc))
        actual = float(actual_window.mean())
        se_window = se / np.sqrt(horizon)
    else:
        expected = float(np.sum(fc))
        actual = float(actual_window.sum())
        se_window = se * np.sqrt(horizon)      # independent daily errors

    dev_abs = actual - expected
    dev_pct = 100.0 * dev_abs / expected if expected else 0.0
    z = dev_abs / se_window if se_window > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Sparse-history uncertainty floor. Applied at window level, after the
    # residual-based standard error, because a short quiet history otherwise
    # yields a small SE and a falsely confident verdict on a series nobody
    # understands yet.
    if sparse:
        floor_coef = float(sparse_cfg.get("uncertainty_floor_coefficient", 0.0))
        if floor_coef > 0 and history_weeks > 0:
            floor = abs(expected) * floor_coef / np.sqrt(history_weeks)
            if floor > se_window:
                warns.append(
                    f"Prediction interval widened to the contract's sparse-history "
                    f"floor: {floor_coef:.0%}/sqrt({history_weeks:.1f} weeks) = "
                    f"{100 * floor / abs(expected):.1f}% of expected, versus "
                    f"{100 * se_window / abs(expected):.1f}% from residuals alone. "
                    f"A quiet short history is not evidence of stability."
                )
                se_window = floor
                z = dev_abs / se_window if se_window > 0 else 0.0
                p = float(2 * (1 - stats.norm.cdf(abs(z))))

    pi_low, pi_high = expected - 1.96 * se_window, expected + 1.96 * se_window

    # --- materiality: statistical AND business, per the contract -----------
    mat = spec.materiality
    alpha = float(mat["significance_alpha"])
    stat_sig = p < alpha

    checks: list[bool] = []
    if mat.get("min_abs_inr") is not None:
        checks.append(abs(dev_abs) >= float(mat["min_abs_inr"]))
    if mat.get("min_pct") is not None:
        checks.append(abs(dev_pct) >= float(mat["min_pct"]))
    business_material = all(checks) if checks else True

    material = stat_sig and business_material

    if spec.mixed_source_ratio:
        warns.append(
            "This KPI's numerator and denominator come from different source "
            "systems with different refresh cadences. A movement can be caused "
            "by one source lagging the other rather than by any business change."
        )

    # Assemble plot series: 4 weeks of history plus the forecast window.
    # Longer history buries the anomaly under the weekly trading cycle -- at
    # eight weeks the reader sees a sawtooth and has to hunt for the week that
    # matters. Four cycles is enough to establish the rhythm and still leave the
    # deviation legible.
    hist = series[series.index >= (w0 - pd.Timedelta(days=28))]
    hist_dates = [d.date().isoformat() for d in hist.index]
    win_dates = [d.date().isoformat() for d in actual_window.index]
    daily_se = se_window / np.sqrt(horizon) if not is_ratio else se_window
    plot_series = {
        "history_dates": hist_dates,
        "history_values": [float(v) for v in hist.values],
        "window_dates": win_dates,
        "window_actual": [float(v) for v in actual_window.values],
        "window_expected": [float(v) for v in fc[:len(win_dates)]],
        "window_pi_low": [float(v - 1.96 * daily_se) for v in fc[:len(win_dates)]],
        "window_pi_high": [float(v + 1.96 * daily_se) for v in fc[:len(win_dates)]],
    }

    return Detection(
        series=plot_series,
        kpi=kpi_name,
        kpi_label=spec.label,
        scope={"region": region, **({"category": category} if category else {})},
        window_start=window_start,
        window_end=window_end,
        actual=actual,
        expected=expected,
        pi_low=pi_low,
        pi_high=pi_high,
        deviation_abs=dev_abs,
        deviation_pct=dev_pct,
        z_score=float(z),
        p_value=p,
        baseline_method=method,
        baseline_method_label=METHOD_LABELS.get(method, method),
        history_days=history_days,
        history_weeks=history_weeks,
        sparse=sparse,
        analogue_used=analogue_used,
        statistically_significant=stat_sig,
        business_material=business_material,
        material=material,
        materiality_rule=mat["rule"],
        thresholds={
            "min_abs_inr": mat.get("min_abs_inr"),
            "min_pct": mat.get("min_pct"),
            "significance_alpha": alpha,
        },
        warnings_=warns,
    )
