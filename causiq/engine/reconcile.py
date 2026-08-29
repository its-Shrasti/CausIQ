"""
Reconciliation — conform four heterogeneous sources onto one governed grain.

This is objective 2 of the brief, and it is where most real BI projects quietly
fail. The three structured sources disagree on all three axes:

    wh_sales       DAILY   grain, T-1 lag,   gold   quality
    web_events     HOURLY  grain, T-4h lag,  silver quality
    ops_inventory  WEEKLY  grain, 240h stale, bronze quality

Conforming them is not just a GROUP BY. Rolling hourly up to daily is lossless.
Spreading weekly down to daily is NOT — it invents within-week detail that was
never measured. So `ops_inventory` is interpolated but every derived row is
stamped `interpolated = TRUE`, and any conclusion that leans on an interpolated
column has its confidence penalised downstream. Being explicit about what the
data cannot support is the point.

All aggregation happens in SQL (DuckDB), not pandas, so the executed query is
capturable as lineage evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from causiq.engine import contract

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
NOW = pd.Timestamp("2026-08-27 09:00:00")   # fixed clock: reproducible demos


# -----------------------------------------------------------------------------
# Freshness
# -----------------------------------------------------------------------------

@dataclass
class Freshness:
    source: str
    system: str
    last_loaded: pd.Timestamp
    age_hours: float
    sla_hours: float
    within_sla: bool
    quality_tier: str
    cadence: str
    grain: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "system": self.system,
            "last_loaded": self.last_loaded.isoformat(),
            "age_hours": round(self.age_hours, 1),
            "sla_hours": self.sla_hours,
            "within_sla": self.within_sla,
            "quality_tier": self.quality_tier,
            "cadence": self.cadence,
            "grain": self.grain,
        }


# -----------------------------------------------------------------------------
# Warehouse
# -----------------------------------------------------------------------------

@dataclass
class Warehouse:
    """A DuckDB session holding the conformed star schema plus its metadata."""
    con: duckdb.DuckDBPyConnection
    freshness: dict[str, Freshness]
    coverage: dict[str, float]
    sql_log: list[str] = field(default_factory=list)

    def q(self, sql: str, label: str | None = None) -> pd.DataFrame:
        """Run a query and record it for the lineage trail."""
        self.sql_log.append((label or "query") + ":\n" + sql.strip())
        return self.con.execute(sql).fetchdf()

    def freshness_table(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_dict() for f in self.freshness.values()])

    def breached_sources(self) -> list[str]:
        return [s for s, f in self.freshness.items() if not f.within_sla]


def _measure_freshness(name: str, last_loaded: pd.Timestamp) -> Freshness:
    spec = contract.source(name)
    age = (NOW - last_loaded).total_seconds() / 3600.0
    return Freshness(
        source=name,
        system=spec["system"],
        last_loaded=last_loaded,
        age_hours=age,
        sla_hours=float(spec["freshness_sla_hours"]),
        within_sla=age <= float(spec["freshness_sla_hours"]),
        quality_tier=spec["quality_tier"],
        cadence=spec["refresh_cadence"],
        grain=", ".join(spec["grain"]),
    )


def build_warehouse() -> Warehouse:
    con = duckdb.connect(":memory:")

    for name in ("wh_sales", "web_events", "ops_inventory", "ctx_docs"):
        con.execute(
            f"CREATE VIEW raw_{name} AS SELECT * FROM read_parquet('{DATA_DIR / (name + '.parquet')}')"
        )

    # --- freshness, measured from the data itself, not asserted -------------
    fresh: dict[str, Freshness] = {}
    fresh["wh_sales"] = _measure_freshness(
        "wh_sales",
        pd.Timestamp(con.execute("SELECT max(date) FROM raw_wh_sales").fetchone()[0])
        + pd.Timedelta(hours=26),
    )
    fresh["web_events"] = _measure_freshness(
        "web_events",
        pd.Timestamp(con.execute(
            "SELECT max(datetime_hour) FROM raw_web_events").fetchone()[0])
        + pd.Timedelta(hours=4),
    )
    fresh["ops_inventory"] = _measure_freshness(
        "ops_inventory",
        pd.Timestamp(con.execute(
            "SELECT max(iso_week) FROM raw_ops_inventory").fetchone()[0])
        + pd.Timedelta(hours=6),
    )
    fresh["ctx_docs"] = _measure_freshness(
        "ctx_docs",
        pd.Timestamp(con.execute(
            "SELECT max(event_datetime) FROM raw_ctx_docs").fetchone()[0]),
    )

    # --- 1. hourly -> daily. Lossless: we are summing measured events -------
    con.execute("""
        CREATE TABLE web_daily AS
        SELECT
            CAST(datetime_hour AS DATE)        AS date,
            region,
            category,
            channel,
            SUM(sessions)                      AS sessions,
            SUM(checkout_starts)               AS checkout_starts,
            SUM(checkout_errors)               AS checkout_errors,
            COUNT(*)                           AS source_rows_rolled_up
        FROM raw_web_events
        GROUP BY 1, 2, 3, 4
    """)

    # --- 2. weekly -> daily. LOSSY: flagged, never silently smoothed --------
    # Each weekly observation is broadcast to its seven days. We do not linearly
    # interpolate between weeks, because that would fabricate a trend the source
    # cannot support. Broadcasting is the honest choice: it says "this is the
    # only resolution we have", and the interpolated flag carries that forward.
    con.execute("""
        CREATE TABLE ops_daily AS
        SELECT
            CAST(o.iso_week AS DATE) + INTERVAL (d.i) DAY  AS date,
            o.region,
            o.category,
            o.in_stock_rate,
            o.days_of_cover,
            o.landed_cogs_index,
            TRUE                                          AS interpolated,
            CAST(o.iso_week AS DATE)                      AS observed_week
        FROM raw_ops_inventory o
        CROSS JOIN (SELECT unnest(generate_series(0, 6)) AS i) d
    """)

    # --- 3. conformed daily fact -------------------------------------------
    # LEFT JOIN from sales: sales is the financially authoritative spine. A NULL
    # from web or ops means "not measured", which is materially different from
    # zero and must survive into the coverage statistics.
    con.execute("""
        CREATE TABLE fact_daily AS
        SELECT
            s.date,
            s.region,
            s.category,
            s.channel,
            s.orders,
            s.net_revenue_inr,
            s.discount_inr,
            s.cogs_inr,
            s.aov_inr,
            s.paid_spend_inr,
            w.sessions,
            w.checkout_starts,
            w.checkout_errors,
            o.in_stock_rate,
            o.days_of_cover,
            o.landed_cogs_index,
            COALESCE(o.interpolated, FALSE)               AS stock_interpolated,
            CASE WHEN w.sessions > 0
                 THEN s.orders / w.sessions END           AS conversion_rate,
            CASE WHEN s.net_revenue_inr > 0
                 THEN (s.net_revenue_inr - s.cogs_inr)
                      / s.net_revenue_inr END             AS gross_margin_pct
        FROM raw_wh_sales s
        LEFT JOIN web_daily w
               ON s.date = w.date AND s.region = w.region
              AND s.category = w.category AND s.channel = w.channel
        LEFT JOIN ops_daily o
               ON s.date = o.date AND s.region = o.region
              AND s.category = o.category
    """)

    # --- 4. coverage: what fraction of each column actually arrived ---------
    cov_df = con.execute("""
        SELECT
            AVG(CASE WHEN sessions      IS NOT NULL THEN 1.0 ELSE 0.0 END) AS sessions,
            AVG(CASE WHEN in_stock_rate IS NOT NULL THEN 1.0 ELSE 0.0 END) AS in_stock_rate,
            AVG(CASE WHEN discount_inr  IS NOT NULL THEN 1.0 ELSE 0.0 END) AS discount_inr,
            AVG(CASE WHEN cogs_inr      IS NOT NULL THEN 1.0 ELSE 0.0 END) AS cogs_inr
        FROM fact_daily
        WHERE date >= DATE '2026-08-01'
    """).fetchdf()
    coverage = {c: float(cov_df[c].iloc[0]) for c in cov_df.columns}

    return Warehouse(con=con, freshness=fresh, coverage=coverage)


# -----------------------------------------------------------------------------
# Scoped reads — entitlements applied HERE, before anything reaches a prompt
# -----------------------------------------------------------------------------

def scoped_fact(wh: Warehouse, role: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read fact_daily through a role's entitlement.

    Row filtering and column denial are applied in SQL. There is no code path
    by which a denied column reaches the narrative layer, so no prompt can be
    talked into revealing it.
    """
    ent = contract.entitlement(role)
    denied = set(ent.get("denied_columns") or [])
    denied |= {k for k in (ent.get("denied_kpis") or [])}

    all_cols = [r[0] for r in wh.con.execute("DESCRIBE fact_daily").fetchall()]
    keep = [c for c in (columns or all_cols) if c not in denied]

    where = ent.get("row_filter") or "1=1"
    sql = f"SELECT {', '.join(keep)} FROM fact_daily WHERE {where}"
    return wh.q(sql, label=f"scoped_fact[{role}]")


def coverage_for(wh: Warehouse, columns: list[str]) -> float:
    """Minimum global coverage across the columns a conclusion depends on."""
    vals = [wh.coverage.get(c, 1.0) for c in columns]
    return min(vals) if vals else 1.0


def coverage_in_scope(wh: Warehouse, columns: list[str], region: str,
                      start: str, end: str) -> dict[str, float]:
    """Coverage measured INSIDE the analysis scope, not across the whole table.

    This distinction decides whether the engine abstains. A load failure that
    nulls 18% of EU rows dilutes to roughly 1% when averaged over four regions
    and three years — so a globally-computed coverage figure would wave the
    defect through. Data quality has to be judged where the question is being
    asked, not on the table as a whole.
    """
    exprs = ", ".join(
        f"AVG(CASE WHEN {c} IS NOT NULL THEN 1.0 ELSE 0.0 END) AS {c}"
        for c in columns
    )
    sql = f"""
        SELECT {exprs}
        FROM fact_daily
        WHERE region = '{region}'
          AND date BETWEEN DATE '{start}' AND DATE '{end}'
    """
    df = wh.q(sql, label="coverage_in_scope")
    return {c: (float(df[c].iloc[0]) if pd.notna(df[c].iloc[0]) else 0.0)
            for c in columns}
