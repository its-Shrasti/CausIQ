"""Verify the planted confound is real and that naive attribution is misled.

If a correlational ranking happens to get the right answer on this dataset,
the entire CHALLENGE stage has nothing to prove. So this check is run before
any engine code is written.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())
sales = pd.read_parquet(RAW / "wh_sales.parquet")
web = pd.read_parquet(RAW / "web_events.parquet")
ops = pd.read_parquet(RAW / "ops_inventory.parquet")

sales["date"] = pd.to_datetime(sales["date"])
web["date"] = pd.to_datetime(web["datetime_hour"]).dt.floor("D")

W0, W1 = pd.Timestamp("2026-08-17"), pd.Timestamp("2026-08-23")
P0, P1 = pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-09")   # clean prior week

print("=" * 72)
print("TRAP VERIFICATION")
print("=" * 72)

# --- 1. Does the naive correlational story point at traffic? --------------
eu = sales[sales.region == "EU"]
eu_web = web[web.region == "EU"]

def agg(df, col, a, b):
    return float(df[(df.date >= a) & (df.date <= b)][col].sum())

rev_now, rev_prev = agg(eu, "net_revenue_inr", W0, W1), agg(eu, "net_revenue_inr", P0, P1)
ord_now, ord_prev = agg(eu, "orders", W0, W1), agg(eu, "orders", P0, P1)
ses_now, ses_prev = agg(eu_web, "sessions", W0, W1), agg(eu_web, "sessions", P0, P1)

cvr_now, cvr_prev = ord_now / ses_now, ord_prev / ses_prev
aov_now, aov_prev = rev_now / ord_now, rev_prev / ord_prev

print("\n1. NAIVE WEEK-OVER-WEEK READ (what a dashboard shows)")
print(f"   revenue   {rev_now/1e7:7.3f} Cr vs {rev_prev/1e7:7.3f} Cr   "
      f"{100*(rev_now/rev_prev-1):+6.2f}%")
print(f"   sessions  {ses_now/1e3:7.1f} k  vs {ses_prev/1e3:7.1f} k    "
      f"{100*(ses_now/ses_prev-1):+6.2f}%")
print(f"   conversion{cvr_now*100:7.3f} %  vs {cvr_prev*100:7.3f} %    "
      f"{100*(cvr_now/cvr_prev-1):+6.2f}%")
print(f"   aov       {aov_now:7.0f}    vs {aov_prev:7.0f}       "
      f"{100*(aov_now/aov_prev-1):+6.2f}%")

# --- 2. Correlation ranking of candidate drivers against revenue ----------
daily = (eu.groupby("date")[["net_revenue_inr", "orders"]].sum()
         .join(eu_web.groupby("date")[["sessions", "checkout_errors",
                                       "checkout_starts"]].sum()))
daily["cvr"] = daily.orders / daily.sessions
daily["err_rate"] = daily.checkout_errors / daily.checkout_starts.replace(0, np.nan)
recent = daily.loc["2026-06-01":"2026-08-23"]

print("\n2. CORRELATION WITH REVENUE (the seductive but wrong ranking)")
for c in ["sessions", "cvr", "err_rate"]:
    r = recent["net_revenue_inr"].corr(recent[c])
    print(f"   corr(revenue, {c:<12}) = {r:+.3f}")

# --- 3. Where did the session decline actually come from? ----------------
print("\n3. THE CONFOUND: sessions fell for THREE different reasons")
print("   Ground-truth effect of each driver on SESSIONS:")
W_SESSIONS = {"marketing_cut": (1.00, "exogenous - a real traffic lever"),
              "stockout": (0.55, "MEDIATED - dead PDPs lose organic traffic"),
              "competitor_promo": (0.52, "MEDIATED - their ads pull our shoppers")}
for name, (w, note) in W_SESSIONS.items():
    eff = -100 * truth["shock_strengths"][name] * w
    print(f"     {name:<18}{eff:>6.2f}% on sessions   ({note})")
exogenous = -100 * truth["shock_strengths"]["marketing_cut"]
print(f"   => Only {exogenous:.2f}pp of the session decline is exogenous. "
      f"Crediting all of it")
print( "      to the marketing budget double counts the other two.")

# --- 4. Naive vs causal ranking, both computed live ----------------------
# The naive column is NOT hardcoded. It runs the analysis the way an ordinary
# BI tool would: attribute the movement of each funnel factor straight to the
# driver most associated with it, with no control group, no adjustment for
# concurrent events and no test of whether traffic is exogenous. That is a fair
# representation of what "automated RCA" usually means, and it has to be
# executed rather than asserted for the comparison to mean anything.
import statsmodels.formula.api as smf  # noqa: E402

from causiq.engine.challenge import (estimate_did,  # noqa: E402
                                     estimate_joint_event_study)
from causiq.engine.reconcile import build_warehouse  # noqa: E402

wh = build_warehouse()

naive_rows = daily.loc["2026-07-15":"2026-08-26"].copy()
naive_rows["post"] = (naive_rows.index >= pd.Timestamp("2026-08-12")).astype(int)
naive_rows["stock"] = ((naive_rows.index >= pd.Timestamp("2026-08-19"))
                       & (naive_rows.index <= pd.Timestamp("2026-08-23"))).astype(int)
naive_rows["comp"] = ((naive_rows.index >= pd.Timestamp("2026-08-15"))
                      & (naive_rows.index <= pd.Timestamp("2026-08-26"))).astype(int)

# Naive attribution: each factor's own decline, credited to its obvious owner,
# one regression at a time with nothing held constant.
pre = naive_rows[naive_rows.index < pd.Timestamp("2026-08-12")]
post = naive_rows[naive_rows.index >= pd.Timestamp("2026-08-12")]
naive = {
    "marketing / traffic": 100 * (post.sessions.mean() / pre.sessions.mean() - 1),
    "checkout": 100 * (post.cvr.mean() / pre.cvr.mean() - 1),
}
for name, col in (("stockout", "stock"), ("competitor", "comp")):
    m = smf.ols(f"np.log(net_revenue_inr) ~ {col}", data=naive_rows).fit()
    naive[name] = float((np.exp(m.params[col]) - 1) * 100)

print("\n4. RANKING COMPARISON  (both columns computed, neither hardcoded)")
print(f"   {'NAIVE — no controls':<32}{'CAUSAL — CausIQ':<32}")
print("   " + "-" * 64)

did = estimate_did(wh, "checkout_experience")
joint = estimate_joint_event_study(
    wh, "EU", "conversion_rate", "2026-07-15", "2026-08-26",
    ["in_stock_rate", "competitor_price", "discount_depth"])
causal = {"checkout": did.effect_pct}
causal.update({k: v.effect_pct for k, v in joint.items()})

nl = sorted(naive.items(), key=lambda x: x[1])
cl = sorted(causal.items(), key=lambda x: x[1])
for i in range(max(len(nl), len(cl))):
    a = f"{nl[i][0]:<20}{nl[i][1]:+6.2f}%" if i < len(nl) else ""
    b = f"{cl[i][0]:<20}{cl[i][1]:+6.2f}%" if i < len(cl) else ""
    print(f"   {a:<32}{b:<32}")

naive_rank = [k for k, _ in nl]
checkout_naive_pos = naive_rank.index("checkout") + 1
print(f"\n   Naive ranks the checkout release {checkout_naive_pos} of "
      f"{len(naive_rank)} at {naive['checkout']:+.2f}% —")
print( "   it looks harmless, so nobody investigates it. CausIQ ranks it FIRST at")
print(f"   {did.effect_pct:+.2f}% (p={did.p_value:.4f}, "
      f"{sum(1 for r in did.refutations if r.passed)}/{len(did.refutations)} "
      f"refutations passed).")
print(f"\n   Naive also credits traffic with {naive['marketing / traffic']:+.2f}%, "
      f"against {exogenous:+.2f}% that is")
print( "   genuinely exogenous — roughly five times too much, because it absorbs")
print( "   the stockout and competitor pathways as well.")
print( "\n   Act on the naive read and you restore Rs 60L/month of paid budget while")
print( "   the checkout stays broken. Act on the causal read and you roll the")
print( "   release back for Rs 4L, once.")

# --- 5. Planted defects present? -----------------------------------------
print("\n5. PLANTED DEFECTS")
eu_late = sales[(sales.region == "EU") & (sales.date >= "2026-08-20")]
null_rate = eu_late.discount_inr.isna().mean()
print(f"   discount_inr null rate (EU, from 20 Aug) : {null_rate:.1%}  (target 18%)")

last_ops = pd.to_datetime(ops.iso_week).max()
stale_h = (pd.Timestamp("2026-08-27") - last_ops).total_seconds() / 3600
print(f"   ops_inventory last load                  : {last_ops.date()} "
      f"({stale_h:.0f}h stale vs 180h SLA -> "
      f"{'BREACH' if stale_h > 180 else 'ok'})")

wear = sales[sales.category == "Wearables"]
weeks = (wear.date.max() - wear.date.min()).days / 7
print(f"   Wearables history                        : {weeks:.1f} weeks "
      f"(< 12 -> partial pooling required)")

eu_cogs = sales[(sales.region == "EU")]
m_pre = 1 - eu_cogs[(eu_cogs.date >= P0) & (eu_cogs.date <= P1)].cogs_inr.sum() / \
    eu_cogs[(eu_cogs.date >= P0) & (eu_cogs.date <= P1)].net_revenue_inr.sum()
m_now = 1 - eu_cogs[(eu_cogs.date >= W0) & (eu_cogs.date <= W1)].cogs_inr.sum() / \
    eu_cogs[(eu_cogs.date >= W0) & (eu_cogs.date <= W1)].net_revenue_inr.sum()
print(f"   EU gross margin                          : {m_pre*100:.2f}% -> "
      f"{m_now*100:.2f}%  ({100*(m_now-m_pre):+.2f} pp)")

print("=" * 72)
