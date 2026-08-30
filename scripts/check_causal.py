"""Compare every causal estimate against the planted ground truth."""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from causiq.engine.reconcile import build_warehouse
from causiq.engine.challenge import estimate_joint_event_study, estimate_did

wh = build_warehouse()
ALL = ["in_stock_rate", "competitor_price", "discount_depth", "marketing_spend"]

did = estimate_did(wh, "checkout_experience")
did_mkt = estimate_did(wh, "marketing_spend")
cvr = estimate_joint_event_study(wh, "EU", "conversion_rate", "2026-07-15", "2026-08-26", ALL)
ses = estimate_joint_event_study(wh, "EU", "sessions", "2026-07-15", "2026-08-26", ALL)

# Ground truth, derived from the generator's shock strengths and pathway weights
T_CVR = {"checkout_experience": -3.68, "in_stock_rate": -2.14,
         "competitor_price": -1.31, "discount_depth": +29.25}
T_SES = {"in_stock_rate": -2.62, "competitor_price": -1.42,
         "marketing_spend": -1.14}
NAN = float("nan")


def row(name, eff, truth, p, verdict, tag=""):
    err = eff - truth if truth == truth else NAN
    e = f"{err:+7.2f}" if err == err else "      -"
    t = f"{truth:+8.2f}%" if truth == truth else "        -"
    print(f"{name:<21}{eff:>+8.2f}%{t}{e}{p:>10.5f}  {verdict} {tag}")


print("=" * 78)
print("OUTCOME = CONVERSION RATE")
print(f"{'driver':<21}{'effect':>9}{'truth':>9}{'err pp':>8}{'p':>10}  verdict")
print("-" * 78)
row("checkout_experience", did.effect_pct, T_CVR["checkout_experience"],
    did.p_value, did.verdict, "(diff-in-diff)")
for k, e in cvr.items():
    row(k, e.effect_pct, T_CVR.get(k, NAN), e.p_value, e.verdict)

print()
print("OUTCOME = SESSIONS   <- the mediation test")
print(f"{'driver':<21}{'effect':>9}{'truth':>9}{'err pp':>8}{'p':>10}  verdict")
print("-" * 78)
row("marketing_spend", did_mkt.effect_pct, T_SES["marketing_spend"],
    did_mkt.p_value, did_mkt.verdict, "(diff-in-diff)")
for k, e in ses.items():
    row(k, e.effect_pct, T_SES.get(k, NAN), e.p_value, e.verdict)

print()
mediators = [k for k, e in ses.items()
             if k != "marketing_spend" and e.verdict == "CONFIRMED"]
print("MEDIATION VERDICT")
if mediators:
    print(f"  sessions is a MEDIATOR: {', '.join(mediators)} significantly move")
    print( "  traffic as well as conversion. The session decline is therefore")
    print( "  partly an EFFECT of those drivers, not an independent cause.")
    print( "  => lever LV-005 (restore paid budget) is BLOCKED by the contract.")
else:
    print("  sessions appears exogenous; the marketing lever is permitted.")
print("=" * 78)
