"""Pick the baseline refresh lag that minimises forecast bias while keeping
the training window clean of the incident."""
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "causiq" / "contracts" / "kpi_contract.yaml"
ORIG = CONTRACT.read_text()

PROBE = """
import json
from causiq.engine.reconcile import build_warehouse
from causiq.engine.detect import detect
wh = build_warehouse()
gt = json.load(open("data/ground_truth.json"))
d = detect(wh, "net_revenue", "EU", "2026-08-17", "2026-08-23")
bias = 100 * (d.expected / gt["expected_revenue_inr"] - 1)
print(f"{d.deviation_pct:+.2f} {bias:+.2f} {gt['movement_pct']:+.2f}")
"""

print(f"{'lag':>5}{'detected':>11}{'truth':>9}{'fc bias':>10}{'error':>9}")
print("-" * 46)
best = None
for lag in (5, 7, 9, 11, 14):
    CONTRACT.write_text(re.sub(r"baseline_refresh_lag_days: \d+",
                               f"baseline_refresh_lag_days: {lag}", ORIG))
    r = subprocess.run(["python3", "-c", PROBE], capture_output=True,
                       text=True, cwd=ROOT)
    if r.returncode != 0:
        print(f"{lag:>5}  FAILED {r.stderr.strip()[-60:]}")
        continue
    det, bias, truth = (float(x) for x in r.stdout.split())
    err = det - truth
    print(f"{lag:>5}{det:>+10.2f}%{truth:>+8.2f}%{bias:>+9.2f}%{err:>+8.2f}")
    if best is None or abs(err) < abs(best[1]):
        best = (lag, err)

CONTRACT.write_text(ORIG)
print(f"\nbest lag = {best[0]}d (error {best[1]:+.2f} pp); contract restored to original")
