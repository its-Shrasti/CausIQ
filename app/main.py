"""
CausIQ decision workspace.

Run:  streamlit run app/main.py

The layout follows the seven beats the engine actually performs, in order, so a
reader can follow the reasoning rather than being handed a conclusion:

    DETECT -> EXPLAIN -> CHALLENGE -> KNOW -> SIMULATE -> DECIDE -> LEARN

Every section carries a chip reading DETERMINISTIC or LLM. That is not
decoration: the brief asks teams to show where a language model is and is not
used, and a label on each step is the most direct way to answer it. On this
workload all seven beats except narration are deterministic, and the telemetry
panel at the foot of the page proves it with measured latency and token counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import charts
from causiq.engine import contract, pipeline
from causiq.engine.reconcile import build_warehouse
from causiq.narrate.narrator import narrate
from causiq.telemetry.tracker import Telemetry

st.set_page_config(page_title="CausIQ", page_icon="◐", layout="wide")

# --- minimal styling; the palette matches the chart layer -------------------
st.markdown("""
<style>
  .stApp { background: #fcfcfb; }
  .chip { display:inline-block; padding:2px 9px; border-radius:11px;
          font-size:11px; font-weight:600; letter-spacing:.03em;
          text-transform:uppercase; margin-left:8px; vertical-align:middle; }
  .chip-det { background:#e8f0fc; color:#1c5cab; }
  .chip-llm { background:#fdeee7; color:#b8451a; }
  .beat { font-size:12px; font-weight:700; letter-spacing:.12em;
          color:#8a8880; text-transform:uppercase; margin-bottom:-6px; }
  .verdict-ok    { color:#0ca30c; font-weight:600; }
  .verdict-bad   { color:#d03b3b; font-weight:600; }
  .verdict-meh   { color:#8a8880; font-weight:600; }
  .masked { font-family:monospace; color:#b0aea6; letter-spacing:2px; }
  .callout { border-left:3px solid #2a78d6; background:#f4f8fe;
             padding:12px 16px; border-radius:0 6px 6px 0; margin:6px 0 14px; }
  .callout-warn { border-left-color:#fab219; background:#fef9ec; }
  .callout-stop { border-left-color:#d03b3b; background:#fdf0f0; }
  code { font-size:12px; }
</style>
""", unsafe_allow_html=True)


def chip(kind: str) -> str:
    return (f'<span class="chip chip-{"llm" if kind == "LLM" else "det"}">'
            f'{"LLM" if kind == "LLM" else "deterministic"}</span>')


def beat(n: str, title: str, kind: str = "DET") -> None:
    st.markdown(f'<div class="beat">{n}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}{chip(kind)}", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def warehouse():
    return build_warehouse()


@st.cache_data(show_spinner="Running the engine…", ttl=3600)
def run_engine(scenario: str, persona_key: str, prior_key: tuple, mode: str):
    """Cached engine run.

    A full pass is roughly seven seconds -- dominated by the difference-in-
    differences fits and their refutation batteries, each of which re-estimates
    the model several times. That is fine for a scheduled alert and far too slow
    to click through in a demo, so results are memoised on the inputs that
    actually change them: scenario, persona, learned priors and narrative mode.
    """
    wh_ = warehouse()
    tel_ = Telemetry()
    ev_ = pipeline.run(wh_, scenario, persona_key, telemetry=tel_,
                       driver_prior=dict(prior_key))
    nar_ = narrate(ev_, persona_key, tel_, mode=mode)
    return ev_, nar_, tel_.summary(), tel_.table()


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.markdown("### ◐ CausIQ")
st.sidebar.caption("KPI intelligence-to-action engine")

# Deep links: ?scenario=margin_abstain&persona=analyst&mode=hallucinate
# Recording a demo means jumping between eight specific states. Clicking through
# dropdowns on camera is slow and easy to fumble; a URL per scene is not.
_qp = st.query_params
_scenarios = list(pipeline.SCENARIOS.keys())
_personas = contract.all_personas()


def _from_query(name: str, options: list[str]) -> int:
    v = _qp.get(name)
    return options.index(v) if v in options else 0


scenario = st.sidebar.selectbox(
    "Scenario", _scenarios,
    index=_from_query("scenario", _scenarios),
    format_func=lambda k: pipeline.SCENARIOS[k]["title"],
)
st.sidebar.caption(pipeline.SCENARIOS[scenario]["demonstrates"])

persona_key = st.sidebar.selectbox(
    "Persona", _personas,
    index=_from_query("persona", _personas),
    format_func=lambda k: (f"{contract.persona(k)['display_name']} — "
                           f"{contract.persona(k)['title']}"),
)
p = contract.persona(persona_key)
ent = contract.entitlement(p["role"])

st.sidebar.markdown(
    f"**Role** `{p['role']}`  \n"
    f"**Channel** {p['channel']}  \n"
    f"**Depth** {ent.get('insight_depth')}"
)
if ent.get("row_filter"):
    st.sidebar.warning(f"Row filter: `{ent['row_filter']}`")
if ent.get("denied_columns"):
    st.sidebar.warning("Masked columns: " + ", ".join(ent["denied_columns"]))

_modes = ["offline", "auto", "hallucinate"]
mode = st.sidebar.radio(
    "Narrative engine", _modes,
    index=_from_query("mode", _modes),
    format_func={"offline": "Offline (deterministic)",
                 "auto": "Gemini if GEMINI_API_KEY set",
                 "hallucinate": "Fault injection — prove the guard"}.get,
    help=("Offline renders from the evidence with no model call. "
          "Fault injection makes the model fabricate a figure so the numeric "
          "guard can be seen rejecting it."),
)

if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if st.sidebar.button("Reset learned priors"):
    st.session_state.feedback = {}

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Contract v{contract.load_contract()['contract_version']} · "
    f"data seed 20260827 · clock pinned 2026-08-27"
)

# =============================================================================
# Run
# =============================================================================

st.title(pipeline.SCENARIOS[scenario]["title"])

try:
    prior_key = tuple(sorted(st.session_state.feedback.items()))
    ev, nar, tel_summary, tel_rows = run_engine(scenario, persona_key,
                                                prior_key, mode)
except PermissionError as e:
    beat("ACCESS CONTROL", "Request denied")
    st.markdown(f'<div class="callout callout-stop"><b>{e}</b></div>',
                unsafe_allow_html=True)
    st.markdown(
        "The denial happens in the semantic contract before any query runs, "
        "not in the interface and not in a prompt. There is no phrasing of a "
        "question that reveals this KPI to this role, because the value is "
        "never loaded into a context it could leak from."
    )
    st.code(f"""entitlements:
  roles:
    {p['role']}:
      row_filter: {ent.get('row_filter')}
      denied_columns: {ent.get('denied_columns')}
      denied_kpis: {ent.get('denied_kpis')}""", language="yaml")
    st.info("Switch persona in the sidebar to view this scenario as an "
            "entitled role.")
    st.stop()

det, conf = ev.detection, ev.confidence
unit = contract.kpi(det["kpi"]).unit

# =============================================================================
# 1 · DETECT
# =============================================================================

beat("01 · DETECT", "Is the movement real, and does it matter?")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Actual", charts._fmt(det["actual"], unit))
c2.metric("Expected", charts._fmt(det["expected"], unit))
# Streamlit colours a delta by whether the STRING starts with "-", so a
# currency prefix makes a shortfall render green with an up arrow. The delta
# therefore leads with the sign and carries its unit as a suffix.
_delta = (f"{det['deviation_abs']/1e7:+.2f} Cr" if unit == "INR"
          else f"{det['deviation_abs']:+.4f}")
c3.metric("Deviation", f"{det['deviation_pct']:+.2f}%",
          delta=_delta, delta_color="normal")
c4.metric("Materiality",
          "MATERIAL" if det["material"] else "NOT MATERIAL")
c4.caption(f"p = {det['p_value']:.4f} · z = {det['z_score']:+.2f}")

st.plotly_chart(charts.baseline_chart(det, unit), use_container_width=True,
                config={"displayModeBar": False})

m1, m2 = st.columns([2, 1])
with m1:
    st.markdown(
        f"**Baseline** {det['baseline_method_label']}  \n"
        f"**History** {det['history_weeks']:.1f} weeks · "
        f"z = {det['z_score']:+.2f} · p = {det['p_value']:.4f}  \n"
        f"**Rule** `{det['materiality_rule']}`"
    )
with m2:
    th = det["thresholds"]
    st.markdown(
        f"**Thresholds**  \n"
        + (f"absolute ≥ ₹{th['min_abs_inr']/1e5:,.0f}L  \n"
           if th.get("min_abs_inr") else "")
        + f"relative ≥ {th['min_pct']}%  \n"
        f"α = {th['significance_alpha']}"
    )

for w in det.get("warnings", []):
    st.markdown(f'<div class="callout callout-warn">{w}</div>',
                unsafe_allow_html=True)

if not det["material"]:
    st.markdown(
        '<div class="callout callout-warn"><b>No action recommended.</b> '
        'The movement is visible but cannot be distinguished from normal '
        'variation given the history available. Analysis stops here rather '
        'than manufacturing an explanation for noise.</div>',
        unsafe_allow_html=True)

# =============================================================================
# 2 · EXPLAIN
# =============================================================================

if ev.explanation:
    st.markdown("---")
    beat("02 · EXPLAIN", "Where did the movement come from?")

    exp = ev.explanation
    if exp.get("paradox_flag"):
        st.markdown(
            f'<div class="callout callout-warn"><b>Aggregation warning.</b> '
            f'{exp["paradox_note"]}</div>', unsafe_allow_html=True)

    e1, e2 = st.columns([3, 2])
    with e1:
        st.markdown("**By category** — each against its own baseline")
        seg = pd.DataFrame(exp["segments"])
        if not seg.empty:
            show = seg[["segment", "actual", "expected", "own_pct_change",
                        "pct_of_total_gap"]].copy()
            show.columns = ["Category", "Actual", "Expected", "Own %", "% of gap"]
            for c in ("Actual", "Expected"):
                show[c] = show[c].map(lambda v: charts._fmt(v, unit))
            show["Own %"] = show["Own %"].map(lambda v: f"{v:+.1f}%")
            show["% of gap"] = show["% of gap"].map(lambda v: f"{v:.1f}%")
            st.dataframe(show, hide_index=True, use_container_width=True)
    with e2:
        st.markdown("**By factor** — mix-free, computed within category")
        fac = pd.DataFrame(exp["factors"])
        if not fac.empty:
            fs = fac[["label", "share_of_movement", "pct_points_of_kpi"]].copy()
            fs.columns = ["Factor", "Share", "pp of KPI"]
            fs["Share"] = fs["Share"].map(lambda v: f"{v:.1%}")
            fs["pp of KPI"] = fs["pp of KPI"].map(lambda v: f"{v:+.2f}pp")
            st.dataframe(fs, hide_index=True, use_container_width=True)
        st.caption(
            "Decomposed inside each category then summed. Computing this on "
            "blended aggregates attributes the movement to basket size, which "
            "is a mix artefact rather than a price effect."
        )

# =============================================================================
# 3 · CHALLENGE
# =============================================================================

if ev.drivers:
    st.markdown("---")
    beat("03 · CHALLENGE", "Did these drivers actually cause it?")

    st.plotly_chart(
        charts.contribution_waterfall(ev.drivers, ev.attribution),
        use_container_width=True, config={"displayModeBar": False})

    att = ev.attribution
    a1, a2, a3 = st.columns(3)
    a1.metric("Confirmed", f"{att['confirmed_pct']:+.2f}%",
              delta=f"{att['confirmed_share_of_movement']:.0f}% of movement",
              delta_color="off")
    a2.metric("Unconfirmed", f"{att['unconfirmed_pct']:+.2f}%")
    a3.metric("Unexplained", f"{att['unexplained_pct']:+.2f}%")
    st.caption(att["note"])

    rows = []
    for d in ev.drivers:
        rows.append({
            "Driver": d["driver_label"],
            "Revenue impact": f"{d.get('revenue_impact_pct', 0):+.2f}%",
            "Method": d["method"].replace("_", " "),
            "Effect": f"{d['effect_pct']:+.2f}%",
            "p": f"{d['p_value']:.4f}" if d["p_value"] == d["p_value"] else "—",
            "Verdict": d["verdict"],
            "Refutations": (f"{sum(1 for r in d['refutations'] if r['passed'])}"
                            f"/{len(d['refutations'])}" if d["refutations"] else "—"),
            "Evidence": ", ".join(d.get("citations") or []) or "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- mediation ---------------------------------------------------------
    med = ev.mediation
    if med.get("sessions_is_mediator"):
        drv = ", ".join(e["driver_label"] for e in med["evidence"])
        st.markdown(
            f'<div class="callout callout-warn"><b>Sessions are a mediator, '
            f'not an independent driver.</b><br>{drv} significantly move traffic '
            f'as well as conversion. {med["implication"]}</div>',
            unsafe_allow_html=True)

    # --- refutation detail -------------------------------------------------
    did = [d for d in ev.drivers if d["method"] == "difference_in_differences"]
    if did:
        lead = did[0]
        with st.expander(f"Refutation battery — {lead['driver_label']}",
                         expanded=True):
            st.caption(lead["method_label"])
            if lead.get("plot_data"):
                st.plotly_chart(charts.parallel_trends(lead["plot_data"]),
                                use_container_width=True,
                                config={"displayModeBar": False})
            for r in lead["refutations"]:
                icon = "✔" if r["passed"] else "✘"
                cls = "verdict-ok" if r["passed"] else "verdict-bad"
                st.markdown(
                    f'<span class="{cls}">{icon} {r["name"].replace("_", " ")}</span> '
                    f'— {r["description"]}<br>'
                    f'<span style="color:#52514e">Threshold: {r["threshold"]} · '
                    f'{r["interpretation"]}</span>',
                    unsafe_allow_html=True)
                st.markdown("")
            st.markdown(f"**Verdict — {lead['verdict']}.** {lead['verdict_reason']}")
            st.markdown("**Assumptions this rests on**")
            for a in lead.get("assumptions", []):
                st.markdown(f"- {a}")

# =============================================================================
# 4 · KNOW
# =============================================================================

st.markdown("---")
beat("04 · KNOW", "How much should anyone trust this?")

k1, k2 = st.columns([1, 2])
with k1:
    st.metric("Confidence", f"{conf['score']:.2f}",
              delta=conf["tier"].upper(), delta_color="off")
    st.markdown(f"**Action gate** `{conf['action']}`")
with k2:
    st.plotly_chart(charts.confidence_bars(conf["components"]),
                    use_container_width=True, config={"displayModeBar": False})

for c in conf["components"]:
    if c["score"] < 0.9:
        st.markdown(f"- **{c['name'].replace('_', ' ')}** "
                    f"({c['score']:.2f}) — {c['detail']}")

if conf["abstained"]:
    st.markdown(
        '<div class="callout callout-stop"><b>CausIQ is abstaining.</b> '
        'The evidence does not support attributing a cause, so none is offered. '
        'Abstention here is the correct output, not a failure to produce one.</div>',
        unsafe_allow_html=True)
    for r in conf["triggered_rules"]:
        st.markdown(f"- {r}")

    st.markdown("#### What would resolve it")
    st.caption("Ranked by expected confidence gain per hour of effort — "
               "Value of Information, not a wish list.")
    voi = pd.DataFrame(conf["next_best_evidence"])
    if not voi.empty:
        v = voi[["action", "why", "owner", "effort_hours",
                 "expected_confidence_uplift"]].copy()
        v.columns = ["Action", "Why", "Owner", "Hours", "Confidence gain"]
        st.dataframe(v, hide_index=True, use_container_width=True)

for c in conf.get("contradictions", []):
    st.markdown(f'<div class="callout callout-warn">{c}</div>',
                unsafe_allow_html=True)

# --- freshness ---------------------------------------------------------------
with st.expander("Source freshness, grain and lineage"):
    fr = pd.DataFrame(ev.freshness)
    fr["status"] = fr["within_sla"].map({True: "within SLA", False: "BREACH"})
    st.dataframe(
        fr[["source", "system", "cadence", "grain", "age_hours", "sla_hours",
            "status", "quality_tier"]],
        hide_index=True, use_container_width=True)
    st.markdown(f"**KPI definition** — {ev.lineage['kpi_definition']}")
    st.markdown(f"**Formula** — `{ev.lineage['formula']}`")
    st.markdown("**Column lineage**")
    for l in ev.lineage["columns"]:
        st.markdown(f"- `{l}`")
    st.caption(f"{ev.lineage['sql_executed']} SQL statements executed under "
               f"contract v{ev.lineage['contract_version']}")

# =============================================================================
# 5 · SIMULATE + 6 · DECIDE
# =============================================================================

if nar.recommendations and not conf["abstained"]:
    st.markdown("---")
    beat("05 · SIMULATE", "Which action is worth taking?")

    rec_rows = []
    for r in nar.recommendations:
        rec_rows.append({
            "Lever": r["lever_id"],
            "Action": r["lever"].replace("_", " "),
            "Expected recovery": f"{r['expected_impact_pct']:+.2f}%",
            "Value": f"₹{r['expected_impact_inr']/1e5:,.1f}L",
            "Cost": f"₹{r['cost_inr']/1e5:,.1f}L",
            "Risk": r["risk"],
            "Owner": r["owner"],
            "Status": "BLOCKED" if r["blocked"] else "available",
        })
    st.dataframe(pd.DataFrame(rec_rows), hide_index=True,
                 use_container_width=True)

    for r in nar.recommendations:
        if r["blocked"]:
            st.markdown(
                f'<div class="callout callout-stop"><b>{r["lever_id"]} blocked '
                f'— {r["lever"].replace("_", " ")}</b><br>{r["blocked_reason"]}'
                f'</div>', unsafe_allow_html=True)

    live = [r for r in nar.recommendations if not r["blocked"]]
    if live:
        st.markdown("---")
        beat("06 · DECIDE", "The recommendation, and who owns it")
        top = live[0]
        st.markdown(f"### {top['action']}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Expected recovery", f"{top['expected_impact_pct']:+.2f}%",
                  delta=f"₹{top['expected_impact_inr']/1e5:,.1f}L",
                  delta_color="off")
        d2.metric("Cost", f"₹{top['cost_inr']/1e5:,.1f}L")
        d3.metric("Risk", top["risk"].upper())
        d4.metric("Confidence", f"{top['confidence']:.2f}")

        st.markdown(
            f"**Owner** {top['owner']}  \n"
            f"**Decision rights** {top['decision_rights']}  \n"
            f"**Reversible** {'yes' if top['reversible'] else 'no'}"
        )
        if top["constraints"]:
            st.markdown("**Constraints**")
            for c in top["constraints"]:
                st.markdown(f"- {c}")
        if top["risk_notes"]:
            st.caption(top["risk_notes"])

        mon = top["monitoring"]
        if mon:
            st.markdown(
                f"**Monitoring plan** — {', '.join(mon.get('metrics', []))} "
                f"at {mon.get('cadence')} cadence for {mon.get('horizon_days')} "
                f"days.  \n**Rollback trigger** — {mon.get('rollback_trigger')}"
            )

# =============================================================================
# Narrative
# =============================================================================

st.markdown("---")
beat("NARRATIVE", f"For {nar.display_name} — via {nar.channel}", "LLM")

st.markdown(f'<div class="callout">{nar.text.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True)

g = nar.guard
if g["passed"]:
    st.markdown(
        f'<span class="verdict-ok">✔ Numeric guard passed</span> — '
        f'{g["numbers_matched"]}/{g["numbers_checked"]} figures traced to the '
        f'evidence package.', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="callout callout-stop"><b>✘ Numeric guard REJECTED this '
        f'narrative.</b><br>The model produced figures that appear nowhere in '
        f'the evidence. Output was discarded and the deterministic rendering '
        f'served instead.</div>', unsafe_allow_html=True)
    for v in g["violations"]:
        st.markdown(f'- `{v["literal"]}` — {v["reason"]}')

st.caption(
    f"provider `{nar.provider}` · model `{nar.model}` · the model receives a "
    f"locked evidence JSON and may only phrase it; it cannot compute, adjust "
    f"or add a number."
)

# =============================================================================
# 7 · LEARN
# =============================================================================

if ev.drivers and not conf["abstained"]:
    st.markdown("---")
    beat("07 · LEARN", "Correct the engine, and watch the ranking move")

    st.caption(
        "Feedback adjusts the PRIOR WEIGHT used to rank candidate drivers on "
        "the next run. It never alters an estimate — a human disagreeing with "
        "a measurement does not change the measurement, only how strongly this "
        "driver is considered in future."
    )

    fb_cols = st.columns(min(len(ev.drivers), 4))
    for i, d in enumerate(ev.drivers[:4]):
        with fb_cols[i]:
            st.markdown(f"**{d['driver_label']}**")
            st.caption(f"{d.get('revenue_impact_pct', 0):+.2f}% · "
                       f"prior ×{d.get('prior_weight', 1.0):.2f}")
            cc1, cc2 = st.columns(2)
            if cc1.button("Agree", key=f"up_{d['driver']}"):
                cur = st.session_state.feedback.get(d["driver"], 1.0)
                st.session_state.feedback[d["driver"]] = min(cur * 1.35, 3.0)
                st.rerun()
            if cc2.button("Reject", key=f"down_{d['driver']}"):
                cur = st.session_state.feedback.get(d["driver"], 1.0)
                st.session_state.feedback[d["driver"]] = max(cur * 0.5, 0.05)
                st.rerun()

    if st.session_state.feedback:
        st.markdown("**Learned priors in effect**")
        st.dataframe(
            pd.DataFrame([{"Driver": k, "Prior weight": f"×{v:.2f}"}
                          for k, v in st.session_state.feedback.items()]),
            hide_index=True, use_container_width=True)

# =============================================================================
# Telemetry
# =============================================================================

st.markdown("---")
beat("RUNTIME", "Latency, model calls, tokens and cost")

t = tel_summary
q1, q2, q3, q4, q5 = st.columns(5)
q1.metric("Total latency", f"{t['total_latency_ms']:,.0f} ms")
q2.metric("Deterministic", f"{t['deterministic_share_pct']:.1f}%",
          delta=f"{t['steps_deterministic']}/{t['steps_total']} steps",
          delta_color="off")
q3.metric("Model calls", t["model_calls"])
q4.metric("Tokens", f"{t['prompt_tokens'] + t['completion_tokens']:,}")
q5.metric("Cost / insight", f"₹{t['cost_per_insight_inr']:.4f}")

st.plotly_chart(charts.telemetry_split(t), use_container_width=True,
                config={"displayModeBar": False})

if t["model_calls"] == 0:
    st.caption(
        f"Offline mode invoked no model, so model calls read zero. The "
        f"{t['prompt_tokens'] + t['completion_tokens']:,} tokens shown are what "
        f"a real call would consume, carried through so the cost projection "
        f"stays meaningful. Switch the narrative engine to Gemini to see live "
        f"call counts and latency."
    )

with st.expander("Per-step trace — the LLM vs non-LLM breakdown"):
    tr = pd.DataFrame(tel_rows)
    tr["engine"] = tr["uses_llm"].map({True: "LLM", False: "deterministic"})
    st.dataframe(
        tr[["stage", "name", "engine", "method", "latency_ms", "model",
            "prompt_tokens", "completion_tokens", "cost_inr"]],
        hide_index=True, use_container_width=True)
    st.caption(
        f"Projected at 500 insights/day: "
        f"₹{t['projected_monthly_inr_at_500_per_day']:,.0f} per month. "
        f"Cache hit rate {t['cache_hit_rate_pct']:.0f}%."
    )
