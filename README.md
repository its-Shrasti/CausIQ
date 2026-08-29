# CausIQ — a KPI intelligence-to-action engine

**Accenture Innovation Challenge 2026 · Problem Track 3 · Team Aura Plus Plus**
IIT Kanpur — Shrasti · Aakriti · Neeru

> Dashboards tell you a KPI moved. CausIQ tells you **why**, tries to **disprove**
> its own answer before showing it to you, and **refuses to answer** when the
> evidence will not support one.

---

## The one-paragraph version

EU net revenue came in 8% under plan. Run the analysis the way an ordinary BI
tool does — one factor at a time, no control group, nothing held constant — and
the **checkout release ranks last of four drivers at +0.58%**. It looks
*helpful*. Nobody would ever investigate it. Meanwhile traffic gets credited
with −5.89% when only −1.14% of that decline is genuinely exogenous, because it
silently absorbs a stockout and a competitor promotion that also suppressed
traffic. The recommendation that falls out is to restore ₹60L/month of paid
budget — pouring visitors into a funnel that is still broken.

CausIQ estimates the checkout release against an untreated control region using
difference-in-differences, survives three falsification tests, and ranks it
**first at −3.57%** against a planted truth of −3.60%. The action is a ₹4L
rollback, once.

Everything in that paragraph is reproducible from this repository, and every
figure is checked against a planted answer key.

---

## Quick start

```bash
pip install -r requirements.txt
python -m causiq.ingest.generate     # build the sources + ground truth (~15s)
python -m pytest tests/ -q           # 30 acceptance tests (~22s)
streamlit run app/main.py            # the decision workspace
```

No API key is required. The narrative layer runs offline by default and
produces deterministic prose from the evidence. To use a real model instead:

```bash
export GEMINI_API_KEY=...            # then pick "Gemini" in the sidebar
```

Two scripts show the core claims without opening the UI:

```bash
python scripts/verify_trap.py        # proves the confound is real
python scripts/check_causal.py       # every estimate vs the answer key
```

---

## Why we can prove this works

A root-cause engine can only be evaluated if you already know the answer. Real
enterprise data never arrives labelled *"the checkout release cost you 3.6% of
revenue"*, so we **plant** the causes and check whether the engine recovers them.

`causiq/ingest/generate.py` builds revenue from an explicit structural model and
defines each event's true contribution as its **exact Shapley value** over all
2⁵ counterfactual worlds. Noise arrays are drawn once and held fixed across
every counterfactual, so toggling an event changes the outcome only through that
event's own causal pathway. The result is `data/ground_truth.json` — an answer
key the engine never sees.

### Accuracy on the hero scenario

Movement detected **−8.59%** against a true **−8.00%** (0.59pp of forecast error).

| Rank | Driver | Estimated | True | Error | Method | Verdict |
|---|---|---|---|---|---|---|
| 1 | Checkout release | −3.57% | −3.60% | **0.03pp** | diff-in-diff | CONFIRMED |
| 2 | Marketing budget cut | −1.50% | −1.10% | 0.41pp | diff-in-diff | CONFIRMED |
| 3 | Accessories promotion | +1.43% | +1.00% | 0.43pp | joint event study | CONFIRMED |
| 4 | Stockout | −1.30% | −2.40% | 1.10pp | joint event study | CONFIRMED |
| 5 | Competitor price cut | −0.70% | −1.90% | 1.19pp | joint event study | *inconclusive* |

The engine confidently explains **58%** of the movement, flags −0.57% as
unconfirmed, and leaves **−3.07% openly unexplained** rather than distributing it
across the known drivers. A decomposition that always sums to 100% is concealing
its error somewhere; we would rather show ours.

---

## Architecture

```
  wh_sales          web_events         ops_inventory        ctx_docs
  Snowflake         BigQuery/GA4       SAP IBP              Jira/Zendesk/scrape
  daily, T-1        hourly, T-4h       weekly, 240h STALE   ad-hoc
  gold              silver             bronze               bronze
       └────────────────┴────────┬───────────┴──────────────────┘
                                 ▼
  1  RECONCILE      conform grain · align calendar · stamp freshness   SQL
                                 ▼
  2  SEMANTIC CONTRACT   definitions · formulas · causal DAG
                         thresholds · lineage · entitlements · levers   YAML
                                 ▼
  3  ANALYSE  ─ DETECT      Holt-Winters + prediction interval          statistics
              ─ EXPLAIN     KPI tree walk · log-additive split          deterministic
              ─ CHALLENGE   diff-in-diff · event study · refutation     causal inference
              ─ RETRIEVE    BM25 over operational documents             retrieval
              ─ SCORE       confidence · contradiction · abstention     business rules
                                 ▼
                  ═══  LOCKED EVIDENCE PACKAGE  ═══
              nothing below this line may write a number
                                 ▼
  4  NARRATE    persona-conditioned prose                               LLM
     GUARD      every figure must exist in the evidence                 deterministic
     ACT        retrieved from the governed lever library               rules + LLM phrasing
                                 ▼
  5  DELIVER    3 personas · RBAC · audit · feedback · telemetry
```

The **locked evidence package** is the load-bearing idea. Every number is
computed before any model is invoked. The LLM receives a frozen JSON payload and
a persona instruction, and may only phrase it.

---

## LLM vs non-LLM — measured, not asserted

| Stage | Method | Engine |
|---|---|---|
| Ingest, conform grain, join | DuckDB SQL | deterministic |
| KPI computation | contract formula | deterministic |
| Baseline forecast | Holt-Winters, damped trend | statistics |
| Anomaly detection | prediction-interval breach | statistics |
| Materiality gate | ₹ floor **AND** significance | business rule |
| Segment & factor decomposition | per-category log-additive split | deterministic |
| Causal estimation | difference-in-differences | causal inference |
| Causal estimation (no experiment) | joint event study, two-way FE | causal inference |
| Falsification | placebo-in-time, negative control, leave-one-out | causal inference |
| Context retrieval | BM25 | retrieval |
| Confidence scoring | weighted formula from contract | deterministic |
| Abstention | contract rules | business rule |
| Action selection | lever-library retrieval | business rule |
| **Narrative generation** | **Gemini Flash** | **LLM** |
| Numeric guard | regex + allow-list | deterministic |

Measured on the hero scenario: **13 steps, 12 deterministic, 1 generative.**
The single generative step consumes ~1,800 prompt and ~180 completion tokens.

### The numeric guard

Prompt instructions cannot enforce numeric fidelity — a model told "use only
these numbers" will still occasionally produce a plausible one, and plausible is
what makes it dangerous. So the check happens **after** generation, in code:
every figure in the narrative is extracted and matched against the evidence
payload, with tolerance derived from display precision. Anything unmatched means
the narrative is discarded and the deterministic rendering is served instead.

Select **"Fault injection"** in the sidebar to watch it reject a fabricated
figure live. A guard nobody has seen fire is indistinguishable from no guard.

---

## Where to see each requirement

| # | Minimum expectation | Where |
|---|---|---|
| 1 | 3–5 connected KPIs across 2–3 sources, different grains/cadences | 5 KPIs, 4 sources (daily / hourly / weekly / ad-hoc) — `kpi_contract.yaml` § sources; freshness table under **KNOW** |
| 2 | KPI/semantic contract: definitions, calculations, drivers, thresholds, lineage, access | `causiq/contracts/kpi_contract.yaml` (all six); lineage shown in the **KNOW** expander |
| 3 | Two+ personas with different narratives/actions | Persona switch — Priya (CMO, 90 words, Slack), Rahul (Analyst, full statistics), Lukas (Regional Manager, EU-scoped, email) |
| 4 | One multi-factor movement with known drivers | Scenario **"EU net revenue down 8%"** — five interacting drivers, planted and verified |
| 5 | One low-confidence scenario: clarify or abstain | Scenario **"EU gross margin"** — abstains on 3 rules at 0.51 confidence, with ranked next-best evidence |
| 6 | One sparse-history / new KPI scenario | Scenario **"EU Wearables"** — 4 weeks of history, switches to partial pooling, declares **not material** despite a −21% movement |
| 7 | One role-based security scenario | Select **Lukas Weber** on the margin scenario → `PermissionError` before any query runs; row filter and column masks visible in the sidebar |
| 8 | Evidence: freshness, method, contribution, confidence, lineage | Driver table under **CHALLENGE**; components under **KNOW**; freshness/lineage expander |
| 9 | Runtime telemetry: latency, model calls, tokens, cost | **RUNTIME** section with per-step trace |

---

## The four scenarios

**1 · Revenue drop (multi-driver).** Five interacting drivers with a deliberate
trap. Aggregate conversion moved **+2.4%** — a dashboard shows green — while
Electronics, 71% of the business, fell **−5.9%**. The mix masks it. Underneath,
`in_stock_rate` and `competitor_price` move *both* traffic and conversion, so
sessions are a **mediator**, not an independent driver. The engine detects this
and **blocks** the "restore paid budget" lever by contract.

**2 · Gross margin (abstention).** Margin fell 2.4pp. Three abstention rules
fire: confidence 0.51 against a 0.60 gate, `ops_inventory` 240h stale against a
180h SLA, and in-scope coverage at 57%. The engine names all three and ranks
what would resolve them by confidence gained per hour of effort.

**3 · Wearables (sparse history).** Conversion appears to collapse **−21%**. With
four weeks of history there is no estimable seasonal baseline, so the engine
switches to partial pooling and applies an uncertainty floor. Verdict: **not
material** (p = 0.13). Refusing to over-claim on a young series is the correct
statistical behaviour, not a limitation.

**4 · Entitlements.** Lukas Weber is denied `gross_margin_pct` outright, sees
only EU rows, and has `cogs_inr` and `discount_inr` masked. Enforcement is in
SQL before evidence assembly — the values are never loaded into a context they
could leak from.

---

## Business case

The measurable claim is not "faster insights". It is **avoiding the wrong
decision**.

| | Uncontrolled analysis | CausIQ |
|---|---|---|
| Checkout release ranked | 4th of 4, at **+0.58%** | **1st**, at −3.57% |
| Traffic credited with | −5.89% | −1.14% exogenous, rest mediated |
| Action implied | Restore paid budget | Roll back the release |
| Cost | ₹60L per month, recurring | ₹4L one-off |
| Effect | Traffic into a broken funnel | ~92% of the attributed loss recovered |

Both columns are computed live by `scripts/verify_trap.py` — neither is
hardcoded, so the comparison can be re-run and checked.

One avoided misdiagnosis pays for the platform many times over. On a business
running ₹10 Cr/week in one region, the 3.6% attributed to the checkout release
is roughly **₹36L per week** while it goes undiagnosed. Traditional RCA takes
24–72h to detect and 3–7 days to investigate; CausIQ produces the same
attribution in about five seconds, with the evidence trail attached.

Cost per insight is effectively zero on the free tier and roughly ₹0.02 at paid
Gemini Flash rates — the analysis is deterministic, so the model is only ever
asked to write two paragraphs.

---

## Roadmap

**Phase 1 — Pilot (0–3 months).** One KPI tree, two source systems, two
personas. Backfill 12 months of history, calibrate materiality thresholds
against what teams actually escalate. Success: analysts agree with the top
driver in 70% of cases.

**Phase 2 — Production (3–9 months).** Warehouse-native deployment
(Snowflake/Databricks). Real entitlement integration via the existing identity
provider. Proactive Slack and email delivery. Feedback loop writing to a
persistent store. Success: median time-to-attribution under 10 minutes.

**Phase 3 — Scale (9–18 months).** Multi-domain KPI trees (supply, finance,
service). Automated experiment proposals where no natural experiment exists —
the engine already knows which drivers lack one. Continuous evaluation against
outcomes of actions actually taken.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Wrong causal claim acted on** | Nothing is presented as a cause until it survives three falsification tests; failures are marked REJECTED and never shown as causes |
| **Over-confidence on thin data** | Abstention gate plus a sparse-history uncertainty floor; the engine refuses rather than hedges |
| **LLM fabricates a figure** | Locked evidence package plus a post-generation numeric guard with deterministic fallback |
| **Semantic drift between teams** | Every definition, threshold and adjustment set lives in one versioned contract; no thresholds in code (enforced by a test) |
| **Sensitive data exposure** | Row and column entitlements applied in SQL before evidence assembly; denied values never enter a prompt |
| **Unknown confounders** | Adjustment sets are declared and surfaced as assumptions, not hidden; the unexplained remainder is always reported |
| **Cost/latency at scale** | 12 of 13 steps are deterministic; model choice, caching and token counts are measured per run |
| **Model or data drift** | Feedback adjusts driver priors; ground-truth harness can be re-run as a regression suite |

---

## What is real and what is simulated

Being explicit, since the brief invites reasonable assumptions:

- **Simulated:** all data. NovaCart Retail Group is fictional. Sources are
  generated from a seeded structural model, though with realistic grain
  mismatches, staleness, null defects and a sparse category.
- **Real:** every analytical method. The Holt-Winters baselines, the
  difference-in-differences estimator, the two-way fixed-effects event study,
  the refutation battery, the confidence formula, the entitlement enforcement,
  the numeric guard and the telemetry all execute as written.
- **Declared, not integrated:** the contract names Snowflake, BigQuery and SAP
  as source systems. The prototype reads Parquet through DuckDB. The interfaces
  are shaped for those systems; the connections are not built.

---

## Six bugs worth reading about

Each was caught by comparing against the answer key, and each is a way
production BI quietly gets it wrong. Without ground truth, all six would have
shipped looking entirely plausible.

1. **Contaminated baseline.** Holt-Winters trained through the incident had
   learned the depressed level and reported "no anomaly" on an 8% drop. Fixed by
   holding the forecast origin behind the earliest known intervention.
2. **Mix artefact.** Blended AOV fell because expensive Electronics shrank while
   cheap Accessories grew — nothing was priced differently. The naive
   decomposition blamed basket size for 60% of the movement. Fixed by
   decomposing within category, then summing.
3. **Additive model, multiplicative world.** Fitting proportional effects with
   additive fixed effects produced a *positive* stockout effect on traffic.
   Fixed by modelling in logs.
4. **Omitted event.** The Accessories promotion had no exposure definition, so
   its +29% conversion spike inflated the date fixed effects and pushed every
   other coefficient negative — the competitor effect read 4× too large.
5. **Specification error.** Marketing spend was in a conversion model the DAG
   says it cannot affect; its step at 10 Aug was nearly collinear with the
   checkout release at 12 Aug and destabilised everything.
6. **A guard that could not fire.** Checking figures against all 336 numbers in
   the evidence package made the allow-list so dense that a fabricated "14.7%"
   matched something unrelated. Fixed by guarding against the ~50 numbers
   actually shown to the model.

---

## Repository layout

```
causiq/
  contracts/       kpi_contract.yaml · lever_library.yaml   the governed layer
  ingest/          generate.py           synthetic sources + planted truth
  engine/          contract · reconcile · detect · explain
                   challenge · retrieve · confidence · pipeline
  narrate/         provider · guard · narrator
  telemetry/       tracker.py
app/               main.py · charts.py   the Streamlit workspace
scripts/           verify_trap.py · check_causal.py · tune_lag.py
tests/             test_engine.py        30 acceptance tests
data/              ground_truth.json     the answer key
```

Roughly 4,600 lines. Reproducible from seed `20260827` with the clock pinned to
2026-08-27, so every figure quoted here regenerates exactly.

---

## Charts

The visualisation palette was validated before any chart code was written —
categorical slots pass adjacent colour-vision separation (ΔE 9.2) and the
normal-vision floor (ΔE 27.6); the diverging pair used for contribution
polarity passes at ΔE 21.6 / 32.3. One y-axis per chart, hues assigned in fixed
order, colour attached to the entity rather than its rank.
