# Demo video script — CausIQ

**Target: 4 minutes.** Scripted, screen-recorded, one take per section.

Record with the app already running and the caches warm — click through every
scenario and persona once beforehand so nothing takes five seconds on camera.
Set the browser to 1440px wide and zoom to 90% so a full section fits on screen.

A note on delivery: the temptation is to narrate what is on screen. Don't. The
viewer can see it. Narrate **what it means and why it is hard**.

---

## 0:00 – 0:25 · The problem, stated as a decision

> "Monday morning. EU revenue came in at ₹9.2 crore against a ₹10 crore plan —
> down 8%. Every dashboard in the company can tell you that. None of them can
> tell you what to do about it.
>
> The obvious answer is wrong, and expensively so. Let me show you."

**On screen:** the DETECT section. Actual, Expected, Deviation, MATERIAL.

---

## 0:25 – 1:00 · Detection, and the dashboard that lies

> "First: is this even real? CausIQ compares against a forecast, not last week —
> and the forecast is deliberately trained to a cut-off *before* the earliest
> known change, because a baseline fitted through an incident learns the
> incident and reports that nothing happened.
>
> The movement clears both gates: statistically significant, and past the ₹50
> lakh floor that makes it worth someone's morning."

**Scroll to EXPLAIN. Pause on the aggregation warning.**

> "Now look at this. Blended conversion went **up** 2.4% that week. A dashboard
> shows green. But Electronics — 71% of the business — fell 5.9%. An Accessories
> promotion was mathematically masking the collapse. The aggregate moved the
> wrong way."

---

## 1:00 – 2:00 · The trap, and the causal work

> "Here is where most tools go wrong. Traffic correlates with revenue at 0.97 —
> almost perfectly. Rank the drivers by correlation and marketing comes first,
> and you'd restore the ₹60 lakh a month you just cut."

**Scroll to CHALLENGE. Point at the mediation callout.**

> "But traffic fell for three different reasons. The stockout killed product
> pages, so they dropped out of organic ranking. The competitor's promotion
> pulled shoppers away. Only part of it was the budget. Sessions are a
> **mediator** — partly an *effect* of other drivers, not an independent cause.
> Attribute the whole traffic drop to marketing and you double count."

**Point at LV-005 BLOCKED.**

> "So the engine blocks that lever. Buying traffic back would push visitors into
> a funnel that is still broken."

**Open the refutation battery. Let the parallel-trends chart sit on screen.**

> "The checkout redesign shipped to EU only, so the US and APAC are a genuine
> control group — a natural experiment. This is difference-in-differences, and
> the two lines track each other until the release date, then separate.
>
> An estimate nobody has tried to break is a number, not a finding. So we
> attack it three ways: a fake treatment date before anything happened, an
> outcome the release physically cannot affect, and dropping each control region
> in turn. All three pass. **Now** it is a cause."

---

## 2:00 – 2:40 · Knowing what you don't know

**Switch scenario to the gross margin case.**

> "Same engine, different question. Margin fell 2.4 points — and CausIQ refuses
> to explain it.
>
> The supply feed is ten days stale against a seven-and-a-half day SLA. Coverage
> in scope is 57%. Confidence scores 0.51 against a 0.60 gate. Three rules fire,
> and the engine abstains.
>
> But it doesn't just shrug. It ranks what would fix it by confidence gained per
> hour of work — re-run the failed load, two hours. That turns 'I don't know'
> into a work item."

**Switch to the Wearables scenario.**

> "And here — a category six weeks old, conversion apparently down 21%. That
> looks alarming. With four weeks of history there is no seasonal baseline to
> compare against, so the engine switches method and widens its interval to
> match what it actually knows. Verdict: not material. It declines to call noise
> a crisis."

---

## 2:40 – 3:20 · Personas, entitlements, and the guard

**Back to the revenue scenario. Switch persona to Priya (CMO).**

> "Same evidence, computed once. The CMO gets ninety words: the rupee number,
> the cause, one action, an owner."

**Switch to Rahul (Analyst).**

> "The analyst gets effect sizes, p-values, the identification strategy, and
> which documents corroborate each driver. Different depth — identical numbers.
> Two people never leave with two versions of the truth."

**Switch to Lukas, then to the margin scenario.**

> "The regional manager is denied this KPI outright. Not hidden in the
> interface — refused in the contract, before any query runs. There is no way to
> phrase the question that reveals it, because the value is never loaded."

**Back to revenue as CMO. Switch narrative engine to Fault injection.**

> "One last thing. The language model here writes prose over a locked evidence
> package — it never computes a number. And we don't trust it to obey that.
> Watch: I'll make it fabricate a figure."

**Point at the rejection.**

> "Fourteen point seven percent appears nowhere in the evidence. The guard
> catches it, discards the narrative, and serves the deterministic version
> instead. That is what 'the LLM is not the source of quantitative truth' looks
> like when it is enforced rather than promised."

---

## 3:20 – 3:50 · Proof and cost

**Scroll to RUNTIME.**

> "Thirteen steps, twelve of them deterministic. One model call, about two
> thousand tokens, effectively zero cost — because the analysis is statistics
> and SQL, and the model is only ever asked to write two paragraphs."

**Cut to a terminal. Run `python scripts/check_causal.py`.**

> "And because this is synthetic data, we planted the answers. Here is every
> estimate against the truth. The checkout release: we estimated −3.57%, the
> planted value was −3.60%. Three hundredths of a percentage point.
>
> We explain 58% of the movement with confidence, and we leave 3 points openly
> unexplained rather than spreading it around to look complete."

---

## 3:50 – 4:00 · Close

> "Dashboards tell you what moved. CausIQ tells you why, tries to prove itself
> wrong first, and tells you when it doesn't know.
>
> Thirty tests, all passing. Everything reproducible from one seed."

---

## Shot list

| # | Scene | Setting |
|---|---|---|
| 1 | DETECT | revenue_drop · Rahul |
| 2 | EXPLAIN + aggregation warning | same |
| 3 | CHALLENGE waterfall + mediation + LV-005 blocked | same |
| 4 | Refutation battery expanded, parallel trends visible | same |
| 5 | KNOW abstention + VOI table | margin_abstain · Rahul |
| 6 | DETECT not-material | wearables_sparse · Rahul |
| 7 | Narrative, three personas in sequence | revenue_drop |
| 8 | PermissionError screen | margin_abstain · Lukas |
| 9 | Guard rejection | revenue_drop · Priya · Fault injection |
| 10 | RUNTIME telemetry | revenue_drop |
| 11 | Terminal: `scripts/check_causal.py` | — |
| 12 | Terminal: `pytest tests/ -q` | — |

## Common mistakes to avoid

- **Don't click around live.** Every transition above is a cut. Record sections
  separately and join them.
- **Don't read the tables aloud.** Say what they mean.
- **Don't apologise for synthetic data.** Lead with it — the planted ground
  truth is the strongest thing in the submission, not a weakness.
- **Don't run over.** If you must cut, drop the Wearables scenario (0:20) before
  anything else; the abstention case makes the same point more forcefully.
