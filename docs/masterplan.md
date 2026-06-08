# QRA Publication-Readiness & Mechanism Master Plan

> **Scope.** This plan carries QRA from "honest, correctly-instrumented pilot" to a **submittable methodology paper** (`docs/methodology.md`) and a **credibility exhibit for the Varela re-application**, then onward through the mechanism roadmap as Cohorts 3–4 accrue. It leads with publication readiness (§0); the mechanism roadmap that the manuscript rests on follows (§1–§3, §6–§8); the code work that makes the paper publishable is §4; the grant pivot is §5.

 
> **Where we are (2026-06-07).** The central mechanism estimator is **re-centered, built, wired, and verified**; the discovery/construct-validation layer (H6) is the strongest contribution; the manuscript has had a **truthing pass** (§0). The pilot is **honestly under-identified at n≈32**; the binding constraints are *data scale* and *one unstarted human-validation pass* (PURER), not instrument design.
>
> **Companions:** `methodology_assessment.md` (the design review — *did we build the right thing*), `experiments/mechanism/RESULTS.md` (the E1–E9 evidence), `methodology.md` (the manuscript), `docs/OUTCOME_INTEGRATION_ROADMAP.md` (the H4 bridge).

---

## 0. Publication-readiness board — start here

**The manuscript-truthing pass is applied** (`docs/methodology.md`, 2026-06-07). Each item below was a place where the prose claimed more than the data/code support; all are now corrected. Remaining work is human-gated (PURER coding) or strengthening (code wiring), not blocking.

| # | Item | Why it mattered | State |
|---|---|---|---|
| **P0-1** | Own the 4-stage VA-MR → 5-class VAAMR extension (abstract, §2.2) | Published paper says "four stages and one barrier"; silent 5-stage attribution = reviewer distrust | ✅ done — extension framed + rationale |
| **P0-2** | Correct PURER validation status to *planned, not started*; gate every therapist claim as directional (§2.3, §4.8, §6.2, §7.2, §7.3, §8.1) | "Underway" was false; the dyadic-mechanism story rests on un-validated cue labels | ✅ done — all instances corrected |
| **P0-3** | Avoidance-barrier report described as both built and not-built (§6.3, §8.2) | Self-contradiction; it **is** built (`mechanism.py` → `avoidance_barrier.txt`) | ✅ done — flipped to ✓ implemented |
| **P0-4** | Capability table still sold the GNN classifier as authoritative-promotable; omitted the probe (§4) | Contradicted the paper's own repositioning + CLAUDE.md | ✅ done — classifier marked default-OFF, promotion retired, probe row added |
| **P1-1** | H3 numbering gap (H1,H2,—,H4,H5,H6) | Silent gap reads as an error | ✅ done — explicit "H3 deferred" placeholder |
| **P1-2** | VCE 54/6 vs 59/7 | Manuscript described Lindahl's 59; we implement a 54-code adaptation | ✅ done — adaptation noted; counts fixed |
| **P1-3** | Leftover "[Verify year…]" editor note in References | Embarrassing in a submission | ✅ done — note removed, Schmidt ref fixed |
| **P2-1/2** | Grant-pivot + sharpened contribution (§10) | Reframes the rejection's GNN-classifier thesis as principled discovery | ✅ done — architecture-lesson paragraph added |
| **P1-4** | Venue: methodology.md (methods venues) vs `docs/ROADMAP.md` ("Journal of Contemplative Studies") | Tone/format consistency | ☐ align ROADMAP to the methods venue |

**The sequence to submission:**
1. **This week — Cohort-3 program-adaptation report.** Achievable now from existing outputs (`efficacy.py`, `mechanism.py`, `01_outcomes/avoidance_barrier.txt`). *Frame all PURER/therapist recommendations as directional* (PURER unvalidated).
2. **+1 week — methodology submission.** The science is ready; the truthing pass is done. Submit as an *honest under-identified pilot of a novel method* — that framing is the asset, not a weakness.
3. **Parallel, human-gated — start the PURER 20% blind-coding now.** It is the one substantive gap; having it *truthfully in progress* strengthens both the paper's next revision and the re-application.

**Do NOT block submission on:** confirmatory mechanism (needs Cohorts 3–4), H4 REDCap linkage, encoder-general H6. *(The `theme_framework→constructs` refactor is complete — all paths updated.)*

---

## 1. Mechanism & hypothesis status board

| Stream | State today | Next move |
|---|---|---|
| **Mechanism estimator** (H2/§7.6 interaction) | ✅ Built + wired + green; reproduces experiment (LR p=0.52, E-values 4.23/3.81/3.49) | **Phase A** — pass the statistical-correctness review → promote |
| **Confound sensitivity** (E-value/Rosenbaum) | ✅ Computed + reported beside the divergence map | Phase A — review the SMD→RR derivation + add CI-limit E-value |
| **H1** developmental progression | ✅ Tested (E9: slope CI [0.007,0.197] excludes 0; barrier 85%) | **Phase C** — wire E9's test into `efficacy.py` production (currently experiment-only) |
| **H6** discriminant validity | ✅ on Qwen; ⚠ **encoder-generality unconfirmed** (E6 flips on MiniLM) | **Phase B** — run the faithful two-encoder `discriminant.py` test |
| **PURER** therapist labels | ⚠ **No human IRR exists**; coding *planned, not started*; per-move ranking fragile (E5) | **Phase B** — human-validate to α≥0.70 (gates *all* therapist-effect claims) |
| **Trajectory / consolidation** | ✅ Built; under-identified at n≈32 (only persistence survives, E4) | **Phase E** — earns power at Cohorts 3–4 |
| **H4** language → therapeutic benefit | ⛔ REDCap not imported | **Phase D** — pre-register directions now; run when data lands |
| **H5** graph-as-scaler | ✅ Settled — refuted at n≈32; distillation ceiling is *data* (best per-rater ensemble κ 0.45/0.36) | Re-open at Cohorts 3–4 N |
| **Bayesian estimator** | ✅ Validated in isolated `.venv_bayes` (numpy≥2 conflict) | Phase A — decide the production dependency posture |

**One-sentence status:** *the instruments are correct and the pilot is honestly under-identified at n≈32; the work is to (A) sign off the statistics and ship, (B) discharge the two claim-blocking dependencies — PURER validation and the H6 encoder test, (C) lock the pre-REDCap defensible product and pre-register, (D) link to benefit when REDCap lands, and (E) confirm at Cohorts 3–4.*

---

## 2. The epistemic chain — the map we are navigating

```
  human raters ⟷ (IRR-validated) ⟷ multi-run LLM consensus      ← LABELS OF RECORD (VAAMR ✅ ; PURER ⚠ α≥0.70 PENDING, coding NOT STARTED — Phase B)
                                          │
                                          ▼
              FROM → CUE → TO triples (cue_blocks.py)  +  participant session trajectories (efficacy.py)
                                          │
            ┌─────────────────────────────┼───────────────────────────────────────────┐
            ▼                             ▼                                              ▼
   ADJACENCY MECHANISM ✅          TRAJECTORY MECHANISM ✅(under-id)            CONSTRUCT VALIDATION
   TO_stage ~ FROM*move           within/between split; needs scale           H6 ✅ on Qwen / ⚠ encoder-general
   + (1|participant)              (Phase E)                                    (Phase B: two-encoder test)
            │                             │                                              │
            └───── bounded by ───────► E-VALUE / ROSENBAUM + identifying assumptions ✅ (§9.4)
                                          │
                                          ▼
              CONVERGENT VALIDITY (H4): language ⟷ REDCap outcomes   ⛔ Phase D — the bridge to BENEFIT
```

*(The former `expected_codes` / `expected_transitions` pre-registration node has been removed — H3 deferred, machinery scrubbed from code and manuscript.)*

---

## 3. The forward roadmap

### Phase A — Sign off the statistics and promote to production *(now; gates the curriculum report)*
The estimator is wired and green; what stands between it and being the **default** mechanism analysis is a statistical-correctness review (the P0/P1 gates, recorded in the mechanism experiments log). Close these, in order:
1. **E-value derivation** (`mechanism_model.sensitivity_bounds` / `stats.smd_to_risk_ratio`): confirm the continuous-effect → SMD → `RR≈exp(0.91·SMD)` → E-value chain and the comparison-group choice; **add the CI-limit E-value** (not just the point estimate).
2. **Ordinal-LR inference** (`stats.ordered_logit`): the in-sample LR p-value is **not participant-cluster-robust** (statsmodels `OrderedModel` limitation); decide whether the participant-grouped CV is sufficient leakage-free inference, or add a **cluster-bootstrap LR / GEE / Bayesian** path. *This is the most likely reviewer pushback.*
3. **Singular-fit handling** in `stats.mixedlm_interaction` (the count of CI-excluding-0 interaction terms must not mislead on a rank-deficient design), plus `rosenbaum_bounds` + `within_between_split` correctness.
4. **Production posture:** confirm frequentist-in-process / Bayesian-isolated (vs. upgrading `transformers`), and that `mechanism.enabled=True` default-on is desired.
- **Exit:** P0/P1 signed off → the curriculum-modification report (§6.3) leads with the hierarchical interaction estimate + E-value bounds; the Bayesian arm stays isolated until the dependency posture is settled.

### Phase B — Discharge the two claim-blocking dependencies *(now; human + infra)*
Neither blocks the *code*; both block calling a result **primary evidence**.
1. **Human-validate PURER to α≥0.70.** Two rater teams blind-code a 20% stratified PURER sample (the VAAMR protocol, reused). **Coding has not started — begin now.** Import via `qra irr`. *Until this clears, every therapist-effect claim carries the E5 caveat and is directional only — the dyadic mechanism story literally rests on these labels.* When it clears, set `mechanism.purer_disagreement_rate` from the measured single-rater rate so E5 uses real noise.
2. **Run the faithful two-encoder H6 test.** Re-run `gnn_layer/discriminant.py` (probe vs Correct-&-Smooth) on **Qwen and a second domain-matched encoder** (the E6 proxy flipped on MiniLM, plausibly a capacity/"No code"-clustering artifact). Needs a working Qwen embedding endpoint (the campaign run failed to load it). *Until then, the manuscript claims H6 on Qwen embeddings only.*

### Phase C — Lock the pre-REDCap defensible product and pre-register *(now)*
1. **Wire E9's H1 test into `efficacy.py` production** (it currently lives only in the experiment): the group slope CI + the avoidance-barrier rate-limiting contrast become a first-class report, not a one-off.
2. **Pre-register the H4 outcome-correlation directions now** in `efficacy.link_to_external` (per `OUTCOME_INTEGRATION_ROADMAP.md`) — pain/TSK-11/ODI/MRPS/MAIA-2, Spearman, directions fixed *before* the data — so Phase D is confirmatory.
3. **Ship the dyadic descriptive structure** (PURER × VAAMR FROM→CUE→TO) as *descriptive*, gated on Phase B.
- **Exit:** a curriculum report + a manuscript draft whose claims are each tagged with an epistemic tier (tested / bounded-hypothesis-generating / leads-only) and whose predictions are code-level pre-registered.

### Phase D — The benefit bridge *(REDCap-gated — the ultimate goal)*
When outcomes are exported to `02_meta/outcomes.csv`: `efficacy.link_to_external` runs the **pre-registered** Spearman directions linking each participant's language trajectory (progression slope, barrier crossing, consolidated Reappraisal) to clinical change, with participant-cluster bootstrap CIs and joint displays. **This is the path from language → therapeutic benefit** — convergent-validity, explicitly *not* efficacy (single-arm). Closes H4.

### Phase E — Confirmatory replication and the model at scale *(Cohorts 3–4)*
1. **Confirmatory test** of the §7.6 directional predictions per (FROM × move) cell, now FDR-powered.
2. **The hierarchical interaction model gets power** — same instrument, re-run; report which interaction credible intervals now exclude 0.
3. **The trajectory/consolidation model earns its place** (E4 under-identified at n≈32) — within- vs between-session split as the closest internal proxy for "learning," feeding Phase D.
4. **Re-open the learned-scaler question** (whether a fine-tuned model or graph earns the LLM-free role at higher N) and **replicate H6** across cohorts.
5. **Enumerate quasi-experimental levers** (session phase, dose, therapist identity, curriculum module) for a within-design contrast that partially breaks selection-on-state.

---

## 4. Code next-steps that make the methodology publishable *(the codebase backlog)*

Tier-tagged. **None block the v1 submission** (the manuscript already discloses each gap), but the first three materially raise both defensibility and the research team's trust, and are cheap/now-runnable.

| Item | Tier | Effort | Payoff |
|---|---|---|---|
| **Wire E9's H1 test into `efficacy.py` production** (group-slope CI + barrier-rate-limiting contrast) — today it's experiment-only | Strengthens | Low | H1 becomes a first-class report behind the Cohort-3 numbers, not a one-off |
| **Confirm the avoidance-barrier dossier renders on `./data/Meta`** (`01_outcomes/avoidance_barrier.txt`, `mechanism_avoidance_barrier.csv`) | Verify | Trivial | The §6.3/§8.2 "✓ implemented" claim is demonstrably true |
| ✅ **Justification-grounding audit** — DONE (default-on, `analysis/reports/justification_grounding.py`; built→code-reviewed→fixed→24 tests green). **78.5% of quoted spans grounded** on Cohorts 1–2 (560/713), per-stage 72.8–80.5%, per-model gemma-4-31b 88.2% > nemotron-3-nano 75.6%; PURER 81.7%. Documented §5.6. Honest caveat: bounds confabulation, *not* correctness; lower bound (lexical) | Upgrade #2 — DONE | "Is the LLM hallucinating?" → a measured number |
| ✅ **Segmentation-sensitivity check** — DONE (opt-in, `analysis/segmentation_sensitivity.py`). H1 direction **STABLE** across the OFAT grid (slope +0.081…+0.093). *Honestly scoped* (code review forced the fix): measures the real mixture coord (not a hard-label proxy), labels **projected not re-classified**, **MiniLM space** (Qwen won't load under the pin), mixedlm **non-converged at n≈32** — all surfaced. Documented §4.1/§9. Stronger re-classifying + Qwen-space arms = future work | Upgrade #3 — DONE (limited claim) | Closes a hidden degree of freedom — directionally |
| **PURER-label-noise robustness** (E5) — perturb at the measured single-rater rate, re-fit, re-rank | Strengthens | Low | Bounds how much un-validated cue labels move the mechanism ranking, pending Phase B |
| **Multi-model consensus for the label of record** (promote the probe campaign's 3 independent checker LLMs to production VAAMR) | Architecture upgrade #1 | Med | Converts the label-of-record reliability from single-model *stability* to cross-model *independence* (§4.3) — the biggest spine upgrade |
| **Two-encoder H6** (Phase B-2) + a working Qwen embedding endpoint | Strengthens | Med (infra) | Makes H6 encoder-general rather than Qwen-only |

**Verifiability framing to state plainly in reports + to the team:** the outputs are **strongly auditable** (every label → provenance + confidence tier + raw ballots; per-item IRR dossiers; content-hashed frozen segments; canonical stat libraries) but only **partially reproducible** (LLM labels are stochastic + model-version-dependent — we freeze the ballots, the human anchor, the prompts, and the model versions, not a regeneration recipe). Say exactly that; it is bulletproof and it builds trust.

**The trust workflow (how to help the research team believe the results):** lead every walkthrough with the line-by-line IRR dossier (machine reasoning beside human reasoning); show the confidence tiers + flagged-for-review queue (the system knows its limits); invite the qualitative team to *find a wrong one* among high-confidence labels; tier every curriculum recommendation with its caveat; foreground the honest negatives (H5 refuted, mechanism under-identified, PURER unvalidated); show convergence across substrates that don't share machinery; and open the artifacts (ballots, human codes, prompts, model versions, analysis code).

---

## 5. The grant pivot — methodology paper → Varela re-application

The rejected application's Aim 1 promised a GNN/CFiCS classifier reaching **κ ≥ 0.70 with humans**, bootstrapping to label 600+ transcripts (Aim 1–2), then fine-tuning MindfulBERT (Aim 3). The pilot **refuted the technical thesis and discovered the defensible one** — and the methodology paper is the evidence of that rigor. The re-application should be re-spec'd accordingly:

1. **Drop the κ ≥ 0.70 target; re-spec reliability to the construct's own ceiling.** Human–human VAAMR agreement is α ≈ 0.47–0.52, so κ ≥ 0.70 was *never achievable* — the target exceeded the construct's ceiling. Re-state the reliability aim as "approach the human band," with the LLM consensus (κ = 0.537 vs human) as the demonstrated engine.
2. **Reframe Aim 1** around the **validated LLM-consensus engine + the H6 discriminant-validity finding** (VAAMR is developmental, not topical) — the graph's classification *failure* is the positive construct result, not a missed deliverable.
3. **Reframe Aim 3 (MindfulBERT)** as "scale the labeled corpus so the *data-limited* ceiling rises" — the distillation campaign proved the ceiling is **data, not method** (three independent methods converge at classifier↔human κ ≈ 0.45). More Varela data is exactly what buys the scaler role; the probe is the assistive, gated pre-labeler in the interim.
4. **Lead the re-application with the methodology paper** as the credibility exhibit: it shows a rigorous pilot that caught its own leakage (κ 0.25→0.05), repositioned on evidence rather than attachment, and reports its negatives as plainly as its positives. That is the panel's strongest reason to fund the scale-up.

---

## 6. Open scientific questions still ahead (the forward experiment queue)

| Question | Why it matters | When |
|---|---|---|
| Cluster-robust ordinal inference (or a cluster-bootstrap LR) | the current in-sample LR p ignores within-participant correlation | Phase A |
| The faithful two-encoder H6 test | H6 is the headline; generality is unconfirmed | Phase B |
| Does any (FROM × move) interaction cell become credibly non-zero at N? | the actual §7.6 claim | Phase E |
| Does the trajectory model identify *consolidation* effects at N? | closest pre-REDCap proxy for benefit | Phase E |
| Are dyadic routines more than move-persistence? | E4 found only persistence at n≈32 | Phase E |
| Quasi-experimental lever for partial de-confounding | the §9.4 confound is bounded, not broken | Phase E |
| Does the probe/learned scaler earn the LLM-free role at higher N? | the scaling bet; ceiling is data | Phase E |

---

## 7. Decision rules — when a result becomes *primary evidence*

- **A mechanism (therapist-effect) claim is primary evidence iff:** the hierarchical interaction estimate is reported with intervals **and** an E-value/Rosenbaum bound **and** PURER has cleared α≥0.70 (or the E5 robustness shows ranking stability) **and** the prediction was code-pre-registered. Until all four hold, it is *bounded, hypothesis-generating*. *(Today: PURER coding not started → 0/4 → directional.)*
- **H1 is tested-and-supported** (E9) — report it as such, flagged underpowered (n=20), monotonicity not yet shown.
- **H6 is primary on Qwen**, *provisional on encoder-generality* until Phase B.
- **The GNN stays scoped** to H6 + lead-generation; the transition counterfactual is a sensitivity lens. Do not re-expand it into the estimator role without first showing the hierarchical model insufficient.
- **No claim links language to clinical benefit** until Phase D (REDCap), and even then it is convergent validity, not efficacy.

---

## 8. Reference — what exists now (the baseline this plan builds on)

Implemented + green:
- `src/analysis/mechanism_model.py` — `MechanismModelConfig`, `fit_adjacency_interaction` (ordinal LR + Gaussian-mixed interaction + earns-its-place CV + opt-in isolated Bayesian), `sensitivity_bounds`, `purer_noise_robustness`, `fit_trajectory`, `run_mechanism_models`.
- `src/analysis/stats.py` — `ordered_logit`, `likelihood_ratio_test`, `mixedlm_interaction`, `e_value`, `smd_to_risk_ratio`, `rosenbaum_bounds`, `within_between_split`.
- `src/analysis/mechanism.py` — report **leads** with the interaction estimate + sensitivity + identifying assumption; additive table demoted; GNN counterfactual = sensitivity lens; **avoidance-barrier dossier** (`_avoidance_barrier` → `01_outcomes/avoidance_barrier.txt`, `mechanism_avoidance_barrier.csv`).
- `src/analysis/efficacy.py` — `compute_group_trajectory`, `compute_participant_slopes`, `compute_barrier_crossing`, `link_to_external` (H4, inert until REDCap).
- `src/gnn_layer/discriminant.py` — H6 probe-vs-content-similarity instrument (default-on).
- `src/classification_tools/probe/probe_classifier.py` — the per-rater-ensemble LLM-free scaler (`qra probe`, `probe_consensus` tier, below the LLM).
- `src/gnn_layer/classifier/` — the GraphSAGE consensus-distillation classifier + gate, **default-OFF** (`gnn_classifier_enabled=False`; H5-refuted, re-adjudicate at higher N).
- `src/process/config.py` — `PipelineConfig.mechanism`.
- `tests/unit/test_mechanism_model.py` — hermetic coverage; backward-compat (`enabled=False` ⇒ unchanged).
- `experiments/mechanism/` — the E1–E9 campaign (+ isolated `run_bayesian_ordinal.py`, `.venv_bayes`); `experiments/classification_scaler/` — the distillation campaign.

**Not yet wired (forward work, Phase C):** the E9 H1 test in `efficacy.py`; the pre-registered H4 directions in `efficacy.link_to_external`. **Pending external (Phase B/D):** PURER human IRR (coding not started); a working Qwen embedding endpoint for the two-encoder H6 test; the REDCap `outcomes.csv` export.
