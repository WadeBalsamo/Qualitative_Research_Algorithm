# Mechanism Master Plan — The Road from Here

> **⚠️ SUPERSEDED IN PART (2026-06-07).** The VAAMR×VCE cross-framework **construct-validity** work referenced in this plan — `expected_codes` / `expected_transitions` pre-registration, the shuffled-stage permutation control, and hypotheses **H3/H3a** — was **deferred to future research and removed** from the code and manuscript. The mechanism re-centering work (the substance of this plan) and the VCE classifier (as optional exploratory enrichment) are **retained**. Current framing: `docs/methodology.md` §3.3, §5.2, §8.2.

> **Where we are (2026-06-07).** The central mechanism estimator has been **re-centered, built, wired into
> production, and verified** (full unit suite green; reproduces the validation numbers exactly). QRA can now
> estimate *stage-moderated therapist effects* with the right instrument and bound them with a formal
> confound-sensitivity analysis. **This document is the forward roadmap** from that baseline to the ultimate
> goal — *associating therapist language with participant progression and, via REDCap, with therapeutic
> benefit* — in a defensible, peer-review-ready way. The retrospective detail lives in the companions; this
> plan looks forward.
>
> **Companions:** `methodology_assessment.md` (the review — *did we build the right thing*), `CODE_REVIEW_REQUEST.md`
> (exactly what changed + the review gates), `experiments/mechanism/RESULTS.md` (the E1–E9 evidence),
> `docs/methodology.md` (the manuscript), `scalable_classification_master_plan.md` (the separate LLM-free
> *labeling* tier), `docs/OUTCOME_INTEGRATION_ROADMAP.md` (the H4 bridge).

---

## 0. Status board — what is done, what gates, what is next

| Stream | State today | Next move |
|---|---|---|
| **Mechanism estimator** (H2/§7.6 interaction) | ✅ Built + wired + green; reproduces experiment (LR p=0.52, E-values 4.23/3.81/3.49) | **Phase A** — pass the statistical-correctness review → promote |
| **Confound sensitivity** (E-value/Rosenbaum) | ✅ Computed + reported beside the divergence map | Phase A — review the SMD→RR derivation + add CI-limit E-value |
| **VCE×VAAMR construct validity** (H3 / `expected_codes` / permutation) | 🚫 DEFERRED to future research; machinery removed (2026-06-07) | — (VCE kept as optional enrichment; no validity claim rests on it) |
| **H1** developmental progression | ✅ Tested (E9: slope CI [0.007,0.197] excludes 0; barrier 85%) | **Phase C** — wire E9's test into `efficacy.py` production (currently experiment-only) |
| **H6** discriminant validity | ✅ on Qwen; ⚠ **encoder-generality unconfirmed** (E6 flips on MiniLM) | **Phase B** — run the faithful two-encoder `discriminant.py` test |
| **PURER** therapist labels | ⚠ **No human IRR exists**; per-move ranking fragile (E5) | **Phase B** — human-validate to α≥0.70 (gates *all* therapist-effect claims) |
| **Trajectory / consolidation** | ✅ Built; under-identified at n≈32 (only persistence survives, E4) | **Phase E** — earns power at Cohorts 3–4 |
| **H4** language → therapeutic benefit | ⛔ REDCap not imported | **Phase D** — pre-register directions now; run when data lands |
| **Bayesian estimator** | ✅ Validated in isolated `.venv_bayes` (numpy≥2 conflict) | Phase A — decide the production dependency posture |

**The one-sentence status:** *the instruments are correct and the pilot is honestly under-identified at
n≈32; the work ahead is to (A) sign off the statistics and ship, (B) discharge the two claim-blocking
dependencies — PURER validation and the H6 encoder test, (C) lock the pre-REDCap defensible product and
pre-register, (D) link to benefit when REDCap lands, and (E) confirm at Cohorts 3–4.*

---

## 1. Where we are — the verified baseline (today's starting point)

**The gap we closed.** QRA's central claim (H2/§7.6) is an **interaction** — a therapist move's effect on the
next participant stage is *moderated by the FROM stage*. The shipped estimator fit a move *main effect only*;
the interaction was tested only by an underpowered per-cell FDR table. We replaced the mechanism lead with a
**hierarchical ordinal interaction model + a formal confound-sensitivity argument**, and scoped the GNN to
**H6 construct validation + lead-generation**.

**What the evidence says now** (campaign: `experiments/mechanism/`, 186 triples / 20 participants / 160 with a
cue move — full detail in `RESULTS.md`):
- **Right instrument, honestly under-identified.** The cue earns its place as a PURER-move *main effect*
  (held-out log-loss 1.553→1.51) but the *interaction* does not (LR p=0.52; Gaussian design singular; FDR
  0/20). The **Bayesian model fits the interaction** (16 terms, 0 divergences) where the MLE is singular, and
  reports 0/16 intervals exclude 0.
- **Two directional positives:** the cue signal is *process, not content* (E3: PURER beats the content
  embedding, which *hurts*), and **H1 holds** (E9: slope CI excludes 0; barrier rate-limiting).
- **Two honesty flags that bound the claims:** the per-move ranking is **fragile under PURER label noise**
  (E5) and **no PURER IRR exists yet**; and **H6's contrast flips on MiniLM** (E6) so it is shown on Qwen
  only.
- **Sensitivity is real and computable:** E-values Avoidance×Education 4.23, AttnReg×Reinforcement 3.81.

**Dependency reality that shapes everything downstream.** The Bayesian stack needs `numpy≥2`; the pipeline's
`transformers==4.42.4` needs `numpy<2`. They cannot coexist, so the **in-process default is the frequentist
ordinal+mixed interaction model** and the **Bayesian arm runs in an isolated `.venv_bayes`**. `bambi` is not a
live dependency.

---

## 2. The epistemic chain — the map we are navigating

```
  human raters ⟷ (IRR-validated) ⟷ multi-run LLM consensus      ← LABELS OF RECORD (VAAMR ✅ ; PURER ⚠ α≥0.70 PENDING — Phase B)
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
              PRE-REGISTERED expected_codes / expected_transitions ✅ wired (Phase A: verify mappings; run before Cohort 3)
                                          │
                                          ▼
              CONVERGENT VALIDITY (H4): language ⟷ REDCap outcomes   ⛔ Phase D — the bridge to BENEFIT
```

---

## 3. The forward roadmap

### Phase A — Sign off the statistics and promote to production *(now; gates the curriculum report)*
The estimator is wired and green; what stands between it and being the **default** mechanism analysis is a
statistical-correctness review (the P0/P1 gates in `CODE_REVIEW_REQUEST.md`). Close these, in order:
1. **E-value derivation** (`mechanism_model.sensitivity_bounds` / `stats.smd_to_risk_ratio`): confirm the
   continuous-effect → SMD → `RR≈exp(0.91·SMD)` → E-value chain and the comparison-group choice; **add the
   CI-limit E-value** (not just the point estimate).
2. **Ordinal-LR inference** (`stats.ordered_logit`): the in-sample LR p-value is **not participant-cluster-
   robust** (statsmodels `OrderedModel` limitation); decide whether the participant-grouped CV is sufficient
   leakage-free inference, or add a **cluster-bootstrap LR / GEE / Bayesian** path. *This is the most likely
   reviewer pushback.*
3. **Singular-fit handling** in `stats.mixedlm_interaction` (the count of CI-excluding-0 interaction terms
   must not mislead on a rank-deficient design), plus `rosenbaum_bounds` + `within_between_split` correctness.
4. **`expected_codes` domain mappings** (`frameworks/VAAMR_FRAMEWORK.md`/`PURER_FRAMEWORK.md`): a domain
   expert verifies the stage→VCE-code and move→transition predictions — this is **pre-registration**, it must
   be right *before* Cohort 3.
5. **Production posture:** confirm frequentist-in-process / Bayesian-isolated (vs. upgrading `transformers`),
   and that `mechanism.enabled=True` default-on is desired.
- **Exit:** P0/P1 signed off → the curriculum-modification report (§6.3) leads with the hierarchical interaction
  estimate + E-value bounds; the Bayesian arm stays isolated until the dependency posture is settled.

### Phase B — Discharge the two claim-blocking dependencies *(now; human + infra)*
Neither blocks the *code*; both block calling a result **primary evidence**.
1. **Human-validate PURER to α≥0.70.** Two rater teams blind-code a 20% stratified PURER sample (the VAAMR
   protocol, reused). Import via `qra irr`. *Until this clears, every therapist-effect claim carries the E5
   caveat and is directional only — the dyadic mechanism story literally rests on these labels.* When it
   clears, set `mechanism.purer_disagreement_rate` from the measured single-rater rate so E5 uses real noise.
2. **Run the faithful two-encoder H6 test.** Re-run `gnn_layer/discriminant.py` (probe vs Correct-&-Smooth)
   on **Qwen and a second domain-matched encoder** (the E6 proxy flipped on MiniLM, plausibly a capacity/"No
   code"-clustering artifact). This needs a working Qwen embedding endpoint (the campaign run failed to load
   it). *Until then, the manuscript claims H6 on Qwen embeddings only.*

### Phase C — Lock the pre-REDCap defensible product and pre-register *(now)*
Assemble the strongest honest deliverable the present data supports, and freeze the predictions that make the
later cohorts confirmatory:
1. **Wire E9's H1 test into `efficacy.py` production** (it currently lives only in the experiment): the group
   slope CI + the avoidance-barrier rate-limiting contrast become a first-class report, not a one-off.
2. **Run + review the §8.2 controls** that are now wired (`construct_validity_checks.json`): the shuffled-
   stage permutation lift and the `expected_codes` mechanical comparison.
3. **Pre-register the H4 outcome-correlation directions now** in `efficacy.link_to_external` (per
   `OUTCOME_INTEGRATION_ROADMAP.md`) — pain/TSK-11/ODI/MRPS/MAIA-2, Spearman, directions fixed *before* the
   data — so Phase D is confirmatory rather than exploratory.
4. **Ship the dyadic descriptive structure** (PURER × VAAMR FROM→CUE→TO) as *descriptive*, gated on Phase B.
- **Exit:** a curriculum report + a manuscript draft whose claims are each tagged with an epistemic tier
  (tested / bounded-hypothesis-generating / leads-only) and whose predictions are code-level pre-registered.

### Phase D — The benefit bridge *(REDCap-gated — the ultimate goal)*
When outcomes are exported to `02_meta/outcomes.csv`: `efficacy.link_to_external` runs the **pre-registered**
Spearman directions linking each participant's language trajectory (progression slope, barrier crossing,
consolidated Reappraisal) to clinical change (pain, kinesiophobia, disability, reappraisal, interoception),
with participant-cluster bootstrap CIs and joint displays. **This is the path from language → therapeutic
benefit** — convergent-validity, explicitly *not* efficacy (single-arm). It also closes H4 and lets the
avoidance-barrier analysis correlate with outcomes (the §8.2 Item-2 hook).

### Phase E — Confirmatory replication and the model at scale *(Cohorts 3–4)*
Cohorts 3–4 provide the confirmatory power the pilot lacks:
1. **Confirmatory test** of the §7.6 directional predictions per (FROM × move) cell, now FDR-powered.
2. **The hierarchical interaction model gets power** — the same instrument, re-run; report which interaction
   credible intervals now exclude 0.
3. **The trajectory/consolidation model earns its place** (E4 was under-identified at n≈32) — the within- vs
   between-session split becomes the closest internal proxy for "learning," feeding Phase D.
4. **Re-open the learned-scaler question** (whether a fine-tuned model or graph earns the LLM-free role at
   higher N) and **replicate H6** across cohorts.
5. **Enumerate quasi-experimental levers** (session phase, dose, therapist identity, curriculum module) for a
   within-design contrast that partially breaks selection-on-state.

---

## 4. Open scientific questions still ahead (the forward experiment queue)

| Question | Why it matters | When |
|---|---|---|
| Cluster-robust ordinal inference (or a cluster-bootstrap LR) | the current in-sample LR p ignores within-participant correlation | Phase A |
| The faithful two-encoder H6 test | H6 is the headline; generality is unconfirmed | Phase B |
| Does any (FROM × move) interaction cell become credibly non-zero at N? | the actual §7.6 claim | Phase E |
| Does the trajectory model identify *consolidation* effects at N? | closest pre-REDCap proxy for benefit | Phase E |
| Are dyadic routines more than move-persistence? | E4 found only persistence at n≈32 | Phase E |
| Quasi-experimental lever for partial de-confounding | the §9.4 confound is bounded, not broken | Phase E |

---

## 5. Decision rules — when a result becomes *primary evidence*

- **A mechanism (therapist-effect) claim is primary evidence iff:** the hierarchical interaction estimate is
  reported with intervals **and** an E-value/Rosenbaum bound **and** PURER has cleared α≥0.70 (or the E5
  robustness shows ranking stability) **and** the prediction was code-pre-registered. Until all four hold, it
  is *bounded, hypothesis-generating*. *(Today: 0/4 of the gates that need Phase B/A are closed → directional.)*
- **H1 is tested-and-supported** (E9) — report it as such, flagged underpowered (n=20), monotonicity not yet
  shown.
- **H6 is primary on Qwen**, *provisional on encoder-generality* until Phase B.
- **The GNN stays scoped** to H6 + lead-generation; the transition counterfactual is a sensitivity lens. Do
  not re-expand it into the estimator role without first showing the hierarchical model insufficient.
- **No claim links language to clinical benefit** until Phase D (REDCap), and even then it is convergent
  validity, not efficacy.

---

## 6. Reference — what exists now (the baseline this plan builds on)

Implemented + green (details + review gates in `CODE_REVIEW_REQUEST.md`):
- `src/analysis/mechanism_model.py` — `MechanismModelConfig`, `fit_adjacency_interaction` (ordinal LR +
  Gaussian-mixed interaction + earns-its-place CV + opt-in isolated Bayesian), `sensitivity_bounds`,
  `purer_noise_robustness`, `fit_trajectory`, `run_mechanism_models`.
- `src/analysis/stats.py` — `ordered_logit`, `likelihood_ratio_test`, `mixedlm_interaction`, `e_value`,
  `smd_to_risk_ratio`, `rosenbaum_bounds`, `within_between_split`.
- `src/analysis/mechanism.py` — report **leads** with the interaction estimate + sensitivity + identifying
  assumption; additive table demoted; GNN counterfactual = sensitivity lens.
- `src/process/cross_validation.py` (+ `orchestrator.py`) — `expected_codes` comparison + shuffled-stage
  permutation control → `construct_validity_checks.json`.
- `src/theme_framework/{theme_schema,markdown_loader}.py` + `frameworks/{VAAMR,PURER}_FRAMEWORK.md` —
  `expected_codes` / `expected_transitions` (markdown-driven pre-registration).
- `src/process/config.py` — `PipelineConfig.mechanism`.
- `tests/unit/test_mechanism_model.py` — hermetic coverage; backward-compat (`enabled=False` ⇒ unchanged).
- `experiments/mechanism/` — the E1–E9 campaign (+ isolated `run_bayesian_ordinal.py`, `.venv_bayes`).

**Not yet wired (forward work, Phase C):** the E9 H1 test in `efficacy.py`; the pre-registered H4 directions
in `efficacy.link_to_external`. **Pending external (Phase B/D):** PURER human IRR; a working Qwen embedding
endpoint for the two-encoder H6 test; the REDCap `outcomes.csv` export.
