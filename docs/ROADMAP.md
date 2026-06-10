# QRA Research Roadmap

> **Scope.** The single forward-looking plan for the QRA research program: finish the pilot's
> remaining instruments, publish the methodology, win the Varela re-application, validate at
> Cohorts 3–4 scale, and deliver the substantive Move-MORE results paper. Organized by **gating
> horizon**, not by legacy phase numbers.
>
> **Supersedes** `docs/masterplan.md` (2026-06, deleted) and the 2025 six-phase roadmap (this
> file's git history). Appendix C maps old section numbers to their new homes.
>
> **Companions:** `docs/methodology.md` (the manuscript / technical monograph),
> `docs/methodology_submission.md` (the journal-length draft), `experiments/mechanism/RESULTS.md`
> + `CAMPAIGN_LOG.md` (mechanism evidence + statistics sign-off), `experiments/docs/graph_experiments.md`
> (graph record), `references/varela-2026.txt` (grant application), `CLAUDE.md` (codebase map).
>
> **Maintenance rule:** when an item completes, strike it with a date; prune struck items at the
> next horizon boundary. History lives in git, not here.

---

## 0. Where we are — baseline (2026-06-10)

**One paragraph.** The instruments are correct and the pilot is honestly under-identified at
n≈32. The manuscript truthing pass is applied (2026-06-07) and the Phase-A statistical-correctness
gates are **closed** (2026-06-10; `experiments/mechanism/CAMPAIGN_LOG.md`). The binding
constraints are **data scale** and **one unstarted human-validation pass (PURER)** — not
instrument design. The 2025 Varela reviews are in hand; every objection has a concrete remedy
(§3). The next eight weeks are: C3 adaptation report → submission + preprint → grant.

### 0.1 Hypothesis / stream scoreboard

| Stream | State (2026-06-10, C1–C3-partial corpus) | Next move |
|---|---|---|
| **H1** developmental progression | **Supported (primary, ordinal-safe):** adaptive-stage occupancy 44%→100% across sessions, Mann-Kendall τ=+0.714, p=.019. Sensitivity: E[stage] slope +0.06/session [−0.01, +0.13], p=.072 (subordinate, interval-scale assumption); 11/14 participants advancing (sign test p=.057) | Replicate at full C3, then C4 (§4) |
| **H2** stage-moderated therapist effect | **Instrumented, under-identified.** Move earns its place as a *main effect* (grouped-CV log-loss 1.553→1.514); interaction does not (1.528); LR in-sample p=.522, cluster-bootstrap p=.934; 0/20 cells survive FDR | Confirmatory test at C3–4 power (§4.4) |
| **H3** VAAMR×VCE construct validity | **Deferred** (2026-06-07): shared-LLM-lexicon confound; machinery scrubbed from code + manuscript | Future work, independent substrate |
| **H4** language ⟷ clinical outcomes | **Scaffolded, inert.** `efficacy.link_to_external` ready; no REDCap export yet. Directions pre-registered (Appendix B) | Importer now (§1.4); run when data lands (§4.3) |
| **H5** learned scaler replaces LLM | **Refuted at n≈32.** Best LLM-free scaler = per-rater ensemble probe, κ 0.450 (human axis) / 0.361 (LLM axis) — gated, assistive, below the LLM | Re-adjudicate at C3–4 N (§4.5) |
| **H6** discriminant validity (stage ≠ topic) | **Headline finding, Qwen-scoped.** Probe ≫ content-similarity on identical folds/features, Δκ +0.170 [0.002, 0.318]; community×stage ARI ≈ 0 | Two-encoder generality test (§4.2) |
| **PURER human IRR** | **NOT STARTED** — the single biggest validity gap; all therapist-side claims directional | **Begin blind coding now** (§1.3) |
| **Label of record** | Multi-run LLM consensus at the human ceiling: κ=0.537 vs human consensus [0.39, 0.68]; human↔human α 0.325–0.523 (median 0.473); justification grounding 78% | Maintain; 20% blind-coding ongoing |
| **Mechanism statistics** | **Signed off** (Gates 1–4 ✅ 2026-06-10): CI-limit E-values (strongest cell point 4.33 / CI-limit 1.86; most cells →1.0), cluster-bootstrap LR, singular-fit handling, frequentist-in-process / Bayesian-isolated | Standing posture (§6) |

### 0.2 The epistemic chain (what gates what)

```
human raters ⟷ (IRR) ⟷ multi-run LLM consensus     ← LABEL OF RECORD (VAAMR ✅ at ceiling; PURER ⚠ α≥0.70 PENDING — §1.3/§4.1)
                        │
                        ▼
   FROM → CUE → TO triples (cue_blocks.py)  +  participant trajectories (efficacy.py)
            │                         │                            │
            ▼                         ▼                            ▼
   ADJACENCY MECHANISM ✅      TRAJECTORY/CONSOLIDATION       CONSTRUCT VALIDATION
   under-identified at n≈32    under-identified (§4.4)        H6 ✅ Qwen / ⚠ encoder-general (§4.2)
            │                                                       
            └── bounded by E-value / Rosenbaum + stated identifying assumption ✅ (methodology §9.4)
                        │
                        ▼
   CONVERGENT VALIDITY (H4): language ⟷ REDCap outcomes   ⛔ the bridge to BENEFIT (§4.3)
```

### 0.3 Cohort position

C1 (8 sessions, n=5) + C2 (8 sessions, n=10) complete and analyzed; C3 partially ingested
(3/8 sessions, n=6) — **C3 completes ~2026-08**; C4 thereafter; trial target N=32. Current
analyzed corpus: 20 participants, 19 sessions, 205 participant segments, 544 therapist segments
(`data/MMORE_Processed/06_reports/00_RESULTS.txt`).

---

## 1. Horizon 1 — Now → Cohort-3 adaptation report (exit ≤ 2026-06-24)

1. **Land the working tree.** Commit the top-level-synthesis refactor as one coherent change
   (`results_brief.py`, `methods_report.py`, `stat_format.py`, `thesis_figures.py`,
   `status_report.py`, tests, the report-tier renumbering) plus the doc merge (this file;
   masterplan + assessment deletions; stale-reference fixes). Delete root debris:
   `run_purer_gemma_watchdog.sh`, `STOP_GEMMA_WATCHDOG`, `.purer_gemma_watchdog.lock`,
   `analyze_meta.log`, `purer_gemma_probe.log`, `purer_rerun.log`.
2. ~~Statistics sign-off~~ — **CLOSED 2026-06-10** (Gates 1–4: E-value chain documented + CI-limit
   E-value shipped; `cluster_bootstrap_lr_test` added and wired; singular-fit suppression;
   production posture confirmed). Record: `experiments/mechanism/CAMPAIGN_LOG.md`. Standing
   residue is posture, not work: frequentist in-process default, Bayesian isolated in
   `.venv_bayes` (numpy≥2 conflict — see Appendix C note).
3. **Start PURER blind coding — this week.** Two rater teams, 20% stratified PURER sample
   (reuse the VAAMR worksheet protocol; `qra testset create --kind purer` → freeze → code →
   `qra irr import`). Target Krippendorff α ≥ 0.70. *Nothing else on this list is human-gated;
   this is.* Starting now also makes "PURER validation truthfully in progress" a citable fact in
   both the paper revision and the grant (§3). Fallback if α lands 0.40–0.69: run the E5
   label-noise ranking-stability check at the measured single-rater disagreement rate and gate
   claims on ranking stability instead of full validation (§6 decision rule).
4. **Build `qra outcomes import`** (the one big missing feature): REDCap CSV + off-repo PHI
   crosswalk → `02_meta/outcomes.csv` per the Appendix A contract, so H4 runs the day the
   coordinator exports outcomes. Confirm `link_to_external` reports Spearman as primary and
   encodes the Appendix B directions.
5. **Exit gate:** the Cohort-3 program-adaptation report (already drafted as
   `c1c2_program_analysis.md` + `c3s1_instructor_report.md`) refreshed on the full current
   corpus, every therapist-move recommendation tier-tagged directional (PURER unvalidated),
   each recommendation carrying its E-value bound — the `00_RESULTS.txt` §6 format.

---

## 2. Horizon 2 — Methods-paper submission + preprint (exit ≤ 2026-07-15)

The manuscript exists at two altitudes: `docs/methodology.md` (the 23k-word technical monograph
— stays the supplement / source of record) and `docs/methodology_submission.md` (the ~9k-word
journal draft). The submission, not the monograph, is what referees see.

1. **Finalize the submission draft** (see its header checklist): co-author pass (Wexler + team),
   then freeze.
2. **Re-freeze statistics on the as-submitted corpus.** The submission reports the C1–C3-partial
   corpus (20 ppts / 205 segments) from `00_RESULTS.txt` + `04_validation/irr/irr_results.json`.
   Sync the monograph's older C1–C2-only numbers where they differ — most consequentially the H1
   paragraph in §3.4 (the current corpus re-orders the H1 statistics: Mann-Kendall on adaptive
   occupancy is now the significant primary, p=.019; the E[stage] slope is the subordinate
   sensitivity, p=.072). **Do not wait for full C3** — the Varela timeline needs "under review"
   by early August; full-C3 replication is the revision-stage strengthener.
3. **Venue (decide at submission; ranked):**
   | Rank | Venue | Why | Risk |
   |---|---|---|---|
   | 1 | **International Journal of Qualitative Methods** | OA; methods-innovation friendly; the venue of the AI-positivism debate the Varela reviewer cited (Chatzichristos 2025) — publish the constructive answer where the critique lives | Qual audience may want less stats; frame stats as transparency |
   | 2 | **BMC Medical Research Methodology** | Trial-embedded methods scope; length-tolerant; OA; strong grant optics | Quant-leaning editors; phenomenology framing needs care |
   | 3 | **JMIR Formative Research / JMIR Mental Health** | Fastest credible turnaround if the clock runs short; AI-pipeline friendly | APC; less methods prestige |
   | later | *JMMR* (after H4 joint displays); *Mindfulness* (the post-C4 results paper, §5.1) | | |
4. **Preprint on submission day** (PsyArXiv; instant DOI) — the grant cites the DOI. Check
   quoted-segment de-identification once more before public posting.
5. **Open-source release checklist** (repo is public; make it citable and clean):
   `LICENSE` (decide: MIT/Apache-2.0), `CITATION.cff`, Zenodo archival DOI tied to the
   submission tag, PHI sweep (`git ls-files data/ references/` must be empty; scan committed
   fixtures for transcript text), README quick-start verified, AI-use/data-handling statement
   (classification ran on locally-served open-weights models via LM Studio — confirm from
   `02_meta/qra_config.json` and state "no transcript text left institutional infrastructure"
   if true).
6. **Submission gates** — blocks: items 1–2 only. Must-NOT-block (the paper discloses each):
   confirmatory mechanism (needs C3–4), H4 pilot result (needs REDCap), encoder-general H6,
   PURER validation *completion* (in-progress status is disclosed). The framing **"an honest,
   correctly-instrumented, under-identified pilot of a novel method" is the asset**, not a
   weakness.

---

## 3. Horizon 3 — Varela re-application (due ~2026-08-10)

The application is drafted at `references/varela-2026.txt`. The pivot, in four moves:
**(i)** retire κ≥0.70 as the reliability aim — it exceeded the construct's measured human
ceiling (α≈0.47–0.52); the demonstrated engine is the LLM consensus at κ=0.537 = human-level;
**(ii)** Aim 1 re-centers on the validated consensus engine + the H6 discriminant-validity
finding (the graph classifier's failure IS the construct result); **(iii)** MindfulBERT becomes
a *data-ceiling* argument — three independent methods converge at classifier↔human κ≈0.45, so
the ceiling is labeled data, which is precisely what the award buys; **(iv)** the methodology
paper (preprint + under review) leads as the rigor exhibit.

### 3.1 Reviewer-objection → remedy map (2025 reviews, on file)

| 2025 objection | 2026 remedy |
|---|---|
| R1+R2: ML collaborator (Balsamo) unsubstantiated — no publications/biosketch/graduate degree visible | Name as co-I **with biosketch + publications**: *Mindfulness* 2026 (2nd author) + first-author methods preprint/under-review + the open-source pipeline itself |
| R2: H1a (stage-distinct language) unsupported; language markers historically unreliable | Lead with measured evidence: human ceiling α 0.33–0.52, LLM at ceiling κ=0.537, H6 (stage is supervised-recoverable but NOT content-similar — Δκ +0.17) |
| R2: models fail on naive data; open-source deployment too quick | Participant-grouped CV that caught our own leakage (κ 0.25→0.05); gated + abstaining scalers; staged release gates; cross-trial held-out validation as an explicit aim |
| R2: AI qualitative analysis = positivism; first-person validity not preserved (Chatzichristos 2025) | Dedicated first-person-validity paragraph: human attention deployed not replaced; justification-grounding audit; per-item IRR dossiers; abstention; neurophenomenology lineage (Varela 1996); cite Chatzichristos directly |
| R2: "Ask ChatGPT" artifact in the application | Impeccable copy; AI-use transparency statement |
| R3: pain/MORE corpus confounds contemplative vs pain-management stages; instructor constraints | Staged multi-site Aim 2: Move-MORE validation Y1 → abbreviated MORE + STAMP Y2 (STAMP includes a **non-MORE mindfulness arm**); PURER fidelity profiles measure instructor variation; English/pain scope named with remediation path |

### 3.2 Evidence-exhibit checklist (assemble by 2026-07-29)

- [ ] Published foundational paper (*Mindfulness* 17:819–833) — Balsamo 2nd author
- [ ] Methods paper: PsyArXiv DOI + "under review at <venue>"
- [ ] Open-source repo + Zenodo DOI (§2.5)
- [ ] Pilot one-pager: κ=0.537 at the human ceiling; H1 MK p=.019; H6 Δκ +0.17; honest negatives (H5 refuted; leakage caught)
- [ ] PURER blind-coding **in progress** (started §1.3)
- [ ] Letters: Wexler (PI), Hanley (abbreviated-MORE data), STAMP team (DUA pathway), ML co-I biosketch
- [ ] Full C3 data milestone statement (completes ~08; strengthens feasibility)

### 3.3 Backward timeline

| Date | Milestone |
|---|---|
| 2026-06-24 | C3 adaptation report out (Horizon 1 exit) |
| 2026-07-15 | Methods paper submitted + preprinted (Horizon 2 exit) |
| 2026-07-29 | Varela full draft + exhibits assembled |
| 2026-08-05 | Internal review (Wexler + mentors) + budget finalized |
| 2026-08-08 | Submission buffer |
| ~2026-08-10 | **Varela application due** |

---

## 4. Horizon 4 — Cohorts 3–4: discharge the claim blockers, then confirm (2026-08 → trial end)

0. **Data milestones.** Full C3 (~2026-08): `qra add-data` → re-run analyses → revision-stage
   replication numbers. C4: corpus reaches trial target N=32.
1. **PURER IRR clearance** (started §1.3). On α ≥ 0.70: therapist-effect claims become eligible
   for primary evidence (§6 rule), and set `mechanism.purer_disagreement_rate` from the measured
   single-rater rate so the E5 noise-robustness check uses real noise. On 0.40–0.69: E5
   ranking-stability gates claims instead. Below 0.40: revise PURER definitions, re-code.
2. **Encoder-general H6.** Fix the embedding endpoint (the pinned `transformers==4.42.4` cannot
   load the Qwen3 embedder — upgrade path or a separate embedding service/venv), then run the
   faithful two-encoder `gnn_layer/discriminant.py` test (Qwen + one domain-matched second
   encoder). Until then every H6 claim stays Qwen-scoped. (The MiniLM proxy sign-flip is a
   capacity artifact hypothesis — test it, don't argue it.)
3. **H4 — the benefit bridge.** When `02_meta/outcomes.csv` lands (importer from §1.4): run the
   **pre-registered** Spearman directions (Appendix B) with participant-cluster bootstrap CIs +
   joint displays. This is **convergent validity, never efficacy** (single-arm). Optional
   reverse export `qra outcomes export-redcap` (per-participant dominant stage, progression
   slope, barrier-crossing session, adaptive occupancy → REDCap-importable CSV) so quantified
   qualitative variables live beside the trial's clinical record.
4. **Confirmatory mechanism at N.** Re-run the hierarchical ordinal interaction model — which
   (FROM×move) credible intervals now exclude 0; FDR-powered directional predictions per cell;
   the trajectory/consolidation model (within- vs between-session split) earns its place;
   enumerate quasi-experimental levers (session phase, dose, therapist identity, curriculum
   module) for a within-design contrast that partially breaks selection-on-state.
5. **Re-adjudicate the learned scaler at N.** Probe vs fine-tuned encoder vs graph — all scored
   on the **human axis** under participant/session-grouped splits (never split a session across
   train/test), same gate. Includes the multi-model-consensus label-of-record promotion
   (3 independent checker LLMs already in production config) and, if a scaler clears its gate,
   the bulk MindfulBERT corpus build (`process/assembly/mindfulbert_dataset.py`).
6. **Forward experiment queue:** does any interaction cell become credibly non-zero at N? · does
   consolidation (between-session) signal emerge? · are dyadic routines more than
   move-persistence (E4)? · does the probe/fine-tune earn the LLM-free role? · H6 replication
   across cohorts and encoders.

---

## 5. Horizon 5 — Post-trial (2027+)

1. **The Move-MORE Qualitative Results paper** — the substantive clinical-phenomenological
   contribution, distinct from the methods paper. Target: ***Mindfulness*** (continuity with the
   foundational paper; the field's home audience). Content: the four-cohort developmental arc
   with between-cohort curriculum changes as a natural experiment; the Avoidance barrier — who
   crosses, when, and the validated therapist language associated with crossing (incl. the
   movement-context contrast and "stuck participant" case studies); the mechanism dossier at
   confirmatory power; convergence with quantitative outcomes (H4 joint displays).
   **Prerequisites:** full corpus, PURER cleared, REDCap linked, scaler re-adjudicated.
2. **MindfulBERT — the long-term vision, data-ceiling framed.** (a) a fine-tuned classifier once
   the labeled corpus is large enough to beat the probe's κ≈0.45 ceiling; (b) corpus scaling
   beyond Move-MORE (abbreviated MORE, STAMP — the Varela Aim 2/3 work); (c) the generative
   capstone (cue-suggestion conditioned on FROM-state and observed Δprogression) is explicitly
   long-horizon: it inherits **no** validity from the analysis pipeline and requires its own
   prospective validation, safety review, and human-in-the-loop deployment before any clinical
   use. No named model/engine commitments until (a) clears its gate.

---

## 6. Standing rules (horizon-independent)

**When a mechanism (therapist-effect) claim becomes primary evidence** — all four, else it is
*bounded, hypothesis-generating*:
1. the hierarchical interaction estimate is reported with intervals, **and**
2. an E-value/Rosenbaum bound accompanies it (point + CI-limit), **and**
3. PURER has cleared α ≥ 0.70 (or the E5 noise check shows ranking stability), **and**
4. the prediction was code-level pre-registered before the data.
*Today: 2/4 (intervals ✅, bounds ✅, PURER ✗, pre-registration ✗ for mechanism cells) → directional.*

- **H1** is tested-and-supported on the ordinal-safe primary (flagged underpowered; monotone
  replication pending C3–4).
- **H6** is primary **on Qwen embeddings**; encoder-general only after §4.2.
- **The GNN stays scoped** to H6 + lead-generation (motifs, communities, coupling); the
  transition counterfactual is a sensitivity lens, never the estimator.
- **No claim links language to clinical benefit** before H4 — and then it is convergent
  validity, not efficacy (single-arm).
- **Verifiability framing (state it everywhere, verbatim):** outputs are **strongly auditable**
  (every label → provenance + confidence tier + raw ballots; per-item IRR dossiers;
  content-hashed frozen segments; canonical stat libraries) but only **partially reproducible**
  (LLM labels are stochastic and model-version-dependent — we freeze ballots, prompts, human
  anchor, and model versions, not a regeneration recipe).
- **Trust workflow with the research team:** lead with the line-by-line IRR dossier; show
  confidence tiers + the flagged-for-review queue; invite the team to find a wrong
  high-confidence label; tier every recommendation with its caveat; foreground the honest
  negatives; open the artifacts.

---

## Appendix A — REDCap outcome integration (the H4 data contract)

**Contract** (`analysis/efficacy.py:load_external_outcomes` — already implemented):
`02_meta/outcomes.csv`, participant-keyed; auto-detects **WIDE** (one row per participant,
`<measure>_pre`/`<measure>_post`, `<measure>_change` computed) or **LONG** (one row per
`participant_id, timepoint`, joined to the per-session VAAMR series). `participant_id` must
match QRA's anonymized IDs; measure columns numeric; missingness explicit, never zero-filled.

**Importer** (`qra outcomes import --redcap-csv <export> --crosswalk <ids>` — build in §1.4):
1. `record_id` → `participant_id` via the speaker-anonymization key; the REDCap↔QRA crosswalk
   stays **off-repo** (PHI).
2. `redcap_event_name` → timepoint (`baseline_arm_1`→`pre`, `week_8_arm_1`→`post`; or session
   numbers for LONG).
3. Select + rename instrument columns to stable measure names; pivot.

**Target instruments:** weekly pain NRS; TSK-11 (the trial's primary behavioral target); ODI;
MRPS (closest to VAAMR Reappraisal); MAIA-2; daily pain/movement/practice diary; EMA;
actigraphy; QST where available.

**Sequencing:** importer → `outcomes.csv` → `qra analyze` (the CONVERGENT VALIDITY section of
`progression_summary.txt` and the results brief populate automatically) → only then describe as
convergent-validity evidence, always with the small-n flag.

**Manuscript framing (verbatim, load-bearing):** *External clinical outcomes are joined on the
anonymized participant key to test the convergent validity of the VAAMR language index — whether
participants whose coded-language trajectory advances also improve on independently measured
outcomes. Correlations (Spearman primary, given the ordinal index and small single-arm n) are
reported as validity evidence with explicit low-power flags, **not** as estimates of program
efficacy, which the single-arm feasibility design cannot support.*

## Appendix B — Pre-registered H4 correlation directions (DO NOT EDIT)

Fixed before any outcome data was seen (carried verbatim from the 2025 roadmap §4.2; amendments
require a dated addendum, never an edit):

| VAAMR-side measure | External measure | Pre-registered direction |
|---|---|---|
| progression slope / adaptive-occupancy trend ↑ | pain NRS change | ↓ pain (negative ρ) |
| progression slope ↑ | TSK-11 change | ↓ kinesiophobia (negative ρ) |
| progression slope ↑ | ODI change | ↓ disability (negative ρ) |
| Reappraisal-stage occupancy ↑ | MRPS change | ↑ reappraisal (positive ρ) |
| Attention-Regulation / Metacognition occupancy ↑ | MAIA-2 change | ↑ interoception (positive ρ) |

Spearman ρ primary, Pearson secondary; effect size + CI with low-power flag (n≈5–8/cohort);
temporal-coupling test (within-week VAAMR shifts vs daily pain/movement/practice, mixed-effects
pooling); joint displays per participant.

## Appendix C — Crosswalk + notes

**Old → new:** masterplan §0 truthing board → done (manuscript, 2026-06-07) · §1/§3 Phase A →
§1.2 (closed) · Phase B → §1.3 + §4.1/§4.2 · Phase C → §1.4 + Appendix B · Phase D → §4.3 ·
Phase E → §4.4 · §4 code backlog → §1/§2 items · §5 grant pivot → §3 · §6 queue → §4.6 · §7
decision rules → §6. Old ROADMAP Phase 1 → done/§2 · Phases 2–3 (AutoResearch fine-tune;
heterogeneous GNN build-out) → **superseded by the H5/probe outcome**, surviving only as §4.5
re-adjudication · Phase 4 → §4.3 + Appendices A/B · Phase 5 → §5.1 · Phase 6 → §5.2.

**Environment note (consequential):** the main venv must stay `numpy==1.26.4`
(`transformers==4.42.4` requires numpy<2); the Bayesian stack (bambi/pymc, numpy≥2) lives only
in `.venv_bayes`. The same pin blocks the Qwen3 embedder load (§4.2's infra task).

**References (trimmed):**
- Wexler, R. S., Balsamo, W., et al. (2026). "Noticing the way that I'm noticing pain." *Mindfulness*, 17, 819–833.
- Wexler, R. S., Balsamo, W., et al. (in review). Development and pilot feasibility testing of Move-MORE.
- Low, D. M., et al. (2024). Text psychometrics. *Psychological Methods*.
- Lindahl, J. R., et al. (2017). The varieties of contemplative experience. *PLOS ONE*, 12(5).
- Schmidt, F., et al. (2025). CFiCS: graph-based classification of common factors. *arXiv:2503.22277*.
- Chatzichristos, G. (2025). Qualitative research in the era of AI: A return to positivism or a new paradigm? *IJQM*, 24.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: the E-value. *Ann Intern Med*, 167(4).
- Garland, E. L. (2024). *Mindfulness-Oriented Recovery Enhancement*. Guilford Press.
