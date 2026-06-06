"""
analysis/reports/reports_guide.py
---------------------------------
Two navigational artifacts written last, after every other report exists:

  • 00_READ_ME.txt        — a guide to the 06_reports/ tree: what each report is,
                            in recommended reading order, plus a live file listing.
  • 07_methods_appendix.txt — how each metric/report is computed and its caveats,
                            sourced from methodology.md (§4.5, §5, §6.3, §8.5) so the
                            outputs and the paper stay consistent.

These exist so a reader (clinician, patient partner, reviewer) can open 06_reports/
cold and know exactly what they are looking at and how far to trust it.
"""

import os
from datetime import date
from typing import Optional

from process import output_paths as _paths


# (subdir or '', filename-or-pattern, one-line description) in reading order.
_REPORT_GUIDE = [
    ('', '00_executive_summary.txt',
     "START HERE — deterministic program-improvement brief (outcomes, mechanism, recommendations, caveats)."),
    ('01_outcomes', 'progression_summary.txt',
     "DESCRIPTIVE single-arm summary (NOT efficacy): adaptive-stage occupancy + ordinal (Mann-Kendall) trend as headline; interval E[stage] slope as caveated sensitivity; per-participant direction; barrier; convergent-validity vs external outcomes (when integrated)."),
    ('01_outcomes', 'longitudinal.txt',
     "Group + per-participant trajectories over sessions; advancing/stable/regressing counts; between-session regression patterns AND advances."),
    ('01_outcomes', 'avoidance_barrier.txt',
     "The central Avoidance→Attention-Regulation barrier: therapist language that helps cross it AND that stalls/pulls back, plus cusp-density trend."),
    ('02_mechanism', 'transitions.txt',
     "Within- and between-session VAAMR transition matrices with exemplars (forward, lateral, backward)."),
    ('02_mechanism', 'cue_response.txt',
     "FROM→CUE→TO synthesis: what therapists do at each transition type (PURER move distribution)."),
    ('02_mechanism', 'purer.txt',
     "PURER × VAAMR influence: lift of each therapist move per stage; transition profiles; empty-cue rates; (GNN motif cross-reference appended when available)."),
    ('02_mechanism', 'mechanism.txt',
     "Signed Δprogression per therapist move WITH inference (cluster-bootstrap CIs, permutation p, FDR); liminality leverage; trajectory typology; mixed-effects model."),
    ('02_mechanism', 'language_atlas.txt',
     "Readable FROM→CUE→TO exemplars for the top forward AND backward/stalling moves; emergent motifs; named coupling factors (the last two need the GNN layer)."),
    ('02_mechanism', 'superposition.txt',
     "Soft stage mixtures: liminal (mixed-stage) segments, cusp density, soft→hard divergence."),
    ('03_per_session', '_overview.txt',
     "Per-session theme-distribution overview across the program (was stage_expressions)."),
    ('03_per_session', 'session_<id>.txt',
     "One file per session: participants, stage distribution, exemplars, transitions, optional therapist summary."),
    ('03_per_session', 'session_summaries.txt',
     "LLM-written therapist-language summary per session (only when summaries are enabled)."),
    ('04_per_participant', 'participant_<id>.txt',
     "One file per participant: stage sequence across sessions, progression trend + interpretation, top exemplars."),
    ('05_per_stage', 'stage_<name>.txt',
     "One file per VAAMR stage: definition, prevalence, longitudinal trend, top exemplars, co-occurring codes."),
    ('05_per_stage', 'codebook_exemplars.txt',
     "Per VCE codebook code: definition, prevalence, top exemplars (when codebook classification ran)."),
    ('06_gnn', 'validation.txt',
     "GNN reliability gate: out-of-sample Cohen's κ vs LLM consensus, per VAAMR stage and PURER move (rare-stage-collapse safeguard)."),
    ('06_gnn', 'triangulation.txt',
     "GNN ↔ LLM ↔ human agreement on VAAMR/PURER (independent-substrate construct validity)."),
    ('06_gnn', 'emergent_motifs.txt',
     "High-influence therapist-language motifs poorly explained by PURER — candidate NEW constructs for human review."),
    ('06_gnn', 'coupling.txt',
     "Latent factors of therapist language and their correlation with subsequent participant forward movement (alliance-like structure, discovered not imposed)."),
    ('06_gnn', 'construct_signal.txt',
     "Ablation: how much independent signal each construct head (VCE/PURER) carries (opt-in)."),
    ('', '07_methods_appendix.txt',
     "How each report/metric is computed, with methodological caveats."),
]


def generate_reports_readme(output_dir) -> Optional[str]:
    """Write 06_reports/00_READ_ME.txt — the guide + a live listing of the tree."""
    root = _paths.human_reports_dir(output_dir)
    os.makedirs(root, exist_ok=True)

    L = []
    L.append("=" * 78)
    L.append("QRA ANALYSIS REPORTS — READ ME")
    L.append("=" * 78)
    L.append(f"Generated: {date.today().isoformat()}")
    L.append("")
    L.append("Reports are tiered by purpose. Recommended reading order, top to bottom:")
    L.append("  00_  start here (executive summary)")
    L.append("  01_outcomes/     how did coded language progress? (descriptive, single-arm)")
    L.append("  02_mechanism/    how does it work — what therapist language moves people?")
    L.append("  03_per_session/  04_per_participant/  05_per_stage/   drill-down detail")
    L.append("  06_gnn/          independent graph-model discovery + validation")
    L.append("  07_  methods appendix (how every number is computed)")
    L.append("")

    current_dir = None
    for subdir, name, desc in _REPORT_GUIDE:
        if subdir != current_dir:
            current_dir = subdir
            label = subdir + '/' if subdir else '(top level)'
            L.append("-" * 78)
            L.append(label)
            L.append("-" * 78)
        # Mark whether the artifact actually exists this run (patterns left as-is).
        flag = ''
        if '<' not in name:
            exists = os.path.isfile(os.path.join(root, subdir, name) if subdir else os.path.join(root, name))
            flag = '   ' if exists else ' (not produced this run)'
        L.append(f"  {name:<26} {desc}{flag}")
    L.append("")

    L.append("Every report is DIRECTIONAL/ASSOCIATIONAL unless human-validated; see each")
    L.append("report's header and 07_methods_appendix.txt for the specific caveats.")

    path = _paths.reports_readme_path(output_dir)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def generate_methods_appendix(output_dir) -> Optional[str]:
    """Write 06_reports/07_methods_appendix.txt — how each metric is computed."""
    L = []
    L.append("=" * 78)
    L.append("METHODS APPENDIX — HOW EACH METRIC IS COMPUTED")
    L.append("=" * 78)
    L.append("")
    L.append("Condensed from methodology.md. Read alongside the reports it describes.")
    L.append("")

    sections = [
        ("VAAMR stage classification (per participant segment)",
         ["Zero-shot multi-run LLM consensus against the 5-stage VAAMR framework, with",
          "majority vote and confidence tiering (high/medium/low). Production runs use",
          "single-model stochastic mode → this is a STABILITY estimate, not independence-",
          "based reliability. Human validation carries the burden of correctness. (§4.3, §5.3)"]),
        ("PURER move classification (per therapist cue block)",
         ["One PURER label per therapist response between consecutive participant turns,",
          "with a documented precedence order for co-occurring moves. (§2.3, §4.8)"]),
        ("Progression coordinate & stage mixture (superposition)",
         ["Each segment carries a soft stage mixture (probabilities over the 5 stages) and a",
          "continuous progression coordinate = E[stage] in 0–4. Entropy of the mixture marks",
          "liminal/mixed-stage segments. Mixture source is the LLM ballots, or the GNN geometry",
          "when the GNN layer is enabled. (analysis/superposition.py, §8.5 Capability A)"]),
        ("Δprogression (mechanism)",
         ["For each FROM→CUE→TO triple, Δprogression = progression_coord(TO) − progression_coord(FROM).",
          "Positive = forward movement, negative = backward. Aggregated per (from_stage, therapist",
          "behavior) with participant-cluster bootstrap 95% CIs, a within-from-stage permutation p,",
          "and Benjamini–Hochberg FDR. A random-participant mixed-effects model is also fit.",
          "Reported in BOTH directions. Associational, not causal. (analysis/mechanism.py, §7.2, §9.2)"]),
        ("Program progression summary (DESCRIPTIVE — not efficacy)",
         ["Single-arm DESCRIPTIVE summary of how LLM-coded VAAMR language moves over sessions. NOT an",
          "efficacy estimate: no control arm, the 'outcome' is the coded language itself (also shaped by",
          "therapist prompting, §9.4). PRIMARY = adaptive-stage (2–4) occupancy by session + a rank-based",
          "Mann–Kendall monotonic trend (ordinal-safe — no equal-spacing assumption). SECONDARY/sensitivity",
          "= E[stage] progression coordinate + linear mixed-effects slope (treats VAAMR as interval; flagged).",
          "Per-participant slope direction + sign test. 'Barrier crossing' = first session Avoidance is no",
          "longer the dominant CODED stage (language-internal). p-values/CIs shown but flagged when",
          "underpowered (power_flag). External outcomes give CONVERGENT VALIDITY only (exploratory), not",
          "efficacy — see docs/OUTCOME_INTEGRATION_ROADMAP.md. (analysis/efficacy.py, §6.3, §8.3, §9)"]),
        ("Cross-framework lift (VAAMR × VCE)",
         ["Lift(stage, code) = P(code | stage) / P(code), filtered at lift ≥ 1.5 and count ≥ 3.",
          "Currently EXPLORATORY: it is not yet compared against pre-specified expected_codes, and a",
          "shuffled-stage permutation control is committed but pending. (process/cross_validation.py, §4.5, §5.2)"]),
        ("GNN layer (representation + discovery + distillation)",
         ["A GraphSAGE model over Qwen3 embeddings + temporal/similarity edges. It (a) yields continuous",
          "superposition, (b) discovers therapist-language MOTIFS scored for forward influence and PURER",
          "purity (low purity ⇒ candidate new construct), (c) triangulates its VAAMR/PURER reads against",
          "LLM and human labels, and (e) extracts latent coupling factors. Its consensus-distillation",
          "classifier may become the label of record ONLY after passing a per-stage out-of-sample κ gate",
          "(no rare-stage collapse) AND explicit analyst promotion. GNN↔LLM agreement is high by",
          "construction; the load-bearing evidence is GNN↔human and the held-out gate. (gnn_layer/, §8.5)"]),
        ("Validation status (what you can trust)",
         ["Linguistic expression ≠ phenomenological state (§9.1). Temporal adjacency ≠ causation (§9.2).",
          "Frameworks derived elsewhere may need contextual adaptation (§9.3). Computational output is",
          "hypotheses for expert review, not a substitute for phenomenological judgment (§9.5). Human",
          "blind-coding targets: VAAMR Krippendorff's α ≥ 0.60, PURER ≥ 0.70 (§5.4, §8.1)."]),
    ]
    for title, body in sections:
        L.append("-" * 78)
        L.append(title)
        L.append("-" * 78)
        for line in body:
            L.append("  " + line)
        L.append("")

    path = _paths.methods_appendix_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path
