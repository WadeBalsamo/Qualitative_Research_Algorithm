"""
analysis/reports/methods_report.py
----------------------------------
08_methods.txt — "how every number was computed."

Every statistic in 00_RESULTS.txt and the tier reports carries an [M#] tag.
This file resolves every tag in full from the canonical METHODS registry in
stat_format.py (the single authoritative source — reports never paraphrase
their own variant), then prints a VALIDATION-STATUS table and pointers.

This replaces reports_guide.py:generate_methods_appendix. The per-metric
caveat prose that the registry does not already capture (the segmentation-
sensitivity opt-in check, the cross-framework lift, and the GNN layer) is
carried forward below as supplementary notes; the registry prose is NOT
duplicated.

Contract (called from analysis/runner.py step 13):
    generate_methods_report(output_dir) -> path | ''
"""

import os
from datetime import date
from typing import Optional

from process import output_paths as _paths
from .stat_format import methods_entries

WRAP = 78


def _wrap(L, text, indent="  "):
    words = text.split()
    if not words:
        L.append("")
        return
    line = indent
    for w in words:
        if len(line) + len(w) + (1 if line.strip() else 0) > WRAP:
            L.append(line.rstrip())
            line = indent + w
        else:
            line = (line + " " + w) if line.strip() else (indent + w)
    if line.strip():
        L.append(line.rstrip())


def _rule(L, char="="):
    L.append(char * WRAP)


def generate_methods_report(output_dir) -> Optional[str]:
    """Write 06_reports/08_methods.txt. Returns the path, or '' on failure."""
    try:
        L = []
        _rule(L, "=")
        L.append("METHODS — HOW EVERY REPORTED NUMBER WAS COMPUTED")
        _rule(L, "=")
        L.append(f"Generated: {date.today().isoformat()}")
        L.append("")
        _wrap(L,
              "Every statistic in 00_RESULTS.txt and the tier reports carries an "
              "[M#] provenance tag. This file resolves each tag in full. The "
              "entries are authoritative: a report cites the tag rather than "
              "restating the method, so the number and its definition can never "
              "drift apart.")
        L.append("")

        # ── The [M#] registry, expanded ────────────────────────────────────
        for tag, key, title, text in methods_entries():
            _rule(L, "-")
            L.append(f"[{tag}]  {title}")
            _rule(L, "-")
            _wrap(L, text)
            L.append("")

        # ── Validation-status table ────────────────────────────────────────
        _rule(L, "=")
        L.append("VALIDATION STATUS — WHAT YOU CAN TRUST, AND HOW FAR")
        _rule(L, "=")
        L.append("")
        rows = [
            ("VAAMR stage labels",
             "HUMAN-LEVEL",
             "LLM consensus agrees with the human consensus at κ≈0.54, at/above "
             "the human↔human band (α 0.33–0.52); 20% human blind-coding "
             "ongoing. Acceptable as label of record. [M2][M12]"),
            ("Participant outcomes",
             "VALIDATED (single-arm)",
             "Adaptive-occupancy Mann-Kendall trend + E[stage] sensitivity + "
             "sign test + barrier crossing rest on the VAAMR labels above. "
             "Descriptive, not efficacy — no control arm. [M4][M5][M11][M7]"),
            ("PURER move labels",
             "DIRECTIONAL",
             "Human validation PLANNED (target α≥0.70), NOT started. Every "
             "therapist-side number is directional. [M3]"),
            ("Mechanism (Δprogression)",
             "DIRECTIONAL + under-identified",
             "Temporal-adjacency association on unvalidated PURER labels; "
             "stage×move interaction does not earn its place out-of-sample; "
             "CI-limit E-values collapse toward 1. [M8][M9][M10]"),
            ("External clinical outcomes (H4)",
             "PENDING",
             "pain NRS / TSK-11 / ODI / MRPS / MAIA-2 not yet linked (REDCap "
             "import pending). No real-world clinical claim can be made."),
            ("GNN consensus classifier",
             "REFUTED AT PILOT n",
             "Grouped-CV κ≈0.05–0.14 < human band at n≈32; a probe ties/beats "
             "it. DEFAULT OFF; re-adjudicable at Cohorts 3–4."),
        ]
        for name, status, note in rows:
            L.append(f"  {name}")
            L.append(f"    status: {status}")
            tmp = []
            _wrap(tmp, note, indent="    ")
            L.extend(tmp)
            L.append("")

        # ── Supplementary per-metric notes not in the registry ─────────────
        _rule(L, "=")
        L.append("SUPPLEMENTARY METRIC NOTES (not in the [M#] registry)")
        _rule(L, "=")
        L.append("")
        extras = [
            ("Segmentation-sensitivity check (opt-in)",
             "Robustness of the headline progression slope to the segmentation "
             "parameters. Over a one-factor-at-a-time grid each arm RE-SEGMENTS "
             "the raw transcripts with LLM refinement forced off and PROJECTS "
             "the existing per-segment progression coordinate + hard labels onto "
             "the new units (token-overlap-weighted; NO re-classification), then "
             "recomputes the slope and barrier-crossing rate against a canonical-"
             "in-same-embedder baseline. The verdict is SCOPE-DISCLOSED "
             "(boundary-placement only); because values are held fixed and only "
             "re-grouped, aggregate stability is PARTLY STRUCTURAL. A re-"
             "classifying arm is future work. (analysis/segmentation_sensitivity.py)"),
            ("Cross-framework lift (VAAMR × VCE codebook)",
             "Lift(stage, code) = P(code | stage) / P(code), filtered at lift "
             "≥ 1.5 and count ≥ 3. EXPLORATORY descriptive co-occurrence only — "
             "VCE is an optional enrichment layer and no construct-validity claim "
             "rests on it (H3 deferred). (process/cross_validation.py, §4.5)"),
            ("GNN discovery + mechanism layer (07_gnn/)",
             "Runs on RAW Qwen3 embeddings, no trained classifier: H6 "
             "discriminant validity (supervised probe vs content-similarity on "
             "the same embeddings), the dyadic FROM→CUE→TO transition model + "
             "confound localization (the mechanism rebuild that replaced the mis-"
             "specified classifier-counterfactual), subtext communities + dyadic "
             "routines, cue motifs, and coupling factors. All hypothesis-"
             "generating, never causal. The GraphSAGE consensus-distillation "
             "CLASSIFIER is a separate concern, DEFAULT OFF. (gnn_layer/, §8.5)"),
        ]
        for title, body in extras:
            _rule(L, "-")
            L.append(title)
            _rule(L, "-")
            _wrap(L, body)
            L.append("")

        # ── Pointers ───────────────────────────────────────────────────────
        _rule(L, "=")
        L.append("POINTERS")
        _rule(L, "=")
        L.append("")
        L.append("  Full inter-rater-reliability dossier")
        L.append("      01_reliability/irr_report.txt")
        L.append("  Justification-grounding audit")
        L.append("      09_supplementary/justification_grounding.txt")
        L.append("  Mechanism CIs / permutation / FDR")
        L.append("      03_mechanism/mechanism.txt")
        L.append("  FROM→CUE→TO exemplars")
        L.append("      03_mechanism/language_atlas.txt")
        L.append("  Full neurophenomenological methodology ... docs/methodology.md")
        L.append("    (§5 reliability & evidence tiers, §6.3 program decisions,")
        L.append("     §8 capabilities, §9 limitations incl. §9.4 elicitation")
        L.append("     confound)")

        path = _paths.reports_methods_path(output_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(L) + "\n")
        return path
    except Exception:
        return ''
