"""
analysis/reports/stat_format.py
-------------------------------
Canonical scientific notation + the [M#] methods registry for every
human-readable report.

Why this exists: the same quantity used to appear as ``Δ=+1.462 [+0.65,+2.21]``
in one report, ``(±0.80)`` in another, and bare in a third. Every report now
formats estimates through this module so a number is written one way,
everywhere, and every headline statistic can cite *how it was computed* via a
shared ``[M#]`` footnote that 08_methods.txt expands in full.

Usage:
    from .stat_format import fmt_est_ci, fmt_kappa, fmt_p, landis_koch, m_ref

    line = f"Δprogression {fmt_est_ci(+1.462, 0.65, 2.21, p=0.042, p_kind='permutation', n_desc='5 blocks / 4 participants')} {m_ref('delta_prog')}"
"""

from collections import OrderedDict


# ── Number formatting primitives ──────────────────────────────────────────

def fmt_signed(x, nd: int = 2) -> str:
    """Signed fixed-point: +1.46 / -0.92 / 0 stays +0.00."""
    try:
        return f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_p(p, nd: int = 3) -> str:
    """APA-style p: p=.042, p<.001; n/a-safe."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "p=n/a"
    if p != p:  # NaN
        return "p=n/a"
    if p < 0.001:
        return "p<.001"
    return ("p=" + f"{p:.{nd}f}").replace("p=0.", "p=.")


def fmt_est_ci(est, lo=None, hi=None, p=None, n_desc: str = None,
               unit: str = "", nd: int = 2, p_kind: str = "") -> str:
    """Canonical estimate string: ``+1.46 [95% CI +0.65, +2.21], permutation p=.042, n=5 blocks / 4 participants``.

    Any missing piece is omitted rather than guessed; a CI is printed only
    when BOTH bounds are present.
    """
    parts = [fmt_signed(est, nd) + (unit or "")]
    if lo is not None and hi is not None:
        try:
            parts[0] += f" [95% CI {fmt_signed(lo, nd)}, {fmt_signed(hi, nd)}]"
        except Exception:
            pass
    if p is not None:
        parts.append((p_kind + " " if p_kind else "") + fmt_p(p))
    if n_desc:
        parts.append(f"n={n_desc}")
    return ", ".join(parts)


# ── Reliability formatting ────────────────────────────────────────────────

def landis_koch(k) -> str:
    """Landis & Koch (1977) verbal band for a κ/α value."""
    try:
        k = float(k)
    except (TypeError, ValueError):
        return "n/a"
    if k != k:
        return "n/a"
    if k < 0.00:
        return "poor"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def fmt_kappa(k, lo=None, hi=None, n=None, stat: str = "κ") -> str:
    """``κ=0.54 [95% CI 0.40, 0.66] (moderate), n=66`` — n/a-safe."""
    try:
        k = float(k)
    except (TypeError, ValueError):
        return f"{stat}=n/a"
    if k != k:
        return f"{stat}=n/a"
    s = f"{stat}={k:+.3f}"
    if lo is not None and hi is not None:
        try:
            s += f" [95% CI {float(lo):+.3f}, {float(hi):+.3f}]"
        except (TypeError, ValueError):
            pass
    s += f" ({landis_koch(k)})"
    if n is not None:
        s += f", n={n}"
    return s


# Pre-registered evidence tiers for reliability (methodology §5.4).
def evidence_tier(alpha) -> str:
    """Map an agreement coefficient onto the pre-registered evidence tiers."""
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return "n/a"
    if a != a:
        return "n/a"
    if a >= 0.60:
        return "PRIMARY-evidence band (α ≥ 0.60)"
    if a >= 0.40:
        return "DIRECTIONAL-evidence band (α 0.40–0.59: use only where ≥2 independent sources converge)"
    return "below the pre-registered floor (α < 0.40: framework revision indicated)"


# ── The [M#] methods registry ─────────────────────────────────────────────
#
# One entry per statistic that appears in a report. 00_RESULTS.txt cites the
# tag inline (``[M4]``); 08_methods.txt prints every entry in full. Keep the
# text here authoritative — reports must not paraphrase their own variant.

METHODS = OrderedDict([
    ('segmentation',
     ("Transcript segmentation (frozen)",
      "Diarized transcripts are segmented semantically (sentence-embedding "
      "similarity with adaptive thresholding + topic clustering, optional LLM "
      "boundary review). Segments are frozen after ingestion; every label is "
      "an overlay keyed to the frozen segment_id, so re-classification never "
      "moves text boundaries.")),
    ('vaamr_labels',
     ("VAAMR stage labels (label of record)",
      "Each participant segment is classified by a zero-shot LLM panel "
      "(multiple runs with model rotation and temperature jitter). Votes are "
      "combined by consensus; the confidence tier integrates cross-run "
      "consistency with per-run confidence (High = unanimous AND confidence "
      ">0.80; Medium = majority AND >0.60; Low = otherwise). The label of "
      "record follows the hierarchy adjudicated > human_consensus > "
      "llm_zero_shot > probe_consensus > gnn_consensus — human labels always "
      "outrank machine labels. VAAMR is the 5-class operationalization of the "
      "published 4-stage VA-MR model (Wexler, Balsamo et al., 2026, "
      "Mindfulness 17(3):819–833), promoting the avoidance barrier to a "
      "distinct class; the developmental arc is unchanged from the source.")),
    ('purer_labels',
     ("PURER therapist-move labels (DIRECTIONAL)",
      "Each cue block (the run of therapist turns between two consecutive "
      "participant turns) receives one PURER move label from a single-run "
      "zero-shot LLM classification with a wide context window. Human "
      "validation of PURER labels is planned (target Krippendorff's α ≥ 0.70) "
      "but has NOT yet begun: every therapist-side statistic in these reports "
      "is directional evidence resting on as-yet-unvalidated cue labels, and "
      "is never used as primary evidence.")),
    ('occupancy_trend',
     ("Adaptive-stage occupancy trend (PRIMARY outcome statistic)",
      "Per session number, the share of participant segments labeled with an "
      "adaptive stage (Attention Regulation / Metacognition / Reappraisal, "
      "stages 2–4). The trend across session numbers is tested with the "
      "rank-based Mann–Kendall test (reported with Kendall's τ and the "
      "Theil–Sen slope). Rank-based, so it makes no equal-spacing assumption "
      "about the ordinal VAAMR scale — this is the primary trend statistic.")),
    ('estage',
     ("E[stage] progression coordinate (SENSITIVITY analysis)",
      "E[stage] is the mixture-weighted mean stage of a segment (0=Vigilance "
      "… 4=Reappraisal), averaged per participant-session. Treating the "
      "ordinal stages as equally spaced is an interval-scale assumption, so "
      "every E[stage] slope is labeled a sensitivity analysis subordinate to "
      "the Mann–Kendall result. Group curves carry participant-cluster "
      "bootstrap 95% CIs; the linear slope comes from a mixed-effects model "
      "with a random participant intercept.")),
    ('cluster_bootstrap',
     ("Participant-cluster bootstrap CIs",
      "All 95% CIs on group means, slopes, and Δprogression aggregates are "
      "computed by resampling PARTICIPANTS with replacement (not segments), "
      "respecting the non-independence of repeated observations within a "
      "person. 2,000+ resamples; percentile intervals.")),
    ('barrier',
     ("Avoidance→Attention-Regulation barrier crossing (descriptive)",
      "Per participant: whether any session after first Avoidance-dominant "
      "expression reaches a stage above Avoidance, and the first-passage "
      "session at which this occurs. The published VA-MR model identifies "
      "this barrier as the central developmental obstacle; crossing rates are "
      "descriptive (single-arm, no counterfactual).")),
    ('delta_prog',
     ("Δprogression per therapist move (DIRECTIONAL mechanism estimate)",
      "For each FROM→CUE→TO triple (participant turn → therapist cue block → "
      "next participant turn), Δprogression = E[stage]_TO − E[stage]_FROM. "
      "Cell means per (FROM-stage × PURER move) carry participant-cluster "
      "bootstrap 95% CIs, within-FROM-stage permutation p-values, and "
      "Benjamini–Hochberg FDR across cells. Temporal adjacency in an "
      "unblinded, uncontrolled setting — association, not causation; PURER "
      "labels are themselves unvalidated [see purer_labels].")),
    ('evalues',
     ("E-value confound sensitivity (VanderWeele & Ding 2017)",
      "For the strongest mechanism cells: the minimum strength (risk-ratio "
      "scale) an unmeasured confounder would need, with both treatment and "
      "outcome, to explain away the association. Point and CI-limit E-values "
      "are reported; on the current sample CI-limit E-values collapse toward "
      "1.0, so the pilot can bound but not establish robustness.")),
    ('interaction_model',
     ("Stage-moderated therapist-effect model (earns-its-place test)",
      "A hierarchical cumulative-logit model of TO-stage with FROM-stage × "
      "move interaction and partial pooling, plus a grouped cross-validation "
      "'earns-its-place' comparison against a FROM-only baseline. On the "
      "current sample the interaction does not improve held-out prediction — "
      "reported as an honest null, not hidden.")),
    ('sign_test',
     ("Per-participant slope direction (exact sign test)",
      "Each multi-session participant's E[stage] trajectory is summarized by "
      "an OLS slope; the count of advancing vs non-advancing participants is "
      "tested with an exact binomial sign test against 0.5.")),
    ('irr',
     ("Inter-rater reliability statistics",
      "Cohen's κ via scikit-learn (pairwise), Fleiss' κ via statsmodels "
      "(complete-case multi-rater), Krippendorff's α via the krippendorff "
      "package (headline multi-rater statistic; tolerates missing ballots). "
      "Verbal bands follow Landis & Koch (1977). Pre-registered evidence "
      "tiers: α ≥ 0.60 primary evidence; 0.40–0.59 directional (requires "
      "convergence with ≥2 independent sources); < 0.40 framework revision. "
      "The human↔human band is the ceiling for any machine rater: no model "
      "can agree with a human consensus more reliably than humans agree among "
      "themselves.")),
    ('grounding',
     ("LLM justification-grounding audit",
      "For every LLM classification justification, the fraction of quoted "
      "spans that appear verbatim in the segment being labeled. Bounds "
      "confabulation (the classifier citing text that is not there); it does "
      "NOT certify label correctness. Full per-item dossier in "
      "09_supplementary/justification_grounding.txt.")),
    ('superposition',
     ("Soft stage mixtures (superposition)",
      "Per-segment stage mixtures derived from the multi-run LLM ballots "
      "(falling back to secondary-stage votes), giving each segment a "
      "probability blend over the five stages. Mixture entropy operationalizes "
      "liminality (between-stage moments); hard labels are never overwritten.")),
    ('lift',
     ("PURER × transition lift",
      "P(move | transition type) / P(move) over cue blocks — how over- or "
      "under-represented each therapist move is at forward, lateral, and "
      "backward participant transitions, relative to its base rate.")),
])


def m_ref(key: str) -> str:
    """Inline footnote tag for a methods entry: m_ref('delta_prog') → '[M8]'."""
    try:
        idx = list(METHODS.keys()).index(key)
    except ValueError:
        return ""
    return f"[M{idx + 1}]"


def methods_entries():
    """Ordered [(tag, key, title, text)] for 08_methods.txt."""
    out = []
    for i, (key, (title, text)) in enumerate(METHODS.items(), start=1):
        out.append((f"M{i}", key, title, text))
    return out


def provenance_header(m_keys, extra: str = "") -> list:
    """Standard 'HOW THESE NUMBERS WERE COMPUTED' block for a report header.

    Returns a list of lines citing the [M#] entries (expanded in
    06_reports/08_methods.txt) so each report carries its provenance without
    duplicating the methods prose.
    """
    tags = " ".join(m_ref(k) for k in m_keys if m_ref(k))
    lines = [
        "HOW THESE NUMBERS WERE COMPUTED: " + tags,
        "  (each [M#] is expanded in 06_reports/08_methods.txt)",
    ]
    if extra:
        lines.append("  " + extra)
    return lines
