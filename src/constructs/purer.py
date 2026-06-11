"""
purer.py
--------
PURER therapist-dialogue classification framework for MORE sessions.

PURER operationalizes five guided-inquiry moves drawn from the 2018 MORE
Manual (Garland) and validated against the qualitative dataset analyzed
by Wexler, Balsamo et al. (2026, Mindfulness, 17, 819–833):

    P  — Phenomenological  (eliciting moment-to-moment experiential
                              breakdown of the participant's practice)
    U  — Utilization       (prompting the participant to identify how
                              the experience can be applied to coping in
                              everyday life)
    R  — Reframing         (transforming the participant's organic report
                              into a direct example of a MORE concept)
    E  — Educate/Expectancy (delivering psychoeducation about pain,
                              stress, or mindfulness mechanisms while
                              imparting an explicit expectation of
                              therapeutic benefit)
    R2 — Reinforcement     (selectively encouraging adaptive responses,
                              insights, or active coping while gently
                              ignoring or pivoting from maladaptive
                              narratives)

These labels apply EXCLUSIVELY to therapist segments. They do NOT apply to
participant utterances; the participant-facing framework is VAAMR (see
constructs/vaamr.py).

================================================================================
CO-OCCURRENCE AND PRECEDENCE
================================================================================
Unlike the VAAMR participant-stage framework, PURER moves frequently
co-occur within a single therapist turn. A debrief such as "That's still
you doing it. That's really positive. That's what the practice is — the
mind drifts and you bring it back" is simultaneously Reinforcement and
Reframing. Most therapist turns of meaningful length contain at least
two PURER moves.

When the analytic unit permits only one label, the following empirical
precedence has been useful:

    1. Reinforcement (R2) is often the WRAPPER around another move.
       When R2 is the outer affective register but a substantive move
       is being executed inside it (P / U / R / E), code the substantive
       move. Code R2 only when affirmation is the entire act.

    2. Utilization (U) takes precedence over Reframing (R) when the
       therapist is asking the participant to apply an insight forward
       in time. "How will you use this tomorrow?" is U even when
       embedded in a reframe of the participant's story.

    3. Reframing (R) takes precedence over Education (E) when the
       therapist is anchoring an explanation to the participant's
       specific story. R = concept landed on this participant's
       experience; E = concept delivered as a general mechanism.

    4. Phenomenological (P) is generally a clean, single-move category
       — questions eliciting a step-by-step experiential breakdown.
       When such a question appears alongside another move (e.g.,
       reinforcement before, reframe after), P is the substantive
       inquiry move.

When a single label must be forced from co-occurring moves whose ballots
are otherwise tied, ``PURER_TIE_BREAK_ORDER`` (below) encodes this
precedence as an explicit highest-precedence-first ordering of move ids.

================================================================================
EXEMPLAR SOURCING
================================================================================
Exemplar utterances are drawn from MORE-for-LRP session recordings
analyzed during framework development. Quotes ending with parenthetical
notes (e.g., "(also reinforces)") are included to demonstrate co-
occurrence; the primary code is the one in the section header.

Structural parallel to constructs/vaamr.py. Exports
get_purer_framework() which returns a ThemeFramework with five
ThemeDefinition objects.
"""

import functools
from pathlib import Path
from .theme_schema import ThemeDefinition, ThemeFramework  # noqa: F401
from .markdown_loader import load_framework_md

_PURER_MD = Path(__file__).resolve().parents[2] / "frameworks" / "PURER_FRAMEWORK.md"

PURER_FRAMEWORK_VERSION = "3.0"

# Highest-precedence-first ordering of PURER move ids, used as the final
# deterministic tie-break when ``coded_plurality`` voting (majority_vote.py)
# must force one label from co-occurring moves whose plurality count and mean
# confidence are both tied. It encodes the precedence documented in the
# CO-OCCURRENCE AND PRECEDENCE section above: Phenomenological first (the
# framework's core elicitation move and a clean single-move category), then the
# documented pairwise orders Utilization > Reframing > Education, with
# Reinforcement (the affective wrapper) last. Move ids: P=0, U=1, R=2, E=3, R2=4.
PURER_TIE_BREAK_ORDER: list[int] = [0, 1, 2, 3, 4]



def get_purer_framework() -> ThemeFramework:
    """Return the PURER therapist-dialogue classification framework (parsed from PURER_FRAMEWORK.md)."""
    return load_framework_md(_PURER_MD)
