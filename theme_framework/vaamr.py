"""
vaamr.py
--------
VAAMR theme framework:
  Vigilance | Avoidance | Attention Regulation | Metacognition | Reappraisal

Conceptual stage ordering:
  Stage 0   — Vigilance           (pre-skill; attentional capture by pain)
  Stage 1 — Avoidance           (maladaptive attentional deployment)
  Stage 2   — Attention Regulation (foundational adaptive skill; Stage 1
                                    of the three mindfulness-based stages)
  Stage 3   — Metacognition        (reflexive observing; Stage 2 of
                                    mindfulness-based stages)
  Stage 4   — Reappraisal          (transformative reinterpretation; Stage 3
                                    of mindfulness-based stages)

Note on stage numbering: Vigilance and Avoidance are pre-mindfulness stages
representing dysfunctional relationships to attention. Attention Regulation,
Metacognition, and Reappraisal are all stages of mindfulness, representing
progressively sophisticated deployments of present-moment awareness. Attention
Regulation is the foundational skill on which Metacognition and Reappraisal
depend. theme_id values are integers (0-4) for schema compatibility; the
conceptual Stage 0.5 of Avoidance is reflected in the definition and
description rather than the integer id.
  

Worked examples of navigating theme overlap between Metacognition and Reappraisal:

  "Noticing the way I'm noticing pain"
      → METACOGNITION
        (pure reflexive observation; no transformation described)

  "Noticing the way I'm noticing pain, my relationship to it is changing"
      → REAPPRAISAL
        (the metacognitive observation is the vehicle through which a
         transformed relationship is being reported; the experience itself
         has shifted, which is the diagnostic feature of Reappraisal)

  "I noticed I can relax around it. I get to notice it's shifting"
      → REAPPRAISAL
        ("shifting" describes a transformation in the sensory experience;
         the metacognitive frame ("I noticed") delivers the report)

  "I focused on the breath and stayed there"
      → ATTENTION REGULATION
        (sustained presence with anchor, no observation of mental processes
         and no described transformation)

  "I focused on the breath and watched my mind want to fight the pain"
      → METACOGNITION
        (the centerpiece is the observed reaction — mind wanting to fight;
         attention regulation is the platform but not the dominant report)

  "I tried to focus on the breath but the pain pulled me right back"
      → VIGILANCE
        (capture dominates the report; attention attempt fails)

  "I focused on the breath instead of the pain"
      → AVOIDANCE
        (escape framing — "instead of" — makes the breath a vehicle for
         redirection away from sensation; this is maladaptive attentional
         deployment, not adaptive sustained presence)

  "It doesn't bother me anymore"
      → AVOIDANCE if no described insight or transformed perception
        REAPPRAISAL only when grounded in described understanding,
        sensory decomposition, or shifted meaning

The presence of *described insight or transformation* is the load-bearing
discriminator between Avoidance and Reappraisal. The presence of a *stable
observing perspective on mental processes* is the load-bearing discriminator
between Attention Regulation and Metacognition.

Stage definitions in brief:

  Vigilance            — Hypervigilant capture by pain combined with absence
                         of successful attentional control. Includes
                         catastrophic cognitions about the meaning, trajectory,
                         or consequences of pain.

  Avoidance            — Maladaptive deployment of emerging attentional skill
                         in the service of escape from internal experience.
                         Includes escapist avoidance (suppression, distraction,
                         flying away) and kinesiophobic avoidance (fear of
                         movement). Diagnostically: unstable attention that
                         cannot remain with discomfort.

  Attention Regulation — The development of the capacity to direct attention
                         as an adaptive skill: stable, sustained, volitional
                         attention that can stay WITH present experience
                         including pain or discomfort, without redirecting
                         away. The foundational mindfulness skill underlying
                         the more sophisticated stages that follow.

  Metacognition        — The reflexive capacity to observe one's own mental
                         processes as they occur. Watching reactions, impulses,
                         and patterns arise; decentering from mental content;
                         recognizing the distinction between pain and reactions
                         to pain. Observation without yet describing
                         transformation of the experience itself.

  Reappraisal          — A transformation of the meaning, structure, or felt
                         quality of pain experience that emerges from insight
                         rather than from suppression. Encompasses sensory
                         reappraisal (decomposition into constituent sensations,
                         jagged-to-dull transformations) and cognitive
                         reappraisal (redefinition of pain's meaning, identity
                         shifts, transformed worldview). Includes equanimity
                         and acceptance grounded in understanding.

Exemplar utterances are drawn from the MORE-for-LRP clinical trial corpus
(Wexler, Balsamo et al., 2026, Mindfulness, 17, 819-833) and from session
recordings of the MORE qualitative study analyzed during framework development.
"""





import functools
from pathlib import Path

from .theme_schema import ThemeFramework, ThemeDefinition  # noqa: F401 — re-exported
from .markdown_loader import load_framework_md

VAAMR_FRAMEWORK_VERSION = "4.0"

_VAAMR_MD = Path(__file__).resolve().parents[1] / 'VAAMR_FRAMEWORK.md'


@functools.lru_cache(maxsize=None)
def get_vaamr_framework() -> ThemeFramework:
    """Return the five-stage VAAMR theme framework (parsed from VAAMR_FRAMEWORK.md)."""
    return load_framework_md(_VAAMR_MD)
