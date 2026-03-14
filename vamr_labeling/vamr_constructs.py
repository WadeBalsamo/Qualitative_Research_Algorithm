"""
vamr_constructs.py
------------------
VA-MR construct definitions for the zero-shot labeling pipeline.

Adapted from srl_constructs.py in the Text Psychometrics repo, which defines:
  - constructs_in_order: ordered list of construct names
  - prompt_names: mapping from internal names to LLM-facing descriptions
  - word_prototypes: single-token exemplars for each construct
  - ctl_tags13_to_srl_name_mapping: cross-reference between naming systems
  - categories: grouping of constructs into higher-order domains
  - colors: visualization color mapping

We replicate this exact pattern for the VA-MR framework, adding the
phenomenology codebook integration from the legacy implementation.
"""

# ---------------------------------------------------------------------------
# Stage identifiers (integer encoding for labels)
# ---------------------------------------------------------------------------
STAGE_IDS = {
    'vigilance': 0,
    'avoidance': 1,
    'metacognition': 2,
    'reappraisal': 3,
}

NUM_STAGES = 4

# Ordered list of stage names (mirrors constructs_in_order)
stages_in_order = [
    'Pain Vigilance and Attention Dysregulation',
    'Attention Regulation applied to Experiential Avoidance',
    'Metacognitive Awareness',
    'Pain Reappraisal',
]

# Short names for display and file naming
stage_short_names = {
    'vigilance': 'Vigilance',
    'avoidance': 'Avoidance',
    'metacognition': 'Metacognition',
    'reappraisal': 'Reappraisal',
}

# Prompt-facing names (mirrors prompt_names in srl_constructs.py)
prompt_names = {
    'vigilance': 'pain vigilance and attention dysregulation',
    'avoidance': 'attention regulation applied to experiential avoidance',
    'metacognition': 'metacognitive awareness',
    'reappraisal': 'pain reappraisal',
}

# ---------------------------------------------------------------------------
# Full operational definitions for LLM prompting
# ---------------------------------------------------------------------------
stage_definitions = {
    'vigilance': {
        'stage_id': 0,
        'name': 'Pain Vigilance and Attention Dysregulation',
        'short_name': 'Vigilance',
        'prompt_name': 'pain vigilance and attention dysregulation',
        'definition': (
            'Expressions of hypervigilance to pain signals combined with '
            'inability to direct or sustain attentional focus. The participant '
            'describes being captured by pain, unable to disengage, and '
            'experiencing attentional fragmentation. Attention is reactive '
            'rather than directed. May include catastrophic cognitions about '
            'pain and hypervigilant monitoring of body signals.'
        ),
        'prototypical_features': [
            'captured by pain',
            'overwhelmed by sensation',
            'cannot focus',
            'attention keeps going back to pain',
            'fragmented awareness',
            'catastrophic thinking about pain',
            'hypervigilant body monitoring',
            'pain dominates awareness',
        ],
        'distinguishing_from_next': (
            'Absence of intentional attentional deployment. '
            'Attention is reactive rather than directed.'
        ),
        'exemplar_utterances': [
            "I can't stop thinking about the pain, it's all I can focus on.",
            "Every little twinge sends me into a spiral.",
            "My mind just keeps going back to it no matter what I try.",
            "I feel like the pain is controlling everything.",
        ],
    },
    'avoidance': {
        'stage_id': 1,
        'name': 'Attention Regulation applied to Experiential Avoidance',
        'short_name': 'Avoidance',
        'prompt_name': 'attention regulation applied to experiential avoidance',
        'definition': (
            'The participant has developed attentional control but deploys it '
            'in the service of avoidance rather than investigation. Descriptions '
            'of deliberately pushing pain away, distracting themselves, or using '
            'newly developed concentration to suppress rather than examine '
            'painful experience. Attention is being directed away from rather '
            'than toward the present experience.'
        ),
        'prototypical_features': [
            'deliberately redirecting attention away from pain',
            'pushing pain away',
            'using breath to escape pain',
            'distraction as strategy',
            'suppressing painful experience',
            'concentration deployed for avoidance',
            'relief-seeking through distraction',
        ],
        'distinguishing_from_next': (
            'Attention is directed away from rather than toward present experience. '
            'The goal is to escape or suppress, not to observe.'
        ),
        'exemplar_utterances': [
            "When the pain comes, I focus really hard on my breathing to push it away.",
            "I've gotten better at not thinking about it.",
            "I use the meditation to take my mind somewhere else.",
            "I concentrate on something pleasant to block out the sensation.",
        ],
    },
    'metacognition': {
        'stage_id': 2,
        'name': 'Metacognitive Awareness',
        'short_name': 'Metacognition',
        'prompt_name': 'metacognitive awareness',
        'definition': (
            'The capacity to observe one\'s own mental processes as they occur. '
            'The participant describes noticing their reactions to pain, watching '
            'thoughts arise and pass, and developing a perspective from which '
            'mental activity can be observed rather than identified with. There '
            'is an observing perspective present without yet describing a '
            'transformed understanding of the experience itself.'
        ),
        'prototypical_features': [
            'noticing reactions to pain',
            'watching thoughts arise and pass',
            'stepping back from experience',
            'observing rather than being embedded in',
            'recognizing thought patterns',
            'present-tense awareness language',
            'seeing the difference between having a thought and being the thought',
        ],
        'distinguishing_from_next': (
            'Presence of an observing perspective without describing a transformed '
            'relationship to the sensory experience itself. Observation without '
            'reinterpretation.'
        ),
        'exemplar_utterances': [
            "I noticed I was getting anxious about the pain, and I could just watch that anxiety.",
            "I saw my mind wanting to fight the sensation, and I just observed that impulse.",
            "There was this moment where I was aware of being aware of the pain.",
            "I could see my thoughts about the pain as just thoughts.",
        ],
    },
    'reappraisal': {
        'stage_id': 3,
        'name': 'Pain Reappraisal',
        'short_name': 'Reappraisal',
        'prompt_name': 'pain reappraisal',
        'definition': (
            'A fundamental reinterpretation of sensory experience. The participant '
            'describes pain as changing, as composed of distinct sensations rather '
            'than monolithic suffering, and as lacking the fixed personal '
            'significance previously attributed to it. Language indicating '
            'decentering from pain-related narratives and expressions of '
            'equanimity or acceptance that emerge from insight rather than '
            'suppression.'
        ),
        'prototypical_features': [
            'pain described as changing or impermanent',
            'pain decomposed into distinct sensations',
            'sensation distinguished from suffering',
            'decentering from pain narratives',
            'equanimity emerging from insight',
            'acceptance through understanding rather than suppression',
            'pain lacking fixed personal significance',
        ],
        'distinguishing_from_next': (
            'Transformed relationship to the sensory experience itself, '
            'not merely observation of reactions.'
        ),
        'exemplar_utterances': [
            "The pain was still there but it was just... sensation. It didn't mean what I thought it meant.",
            "I realized it's not one solid thing, it's like waves that come and go.",
            "It's interesting, when I really look at it, the 'pain' is actually many different feelings.",
            "I don't have to make it mean something about my life anymore.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Word prototypes for CTS baseline (mirrors word_prototypes in srl_constructs)
# ---------------------------------------------------------------------------
word_prototypes = {
    'vigilance': ['pain', "can't focus", 'overwhelmed', 'captured'],
    'avoidance': ['push away', 'distract', 'escape', 'block out'],
    'metacognition': ['noticed', 'observed', 'watched', 'aware of'],
    'reappraisal': ['just sensation', 'changing', 'impermanent', "doesn't mean"],
}

# ---------------------------------------------------------------------------
# Categories for grouping (mirrors categories dict in srl_constructs.py)
# ---------------------------------------------------------------------------
categories = {
    'Early stages': [
        'Pain Vigilance and Attention Dysregulation',
        'Attention Regulation applied to Experiential Avoidance',
    ],
    'Late stages': [
        'Metacognitive Awareness',
        'Pain Reappraisal',
    ],
}

# ---------------------------------------------------------------------------
# Visualization colors (mirrors colors_severity in srl_constructs.py)
# ---------------------------------------------------------------------------
colors_stages = {
    0: '#DC267F',  # Vigilance  - magenta
    1: '#FE6100',  # Avoidance  - orange
    2: '#648FFF',  # Metacognition - blue
    3: '#785EF0',  # Reappraisal   - purple
}

# ---------------------------------------------------------------------------
# VA-MR to Phenomenology Codebook mapping
# Integrates the legacy phenomenology codebook from
# apply_codebook_legacy_implementation.py with the VA-MR framework.
# These mappings establish which phenomenology codes are theoretically
# expected to co-occur with each VA-MR stage.
# ---------------------------------------------------------------------------
vamr_to_phenomenology_mapping = {
    'vigilance': [
        'Fear, Anxiety, Panic, or Paranoia',
        'Agitation or Irritability',
        'Pain',
    ],
    'avoidance': [
        'Pain Avoidance',
        'Fear Avoidance',
        'Affective Flattening, Emotional Detachment, or Alexithymia',
    ],
    'metacognition': [
        'Meta-Cognition',
        'Clarity',
        'Change in Narrative Self',
    ],
    'reappraisal': [
        'Reappraisal',
        'Change in Worldview',
        'Change in Self-Other or Self-World Boundaries',
        'Disintegration of Conceptual Meaning Structures',
    ],
}

# ---------------------------------------------------------------------------
# Stage name to ID lookups (for parsing LLM outputs)
# ---------------------------------------------------------------------------
STAGE_NAME_TO_ID = {
    # Full names
    'pain vigilance and attention dysregulation': 0,
    'attention regulation applied to experiential avoidance': 1,
    'metacognitive awareness': 2,
    'pain reappraisal': 3,
    # Short names (case-insensitive matching in parser)
    'vigilance': 0,
    'avoidance': 1,
    'experiential avoidance': 1,
    'metacognition': 2,
    'reappraisal': 3,
}

STAGE_ID_TO_SHORT = {0: 'Vigilance', 1: 'Avoidance', 2: 'Metacognition', 3: 'Reappraisal'}


def get_stage_definitions_json():
    """Export stage definitions as the Output 4 JSON structure."""
    return {
        'stages': [
            stage_definitions[key]
            for key in ['vigilance', 'avoidance', 'metacognition', 'reappraisal']
        ]
    }
