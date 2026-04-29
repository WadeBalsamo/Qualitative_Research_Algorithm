"""
vamr.py
-------
VA-MR (Vigilance-Avoidance Metacognition-Reappraisal) theme framework.

Defines the four stages of contemplative transformation used to classify
therapeutic dialogue.

Deprecated: use get_vammr_framework() from vammr.py for the current
five-stage VAMMR framework. This module is retained for backward
compatibility with existing config files that specify "preset": "vamr".
"""

from .theme_schema import ThemeFramework, ThemeDefinition


def get_vamr_framework() -> ThemeFramework:
    """Return the VA-MR theme framework."""

    vigilance = ThemeDefinition(
        theme_id=0,
        key='vigilance',
        name='Pain Vigilance and Attention Dysregulation',
        short_name='Vigilance',
        prompt_name='pain vigilance and attention dysregulation',
        definition=(
            'Expressions of hypervigilance to pain signals combined with '
            'inability to direct or sustain attentional focus. The participant '
            'describes being captured by pain, unable to disengage, and '
            'experiencing attentional fragmentation. Attention is reactive '
            'rather than directed. May include catastrophic cognitions about '
            'pain and hypervigilant monitoring of body signals.'
        ),
        prototypical_features=[
            'captured by pain',
            'overwhelmed by sensation',
            'cannot focus',
            'attention keeps going back to pain',
            'fragmented awareness',
            'catastrophic thinking about pain',
            'hypervigilant body monitoring',
            'pain dominates awareness',
        ],
        distinguishing_criteria=(
            'Absence of intentional attentional deployment. '
            'Attention is reactive rather than directed.'
        ),
        exemplar_utterances=[
            "I can't stop thinking about the pain, it's all I can focus on.",
            "Every little twinge sends me into a spiral.",
            "My mind just keeps going back to it no matter what I try.",
            "I feel like the pain is controlling everything.",
        ],
        subtle_utterances=[
            "I keep checking in on how my back feels, even when I'm trying to relax.",
            "It's hard to concentrate on anything else when it flares up.",
            "I notice I'm always scanning for whether it's going to get worse.",
        ],
        adversarial_utterances=[
            "I try to push the pain away but it keeps grabbing my attention.",  # Could be avoidance
            "I'm aware that my mind keeps going to the pain.",  # Could be metacognition
            "Sometimes I just sit with the discomfort and wait for it to pass.",  # Could be reappraisal
        ],
        word_prototypes=['pain', "can't focus", 'overwhelmed', 'captured'],
        color='#DC267F',
    )

    avoidance = ThemeDefinition(
        theme_id=1,
        key='avoidance',
        name='Attention Regulation applied to Experiential Avoidance',
        short_name='Avoidance',
        prompt_name='attention regulation applied to experiential avoidance',
        definition=(
            'The participant has developed attentional control but deploys it '
            'in the service of avoidance rather than investigation. Descriptions '
            'of deliberately pushing pain away, distracting themselves, or using '
            'newly developed concentration to suppress rather than examine '
            'painful experience. Attention is being directed away from rather '
            'than toward the present experience.'
        ),
        prototypical_features=[
            'deliberately redirecting attention away from pain',
            'pushing pain away',
            'using breath to escape pain',
            'distraction as strategy',
            'suppressing painful experience',
            'concentration deployed for avoidance',
            'relief-seeking through distraction',
        ],
        distinguishing_criteria=(
            'Attention is directed away from rather than toward present experience. '
            'The goal is to escape or suppress, not to observe.'
        ),
        exemplar_utterances=[
            "When the pain comes, I focus really hard on my breathing to push it away.",
            "I've gotten better at not thinking about it.",
            "I use the meditation to take my mind somewhere else.",
            "I concentrate on something pleasant to block out the sensation.",
        ],
        subtle_utterances=[
            "I've found that if I keep myself busy, it doesn't bother me as much.",
            "During the meditation I kind of redirect myself away from the discomfort.",
            "I've learned to just not go there when those feelings come up.",
        ],
        adversarial_utterances=[
            "I notice myself wanting to push the pain away and I just let that happen.",  # Could be metacognition
            "The pain feels less important when I focus on my breath.",  # Could be reappraisal
            "I can't escape it no matter how hard I try to focus elsewhere.",  # Could be vigilance
        ],
        word_prototypes=['push away', 'distract', 'escape', 'block out'],
        color='#FE6100',
        aliases=['experiential avoidance'],
    )

    metacognition = ThemeDefinition(
        theme_id=2,
        key='metacognition',
        name='Metacognitive Awareness',
        short_name='Metacognition',
        prompt_name='metacognitive awareness',
        definition=(
            "The capacity to observe one's own mental processes as they occur. "
            "The participant describes noticing their reactions to pain, watching "
            "thoughts arise and pass, and developing a perspective from which "
            "mental activity can be observed rather than identified with. There "
            "is an observing perspective present without yet describing a "
            "transformed understanding of the experience itself."
        ),
        prototypical_features=[
            'noticing reactions to pain',
            'watching thoughts arise and pass',
            'stepping back from experience',
            'observing rather than being embedded in',
            'recognizing thought patterns',
            'present-tense awareness language',
            'seeing the difference between having a thought and being the thought',
        ],
        distinguishing_criteria=(
            'Presence of an observing perspective without describing a transformed '
            'relationship to the sensory experience itself. Observation without '
            'reinterpretation.'
        ),
        exemplar_utterances=[
            "I noticed I was getting anxious about the pain, and I could just watch that anxiety.",
            "I saw my mind wanting to fight the sensation, and I just observed that impulse.",
            "There was this moment where I was aware of being aware of the pain.",
            "I could see my thoughts about the pain as just thoughts.",
        ],
        subtle_utterances=[
            "It was like I could step back a little and see what was happening in my mind.",
            "I started to recognize that the worry and the pain are two different things.",
            "I caught myself reacting and that catching itself felt different.",
        ],
        adversarial_utterances=[
            "I watched the pain change and realized it wasn't as solid as I thought.",  # Could be reappraisal
            "I observed myself pushing the pain away during the meditation.",  # Could be avoidance
            "I was so focused on noticing my thoughts that I lost track of the pain.",  # Could be avoidance
        ],
        word_prototypes=['noticed', 'observed', 'watched', 'aware of'],
        color='#648FFF',
    )

    reappraisal = ThemeDefinition(
        theme_id=3,
        key='reappraisal',
        name='Pain Reappraisal',
        short_name='Reappraisal',
        prompt_name='pain reappraisal',
        definition=(
            'A fundamental reinterpretation of sensory experience. The participant '
            'describes pain as changing, as composed of distinct sensations rather '
            'than monolithic suffering, and as lacking the fixed personal '
            'significance previously attributed to it. Language indicating '
            'decentering from pain-related narratives and expressions of '
            'equanimity or acceptance that emerge from insight rather than '
            'suppression.'
        ),
        prototypical_features=[
            'pain described as changing or impermanent',
            'pain decomposed into distinct sensations',
            'sensation distinguished from suffering',
            'decentering from pain narratives',
            'equanimity emerging from insight',
            'acceptance through understanding rather than suppression',
            'pain lacking fixed personal significance',
        ],
        distinguishing_criteria=(
            'Transformed relationship to the sensory experience itself, '
            'not merely observation of reactions.'
        ),
        exemplar_utterances=[
            "The pain was still there but it was just... sensation. It didn't mean what I thought it meant.",
            "I realized it's not one solid thing, it's like waves that come and go.",
            "It's interesting, when I really look at it, the 'pain' is actually many different feelings.",
            "I don't have to make it mean something about my life anymore.",
        ],
        subtle_utterances=[
            "I started to see the pain differently, like it's just part of what's happening.",
            "It's not that the pain went away, but something about my relationship to it shifted.",
            "The sensation is still there but it feels lighter somehow.",
        ],
        adversarial_utterances=[
            "I noticed I was seeing the pain as just sensation, which was interesting.",  # Could be metacognition
            "I focused on breaking the pain into smaller pieces to make it manageable.",  # Could be avoidance
            "The pain comes and goes and I just let it do its thing.",  # Could be metacognition
        ],
        word_prototypes=['just sensation', 'changing', 'impermanent', "doesn't mean"],
        color='#785EF0',
    )

    return ThemeFramework(
        name="Vigilance-Avoidance Metacognition-Reappraisal (VA-MR)",
        version="1.0",
        description=(
            "Four stages of contemplative transformation that participants "
            "express in their language during Mindfulness-Oriented Recovery "
            "Enhancement (MORE) therapy sessions."
        ),
        themes=[vigilance, avoidance, metacognition, reappraisal],
        categories={
            'Early stages': ['Vigilance', 'Avoidance'],
            'Late stages': ['Metacognition', 'Reappraisal'],
        },
    )
