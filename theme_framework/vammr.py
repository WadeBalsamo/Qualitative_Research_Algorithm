"""
vammr.py
--------
VAMMR theme framework: Vigilance, Avoidance, Mindfulness, Metacognition,
Reappraisal.

Five stages of contemplative transformation expressed in participant language
during Mindfulness-Oriented Recovery Enhancement (MORE) therapy for chronic
pain. The framework characterizes the developmental trajectory by which
participants restructure their relationship to pain through mindfulness
practice, from initial attentional capture by pain through the transformative
reinterpretation of pain experience.

The five stages are operationalized for classification of participant segments
in qualitative research and computational mixed-methods pipelines. Each stage
specifies what kind of language indicates it, what features distinguish it
from adjacent stages, and prototypical, subtle, and adversarial exemplar
utterances drawn from the MORE clinical trial corpus.

Stage definitions in brief

  Vigilance        — Hypervigilant capture by pain combined with absence of
                     successful attentional control. Includes catastrophic
                     cognitions about the meaning, trajectory, or
                     consequences of pain.

  Avoidance        — Maladaptive deployment of emerging attentional skill in
                     the service of escape from internal experience.
                     Includes escapist avoidance (suppression, distraction,
                     flying away) and kinesiophobic avoidance (fear of
                     movement). Diagnostically: unstable attention that
                     cannot remain with discomfort.

  Mindfulness      — Stable, sustained, volitional attention that can stay
                     WITH present experience including pain or discomfort.
                     The development of mindfulness as an adaptive skill:
                     reduced effortful struggle, the capacity to enter and
                     remain in present awareness, attention as anchor rather
                     than escape vehicle.

  Metacognition    — The reflexive capacity to observe one's own mental
                     processes as they occur. Watching reactions, impulses,
                     and patterns arise; decentering from mental content;
                     recognizing the distinction between pain and reactions
                     to pain. Observation without yet describing
                     transformation of the experience itself.

  Reappraisal      — A transformation of the meaning, structure, or felt
                     quality of pain experience that emerges from insight
                     rather than from suppression. Encompasses sensory
                     reappraisal (decomposition into constituent sensations,
                     jagged-to-dull transformations) and cognitive
                     reappraisal (redefinition of pain's meaning, identity
                     shifts, transformed worldview). Includes equanimity
                     and acceptance grounded in understanding.

Exemplar utterances are drawn from the MORE-for-LRP clinical trial corpus
(Wexler, Balsamo et al., 2026, Mindfulness, 17, 819-833) and from session
recordings analyzed during framework development.
"""

from .theme_schema import ThemeFramework, ThemeDefinition


def get_vammr_framework() -> ThemeFramework:
    """Return the five-stage VAMMR theme framework."""

    # =========================================================================
    # Vigilance
    # =========================================================================
    vigilance = ThemeDefinition(
        theme_id=0,
        key='vigilance',
        name='Pain Vigilance and Catastrophizing',
        short_name='Vigilance',
        prompt_name='pain vigilance and catastrophizing',
        definition=(
            "Hypervigilant monitoring of pain signals coupled with the absence "
            "of successful attentional deployment. The participant is captured "
            "or overwhelmed by pain, attention fragments or scatters, and the "
            "participant cannot disengage from pain or sustain focus on chosen "
            "objects of awareness. Catastrophic cognitions about pain may be "
            "present: pain framed as endless, defining, controlling, or "
            "disabling; doubt about one's capacity for practice; helplessness "
            "about pain trajectory. Restlessness and inability to remain still "
            "during practice are common physical correlates. The participant "
            "has not yet developed even a maladaptive strategy for managing "
            "where attention goes — pain organizes the field of awareness "
            "without volitional control."
        ),
        prototypical_features=[
            'captured or overwhelmed by pain',
            'attention returns to pain involuntarily',
            'fragmented, scattered, or restless awareness',
            "cannot focus during practice or daily activity",
            'must move or shift due to discomfort during meditation',
            'distractions (sounds, thoughts, environment) repeatedly disrupt attention',
            'catastrophic cognitions: pain as life sentence, never-ending, controlling',
            "doubt about one's capacity for mindfulness practice",
            'hypervigilant body monitoring or scanning for symptom change',
            'pain is mentally amplified by talking or thinking about it',
            'helplessness about ability to manage pain after the program ends',
            'pain dominates awareness without any felt volitional control',
        ],
        distinguishing_criteria=(
            "Absence of any successful attentional deployment, adaptive or "
            "maladaptive. Distinguished from Avoidance: the participant has "
            "not yet developed even a maladaptive redirection strategy — "
            "pain captures awareness despite efforts to engage with practice. "
            "Distinguished from Mindfulness: there is no sustained presence "
            "with any chosen object of attention. Distinguished from "
            "Metacognition: the participant may notice that pain has captured "
            "attention, but this is reportage of capture rather than the "
            "establishment of an observing perspective."
        ),
        exemplar_utterances=[
            "...maybe I had doubt early on because I have so many attention issues, [I doubted] if I would be able to do this practice well.",
            "The most difficult part... I mean, I feel like it's this every time, is just sort of staying, like keeping my mind on track with it, just not getting distracted by other things.",
            "...if I'm still, my body, the pain in the hips will flare up. So I'll have to move. And it's like a butterfly effect with my brain. Once I have that pain and I move, my focus is lost.",
            "I went to the part that was painful. And I was okay there, but then I think it's, I don't know. I had to keep moving. I can't, I can't lay still.",
            "I have the attention span of a squirrel and about the energy level as well.",
            "I just had a hard time getting settled. I mean, I couldn't get comfortable. Even though I was, you know, being one position and I'd have to lean over.",
            "It's just hard to kind of quiet my mind. I know that when we're meditating that the mind will wander, but if the mind is really jazzed up, it's really hard to wander and then come back.",
            "the hardest part is always for me just getting the distractions. You know, just, the focus, the focused present, like getting distracted. So it seems to be the hardest. noises and cats and whatever scratches and sneezes and stuff like that.",
            "Now that we're talking about it again though my leg will start firing. It's just when I think about it or talk about it too much...I think it's something I made up but sometimes it will hurt more.",
            "I just kind of really rely on this class to kind of give me a push. And I'm not anticipating having that push after this class is over so I'm not sure how to make that work for me.",
        ],
        subtle_utterances=[
            "I keep checking in on how my back feels, even when I'm trying to relax.",
            "It's hard to concentrate on anything else when it flares up.",
            "I notice I'm always scanning for whether it's going to get worse.",
            "I tried to do mindfulness all day and nothing helped.",
            "I had a lot of when I went to my spine, it was very painful, it was like a curvature... it was hot, you know, and so I was trying to figure out how to come out of it a couple times.",
            "no matter how spiritually evolved I think I am, it's very destabilizing.",
        ],
        adversarial_utterances=[
            # Vigilance vs Avoidance: an attempted redirection that fails.
            # Code Vigilance when capture dominates; Avoidance when
            # the redirection itself is the focus and is being maintained.
            "I try to push the pain away but it keeps grabbing my attention.",
            # Vigilance vs Metacognition: noticing capture without an
            # observing perspective. No reflective distance, no decentered
            # stance — the report is OF the capture, not FROM a position
            # outside it. Code Vigilance.
            "I'm aware that my mind keeps going to the pain.",
            # Vigilance vs Mindfulness: attempted anchoring that fails to
            # sustain. Code Vigilance when the failure dominates the report;
            # Mindfulness only when the participant describes successful
            # return and sustained presence.
            "I tried to focus on the breath but the pain pulled me right back.",
            # Vigilance vs Reappraisal (catastrophic frame): described as a
            # life sentence with no insight or transformed perception. The
            # absence of insight markers is what makes this Vigilance rather
            # than Reappraisal.
            "It feels like a life sentence, like it's not going to end.",
        ],
        word_prototypes=[
            'captured', 'overwhelmed', 'scattered', 'distracted',
            "cannot focus", "can't focus", "can't concentrate",
            'mind wanders', 'attention pulled', 'butterfly effect',
            'lost focus', 'all I can think about', 'life sentence',
            'never end', 'useless', 'jazzed up', 'restless', 'agitated',
            "can't get comfortable", "can't be still", "can't lay still",
            'attention span of a squirrel', 'all over the place',
            'spiraling', 'flare up', 'firing', 'doubt',
            'pulled me right back', 'hard time getting settled',
            'crashes into me', 'destabilizing',
        ],
        color='#DC267F',
        aliases=['catastrophizing', 'attention dysregulation', 'helplessness'],
    )

    # =========================================================================
    # Avoidance
    # =========================================================================
    avoidance = ThemeDefinition(
        theme_id=1,
        key='avoidance',
        name='Experiential Avoidance',
        short_name='Avoidance',
        prompt_name='experiential avoidance',
        definition=(
            "A narrowly defined maladaptive deployment of emerging attentional "
            "skill in the service of escape from internal experience. Two "
            "subtypes are included. ESCAPIST AVOIDANCE: deliberately ignoring, "
            "suppressing, distracting from, or 'flying away from' pain "
            "sensations, thoughts, or emotions. KINESIOPHOBIC AVOIDANCE: "
            "fear-based avoidance of movement, posture, or activity expected "
            "to provoke pain. The diagnostic feature is unstable attention — "
            "the participant cannot remain with discomfort and must redirect "
            "away. The goal of the redirection is to NOT FEEL, not to "
            "investigate. Relief, when reported, comes from suppression, "
            "disconnection, or distraction rather than from understanding. "
            "Includes language framing mindfulness itself as a tool for "
            "escape (e.g., as a 'superpower' to fly somewhere else)."
        ),
        prototypical_features=[
            'deliberately redirecting attention away from pain or discomfort',
            'using breath or imagery as escape rather than as anchor for presence',
            'language of ignoring, blocking out, pushing away, or getting away from pain',
            'distraction strategies framed as pain-management technique',
            'kinesiophobic avoidance: not moving, holding still, or avoiding postures for fear of pain',
            'unstable attention: cannot remain with discomfort, must shift away',
            'agitation or restlessness driving disengagement from practice itself',
            'mindfulness reframed as a way to "fly away" or escape',
            'forgetting pain framed as the goal, not understanding it',
            'disconnection, numbing, or floating-away described as relief',
            'absence of insight: relief reported as not-feeling rather than transformed perception',
            'suppressive coping (sleeping too much, overeating) used to manage pain',
        ],
        distinguishing_criteria=(
            "Attention is deployed AWAY from present experience rather than "
            "WITH it. Distinguished from Mindfulness by both goal (escape vs. "
            "presence) and stability (cannot stay with discomfort vs. "
            "sustained awareness even of unpleasant sensations). Distinguished "
            "from Reappraisal: relief comes from suppression or distraction, "
            "NOT from insight, transformed perception, or decomposed sensory "
            "experience. The absence of described insight is the most "
            "important boundary marker. Distinguished from Vigilance: a "
            "deliberate redirection strategy is being enacted, however "
            "unsustainably."
        ),
        exemplar_utterances=[
            "It was hard for me to get past the pain that I have. I kept trying to circle back and try to make it go away, but I was having a really hard time… I'm trying to not think about it. Just think about the breathing and staying with the breathing.",
            "So I'm trying to just listen to you and ignore [the pain]... Ignoring the pain instead of going into the pain. It makes it hurt more…When I ignore the pain, it makes it hurt less.",
            "I'm just kind of agitated, it's like my mind is in an agitated state when I'm just laying there... I notice myself not wanting to do it… feeling very much wanting to distract myself.",
            "It [mindfulness] becomes like a superpower... to fly you somewhere else away from your sorrows and away from the pain.",
            "I was trying not to do it. It was just good to take the time to try and let everything go.",
            "I just feel like everything went away for a minute. You know, like I wasn't feeling my neck, that's what I've been feeling. And it just felt very calming.",
            "I experienced the floatingness, disconnection from the world around me. Kind of a numb buzzing feeling in my body.",
            "I tend to forget about my pain, that's how I manage my pain for years.",
            "yeah, like it felt like it gave me permission to not think about it. It's just released my mind from it for a little bit.",
            "if I not thought about my pain, I went to my heart. It was much better.",
            "They're a good distraction. You'll forget about your sorrows for a minute. You won't feel as much pain, you know?",
            "I'll sleep at night, which is great, but sometimes if I'm in too much pain or if you add that with the depression, I'll sleep. The overeating.",
        ],
        subtle_utterances=[
            "I've found that if I keep myself busy, it doesn't bother me as much.",
            "During the meditation I kind of redirect myself away from the discomfort.",
            "I don't want to bend that way because I know it's going to hurt.",
            "It really helped to distract me from my pain.",
            "And I feel that to get away with my pain more because I was focusing on something joyful.",
            "I had to come out of it for a while.",
        ],
        adversarial_utterances=[
            # Avoidance vs Mindfulness: same surface behavior ("focused on the
            # breath") with opposite intent. Code Avoidance when attention is
            # AWAY from present sensation and frame is escape; Mindfulness
            # when breath is the anchor for staying WITH present experience.
            "I focused on the breath and the pain felt less important.",
            # Avoidance vs Metacognition: observing the avoidance impulse
            # without enacting it leans Metacognition; enacting the impulse
            # without observation is Avoidance. The discriminator is whether
            # an observing perspective is present.
            "I noticed myself wanting to push the pain away.",
            # Avoidance vs Reappraisal: "doesn't bother me" without described
            # insight is Avoidance (suppression); with described insight or
            # transformed perception, it is Reappraisal.
            "It doesn't bother me as much anymore.",
            # Avoidance vs Mindfulness: "let everything go" can be either
            # adaptive non-attachment (Mindfulness/Reappraisal) or escape
            # framing (Avoidance). The frame here ("trying not to do it")
            # marks it as Avoidance.
            "It was just good to take the time to try and let everything go.",
        ],
        word_prototypes=[
            'push away', 'pushing away', 'ignore', 'ignoring',
            'block out', 'escape', 'fly away', 'fly somewhere else',
            'distract', 'distract myself', 'distraction',
            'get away from', 'not think about', "don't think about",
            "trying not to", 'circle back to make it go away',
            'check out', 'disconnect', 'disconnected', 'detach',
            'detached', 'numb', 'buzzing', 'floatingness',
            "wasn't feeling", 'everything went away', 'gone',
            'went somewhere else', 'afraid to move', "don't want to bend",
            'superpower', 'forget about', 'forget my pain',
            'permission to not think', 'released my mind',
            'wanting to distract',
        ],
        color='#FE6100',
        aliases=[
            'escapist avoidance',
            'kinesiophobic avoidance',
            'fear-avoidance',
            'experiential avoidance of pain',
            'thought suppression',
        ],
    )

    # =========================================================================
    # Mindfulness
    # =========================================================================
    mindfulness = ThemeDefinition(
        theme_id=2,
        key='mindfulness',
        name='Mindfulness',
        short_name='Mindfulness',
        prompt_name='mindfulness',
        definition=(
            "The development of mindfulness as an adaptive skill: stable, "
            "sustained, volitional attention that can stay WITH present "
            "experience including pain or discomfort, without redirecting "
            "away. The participant describes increasing capacity to enter "
            "and remain in present awareness, reduced effortful struggle "
            "between attention and distraction, and the ability to bring "
            "attention to chosen objects (breath, body, sensation) and hold "
            "it there. Attention functions as anchor rather than vehicle for "
            "escape: the participant can return to the breath after "
            "distraction without using the breath to flee from sensation. "
            "This is the foundational mindfulness skill underlying the more "
            "sophisticated stages that follow — present awareness as it is, "
            "without yet describing observation of mental processes "
            "themselves (Metacognition) or transformation of pain experience "
            "(Reappraisal)."
        ),
        prototypical_features=[
            'stable, sustained attention during practice',
            'reduced internal struggle or fight to focus',
            'increased capacity to enter and remain in meditative state',
            'attention can stay WITH discomfort rather than fleeing from it',
            'volitional, directed attention to breath, body, or sensation',
            'descriptions of attention "operating smoother" or "less back and forth"',
            'capacity to return to the anchor after distraction',
            'present-moment awareness without commentary or reinterpretation',
            'reports of progress: easier to settle, getting better at it',
            'breath as anchor for presence (not as escape from sensation)',
            'staying with the body in a non-clenched relaxed state',
            'practice generalizing to daily life as steady awareness',
        ],
        distinguishing_criteria=(
            "Sustained attention WITH experience rather than away from it "
            "(distinguishing from Avoidance). The breath and body are anchors "
            "for presence, not vehicles for escape; the participant can stay "
            "with discomfort if it arises. Reports of attentional competence "
            "WITHOUT yet describing observation of mental processes themselves "
            "(distinguishing from Metacognition). NO transformation of pain "
            "meaning or sensory structure is described (distinguishing from "
            "Reappraisal) — pain may still be pain, but the participant can "
            "be present with what is."
        ),
        exemplar_utterances=[
            "There's less fight going on internally as far as the intention goes than it was four weeks ago. It's almost like a switch now. Whereas before, it was really back and forth, back and forth. And now it's operating much smoother.",
            "When I first started… I was really scattered, and I had a hard time staying in tune with it. And this time, well, I really noticed the whole time, it's my ability to like jump in and get into that state and stay in that state better than before.",
            "So once I get through the power struggle of attention deficit, then I can get more into it. But once I'm able to kind of relinquish and let go, then I feel really present with your words and what you're saying and kind of everything else is just gone by the wayside.",
            "I do think I'm getting slowly better at it... I am enjoying that the more that I do it, the easier it seems to become.",
            "I just feel like I could just like stay there.",
            "Maybe just having done it for four weeks and gone kind of more and a half of it. Now I don't feel like there's a conflict so much.",
            "It's kind of like I finally get into it... it seems like every time you get to that point, it's like, sigh, enough time is past and I've gotten enough into it that I can continue without any disruption or distraction.",
            "Even after that little bit of mindfulness, my leg isn't in as much pain as before we started.",
            "Well, the only thing that worked really is I said, okay, this isn't working. I'll come back to this. And I did come back to it and it did work.",
            "I lose track of time until you say it's time to come back. You know, I mean, I'm not sleeping. I'm just relaxed.",
            "And just I guess through having my body just one piece at a time getting my whole body in a non clenched relaxed state.",
            "Focusing on the breath is really the easiest way for me to do that... I just focus on the breath to the pain without saying it if that makes sense.",
        ],
        subtle_utterances=[
            "It's getting easier to settle in.",
            "I can come back to the breath when I get distracted now.",
            "I stayed present with what I was feeling for longer than I used to.",
            "I started to do that and just pay attention to what it was doing.",
            "I really liked their reminder and the focus of the breath on the nostrils. I feel like having a thing to anchor my attention is really helpful.",
            "I was listening to you and your guidance has kind of helped me bring me back for that. I'd find I'd get distracted again.",
        ],
        adversarial_utterances=[
            # Mindfulness vs Avoidance: hinges on whether breath/body is
            # anchor for presence or vehicle for escape. Code Mindfulness
            # when participant describes staying WITH experience; Avoidance
            # when frame is escape and attention destabilizes around
            # discomfort or the goal is not-feeling.
            "I focused on the breath instead of the pain.",
            # Mindfulness vs Reappraisal: "stayed with the sensation" is
            # Mindfulness; if accompanied by sensory decomposition or
            # described meaning shift, code Reappraisal.
            "I just stayed with the sensation.",
            # Mindfulness vs Metacognition: meta-attention to one's own
            # attention is the boundary case. Code Mindfulness when emphasis
            # is on sustained control of attention; Metacognition when
            # emphasis is on the observing perspective itself.
            "I noticed I could keep my attention where I wanted it.",
            # Mindfulness vs Avoidance: relaxation that displaces pain
            # awareness can be either adaptive presence with body or
            # disengagement from pain. Code based on whether the participant
            # describes staying with what arises or describes escape.
            "I felt really relaxed and the pain just wasn't there.",
        ],
        word_prototypes=[
            'stay with', 'staying with', 'stayed there', 'stay there',
            'sustained', 'settled', 'steady', 'present', 'in the moment',
            'present moment', 'anchor', 'come back to', 'return',
            'returning', 'come back', 'jump in', 'drop in', 'dropping in',
            'operating smoother', 'less fight', 'less back and forth',
            'relinquish', 'easier to settle', 'getting better at it',
            'kept my attention', 'pay attention', 'paying attention',
            'soaking in', 'just stayed', 'finally let go',
            'less conflict', 'come back to it', 'bring me back',
            'breath to the pain', 'gone by the wayside',
        ],
        color='#FFB000',
        aliases=[
            'attention regulation',
            'present awareness',
            'concentration',
            'adaptive coping',
            'sustained attention',
            'mindfulness skill',
            'meditation skill',
        ],
    )

    # =========================================================================
    # Metacognition
    # =========================================================================
    metacognition = ThemeDefinition(
        theme_id=3,
        key='metacognition',
        name='Metacognitive Awareness',
        short_name='Metacognition',
        prompt_name='metacognitive awareness',
        definition=(
            "The reflexive capacity to observe one's own mental processes as "
            "they occur. The participant describes noticing reactions to "
            "pain, watching thoughts and impulses arise and pass, recognizing "
            "characteristic patterns in their own responses, and adopting an "
            "observing perspective from which mental activity can be examined "
            "rather than identified with. This is decentering: recognizing "
            "that one is HAVING a thought rather than BEING the thought; "
            "recognizing the difference between pain and the reaction to "
            "pain; recognizing that anticipation can amplify pain before "
            "the sensation even arrives. Metacognition presupposes the "
            "attentional stability of Mindfulness but adds a reflexive "
            "observing position. It does not yet involve described "
            "transformation of the experience itself; the participant "
            "observes their experience rather than reinterprets it."
        ),
        prototypical_features=[
            'noticing reactions, impulses, or characteristic patterns in oneself',
            'watching thoughts arise and pass without being captured',
            'observing rather than being identified with mental content',
            'decentering: thoughts seen as thoughts, reactions seen as reactions',
            'recursive awareness ("noticing the way I am noticing")',
            'recognizing the difference between pain and reactions to pain',
            'catching anticipation or bracing before it shapes the experience',
            'pulling apart a complex emotional reaction into its components',
            'present-tense observational language',
            'a stable observing perspective on the flow of experience',
            'recognizing recurring tendencies ("that\'s how I am with everything")',
            'noticing the relationship between thought and bodily response',
        ],
        distinguishing_criteria=(
            "Presence of an observing perspective on mental processes, "
            "without yet describing transformation of the sensory or affective "
            "experience itself. Distinguished from Mindfulness by the explicit "
            "observation of mental activity rather than mere sustained "
            "attention — meta-attention rather than just attention. "
            "Distinguished from Reappraisal because the participant observes "
            "their experience but does not describe reinterpreting it; pain "
            "may be observed as still being pain, but observed FROM somewhere "
            "rather than FROM WITHIN. Distinguished from Vigilance: the "
            "noticing is from a stable observing position, not a report of "
            "having been captured."
        ),
        exemplar_utterances=[
            "[I'm] noticing the way that I'm noticing pain... I feel like my relationship to it is changing.",
            "I caught myself at one point... during the scan when you were going down the body and into the lower leg. I have a lot of pain there. And I noticed like tension right here as you were approaching that... and I was like, that's anticipation. Like I'm in this. I'm already thinking too much about what that's going to feel like... my brain is already deciding that it's kind of hurt before I even get there.",
            "I think like the times when I've noticed I can relax around it, it helps with the pain. Because then I get to notice it's shifting. I'm not bracing against it or like trying to avoid it.",
            "[It was] quite enjoyable to be able to just kind of soak in and just kind of feel or experience what my body's experiencing because I've really kind of been detached and, more trying to make it go away and not really focusing on feeling it. So it was good to be able to just recognize what my body's doing.",
            "So I thought about all those different things that happened just in those seconds. Got out my stomach, in my head, in my heart. Oh, it's how I am with everything like that. That how much just a thought can control what happens in my body. To pull it apart, it was really useful.",
            "the two things that are stuck with me is realizing that, you know, I have a lot of skill and ability with being able to really quickly instantly reframe things emotionally. And thinking, well, you know, I'm really skillful on that. How can I apply that skill to this?",
            "I feel like it was just kind of like a kept trying to move like further inward with it. But I don't know if I ever like reached a destination with that, just sort of like this cyclical inward kind of noticing.",
            "I'm kind of more aware of the bigger effect when I'm not in session, where I'm not doing the exercise, but rather just trying to do my life and moving.",
            "I noticed I was getting anxious about the pain, and I could just watch that anxiety.",
        ],
        subtle_utterances=[
            "It was like I could step back a little and see what was happening in my mind.",
            "I started to recognize that the worry and the pain are two different things.",
            "I caught myself bracing and that catching itself felt different.",
            "I was thinking about my breathing more than the pain.",
            "I saw my mind wanting to fight the sensation, and I just observed that impulse.",
        ],
        adversarial_utterances=[
            # Metacognition vs Reappraisal: observation accompanied by a
            # realization about pain's structure. Code Reappraisal when the
            # realization transforms what pain IS for the participant
            # (sensory decomposition, lost solidity). Code Metacognition
            # when the report stays at observed change without described
            # reinterpretation.
            "I watched the pain change and realized it wasn't as solid as I thought.",
            # Metacognition vs Avoidance: observing an avoidance impulse
            # WITHOUT enacting it is Metacognition. The observing perspective
            # is the dominant feature.
            "I observed myself pushing the pain away during the meditation.",
            # Metacognition vs Mindfulness: noticing one's own attentional
            # process is meta-attention. Code Metacognition when the
            # observing perspective is dominant; Mindfulness when sustained
            # attention itself is the dominant report.
            "I saw my mind wanting to fight the sensation, and I just observed that impulse.",
            # Metacognition vs Vigilance: noticing capture from within
            # capture is Vigilance. Noticing from a stable observing
            # position is Metacognition. The key marker is reflective
            # distance.
            "I noticed I was completely caught up in the pain and couldn't get out.",
        ],
        word_prototypes=[
            'noticing', 'noticed', 'notice', 'observing', 'observed',
            'observe', 'watching', 'watched', 'recognized', 'recognizing',
            'recognize', 'aware of', 'awareness', 'caught myself',
            'caught', 'stepping back', 'step back', 'stepped back',
            'pull it apart', 'pulling apart', 'noticing the noticing',
            'saw myself', 'see myself', 'realize', 'realized',
            'realizing', 'two different things', 'separate',
            "that's how I am", 'anticipation', "that's anticipation",
            'relationship to it is changing', 'recognize what my body',
            'more aware of', 'bracing',
        ],
        color='#648FFF',
        aliases=[
            'self-awareness',
            'decentering',
            'observing perspective',
            'reflective awareness',
            'meta-awareness',
            'metacognitive awareness',
        ],
    )

    # =========================================================================
    # Reappraisal
    # =========================================================================
    reappraisal = ThemeDefinition(
        theme_id=4,
        key='reappraisal',
        name='Cognitive and Sensory Reappraisal',
        short_name='Reappraisal',
        prompt_name='cognitive and sensory reappraisal',
        definition=(
            "A transformation of the meaning, structure, or felt quality of "
            "pain experience that emerges from insight rather than from "
            "suppression. Two complementary domains are encompassed. SENSORY "
            "REAPPRAISAL: pain decomposed into constituent sensations (heat, "
            "tingling, pressure; jagged versus dull; sharp versus soft; hot "
            "versus cool), described as changing or impermanent rather than "
            "monolithic, with sensation distinguished from suffering. The "
            "felt quality of pain shifts as the participant attends to it "
            "directly. COGNITIVE REAPPRAISAL: pain redefined in terms of "
            "identity, life trajectory, or self-narrative; pain no longer "
            "defining the self; new sense of freedom, possibility, or "
            "worldview; pain reframed as a challenge rather than a "
            "struggle. Includes equanimity and acceptance that emerge from "
            "described understanding rather than from effortful suppression "
            "— this is the critical boundary against Avoidance."
        ),
        prototypical_features=[
            'pain decomposed into distinct constituent sensations',
            'pain described as changing, shifting, or impermanent',
            'sensory transformations during practice (e.g., jagged-to-dull, hot-to-cool, red-to-blue)',
            'sensation explicitly distinguished from suffering or threat',
            'pain redefined: not defining self, not a "life sentence"',
            'sense of freedom, new lease on life, or shifted identity',
            'equanimity emerging from described insight, not from blocking out',
            'acceptance grounded in understanding, not in resignation or suppression',
            'pain becoming "secondary" rather than dominant in awareness',
            'language of "reframe" applied to pain or to emotional experience',
            'pain met as introducing itself ("oh, you\'re noticing me, here I am")',
            'capacity to walk, move, or function in ways previously prevented by pain',
        ],
        distinguishing_criteria=(
            "Transformation of the experience itself — its meaning, "
            "structure, or felt quality — not merely sustained attention "
            "(Mindfulness) or observation of reactions (Metacognition). "
            "CRITICAL boundary against Avoidance: when participants report "
            "that pain bothers them less or has shifted, code Reappraisal "
            "ONLY when this is grounded in described insight, understanding, "
            "or transformed perception — NOT in suppression, distraction, "
            "or simple absence of feeling. The presence of described insight "
            "or sensory transformation is the discriminating feature. "
            "Distinguished from Metacognition: observation has crossed into "
            "reinterpretation; the experience itself is reported as "
            "different, not merely observed."
        ),
        exemplar_utterances=[
            # Sensory reappraisal
            "When I went into the pain part of it, it was very jagged and sharp and hot and red... and then, as I breathed more into it, it got less sharp, more of a dull. And the color changed for me from a red to a dark blue.",
            "the more I focused on the breath at that specific point it just seemed like the intensity of the pain decreased into a cool feeling... to where I could kind of calm things down by utilizing the breath.",
            "the pain went from feeling hot to just kind of like a cooling sensation as if I had put cold sheets... it wasn't cold like an ice pack. it was cold like crisp sheets... and it just kind of like numbed down that hot sensation.",
            "Well I thought it was kind of interesting like being able to fully tune in and then notice how the pain had changed... to notice if it had a center or if it had edges or if it was tingling.",
            "Every part of the body that we went to, my body, like, I felt more pain... and it wasn't like constant and it wasn't like excruciating. It was just like, oh, you're noticing me and here I am.",

            # Cognitive reappraisal
            "I have to redefine my definition of pain... that's kind of what I've been set out to do is to redefine my definition of pain and not let it define me and control me and keep me just useless.",
            "I noticed that like today, I was walking. And I don't remember the last time I walked and didn't immediately notice pain. I walked out of my house... I had gone like a block and a half. And I was like, oh my gosh, I'm not experiencing the same kind of pain.",
            "Even though I was thinking about it, it kind of became secondary because of the breathing.",
            "It's a challenge not a struggle. The challenge is a climb. A struggle is a flail. If you're struggling, you don't have a focus. If you're challenged, you can have a focus.",

            # Equanimity from insight
            "I have a lot of skill and ability with being able to to really quickly instantly reframe things emotionally... how can I apply that skill to this physical experience of pain?",
            "I ended up in a place of hope. When normally looking at these pictures, all I can feel is a sadness... I'm grateful for your guidance because it took me to a place of hope.",
            "noticing the way that I'm noticing pain. I feel like my relationship to it is changing. Which is great. I mean, it's still there. But I noticed that today, I was walking, and I don't remember the last time I walked and didn't immediately notice pain.",
        ],
        subtle_utterances=[
            "It's not that the pain went away, but something about my relationship to it shifted.",
            "I started to see the pain differently, like it's just part of what's happening.",
            "The sensation is still there but it feels lighter somehow.",
            "It wasn't just one. It's like all those different types of pain.",
            "I felt like a gift. Just like, oh, look at that.",
            "It just brought... I ended up in a place of hope. When normally looking at these pictures, all I can feel a sadness.",
        ],
        adversarial_utterances=[
            # Reappraisal vs Avoidance: the discriminating feature is whether
            # described insight grounds the relief. Without described insight
            # or transformed perception, "doesn't bother me" is Avoidance;
            # with described insight, it is Reappraisal.
            "It doesn't bother me anymore.",
            # Reappraisal vs Metacognition: "noticed the pain was changing"
            # is Metacognition if framed as observation. It is Reappraisal
            # if the change describes a transformation in the experience
            # itself (e.g., into distinct sensations, into something
            # qualitatively different, with described meaning shift).
            "I noticed the pain was changing.",
            # Reappraisal vs Avoidance via equanimity: equanimity grounded
            # in insight is Reappraisal; equanimity as disengagement strategy
            # is Avoidance. The marker is whether the participant connects
            # the equanimity to insight or transformed perception.
            "The pain comes and goes and I just let it do its thing.",
            # Reappraisal vs Mindfulness: pain becoming "secondary" can be
            # Mindfulness (sustained attention with the breath) or Reappraisal
            # (the meaning of pain has shifted such that it is no longer
            # central). The presence of the explicit shift in centrality
            # marks it as Reappraisal here.
            "Even though I was thinking about it, it kind of became secondary because of the breathing.",
        ],
        word_prototypes=[
            'just sensation', 'changing', 'shifting', 'shifted',
            'impermanent', "doesn't mean", 'redefine', 'redefined',
            'redefining', "doesn't define me", "not let it define me",
            'new lease', 'new lease on life', 'freedom',
            'sense of freedom', 'lighter', 'softer', 'cool feeling',
            'jagged to dull', 'sharp to soft', 'hot to cool',
            'red to blue', 'less sharp', 'less intense', 'decreased',
            'became secondary', 'secondary', 'waves', 'comes and goes',
            'place of hope', 'challenge not a struggle', 'reframe',
            'reframing', 'gift', 'like a gift', 'here I am',
            "you're noticing me", "calm things down", "different kind of pain",
            'crisp sheets', 'numbed down',
        ],
        color='#785EF0',
        aliases=[
            'cognitive reappraisal',
            'sensory reappraisal',
            'mindful reappraisal',
            'mindful reappraisal of pain sensations',
            'pain reappraisal',
            'equanimity from insight',
            'reframing',
        ],
    )

    return ThemeFramework(
        name="VAMMR — Vigilance, Avoidance, Mindfulness, Metacognition, Reappraisal",
        version="2.0",
        description=(
            "Five-stage framework of contemplative transformation in "
            "Mindfulness-Oriented Recovery Enhancement (MORE) for chronic "
            "pain. Characterizes the developmental trajectory by which "
            "participants restructure their relationship to pain through "
            "mindfulness practice: from initial attentional capture by pain "
            "(Vigilance), through maladaptive use of emerging attentional "
            "skill for escape (Avoidance), to the development of stable "
            "present awareness as adaptive skill (Mindfulness), to the "
            "reflexive observing of mental processes themselves "
            "(Metacognition), to the transformation of pain experience "
            "through insight (Reappraisal)."
        ),
        themes=[
            vigilance,
            avoidance,
            mindfulness,
            metacognition,
            reappraisal,
        ],
        categories={
            'Dysregulation': ['Vigilance', 'Avoidance'],
            'Attention Regulation': ['Mindfulness', 'Metacognition'],
            'Transformational Experiences': ['Reappraisal'],
        },
    )