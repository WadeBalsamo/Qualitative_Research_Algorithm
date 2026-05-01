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
theme_framework/vaamr.py).

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

================================================================================
EXEMPLAR SOURCING
================================================================================
Exemplar utterances are drawn from MORE-for-LRP session recordings
analyzed during framework development. Quotes ending with parenthetical
notes (e.g., "(also reinforces)") are included to demonstrate co-
occurrence; the primary code is the one in the section header.

Structural parallel to theme_framework/vaamr.py. Exports
get_purer_framework() which returns a ThemeFramework with five
ThemeDefinition objects.
"""

from theme_framework.theme_schema import ThemeDefinition, ThemeFramework


_PURER_DEFINITIONS = [
    # =========================================================================
    # P — Phenomenological
    # =========================================================================
    ThemeDefinition(
        theme_id=0,
        key='P',
        name='Phenomenological',
        short_name='Phenom',
        prompt_name='Phenomenological (P)',
        definition=(
            "The therapist prompts the participant to break their experience "
            "down into clear, specific, sequenced steps that another person "
            "could practically follow. The move treats subjective internal "
            "experience — pain, urges, emotions, calmness — as an observable, "
            "decomposable process rather than a global narrative. "
            "Phenomenological inquiry orients the participant's attention "
            "toward the direct sequence of sensory, affective, or cognitive "
            "events: what happened first, what happened next, what was the "
            "quality of the sensation, where were the edges. The 2018 manual "
            "frames this as the clinician acting like a mechanic asking the "
            "participant to open the hood on their experience. Therapist "
            "modeling — walking the participant through a phenomenological "
            "breakdown when the participant cannot do it themselves — is "
            "also Phenomenological, even though it is delivered as a "
            "demonstration rather than a question."
        ),
        prototypical_features=[
            'open questions eliciting step-by-step breakdown of experience',
            'chronological prompts ("what happened first," "and then what")',
            'inquiry into the sequence of sensory, affective, or cognitive events',
            'questions targeting specific sensory qualities (edges, center, intensity, color, temperature)',
            'questions about the temporal unfolding of an experience during practice',
            'therapist modeling a phenomenological breakdown using their own example',
            'inquiry that locates a specific moment within the meditation ("zoom into the part when...")',
            'questions decomposing a global report into discrete components',
        ],
        distinguishing_criteria=(
            "Phenomenological is distinguished from Education (E) by the "
            "direction of information flow: P elicits a detailed account from "
            "the participant; E delivers an explanation to the participant. "
            "Distinguished from Reframing (R) by the absence of a conceptual "
            "translation — P asks what happened, R labels what happened as an "
            "instance of a concept. Distinguished from Reinforcement (R2): "
            "even when a P question follows praise (\"Great share. Now walk "
            "me through the moment when the calm started\"), the inquiry "
            "itself is the substantive move and is coded P. Closed clinical "
            "assessment questions (\"Does your back still hurt?\") are NOT "
            "Phenomenological — they ask for a status report rather than an "
            "experiential breakdown."
        ),
        exemplar_utterances=[
            # All quotes drawn from MORE-for-LRP therapist dialogue.
            "Do you remember during the meditation when you felt the calm start to hit or became aware of calmness?",
            "Do you remember what happened next after you started to hear that and the calming happens?",
            "After you started feeling the calmness and you started to notice the breath. Do you remember a next step after that?",
            "I was wondering if maybe you could zoom into a particular part of the meditation when that was happening and tell me how that happened with you and your mind and your breath?",
            "How did the pain in your back change? And were you feeling the need to shift at all during this picture?",
            "Where did your mind go to with your attention when you're asked to notice where you're noticing?",
            "How does the body respond? You said relaxation. What else is occurring?",
            "Can you tell me in the meditation though when that was happening, was it the whole time? Was there a particular thing?",
            "When did that start for you in the meditation?",
            "I'm curious about how were you able, you know, what we were doing the practice. How are you able to see the bike and how were you able to feel it in your body, the connections and memories in the joy?",
            "Were you able to tap into that? While you're sitting there, you can feel it in your body?",
            "And did you notice when you returned any difference in the quality of discomfort in your body?",
        ],
        subtle_utterances=[
            "And then what happened in your body?",
            "What were we doing? What was the instruction when that was happening?",
            "Can you walk me through it a little bit?",
            "Where in your body you felt it when that was happening?",
            "When I savor campfires, I take a few moments of mindful breathing, and then focus on the beauty of the flames... Then I become aware of feeling peaceful and calm. (Therapist modeling phenomenology when the participant cannot break their experience down)",
        ],
        adversarial_utterances=[
            # P vs E: closed clinical assessment is not P. P asks for an
            # experiential breakdown; closed status questions ask for a
            # report against a clinical metric.
            "Does your back still hurt today? (→ closed clinical question, not P)",
            # P vs R: notice the difference between asking what happened
            # and labeling what happened. R takes the participant's
            # account and tells them what concept it instantiates.
            "What you're describing is a very good reflection of the mind being busy. (→ REFRAMING, not P)",
            # P vs E: instructions delivered to the participant are not P.
            # P invites participant report; instructions deliver therapist
            # guidance.
            "Notice how the sensation changes as you breathe into it. (→ EDUCATION/instruction, not P)",
            # P vs R2: praise alone is not P, even when followed by an
            # invitation to share. Code P only when the inquiry is the
            # substantive move.
            "That is a great share. (→ REINFORCEMENT, not P)",
        ],
        word_prototypes=[
            'walk me through', 'zoom into', 'first', 'next', 'then what',
            'do you remember', 'can you tell me', 'what was happening',
            'where did your', 'where in your body', 'what was the instruction',
            'when did that', 'how did', 'what did you experience',
            'what did you do in your mind', 'did you notice',
            'tell me about', 'describe', 'walk me', 'edges', 'center',
            'change in quality',
        ],
        color='#5B8DB8',
        aliases=['phenomenology', 'experiential breakdown', 'step-by-step inquiry'],
    ),

    # =========================================================================
    # U — Utilization
    # =========================================================================
    ThemeDefinition(
        theme_id=1,
        key='U',
        name='Utilization',
        short_name='Utilize',
        prompt_name='Utilization (U)',
        definition=(
            "The therapist helps the participant identify how a specific "
            "experience, insight, or learned skill can be applied as a "
            "practical coping tool in everyday life. Utilization is the "
            "bridge between in-session realization and real-world habit "
            "formation. The move is forward-looking: it asks the participant "
            "to imagine future moments — triggers, stressors, pain flares, "
            "interpersonal situations — in which the just-experienced skill "
            "could be deployed. The 2018 manual emphasizes that Utilization "
            "is patient-generated: the therapist prompts the participant to "
            "identify the application rather than instructing them on when "
            "to use it. Classic Utilization moves include linking mindfulness "
            "to specific daily triggers, prompting reflection on how "
            "in-session changes might generalize, and asking the participant "
            "to anticipate how a skill could be deployed before a known "
            "stressor."
        ),
        prototypical_features=[
            'forward-looking questions about how a skill can be deployed in daily life',
            'explicit links between an in-session experience and specific future situations',
            'prompts asking the participant to identify their own application',
            'questions about how a skill could enhance traditional pain management',
            'inquiry about generalization across contexts ("outside the session," "tomorrow")',
            'reflection prompts that compare in-session change to everyday function',
            'invitations to imagine deploying the skill before an anticipated stressor',
        ],
        distinguishing_criteria=(
            "Utilization is distinguished from Reinforcement (R2) by its "
            "future orientation — U asks how a success will be deployed "
            "next time, while R2 affirms a success that has already "
            "happened. Distinguished from Education (E) because U asks "
            "the participant to generate the application rather than "
            "instructing the participant on when to use the skill. "
            "Distinguished from Reframing (R): R repositions a past "
            "experience as an instance of a concept; U asks the participant "
            "to imagine using the experience or insight forward in time. "
            "Per the precedence note, when an utterance both reframes a "
            "past experience AND prompts forward application, U is the "
            "substantive move."
        ),
        exemplar_utterances=[
            "I want to take some time today, as we are debriefing that experience, to reflect on how things have changed over the course of these 8 weeks. How does that compare to some of the earlier experiences you had in the program?",
            "What can you imagine would be a way that that would change your ability to relate to your life that you could carry the peaceful experience with you and other parts of your life? How do you think that may change your interactions, your day to day habits?",
            "How can you use what you learned during this experience to help yourself in future moments?",
            "How are you going to use mindfulness to enhance your traditional pain management strategies?",
            "What will you do, specifically, the next time one of these triggering situations comes up?",
            "When you're attentive to the area around it and when you're feeling it the tightest, the question's going to be then, how do you bring that same quality to that time?",
        ],
        subtle_utterances=[
            "Where else in your week could you apply that?",
            "Think about how you might use that 3-minute breathing space before taking your medication.",
            "How does that compare to some of the earlier experiences you had?",
        ],
        adversarial_utterances=[
            # U vs R2: praise of past action is reinforcement. U asks
            # about future deployment.
            "It's great that you used the breathing technique yesterday. (→ REINFORCEMENT, not U)",
            # U vs E: instruction or homework assignment is education,
            # not utilization. U asks the participant to generate the
            # application themselves.
            "You should practice this before you take your medication. (→ EDUCATION/instruction, not U)",
            # U vs R: framing a current experience as practicing
            # mindfulness is reframing. U asks about the future.
            "That is practicing mindfulness and applying it. (→ REFRAMING, not U)",
            # U vs P: phenomenological inquiry into a present or past
            # experience is not utilization, even when the participant
            # later applies the insight.
            "What did that calmness feel like in your body? (→ PHENOMENOLOGICAL, not U)",
        ],
        word_prototypes=[
            'how can you use', 'in future moments', 'in everyday life',
            'apply', 'enhance', 'next time', 'tomorrow',
            'triggering situations', 'how do you bring',
            'compare to earlier', 'carry that with you',
            'in other parts of your life', 'going forward',
            'before you take', 'imagine yourself',
        ],
        color='#8DA87E',
        aliases=['application prompt', 'future coping', 'real-world utilization'],
    ),

    # =========================================================================
    # R — Reframing
    # =========================================================================
    ThemeDefinition(
        theme_id=2,
        key='R',
        name='Reframing',
        short_name='Reframe',
        prompt_name='Reframing (R)',
        definition=(
            "The therapist transforms the participant's organic report "
            "into a direct example of a psychological concept being taught "
            "in MORE. Reframing connects the participant's specific narrative "
            "to the curriculum: difficulty becomes practice, distraction "
            "becomes mindfulness in action, an urge becomes a sensation, "
            "frustration becomes the second arrow. The 2018 manual notes "
            "that participants will often share stories that feel unrelated "
            "or entirely negative; Reframing uses these moments as teaching "
            "tools without invalidating the experience. The change is in "
            "meaning — the same event is repositioned through a MORE-relevant "
            "lens. Common reframes include: normalizing mind-wandering as "
            "the nature of mind, normalizing struggle as the practice of "
            "mindfulness, separating sensation from suffering, and recasting "
            "challenge versus struggle. Reframing is always anchored to the "
            "participant's just-shared story — that anchoring is what "
            "distinguishes it from generalized education."
        ),
        prototypical_features=[
            'translating a participant complaint into a normal mechanism of mind/body',
            'normalizing struggle, distraction, or difficulty as the practice of mindfulness',
            'identifying the participant\'s story as an instance of a specific MORE concept',
            'separating hurt from harm, sensation from suffering, urge from action',
            'recasting "failure" at meditation as success at mindfulness',
            'reframing difficulty as a teacher, challenge, or growth opportunity',
            'using the participant\'s organic vocabulary as a hook for a concept',
        ],
        distinguishing_criteria=(
            "Reframing is distinguished from Education (E) because it is "
            "explicitly anchored to the participant's immediate story. E "
            "delivers a concept generally; R takes what the participant just "
            "said and labels it as the concept. Distinguished from "
            "Reinforcement (R2) because R provides a new cognitive lens or "
            "translation, not just affirmation of the action; however, R "
            "and R2 frequently co-occur (\"That's still you doing it — "
            "that's what the practice IS\" is both reframing distraction as "
            "mindfulness AND reinforcing the participant's effort). "
            "Distinguished from Phenomenological (P) by the direction of "
            "movement: P invites the participant to elaborate a description; "
            "R consolidates what they have already described under a "
            "concept."
        ),
        exemplar_utterances=[
            # All drawn from MORE-for-LRP therapist dialogue.
            "From what you're describing to me what I'm hearing, it sounds like you're doing the practice. That was practicing mindfulness in this way. It's like crossing a threshold into that space.",
            "What you're describing is a very good reflection of the mind being busy. Because it hears sounds. You said your child laughing and remembering tasks from work.",
            "That is a great share. That is practicing mindfulness and applying it.",
            "Well, that's practicing mindfulness. For sure. That's if you're aware that you were getting distracted.",
            "It's not doing it perfectly. It's the mind get distracted and then you have to return and you have to sometimes move because it's painful. And the new return.",
            "That's still you doing it. But that's really positive. I mean, that's great. That's what the practice is. The mind is in a drift and then you're going to bring it back in. (also reinforces)",
            "It's a challenge not a struggle. The challenge is a climb. A struggle is a flail.",
            "That's such a normal experience. There's a lot of techniques taught on how to work with the mind when it's overstimulated.",
            "So you found that every time you went back to observe the pain, there was a little bit less there to notice?",
        ],
        subtle_utterances=[
            "So the fact that you noticed you were distracted actually means you were being mindful.",
            "It's not the same as taking pain medication — what we're doing is helping you reevaluate it.",
            "That feeling of wanting to push the pain away is exactly what we've been talking about.",
        ],
        adversarial_utterances=[
            # R vs E: a general mechanism delivered without anchoring to
            # the participant's specific story is education, not reframing.
            "The brain has a negativity bias that makes pain feel worse under stress. (→ EDUCATION, not R)",
            # R vs R2: warm validation without conceptual translation is
            # reinforcement only.
            "That sounds really hard to deal with. (→ REINFORCEMENT/empathic, not R)",
            "Great job noticing that. (→ REINFORCEMENT, not R)",
            # R vs P: a question seeking elaboration is phenomenological.
            # R consolidates rather than expands.
            "Can you tell me more about what that felt like? (→ PHENOMENOLOGICAL, not R)",
        ],
        word_prototypes=[
            "that's practicing mindfulness", "that is practicing mindfulness",
            "what you're describing", "that's still you doing it",
            "that is what we call", "that's exactly", "what we mean by",
            "perfect example", "instance of", "that's mindfulness",
            "challenge not a struggle", "that's the practice",
            "reflection of the mind", "applying it", "doing the practice",
            "crossing a threshold", "actually means",
        ],
        color='#C97B3A',
        aliases=['conceptual translation', 'normalizing', 'concept mapping'],
    ),

    # =========================================================================
    # E — Educate and Build Expectancy
    # =========================================================================
    ThemeDefinition(
        theme_id=3,
        key='E',
        name='Educate and Build Expectancy',
        short_name='Educate',
        prompt_name='Educate and Build Expectancy (E)',
        definition=(
            "The therapist provides psychoeducational information about "
            "pain, stress, mindfulness, or contemplative practice, AND/OR "
            "imparts an explicit expectation of therapeutic benefit. The "
            "2018 manual frames Education as a combination of cognitive "
            "psychoeducation and the installation of hope. Patients with "
            "chronic pain often feel broken or helpless; explaining the "
            "neurobiology of pain, attention, and stress demystifies "
            "suffering, while building expectancy gives the participant "
            "confidence that consistent practice will yield benefit through "
            "neuroplasticity. Common Education moves include: explaining "
            "pain as comprising nociception, pain perception, suffering, "
            "and pain behavior; describing the thalamus or pain gate; "
            "introducing the hurt-versus-harm distinction; presenting the "
            "two-arrows metaphor; comparing mindfulness practice to "
            "physical exercise (\"curling a dumbbell\"); and explaining "
            "that meditation can bring up strong emotions. Expectancy-"
            "building includes statements that practice will become easier, "
            "that benefit will accumulate, and that effort will pay off."
        ),
        prototypical_features=[
            'didactic explanation of pain, stress, attention, or neurobiology',
            'introduction of MORE-curriculum metaphors (pain gate, two arrows, dumbbell)',
            'statements explicitly linking practice to expected future benefit',
            'authoritative delivery of generalized knowledge about practice',
            'explanation of mind-wandering, mind-states, or contemplative phenomena as universal',
            'preparation of participants for emotional or somatic responses to practice',
            'instruction on what to expect from a particular skill or technique',
            'explicit hope-installation language ("with practice this will get easier")',
        ],
        distinguishing_criteria=(
            "Education is distinguished from Reframing (R) by its generality "
            "— E is delivered as a rule or mechanism of human experience, "
            "while R is anchored to the participant's specific story. "
            "The presence of explicit expectancy-building language (\"with "
            "practice...\", \"over time...\", \"you will start to notice...\") "
            "is a strong marker of E. Distinguished from Phenomenological (P) "
            "by direction: E delivers information to the participant, P "
            "elicits information from the participant. Distinguished from "
            "Reinforcement (R2) because E teaches a concept rather than "
            "praising an action, though E and R2 may co-occur (e.g., "
            "explaining the dumbbell metaphor while praising the participant's "
            "effort)."
        ),
        exemplar_utterances=[
            # All drawn from MORE-for-LRP therapist dialogue or directly
            # from the 2018 manual where therapists deploy them verbatim.
            "I would say that the way that we're doing mindfulness here is to help us reevaluate it, but it won't necessarily make the pain magically disappear. It's not the same as taking pain medication.",
            "Suzuki Roshi said that you're only practicing meditation if you can do it while sitting under a highway... because our minds and our distraction, that distractibility can be very much like that.",
            "It is really important for all of us to recognize one thing for mindfulness and meditation practice: these practices can bring up really strong emotional experiences. Group setting allows us to tap into things which, alone, might feel even more overwhelming.",
            "Consider that stream of attention, because we've got a practice of retrospective mindfulness, and it helps you to continue the stream of mindfulness training — to go back and ask when did that change, was I awake, was I present, when did I fall into distraction, and when did I return?",
            "An idea that may be helpful for you in terms of struggling with the idea of sleep and drowsiness... the nature of that spaciousness is described as clear, as luminous, and as non-conceptual. Entering that non-conceptual space can feel like we're not awake the way we're normally awake.",
            "Every time you bring your mind back, it is like lifting weights. You are making your mind stronger.",
            "Pain and suffering are like getting hit by two arrows.",
            "Because mindfulness changes the way the brain functions, and the brain is what processes pain, mindfulness can actually decrease pain.",
            "Hurt doesn't necessarily mean that your body has been harmed.",
            "As you start to practice this, you will begin to notice it gets easier and easier over time.",
        ],
        subtle_utterances=[
            "It might feel awkward at first, but with time it becomes automatic, just like tying your shoes.",
            "The thalamus acts like a pain gate that turns the volume of pain up or down.",
            "When people use active coping strategies, they are five times less likely to develop disabling pain.",
        ],
        adversarial_utterances=[
            # E vs R: a concept anchored to the participant's specific
            # story is reframing, not generalized education.
            "Your story about the campfire is a great example of savoring. (→ REFRAMING, not E)",
            # E vs P: a question eliciting experiential detail is
            # phenomenological.
            "Where did you feel the tension in your body? (→ PHENOMENOLOGICAL, not E)",
            # E vs R2: pure logistic instruction without a theoretical or
            # mechanistic component is not E. E carries either explanatory
            # or expectancy content.
            "Please practice the body scan for 15 minutes daily. (→ logistic instruction, not E)",
            # E vs U: education tells the participant about the mechanism;
            # U asks the participant to identify their own application.
            "How might you use mindfulness when stress comes up tomorrow? (→ UTILIZATION, not E)",
        ],
        word_prototypes=[
            'brain', 'nervous system', 'pain gate', 'thalamus', 'two arrows',
            'hurt versus harm', 'dumbbell', 'curling a dumbbell',
            'research shows', 'over time', 'easier and easier',
            'with practice', 'you will start to notice', 'expectation',
            'neuroplasticity', 'is the nature of', 'these practices can',
            'going to be', 'is not the same as', 'the way we',
        ],
        color='#9B6BA8',
        aliases=['psychoeducation', 'expectancy building', 'didactic mechanism'],
    ),

    # =========================================================================
    # R2 — Reinforcement
    # =========================================================================
    ThemeDefinition(
        theme_id=4,
        key='R2',
        name='Reinforcement',
        short_name='Reinforce',
        prompt_name='Reinforcement (R2)',
        definition=(
            "The therapist actively encourages and supports the participant's "
            "practice using selective positive reinforcement. The 2018 manual "
            "explicitly instructs clinicians to use selective attention and "
            "ignoring to shape discussions: praise and validate adaptive "
            "behaviors, insights, and active coping; gently ignore or pivot "
            "from language that fuels catastrophizing, victimhood, or passive "
            "coping. Reinforcement also includes bolstering the participant's "
            "identity as someone capable of healing — affirmations of their "
            "courage, strength, fortitude, and capacity. The move is "
            "fundamentally backward-looking and validating: it consolidates "
            "behavioral change by making the just-completed adaptive action "
            "salient. Reinforcement frequently co-occurs with Reframing (a "
            "single therapist turn often both labels the participant's act "
            "as practice AND praises them for it), and the boundary between "
            "the two is whether substantive conceptual translation is "
            "present."
        ),
        prototypical_features=[
            'explicit praise or validation of practice or insight',
            'affirmation of the participant\'s courage, effort, or fortitude',
            'reinforcement of identity as a capable practitioner',
            'selectively highlighting adaptive moves over passive ones',
            'warm, affirming register designed to consolidate behavioral change',
            'gentle ignoring or pivoting from maladaptive narratives',
            'quick verbal rewards ("yes, exactly," "great share") that reinforce a specific moment',
        ],
        distinguishing_criteria=(
            "Reinforcement is distinguished from Reframing (R) because it "
            "affirms an action rather than translating its meaning. R adds "
            "a conceptual lens; R2 simply consolidates the act. Distinguished "
            "from Utilization (U) because R2 looks backward at a success "
            "that just happened, while U asks the participant to imagine "
            "deploying it forward in time. Distinguished from Education (E) "
            "because R2 praises the participant rather than teaching them. "
            "Per the precedence note, when R2 is the affective wrapper "
            "around another move (P, U, R, or E), the substantive move is "
            "coded; pure R2 is reserved for utterances where affirmation is "
            "the entire act."
        ),
        exemplar_utterances=[
            # All drawn from MORE-for-LRP therapist dialogue.
            "That is a great share.",
            "Well, that's practicing mindfulness. For sure. (also reframes)",
            "Great. That's practicing mindfulness. Thank you very much. Great job. (also reframes)",
            "No, that's still you doing it. But that's really positive. I mean, that's great. That's what the practice is. (also reframes)",
            "That sounds empowering. As somebody who really is stomach oriented and enjoys good food, that sounds very empowering to me.",
            "That's amazing for you to be able to do that.",
            "It sounds like you really embodied that idea that we did in the practice of considering why you're motivated.",
            "Can you talk about joy? I would want to hear more of that. I love the memory share because that's part of the savoring, right?",
            "I admire you for your courage and strength. You have what it takes to sustain these changes.",
        ],
        subtle_utterances=[
            "Mm-hmm, yes, that is exactly right.",
            "Great share.",
            "That's beautiful.",
            "Yeah, that's the application.",
        ],
        adversarial_utterances=[
            # R2 vs U: forward-application prompts are utilization, not
            # reinforcement.
            "How can you use that success tomorrow? (→ UTILIZATION, not R2)",
            # R2 vs E: explanation of mechanism is education, even when
            # it follows praise.
            "That means your parasympathetic nervous system was activating. (→ EDUCATION, not R2)",
            # R2 vs P: questions eliciting experiential detail are
            # phenomenological, not reinforcement, even when warmly framed.
            "What did that success feel like in your body? (→ PHENOMENOLOGICAL, not R2)",
            # R2 vs R: a label that translates the participant's act into
            # a concept is reframing, even if also affirming. The
            # discriminator is whether substantive conceptual translation
            # is present. When in doubt, code the substantive move (R).
            "That is practicing mindfulness and applying it. (→ REFRAMING with R2 wrapper; primary code R)",
        ],
        word_prototypes=[
            'great share', 'great job', 'fantastic', 'that is great',
            "that's great", "that's really positive", 'wonderful',
            'beautiful', 'admire', 'courage', 'strength', 'fortitude',
            'amazing', 'empowering', "that's amazing", 'proud of you',
            "you've got it", 'exactly right', 'exactly', 'thank you for',
            'i love that', 'huge step', 'really proud',
        ],
        color='#D4A853',
        aliases=['selective reinforcement', 'validation', 'behavior shaping'],
    ),
]


def get_purer_framework() -> ThemeFramework:
    """Return the PURER therapist-dialogue classification framework."""
    return ThemeFramework(
        name='PURER',
        version='3.0',
        description=(
            "Five therapist guided-inquiry constructs for Mindfulness-"
            "Oriented Recovery Enhancement (MORE) sessions, refined to "
            "match the 2018 MORE Manual (Garland) and validated against "
            "the qualitative dataset of Wexler, Balsamo et al. (2026, "
            "Mindfulness, 17, 819-833). Classifies therapist segments "
            "only. P=Phenomenological, U=Utilization, R=Reframing, "
            "E=Educate/Expectancy, R2=Reinforcement. PURER moves "
            "frequently co-occur within a single therapist turn; when "
            "the analytic unit permits one label, the precedence is: "
            "R2 wraps the substantive move (P/U/R/E); U > R when forward "
            "application is being prompted; R > E when the concept is "
            "anchored to the participant's specific story. See module "
            "docstring for full precedence rules and worked examples."
        ),
        themes=_PURER_DEFINITIONS,
    )