---
codebook: microcounseling
version: "1.0"
codebook_description: >
  Therapist-side microcounseling-skills codebook for the Qualitative Research
  Algorithm (QRA). These eight codes describe concrete, surface-observable
  therapist behaviours and form the granular, multi-label *behavioural* sub-layer
  of the PURER therapist framework — the structural twin of how the Varieties of
  Contemplative Experience (VCE) codebook is the content sub-layer of the
  participant-side VAAMR framework. A single therapist turn typically receives a
  PURER move label (its rhetorical function) AND one or more microcounseling
  codes (the discrete skills used to execute it). The codes are grounded in the
  microcounseling tradition (Ivey) and motivational interviewing (Miller &
  Rollnick); they are authored independently of any external dataset and are
  intended for cross-validation and sub-typing of PURER moves, not as a competing
  primary framework. Codes are observable from the therapist's language alone and
  make no inference about the participant's internal state.
---

<!--
  Format contract (parsed by codebook/markdown_loader.load_codebook_md):
    - YAML frontmatter keys: codebook, version, codebook_description
    - One block per code, opened by:  ## Code N — code_id — Category Name
      (em-dash " — " separated; code_id contains no spaces)
    - A ```yaml fenced block with: code_id, category, domain
    - H3 sections: ### Description / ### Inclusive Criteria /
      ### Exclusive Criteria / ### Exemplar Utterances
    - Exemplars are "> " blockquotes; SEPARATE each exemplar with a blank line so
      the parser (_parse_blockquotes) emits them as distinct items.
  This mirrors PHENOMENOLOGY_CODEBOOK.md so no loader change is required.
-->

## Code 1 — reflective_listening — Reflective Listening

```yaml
code_id: reflective_listening
category: Reflective Listening
domain: Reflective
```

### Description

The therapist mirrors, paraphrases, or distills the content or feeling of what the
participant has just expressed, demonstrating accurate understanding and inviting
the participant to continue or correct. Reflections may be simple (restating the
surface content) or complex (naming the implied feeling, meaning, or the two sides
of an ambivalence). The defining feature is that the therapist's utterance is a
statement *about the participant's own expression*, offered back to them, rather
than a new question, evaluation, or piece of information.

### Inclusive Criteria

Apply when the therapist restates, paraphrases, summarizes, or names the feeling
or meaning of the participant's preceding contribution; when the therapist offers
a "so what I'm hearing is…", "it sounds like…", or a declarative mirror of the
participant's words or affect; when the therapist consolidates several participant
statements into a brief summary that hands the thread back to the participant.

### Exclusive Criteria

Do not apply when the utterance is primarily a question seeking new information
(code Open-ended Question), an affirmation of the participant's strength or effort
(code Affirmation), an explicit endorsement that the participant's reaction is
understandable or legitimate (code Validation), or the delivery of new conceptual
content (code Neutral / psychoeducation). A reflection that mainly evaluates or
praises is Affirmation, not Reflective Listening.

### Exemplar Utterances

> So it sounds like the moment you noticed the tightening in your back, your whole attention went straight to it and stayed there.

> What I'm hearing is that part of you wants to keep moving, and another part is bracing for the pain to get worse.

> Let me make sure I've got this — you tried staying with the sensation, and it shifted from one solid block into something more like waves.

> You're saying the breathing helped, but only once you stopped trying to make the pain go away with it.

---

## Code 2 — validation — Validation

```yaml
code_id: validation
category: Validation
domain: Reflective
```

### Description

The therapist explicitly communicates that the participant's experience, reaction,
or difficulty is understandable, legitimate, or makes sense given their situation.
Validation normalizes the experience and reduces the sense that the participant is
doing it wrong or is alone in it. It goes beyond reflecting *what* was said to
endorsing that the experience is a reasonable or expectable one.

### Inclusive Criteria

Apply when the therapist states or strongly implies that a reaction is normal,
understandable, expected, or shared by others ("that makes complete sense", "of
course that's hard", "most people in chronic pain do exactly that"); when the
therapist legitimizes a struggle or an emotion rather than treating it as a problem
to be corrected.

### Exclusive Criteria

Do not apply when the therapist merely paraphrases without endorsing legitimacy
(code Reflective Listening), praises a specific effort or character strength (code
Affirmation), or explains the mechanism behind why something happens as teaching
content (code Neutral). Validation targets the *legitimacy* of the experience, not
the participant's *agency or effort*.

### Exemplar Utterances

> It makes total sense that you'd want to avoid the movement — that's what the body does to protect itself when it expects pain.

> Of course that was frustrating. Anyone who'd been told to "just relax" for ten years would feel that.

> What you're describing is incredibly common in this work; you're not failing at it.

> Feeling discouraged after a tough week is a completely reasonable response, not a setback in your practice.

---

## Code 3 — affirmation — Affirmation

```yaml
code_id: affirmation
category: Affirmation
domain: Affirmative
```

### Description

The therapist recognizes, appreciates, or reinforces a specific strength, effort,
intention, or constructive step taken by the participant. Affirmation is directed
at the participant's *agency* — what they did, tried, or are capable of — and
functions to build confidence and consolidate progress.

### Inclusive Criteria

Apply when the therapist names and appreciates a concrete effort, choice, value,
or capability ("it took real courage to stay with that", "you've been consistent
with the practice", "that's a strength you can lean on"); when the therapist
reinforces an insight or behaviour the participant has just demonstrated.

### Exclusive Criteria

Do not apply when the therapist endorses the legitimacy of a difficulty without
crediting the participant's agency (code Validation), reflects content back without
appreciation (code Reflective Listening), or gives generic conversational
acknowledgement with no specific strength named (code Neutral).

### Exemplar Utterances

> You noticed the urge to push the pain away and you didn't act on it — that's a real skill you're building.

> You showed up every week even when it was hard. That persistence is going to serve you.

> The way you described that just now shows how closely you've started paying attention to your own experience.

> That was a brave thing to try with such a sensitive area of your back.

---

## Code 4 — genuineness — Genuineness

```yaml
code_id: genuineness
category: Genuineness
domain: Relational
```

### Description

The therapist responds in a personally authentic, congruent, transparent way —
sharing a genuine reaction, being honest about uncertainty, or speaking from their
own perspective in a warm and non-scripted manner. Genuineness conveys a real human
presence rather than a purely technical or formulaic stance.

### Inclusive Criteria

Apply when the therapist offers an authentic personal reaction or self-disclosure
in service of the relationship ("I'm genuinely moved by what you just shared"),
admits uncertainty or limits honestly ("I don't have a clean answer for that"), or
speaks transparently about the process in a warm, congruent register.

### Exclusive Criteria

Do not apply to routine empathic reflections (code Reflective Listening) or to
warmth expressed purely as endorsement of legitimacy (code Validation). Genuineness
is marked by the therapist's *own* authentic presence/disclosure, not by mirroring
or normalizing the participant.

### Exemplar Utterances

> I'll be honest — when you described that moment of relief, it gave me chills.

> I don't have a tidy answer for why it comes back some days and not others, and I don't want to pretend I do.

> Speaking personally, that's something I still have to work at in my own practice too.

> I really mean it when I say I look forward to hearing how your week went.

---

## Code 5 — respect_for_autonomy — Respect for Autonomy

```yaml
code_id: respect_for_autonomy
category: Respect for Autonomy
domain: Autonomy-Supportive
```

### Description

The therapist explicitly affirms the participant's freedom of choice, control, and
self-direction — emphasizing that decisions about practice, pace, and goals belong
to the participant. This supports the autonomy need and counters a directive or
coercive stance.

### Inclusive Criteria

Apply when the therapist underscores the participant's choice or control ("it's
completely up to you", "you're the expert on your own body", "you can take this at
whatever pace feels right", "there's no wrong way to do this"); when the therapist
offers options rather than instructions and locates the decision with the
participant.

### Exclusive Criteria

Do not apply when the therapist asks for consent to proceed or to offer something
(code Asking for Permission) — that is a distinct, narrower move. Do not apply to
generic encouragement of effort (code Affirmation) or to delivery of neutral
instructions without an autonomy emphasis (code Neutral).

### Exemplar Utterances

> This is your practice — you get to decide how far into the stretch feels right today.

> There's no wrong way to do this; whatever you notice is the right thing to notice.

> You know your body better than anyone in this room, so trust what it's telling you.

> Whether you use the breath or the body scan tonight is entirely your call.

---

## Code 6 — asking_permission — Asking for Permission

```yaml
code_id: asking_permission
category: Asking for Permission
domain: Autonomy-Supportive
```

### Description

The therapist asks the participant's consent before offering information, advice, a
suggestion, or a shift in activity. Asking permission is a specific, bounded move
that hands a momentary choice to the participant before the therapist proceeds.

### Inclusive Criteria

Apply when the therapist requests consent to share, suggest, or transition ("would
it be okay if I offered an idea?", "can I share something that might help?", "are
you open to trying something?", "is it alright if we move to the practice now?").

### Exclusive Criteria

Do not apply to broad statements affirming the participant's general control (code
Respect for Autonomy) — that is a stance, whereas Asking for Permission is a
discrete consent request. Do not apply to open-ended exploratory questions seeking
the participant's experience (code Open-ended Question).

### Exemplar Utterances

> Would it be okay if I offered one way of thinking about that?

> Can I share something that other people have found useful here?

> Are you open to trying a short experiment with the sensation right now?

> Is it alright if we pause the discussion and move into the practice?

---

## Code 7 — open_ended_question — Open-ended Question

```yaml
code_id: open_ended_question
category: Open-ended Question
domain: Inquiry
```

### Description

The therapist poses a question that invites an expansive, descriptive response and
opens space for the participant to explore their experience, rather than a question
answerable with yes/no or a single fact. In MORE this is the engine of
phenomenological inquiry — "what did you notice?" style prompts.

### Inclusive Criteria

Apply when the therapist asks a question beginning with or functioning as what /
how / what was it like / tell me about / describe; when the question invites the
participant to elaborate on their experience, meaning, or process.

### Exclusive Criteria

Do not apply to closed questions seeking consent (code Asking for Permission) or a
single yes/no/factual answer. Do not apply to rhetorical questions that are really
teaching statements (code Neutral). A question that only confirms a reflection back
("…is that right?") is Reflective Listening, not an Open-ended Question.

### Exemplar Utterances

> What did you notice in your body as you stayed with that sensation?

> How was that different from the way you usually relate to the pain?

> Tell me more about what "letting it be there" was actually like for you.

> What changed, if anything, between the start of the practice and the end?

---

## Code 8 — neutral — Neutral

```yaml
code_id: neutral
category: Neutral
domain: Neutral
```

### Description

Therapist speech that does not enact a distinct relational microskill: logistical
or procedural talk, neutral acknowledgements, transitions, and the delivery of
didactic / psychoeducational content (explanations, instructions, guided-meditation
scripting). This is the residual category that prevents over-coding ordinary or
content-delivery speech as a relational skill.

### Inclusive Criteria

Apply when the therapist gives logistics or housekeeping ("we'll start at the top
of the hour"), neutral back-channels ("okay", "mm-hm", "right"), procedural
transitions, or delivers conceptual/educational content and instructions without an
embedded reflective, affirming, autonomy-supportive, or inquiring microskill.

### Exclusive Criteria

Do not apply when any of the other seven codes is clearly present in the utterance;
those take precedence. Neutral is the fallback only when no relational microskill is
expressed. A single turn may carry both a Neutral/teaching segment and a microskill;
code each skill that is present.

### Exemplar Utterances

> Pain and harm aren't the same thing — hurt doesn't always mean damage is happening.

> Today we're going to work through the body scan and then debrief as a group.

> Go ahead and find a comfortable position, and when you're ready, let the eyes close.

> Okay. Let's come back to that in a moment.
