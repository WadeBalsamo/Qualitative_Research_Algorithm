# PURER Framework — Implementation Specifications

**PURER**: Phenomenology · Utilization · Reframing · Education/Expectancy · Reinforcement

This document specifies the PURER therapist-dialogue classification system: the full construct definitions, the code architecture, every file that must be created or modified, the wizard rewiring, and the cross-validation analysis design. It is a specification document — no code is changed here.

---

## 1. Why PURER

QRA currently treats therapist dialogue as read-only context. Therapist segments are created as `Segment` objects with `speaker='therapist'`, interleaved back into the session timeline by timestamp, and then filtered out before VAMMR classification. They are stored in `master_segments.jsonl` with all classification fields null. They serve one purpose: providing conversational context for the LLM when it classifies adjacent participant segments.

This is a waste. The therapist cues *are* the intervention mechanism. The transition reports already surface raw therapist text between stage transitions — but they cannot answer the structural question: *which kinds of therapist moves precede which participant stage transitions?* That question requires labeled therapist segments.

PURER operationalizes the five therapist guided-inquiry moves in Mindfulness-Oriented Recovery Enhancement (MORE) sessions:

| Code | Construct | Core action |
|------|-----------|-------------|
| **P** | Phenomenology | Elicit direct sensory/experiential description |
| **U** | Utilization | Reflect participant's own resources back as capacity |
| **R** | Reframing | Introduce alternative interpretive frame |
| **E** | Education/Expectancy | Deliver psychoeducation or set practice expectations |
| **R** | Reinforcement | Validate or affirm just-completed insight or skill |

Adding PURER classification produces:

1. A labeled therapist corpus for supervised fine-tuning (Phase 2 autoresearch)
2. PURER × VAMMR influence tables — empirical lift measures of which therapist moves precede which participant stage transitions
3. Per-session therapist fidelity profiles showing PURER move frequency and distribution
4. The training signal for the CFiCS-style therapist-participant interaction GNN (Phase 3)

---

## 2. PURER Construct Definitions

These are the `ThemeDefinition` specifications for `theme_framework/purer.py`. Each construct maps directly to a `ThemeDefinition` object using the schema defined in `theme_framework/theme_schema.py`. The `theme_id` integers 0–4 are the PURER stage IDs, analogous to VAMMR stage IDs 0–4 but in a separate namespace (stored in `purer_*` Segment fields, never conflated with VAMMR fields).

---

### P — Phenomenology (`theme_id=0`, `key='P'`)

**Definition**

The therapist prompts the participant to describe, attend to, or elaborate their present-moment sensory, affective, or felt experience with specificity. The move orients the participant's attention toward the direct texture of experience rather than interpretation, narrative, or evaluation. The therapist is asking the participant to slow down and notice — to report from the inside rather than explain from the outside.

**Prototypical features**

- Direct open question targeting body sensation, breath, or present-moment awareness
- Invitation to linger with or stay inside an experience rather than comment on it
- Reflective prompts that widen the phenomenological field ("What else do you notice?", "And then what happened?")
- Deliberate deceleration of the conversational pace to allow experiential reporting
- Second-person orientation toward the participant's first-person access ("What do *you* notice...")

**Distinguishing criteria**

Phenomenology is distinguished from Education (E) by the absence of new information being delivered — the therapist is not instructing but inquiring. Distinguished from Reinforcement (second R) by its forward-looking, eliciting orientation: Phenomenology opens experiential space, Reinforcement closes it with validation. Distinguished from Reframing (R) by the absence of an alternative frame being introduced — the therapist is asking what is there, not suggesting how to interpret it.

The key test: Is the therapist asking the participant to report from experience, or is the therapist providing something (information, frame, affirmation, capacity reflection)?

**Exemplar utterances**

- "What do you notice in your body right now?"
- "Can you stay with that sensation for a moment — what's there?"
- "Where do you feel that?"
- "Tell me more about what that experience was like."
- "Can you describe that in terms of physical sensations?"
- "What happens in your body when you bring attention to that area?"
- "What are you aware of as you say that?"

**Subtle utterances**

Minimal phenomenological prompts that hold experiential space without elaboration:
- "And?" (following a participant's description)
- "Stay with that."
- "What happened next — in your body?"
- A brief pause followed by: "What's here?"

**Adversarial utterances (not P)**

- "Does it still hurt?" — closed question, no phenomenological elicitation
- "Did you practice this week?" — logistical/compliance, not experiential
- "What did you think about that?" — cognitive, not somatic/experiential
- "What does that remind you of?" — associative, not present-moment
- "That's great — so you noticed the breath?" — validation + yes/no, not P

---

### U — Utilization (`theme_id=1`, `key='U'`)

**Definition**

The therapist reflects the participant's own stated experience, coping history, or demonstrated capacity back to them as evidence of an existing resource or strength. The move turns the participant's narrative into a demonstrated competency. The therapist is not teaching a new skill but naming one the participant already possesses or has already employed. The source material for Utilization is always the participant's own words or history.

**Prototypical features**

- Explicit acknowledgment of a coping strategy the participant has previously described
- Mirroring of participant language to frame an experience as skill ("You said you found a moment of quiet — that's exactly the skill")
- "You've been able to..." / "You know how to..." / "You've already..." constructions
- Linking a present difficulty to a past demonstrated success
- Making implicit competency explicit without adding new interpretive content

**Distinguishing criteria**

Utilization is distinguished from Reinforcement by temporal orientation. Reinforcement validates a *just-completed* action or insight in the current interaction. Utilization draws on *history or existing capacity* — it says "you have this" rather than "you just did this." They can occur together (acknowledging what was just done *and* noting it as a broader capacity), but the defining move is different.

Utilization is distinguished from Reframing (R) by not introducing a new interpretive lens. Utilization works within the participant's own frame and says "this is a resource." Reframing says "here is a different way to see this."

**Exemplar utterances**

- "It sounds like you've already found a way to work with that."
- "You mentioned using your breath last week — that's something you know how to do."
- "That experience you described — you drew on something real there."
- "You said you 'found a quiet moment' — that's the skill we're talking about."
- "You've been carrying this practice even when it's been hard."
- "The fact that you're here, practicing, even with the pain — that's utilization."

**Subtle utterances**

- Therapist quotes participant's exact phrasing back with an affirming inflection, implying strength
- "That sounds like exactly what you did the time you mentioned..."
- "You already know this — you described it."

**Adversarial utterances (not U)**

- Generic encouragement not anchored to participant history: "You're doing great!"
- Pure skill instruction: "Here's how you use the breath..."
- Reinforcement of a just-completed action without evoking capacity: "Nice work on that last exercise."
- Reframing that introduces a new lens, even if it incorporates participant language

---

### R — Reframing (`theme_id=2`, `key='R1'`)

**Definition**

The therapist introduces an alternative interpretive frame for the participant's experience — repositioning pain, emotion, thought, or sensation from its habitual meaning to a new relational or conceptual context. The move is conceptual: it offers the participant a way to hold or relate to their experience differently, rather than asking them to describe it (Phenomenology) or telling them a general fact about it (Education). The therapist is providing a lens.

**Prototypical features**

- Explicit recontextualization using "What if...," "Instead of...," "Maybe what's happening is..."
- Introduction of mindfulness or pain science concepts applied directly to the participant's reported experience
- Decoupling sensation from suffering narrative ("That sensation doesn't have to mean...")
- Positioning experience as information, process, or impermanent event rather than threat or fixed condition
- "Just sensation" / "just information" / "just this" constructions that reduce evaluative loading

**Distinguishing criteria**

Reframing is distinguished from Education (E) by being applied to the participant's specific experience in the current moment rather than delivered as transferable general knowledge. If the therapist says "the brain amplifies pain signals in response to threat," that is Education. If the therapist says "that sensation you just described — what if that's your nervous system doing its job rather than something going wrong?", that is Reframing.

Reframing is distinguished from Phenomenology (P) by providing rather than eliciting. Phenomenology opens inquiry; Reframing introduces a frame.

**Exemplar utterances**

- "What if that tension is just the body doing its job?"
- "Sometimes what we call pain is the mind's resistance to what's happening."
- "That feeling doesn't have to mean anything about the future."
- "Instead of fighting it, what happens if you just notice it?"
- "What you're calling suffering might be the sensation plus your relationship to it."
- "That discomfort — it's real, and it's also just information."
- "You don't have to make it go away. What if you could just be with it?"

**Subtle utterances**

- Soft conceptual openings: "I wonder if there's another way to hold what's happening."
- Minimal reframes embedded in reflections: "So the sensation is there, and it's also changing."
- Question-form reframes that don't announce themselves: "What is the sensation itself, apart from what you're telling yourself about it?"

**Adversarial utterances (not R)**

- Direct validation without recontextualization: "That sounds really hard" — this is Reinforcement or empathic reflection, not Reframing
- Phenomenology probes that do not introduce new meaning
- Education delivered generically, not applied to participant's experience
- Utilization drawing on participant history without introducing a new frame

---

### E — Education/Expectancy (`theme_id=3`, `key='E'`)

**Definition**

The therapist delivers psychoeducational content about pain neuroscience, mindfulness mechanisms, the theoretical basis of MORE, or the structure of a specific practice; or sets explicit expectations about what participants will experience during or after an exercise. The move transmits knowledge or prepares the participant cognitively/emotionally for what is about to happen. The defining characteristic is that the content is *transferable* — it would mean the same thing regardless of which participant received it.

**Prototypical features**

- Didactic explanation with hedged-but-authoritative register
- References to brain, nervous system, attention, neuroplasticity, research
- "This is normal" / "Many people find..." expectancy-setting
- Preview of the experiential structure of an upcoming exercise ("In this practice, you may notice...")
- Explanation of *why* a technique works, not just how to do it
- Reference to MORE program rationale or session curriculum content

**Distinguishing criteria**

Education is distinguished from Reframing (R) by generality. Education delivers content that applies broadly to the human pain/mindfulness system; Reframing applies a new lens to this participant's specific reported experience. The test: could this utterance be delivered word-for-word to every participant in every session? If yes, it is Education.

Education is distinguished from Reinforcement (second R) by informing rather than validating. Distinguished from Phenomenology (P) by providing information rather than eliciting experience.

**Exemplar utterances**

- "The brain has a negativity bias that tends to amplify pain signals."
- "Research shows that mindfulness can actually change the way the brain processes pain."
- "In this next exercise, you may notice the urge to resist or fix what you're feeling — that's completely normal."
- "MORE is built on the idea that we can train attention itself."
- "When we're in pain, the nervous system is often in a state of threat response — that's what we're working with."
- "This feeling of wanting to stop — it often comes up right at the edge of growth."
- "Savoring and mindfulness activate overlapping neural pathways."

**Subtle utterances**

- Brief expectancy-setting before an exercise: "You might find this challenging at first, and that's okay."
- Embedded science: "...because the brain's attention system works this way..."
- Single-sentence psychoeducation delivered mid-conversation

**Adversarial utterances (not E)**

- Reframing applied to participant's specific experience (even if it uses pain science concepts)
- Behavioral instructions without explanatory content: "Take a slow breath and notice."
- Reinforcement: "You understood that really well."
- Logistical information: "We'll meet again next Tuesday."

---

### R — Reinforcement (`theme_id=4`, `key='R2'`)

**Definition**

The therapist validates, affirms, or encourages the participant's just-completed insight, disclosure, or skill application. The move consolidates what has just occurred in the interaction — it names, acknowledges, or celebrates a specific moment of progress. The defining characteristic is immediacy: the therapist is responding to something that just happened.

**Prototypical features**

- Explicit positive acknowledgment of what the participant just described, did, or realized
- Language that names the skill or process at work ("That's mindfulness." / "That's exactly reappraisal.")
- Bridge to future practice: "You can use that." / "Take that with you."
- Celebratory, warmly affirming, or gently triumphant register
- Reflective summary that implicitly validates: "So you noticed the urge, you stayed with it, and it passed."

**Distinguishing criteria**

Reinforcement is distinguished from Utilization (U) by temporal orientation. Both involve acknowledgment, but Reinforcement closes around something *just done in this exchange*; Utilization draws on *history or existing capacity*. They can co-occur, but the primary classifiable move is distinct.

Reinforcement is distinguished from Phenomenology (P) by closing rather than opening the experiential field. Reinforcement lands; Phenomenology inquires. Reinforcement is the move that ends a sequence; Phenomenology is the move that begins one.

**Exemplar utterances**

- "That's exactly it."
- "You just did something really important."
- "Notice what you were able to do there."
- "That's the skill working — you brought your attention back."
- "That's real progress."
- "Do you feel that? That's what we're going for."
- "Yes — and you found it yourself."
- "Perfect. That's savoring."

**Subtle utterances**

- Nonverbal affirmations transcribed as text: "Mm-hmm." / "Yes." (in context, following participant's insight)
- Brief process-naming validation: "Right — you let it be there."
- Therapist silently nodding coded as a short verbal acknowledgment in the transcript

**Adversarial utterances (not R)**

- Generic social politeness not anchored to skill: "Thanks for sharing."
- Utilization invoking prior history rather than present accomplishment
- Phenomenology probes that follow validation but become the primary move
- Education delivered *after* a participant's disclosure about what just happened

---

## 3. Architecture Overview

### Current pipeline flow (therapist segments)

```
transcript_ingestion.py
  └── _filter_sentences_by_speaker()     — strips therapist sentences before embedding
  └── extract_therapist_segments()        — creates Segment(speaker='therapist') objects

orchestrator.py
  └── Stage 2: Semantic Segmentation     — participant sentences only
  └── all_segments = merge + interleave  — therapist segs reinserted by timestamp (lines 288–300)
  └── Stage 3: VAMMR Classification
        └── _apply_speaker_filter()       — filters OUT therapist segs (line 387)
        └── classify_segments_zero_shot() — participant segs only
        └── parse_all_results()           — VAMMR fields populated on participant segs
  └── Stage 3b: VCE Codebook             — participant segs only
  └── master_segments.jsonl              — therapist segs stored with all classification fields = None
```

### Target flow with PURER

```
orchestrator.py (modified)
  └── Stage 3: VAMMR (unchanged)
        └── _apply_speaker_filter() — still excludes therapists from VAMMR
        └── parse_all_results()     — VAMMR fields on participant segs
  └── Stage 3c: PURER (NEW — between Stage 3 and Stage 3b)
        └── therapist_segs = [s for s in all_segments if s.speaker == 'therapist']
        └── classify_segments_zero_shot(therapist_segs, purer_framework, purer_config)
        └── parse_purer_results()   — purer_* fields on therapist segs
  └── Stage 3b: VCE Codebook (unchanged)
  └── master_segments.jsonl — therapist segs now have purer_* fields populated
```

The key architectural insight: `speaker_filter.mode='exclude'` is *correct and should not change*. It means "exclude from VAMMR classification." PURER is a parallel classification path, not a replacement. The `_apply_speaker_filter()` call in Stage 3 selects participant segments; Stage 3c explicitly selects therapist segments. They are two separate classification passes on two disjoint sets of segments.

---

## 4. Code Specifications

### 4.1 `theme_framework/purer.py` — NEW FILE

A module structurally parallel to `theme_framework/vammr.py`. Exports a single function `get_purer_framework()` that returns a `ThemeFramework` containing five `ThemeDefinition` objects, one per PURER construct.

**Module structure:**

```python
from theme_framework.theme_schema import ThemeDefinition, ThemeFramework

_PURER_DEFINITIONS = [
    ThemeDefinition(
        theme_id=0,
        key='P',
        name='Phenomenology',
        short_name='Phenom',
        prompt_name='Phenomenology (P)',
        definition='...',
        prototypical_features='...',
        distinguishing_criteria='...',
        exemplar_utterances=[...],
        subtle_utterances=[...],
        adversarial_utterances=[...],
        word_prototypes=['notice', 'feel', 'body', 'sensation', 'aware', 'describe', 'what do you notice'],
        color='#5B8DB8',
    ),
    ThemeDefinition(theme_id=1, key='U', name='Utilization', ...),
    ThemeDefinition(theme_id=2, key='R', name='Reframing', short_name='Reframe', ...),
    ThemeDefinition(theme_id=3, key='E', name='Education/Expectancy', short_name='Education', ...),
    ThemeDefinition(theme_id=4, key='R2', name='Reinforcement', short_name='Reinforce', ...),
]

def get_purer_framework() -> ThemeFramework:
    return ThemeFramework(
        name='PURER',
        version='1.0',
        description='Therapist guided-inquiry constructs for MORE sessions',
        themes=_PURER_DEFINITIONS,
    )
```

All definition field text should be drawn directly from Section 2 of this document.

**word_prototypes by construct** (seeds for embedding proximity, not classification rules):

| Construct | word_prototypes |
|-----------|----------------|
| P | notice, feel, body, sensation, aware, describe, experience, attention, noticing, where |
| U | you've, already, able, skill, you know, capacity, you mentioned, you found, resource |
| R | instead, what if, rather than, reframe, just sensation, just information, relationship, perspective |
| E | research, brain, nervous system, tends to, naturally, normal, program, mindfulness works, MORE |
| R2 (Reinforcement) | exactly, that's it, great, yes, perfect, well done, you did, progress, right there |

---

### 4.2 `classification_tools/data_structures.py` — MODIFY

Add PURER fields to the `Segment` dataclass. Insert after the `codebook_confidence` field (currently ~line 77) and before the `# Validation fields` block:

```python
# PURER label fields (populated by Stage 3c; only set when speaker == 'therapist')
purer_primary: Optional[int] = None
purer_secondary: Optional[int] = None
purer_confidence_primary: Optional[float] = None
purer_confidence_secondary: Optional[float] = None
purer_justification: Optional[str] = None
purer_run_consistency: Optional[int] = None
purer_agreement_level: Optional[str] = None
purer_agreement_fraction: Optional[float] = None
purer_needs_review: bool = False
purer_rater_ids: Optional[List[str]] = None
purer_rater_votes: Optional[List[Dict]] = None
```

These fields are always `None` / `False` for participant segments. No conditional logic needed — unset defaults are sufficient.

---

### 4.3 `process/config.py` — MODIFY

**Addition 1**: New feature flag in `PipelineConfig` (after `run_codebook_classifier` at ~line 112):

```python
run_purer_labeler: bool = True
```

**Addition 2**: New sub-config field in `PipelineConfig` (after `theme_classification` at ~line 117):

```python
purer_classification: ThemeClassificationConfig = field(default_factory=ThemeClassificationConfig)
```

**Addition 3**: Register in the nested config deserializer dict (~line 162):

```python
'purer_classification': ThemeClassificationConfig,
```

No new config class needed. `ThemeClassificationConfig` from `theme_framework/config.py` already handles all parameters (model, n_runs, per_run_models, temperature, backend, etc.). The `purer_classification` sub-config allows independent model and run-count settings for the therapist classification pass — useful if a smaller/cheaper model is appropriate for PURER.

---

### 4.4 `process/orchestrator.py` — MODIFY

**Import addition** (near top, alongside `parse_all_results`):

```python
from classification_tools.response_parser import parse_all_results, parse_purer_results
```

**Stage 3c insertion** (after Stage 3 `observer.on_stage_complete` at ~line 419, before Stage 3b check at ~line 427):

```python
# ------------------------------------------------------------------
# Stage 3c: PURER Therapist-Dialogue Classification
# ------------------------------------------------------------------
if config.run_purer_labeler:
    observer.on_stage_start(
        "PURER Therapist Classification", "3c",
        explanation_key='purer_classification',
    )

    from theme_framework.purer import get_purer_framework
    purer_framework = get_purer_framework()

    purer_config = config.purer_classification
    purer_config.output_dir = _paths.llm_raw_dir(output_dir)

    therapist_segs = [s for s in all_segments if s.speaker == 'therapist']
    observer.on_stage_progress(
        "PURER Therapist Classification",
        f"{len(therapist_segs)} therapist segments to classify",
    )

    purer_results, purer_metadata = classify_segments_zero_shot(
        segments=therapist_segs,
        framework=purer_framework,
        config=purer_config,
        resume_from=config.resume_from,
        process_logger=plog,
    )

    observer.on_stage_progress(
        "PURER Therapist Classification",
        "Parsing PURER responses...",
    )

    all_segments, purer_parse_stats = parse_purer_results(purer_results, all_segments)

    observer.on_stage_complete(
        "PURER Therapist Classification",
        f"PURER classification complete — "
        f"{purer_parse_stats.get('parsed', 0)} of {purer_parse_stats.get('total', 0)} classified",
    )
else:
    observer.on_stage_progress(
        "PURER Therapist Classification",
        "Skipping PURER classification (run_purer_labeler=False)",
    )
```

---

### 4.5 `classification_tools/response_parser.py` — MODIFY

Add `parse_purer_results()` function after `parse_all_results()`. The function is structurally identical to `parse_all_results()` but writes to `purer_*` fields instead of VAMMR fields. It does not need a `name_to_id` parameter (the PURER IDs 0–4 are already integer-encoded in the consensus output, same as VAMMR).

**Signature:**

```python
def parse_purer_results(
    results_all: Dict[str, Any],
    segments: List[Segment],
) -> Tuple[List[Segment], Dict[str, Any]]:
```

**Field mapping** (consensus dict → Segment field):

| consensus key | Segment field |
|---------------|---------------|
| `primary_stage` | `purer_primary` |
| `secondary_stage` | `purer_secondary` |
| `primary_confidence` | `purer_confidence_primary` |
| `secondary_confidence` | `purer_confidence_secondary` |
| `justification` | `purer_justification` |
| `n_agree` | `purer_run_consistency` |
| `agreement_level` | `purer_agreement_level` |
| `n_agree / n_raters` | `purer_agreement_fraction` |
| `needs_review` | `purer_needs_review` |
| `rater_ids` | `purer_rater_ids` |
| `rater_votes` | `purer_rater_votes` |

The function should maintain the same error counting and stats return dict structure as `parse_all_results()` for consistency with progress reporting.

---

### 4.6 `process/setup_wizard.py` — MODIFY

**Target**: `_step_2_speaker_filter()` method, lines 215–287.

**Current behavior**: After identifying therapist speakers, asks "Exclude therapist utterances from classification?" with default True. Sets `speaker_filter.mode = 'exclude'` or `'none'`.

**New behavior**: Split into two concerns.

**Part A** (unchanged except messaging): Identify which speakers are therapists. Set `speaker_filter.mode = 'exclude'` always when therapists are identified. The old "exclude from classification" question is removed — its answer is always yes for VAMMR.

Update the explanatory text from:
> "Therapist utterances can be excluded from classification while still being used as conversational context."

To:
> "Therapist utterances will be excluded from VAMMR participant-stage classification. They will be classified separately using the PURER therapist-dialogue framework."

**Part B** (new sub-step after therapist identification): Enable PURER.

```
--- PURER Therapist Classification ---
    The PURER framework classifies therapist guided-inquiry moves:
      P  — Phenomenology      (eliciting direct sensory description)
      U  — Utilization         (reflecting participant resources back)
      R  — Reframing           (introducing new interpretive frames)
      E  — Education/Expectancy (psychoeducation and expectancy-setting)
      R  — Reinforcement       (validating just-completed insight or skill)

    PURER labels enable analysis of which therapist moves precede
    participant stage transitions in VAMMR.

Enable PURER classification for therapist dialogue? [Y/n]
```

If yes: `config_data['run_purer_labeler'] = True`
If no: `config_data['run_purer_labeler'] = False`

**Step header update**: "Step 2/12: Speaker Role Identification" → "Step 2/12: Speaker Roles & PURER Setup"

**What does NOT change**: The `speaker_filter` config structure and serialization. `SpeakerFilterConfig` with `mode='exclude'` is correct and should stay.

---

### 4.7 `process/cross_validation.py` — MODIFY

Add `compute_purer_vammr_influence()` function. This is the PURER × VAMMR lift analysis, the analog of the VCE × VAMMR lift table already produced for participant segments.

**Core logic**:

1. Load `master_segments.jsonl` (or accept `List[Segment]`)
2. For each session, sort segments by `segment_index`
3. For each therapist segment `t` with `purer_primary` set, find the immediately following participant segment `p` (next `segment_index` where `speaker != 'therapist'` and `primary_stage` is not null)
4. Record the `(purer_primary, primary_stage)` pair as a co-occurrence
5. Compute lift: `lift(p, s) = P(s | p) / P(s)` where `P(s)` is the marginal base rate of VAMMR stage `s` across all participant segments in the corpus

**Output**: Three artifacts:
- `purer_vammr_influence_raw.csv` — raw co-occurrence counts, columns: `purer_construct, vammr_stage, count, purer_total, vammr_base_rate, lift`
- `purer_vammr_influence_pivot.csv` — 5×5 pivot table (PURER × VAMMR stage) of lift values
- Section in `cross_validation_report.md` describing the influence table with interpretive notes

**Lift interpretation note** for the report: Lift > 1.0 means the VAMMR stage is more common *following* that therapist move than in the base rate. A lift of 2.0 for (Reframing, Reappraisal) means Reappraisal is twice as likely to appear after a therapist Reframing move than it is overall — suggesting a direct mechanism.

---

### 4.8 `process/dataset_assembly.py` — MODIFY

Add PURER fields to the master dataset CSV export. The `purer_*` Segment fields are already populated (or null) before `dataset_assembly.py` runs. It is a matter of including them in the column list when constructing the output DataFrame.

Add to the participant/therapist unified export:

```
purer_primary, purer_secondary, purer_confidence_primary, purer_confidence_secondary,
purer_justification, purer_run_consistency, purer_agreement_level, purer_agreement_fraction,
purer_needs_review
```

Note: `purer_rater_ids` and `purer_rater_votes` are list/dict fields and should be serialized as JSON strings, consistent with how `rater_votes` is handled for VAMMR.

---

## 5. Wizard Rewiring — Complete Design

The wizard question flow for Step 2 changes as follows. The `_step_2_speaker_filter` method should be refactored into:

**Phase A: Discover and identify therapist speakers** (largely unchanged)
1. Scan transcripts for speaker names
2. Display discovered speakers with [therapist] / [participant] tags
3. Confirm or override defaults → `therapists` list

**Phase B: Configure speaker filter** (simplified — no user choice)
```python
if therapists:
    self.config_data['speaker_filter'] = {
        'mode': 'exclude',
        'speakers': therapists,
    }
    print(f"\n    {len(therapists)} therapist speaker(s) identified.")
    print("    Therapist segments will be excluded from VAMMR participant-stage classification.")
    print("    They will be classified separately using the PURER framework (see below).")
else:
    self.config_data['speaker_filter'] = {'mode': 'none', 'speakers': []}
```

**Phase C: PURER enable prompt** (new)
```python
if therapists:
    print()
    print("--- PURER Therapist Classification ---")
    print("    Classifies therapist guided-inquiry moves:")
    print("      P — Phenomenology      (eliciting present-moment description)")
    print("      U — Utilization         (reflecting participant resources)")
    print("      R — Reframing           (introducing new interpretive frames)")
    print("      E — Education/Expectancy (psychoeducation and expectancy-setting)")
    print("      R — Reinforcement       (validating insight or skill)")
    print()
    print("    Enables PURER × VAMMR influence analysis: which therapist moves")
    print("    precede which participant stage transitions.")
    print()
    run_purer = _prompt_yes_no("Enable PURER classification for therapist dialogue?", True)
    self.config_data['run_purer_labeler'] = run_purer
    if run_purer:
        print("    PURER classification enabled.")
    else:
        print("    Therapist dialogue will be stored without PURER labels.")
else:
    self.config_data['run_purer_labeler'] = False
```

Note: Advanced PURER model configuration (separate model, n_runs, per_run_models for the PURER pass) would appear in a later configuration step or be inferred from the VAMMR classification config as defaults. The wizard should not over-configure Step 2.

---

## 6. PURER × VAMMR Influence Analysis — Research Specification

### Motivation

VAMMR characterizes *where participants are* in therapeutic progression. PURER characterizes *what the therapist did before that*. The influence analysis asks: are certain PURER moves systematically associated with certain VAMMR outcomes in the following participant turn?

### Hypotheses (to test with lift data)

| Hypothesis | Direction |
|-----------|-----------|
| Phenomenology precedes Metacognition | Lift(P → M) > 1 |
| Reframing precedes Reappraisal | Lift(R → Reap) > 1 |
| Education precedes Vigilance responses (explaining doesn't resolve it) | Lift(E → V) > 1 |
| Reinforcement precedes another Metacognition/Reappraisal (consolidates) | Lift(Reinf → M or Reap) > 1 |
| Utilization precedes Metacognition (capacity-invoking promotes observing) | Lift(U → M) > 1 |

These are directional hypotheses grounded in the theoretical structure of PURER. The lift statistics provide an empirical test. Because the analysis is cross-cohort (multiple sessions, multiple therapists, many transitions), lift values are meaningful even at small-N study scales.

### Adjacency definition

"Following" is defined as the next participant segment by `segment_index` in the same session. This is conservative — it requires direct adjacency. A lookahead of 2–3 segments (allowing therapist-to-therapist transitions before the participant responds) may be explored as an extended analysis. The primary table uses direct adjacency.

### Reporting format

The influence table section in the cross-validation report should include:
1. The 5×5 lift matrix with cells colored by lift magnitude
2. A plain-language interpretation of the top 3 highest-lift cells
3. A note on sample size per PURER construct (number of therapist segments contributing to each row) to flag under-powered cells
4. A comparison across cohorts if ≥2 cohorts of data are available

---

## 7. Validation Plan

### Human coding of therapist segments

Extend the existing human validation workflow to include therapist segments. The human validation worksheet generator (`analysis/reports/validation_worksheet.py` or equivalent) should produce a PURER worksheet alongside the VAMMR participant worksheet.

Target: N=30 therapist segments per PURER construct (150 total) blind-coded by two independent raters (research team members with MORE training). Krippendorff's alpha ≥ 0.70 is the reliability threshold, consistent with the VAMMR standard.

### Construct validity checks

For each PURER construct, verify:
1. **Content validity**: Each exemplar utterance in the construct definition is classified correctly by the LLM at rate ≥ 90% in zero-shot testing
2. **Discriminant validity**: Each adversarial utterance in the construct definition is rejected at rate ≥ 90%
3. **Consistency**: Mean `purer_run_consistency` per construct ≥ 2.5 out of 3 runs (consistent with VAMMR threshold)

### Session-level fidelity profiles

Aggregate PURER labels per session to compute:
- PURER move frequency distribution (% of therapist segments per construct)
- Temporal distribution of each move across the session arc (early/middle/late)
- Per-therapist PURER profiles if multiple therapists are in the dataset

These profiles can be compared against theoretical PURER move distribution (e.g., whether Phenomenology dominates early sessions, Reinforcement scales with participant skill level).

---

## 8. File Map Summary

| Action | File | Location |
|--------|------|----------|
| CREATE | `theme_framework/purer.py` | New file, parallel to `theme_framework/vammr.py` |
| MODIFY | `classification_tools/data_structures.py` | After line 77 — add 11 `purer_*` fields to Segment |
| MODIFY | `process/config.py` | Lines 111–117 — add `run_purer_labeler` flag + `purer_classification` sub-config |
| MODIFY | `process/orchestrator.py` | After line 419 — insert Stage 3c block; add import |
| MODIFY | `classification_tools/response_parser.py` | After line ~160 — add `parse_purer_results()` |
| MODIFY | `process/setup_wizard.py` | Lines 263–286 — remove old exclude prompt; add Phase B + Phase C |
| MODIFY | `process/cross_validation.py` | New function `compute_purer_vammr_influence()` |
| MODIFY | `process/dataset_assembly.py` | Add `purer_*` fields to CSV export column list |

---

## 9. Out of Scope for This Iteration

The following items are intentionally deferred:

- **Multi-label PURER**: A single therapist utterance can theoretically contain more than one PURER move (e.g., brief Education followed by Reinforcement). The first iteration classifies primary and secondary only (same pattern as VAMMR). Multi-label PURER is Phase 2 scope once the single-label corpus is built and validated.
- **PURER × VCE cross-validation**: What phenomenological experiences follow which therapist moves? Technically feasible (therapist PURER → participant VCE codes), but requires both PURER and VCE to be validated first. Deferred to post-Cohort 2.
- **Wizard-level PURER model configuration**: The `purer_classification` sub-config defaults to the same settings as `theme_classification`. Exposing separate PURER model/n_runs settings in the wizard is deferred until there is evidence that different settings are needed.
- **Real-time fidelity feedback**: A separate tool that provides session-by-session PURER profiles to the clinical team between cohorts. Architecturally trivial once the labels exist; deferred to Phase 2 reporting tooling.
