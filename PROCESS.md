# QRA Technical Process Document

## Qualitative Research Algorithm — End-to-End Pipeline Reference

This document describes the complete logical flow of the QRA pipeline: how data enters, what computations happen at each stage, and exactly what every output file contains and how each measure was derived. It is written for a researcher or engineer who needs to understand *why* the pipeline produces the outputs it does, not just *that* it does.

---

## Overview

QRA is a machine-assisted computational pipeline for analyzing psychotherapy session transcripts from mindfulness-based interventions. It operationalizes two complementary phenomenological frameworks—VA-MR (Vigilance–Avoidance–Metacognition–Reappraisal) and VCE (Varieties of Contemplative Experience)—into an automated classification system that can produce defensible qualitative analysis within days rather than weeks.

The pipeline has two philosophically distinct purposes running simultaneously:

1. **Classification**: Assign every participant utterance in every transcript a VA-MR stage label (and optionally a set of VCE phenomenology codes). This produces a coded dataset suitable for statistical analysis.

2. **Validation**: Generate the artifacts a human researcher needs to audit, verify, and calibrate the machine-assigned labels—blind-coding worksheets, interrater reliability metrics, cross-framework lift statistics, and flagged items.

The underlying epistemological design is called **Computational Mutual Constraints**: two independent frameworks are applied to the same corpus. Their empirical cross-validation—measured via lift statistics—tests whether both frameworks track genuine phenomenological distinctions in the data rather than artefacts of prompt engineering or researcher expectation.

---

## Input

The pipeline accepts **diarized therapy session transcripts** in one of two formats:

### JSON (Whisper + diarization)
```json
{
  "metadata": {
    "trial_id": "standard",
    "participant_id": "participant_1",
    "session_id": "c1s1",
    "session_number": 1,
    "cohort_id": 1
  },
  "sentences": [
    {
      "text": "How are you feeling today?",
      "speaker": "Therapist",
      "start": 0.5,
      "end": 3.2
    },
    {
      "text": "I'm doing pretty well overall.",
      "speaker": "Participant_1",
      "start": 3.5,
      "end": 6.8
    }
  ]
}
```

Each `sentence` is one sentence-level utterance as output by an automatic speech recognition system with speaker diarization. The `speaker` field uses real speaker names (these are anonymized later in the pipeline). The `start` and `end` fields are timestamps in seconds.

### WebVTT (alternative)
Subtitle-format files with speaker-prefixed caption blocks. The filename stem is parsed to extract session metadata (e.g., `c1s3` → cohort 1, session 3).

---

## The 8-Stage Pipeline

### Stage 1: Transcript Ingestion and Semantic Segmentation

**What it does:** Converts the raw sentence stream from each transcript into a list of semantically coherent *segments*—multi-sentence units of meaning that become the atomic unit of classification throughout the rest of the pipeline.

**Why segmentation is necessary:** Individual sentences from an ASR transcript are too short and fragmentary to classify reliably. A single VA-MR stage is typically expressed over several sentences. Grouping sentences into coherent chunks gives the LLM enough context to make a defensible classification and gives the human coder enough text to verify it.

**The segmentation algorithm** (`process/transcript_ingestion.py`, class `ConversationalSegmenter`):

1. **Load and parse the transcript.** The `load_diarized_session()` or `load_vtt_session()` function reads the file and normalizes the sentence list. Each sentence has: text, speaker name, start time (ms), end time (ms).

2. **Speaker normalization.** The `SpeakerNormalizer` class maps raw speaker names to stable anonymized identifiers: `participant_1`, `participant_2`, `therapist_1`, etc. If a `speaker_anonymization_key.json` was imported from a previous run, the same IDs are reused across sessions, ensuring consistent participant identity throughout a longitudinal dataset.

3. **Speaker filtering.** Sentences from excluded speakers (typically therapists, configured in `SpeakerFilterConfig`) are not placed into participant segments. They are retained as *context* only—they can appear as preamble to help the LLM understand what the participant is responding to, but they are not themselves classified.

4. **Sentence embedding.** Each remaining participant sentence is encoded into a dense vector using a sentence-transformer model (default: `Qwen/Qwen3-Embedding-8B`, 4096 dimensions, loaded at float16 to halve VRAM).

5. **Semantic similarity curve.** For each pair of consecutive sentences, the cosine similarity between their embeddings is computed. A high similarity between sentence *i* and sentence *i+1* means they are probably part of the same semantic unit. A sudden drop in similarity indicates a topic shift.

6. **Adaptive boundary detection.** Rather than using a fixed similarity threshold, the algorithm finds **local minima** in the similarity curve. Two window sizes are used simultaneously:
   - A narrow window (3 sentences) that is sensitive to fine-grained topic shifts
   - A broad window (7 sentences) that is robust to noisy one-sentence asides

   The broad-window minima are prioritized. A minimum is only treated as a segment boundary if its prominence (how far it drops below neighboring values) exceeds `min_prominence` (default 0.05). This prevents boundaries from being inserted inside a continuous train of thought.

7. **Optional topic clustering.** If `use_topic_clustering` is enabled, agglomerative clustering is run on the sentence embeddings to group sentences by topic. Cluster boundaries that agree with similarity-curve minima are used to confirm or reinforce boundary decisions.

8. **Size and duration constraints.** After boundary detection, segments are merged or split to satisfy practical constraints:
   - Minimum 60 words (short segments are merged into neighbors)
   - Maximum 500 words
   - Maximum gap between utterances within a segment: 15 seconds
   - Maximum segment duration: 60 seconds

9. **LLM refinement (optional).** If `use_llm_refinement` is enabled, the same LLM backend used for classification reviews ambiguous boundary decisions—those where the similarity score was close to the detection threshold. The LLM is given a short window of sentences and asked whether the current split point is a natural boundary (`mode='boundary_review'`), whether surrounding context should be included (`mode='context_expansion'`), or whether the segment stays on one topic (`mode='coherence_check'`). The LLM uses `no_reasoning=True` for this task—segmentation prompts require simple true/false judgments and reasoning tokens would waste context budget.

10. **Therapist segment extraction.** After participant segments are finalized, therapist utterances are also extracted as separate `Segment` objects (speaker = `therapist`). These are interleaved with participant segments in chronological order by `start_time_ms`. This interleaving is what allows the analysis module to surface the therapist's exact words between any two participant stage expressions—an important feature for understanding the cue-response dynamics of the therapy session.

**Output of Stage 1:** A list of `Segment` objects, one per semantically coherent unit of speech. Each segment carries:
- `segment_id`: globally unique string, e.g., `c1s1_p1_seg5`
- `participant_id`, `session_id`, `session_number`, `cohort_id`
- `speaker`: `participant` or `therapist`
- `text`: the joined text of all sentences in the segment
- `word_count`
- `start_time_ms`, `end_time_ms`
- `segment_index`: position in the session (combining both participant and therapist segments)
- `total_segments_in_session`: filled after all sessions are processed

The speaker anonymization key (`07_meta/speaker_anonymization_key.json`) is written atomically at Stage 7 (not here), so a failed run between ingestion and report generation cannot leave a stale key.

---

### Stage 2: Construct Operationalization

**What it does:** Loads the framework definitions and generates the *content validity test set*—a curated set of items that can be used to evaluate whether the classifier covers the intended construct space before and after any refinements.

**The VA-MR framework** (`constructs/vamr.py`) defines four stages as `ThemeDefinition` objects:

| ID | Key | Name |
|----|-----|------|
| 0 | `vigilance` | Pain Vigilance and Attention Dysregulation |
| 1 | `avoidance` | Attention Regulation applied to Experiential Avoidance |
| 2 | `metacognition` | Metacognitive Awareness and Observing Stance |
| 3 | `reappraisal` | Experiential Reappraisal and Sensory Deconstruction |

Each `ThemeDefinition` contains:
- `definition`: a full phenomenological description
- `prototypical_features`: 6–8 canonical linguistic markers that reliably indicate this stage
- `distinguishing_criteria`: what separates this stage from the adjacent ones
- `exemplar_utterances`: 4 clear, unambiguous examples
- `subtle_utterances`: 3 examples that are harder to classify correctly
- `adversarial_utterances`: 3 boundary-crossing examples that could plausibly belong to an adjacent stage (e.g., "I try to push the pain away" could be Vigilance or Avoidance)

**Content validity test set** (`create_content_validity_test_set()`): Pulls the exemplar, subtle, and adversarial utterances from each theme definition and assembles them into a JSONL dataset. Each item has the true label known in advance. This test set can be run through the classifier before the main data to verify that the classifier correctly handles both the easy cases (exemplars) and the hard cases (adversarial). It is the primary tool for prompt engineering—if the classifier fails on adversarial items, the framework definitions or the prompt can be refined before wasting compute on the full corpus.

**Outputs:**
- `07_meta/theme_definitions.json`: The full framework serialized to JSON. Contains every field of every `ThemeDefinition`. This is the ground truth document that the analysis module uses to reconstruct the framework without the Python objects.
- `05_validation/content_validity_test_set.jsonl`: One line per test item. Fields: `text`, `expected_label` (int), `difficulty` (exemplar/subtle/adversarial), `theme_key`, `theme_name`.

---

### Stage 3: Zero-Shot LLM Theme Classification

**What it does:** For every participant segment, asks an LLM to classify it into one of the four VA-MR stages. Multiple independent classification passes are run and then aggregated into a consensus label with an interrater reliability score.

#### 3A. Prompt construction

For each segment, a prompt is assembled from:

1. **Framework description.** The name and full definition of each of the four stages, plus their exemplar utterances. This is the "codebook" provided to the LLM as a zero-shot classifier.

2. **Context preamble (optional).** The 2–3 sentences immediately preceding the segment in the transcript, drawn from both participant and therapist turns. This is capped at 300 words to avoid bloating the prompt. Context is crucial for disambiguation: an utterance like "Yes, I can feel that separation" means completely different things depending on what the therapist just said.

3. **The segment text itself**, clearly delimited.

4. **Output schema.** The LLM is instructed to return a JSON object with exactly these fields:
   ```json
   {
     "primary_stage": "<prompt_name of stage or null>",
     "primary_confidence": 0.0,
     "secondary_stage": "<prompt_name or null>",
     "secondary_confidence": null,
     "justification": "<brief explanation>"
   }
   ```
   A `null` `primary_stage` is a valid response—it means the LLM judges the utterance to be irrelevant to the VA-MR framework (e.g., logistical small talk). This becomes an ABSTAIN ballot.

5. **Theme order randomization.** If temperature > 0, the order in which the four stage definitions appear in the prompt is randomized between runs. This prevents the LLM from defaulting to the first or last option and reduces systematic position bias.

#### 3B. Multi-run / multi-model classification

Each segment is classified `n_runs` times (default: 3). How those runs are distributed depends on configuration:

- **Single model, stochastic runs**: The same model is called three times with slight temperature jitter. This tests intra-model consistency.
- **Multi-model runs** (`per_run_models`): Each run uses a different LLM (e.g., GPT-4o for run 1, Claude Sonnet for run 2, Gemini Flash for run 3). This provides true multi-rater interrater reliability between independent models.

Each run produces a *ballot*: a parsed JSON response with `vote` = `CODED`, `ABSTAIN`, or `ERROR`.

#### 3C. Consensus voting (`classification_tools/majority_vote.py`)

`vote_single_label()` aggregates all ballots for a segment:

1. **Valid ballots only.** ERROR ballots (parse failures) are excluded from the count. ABSTAIN ballots count as a valid vote for "this segment is not classifiable."

2. **Majority rule.** The winning stage is the one with the most votes. A result is:
   - `unanimous`: all raters agreed (and no errors)
   - `majority`: strictly more than half the raters agreed
   - `split`: no stage reached majority — flagged for human review
   - `none`: all raters failed (all ERRORs)

3. **Tie-breaking.** When two stages tie, CODED stages are preferred over ABSTAIN (the pipeline is biased toward assigning a label rather than dropping the segment). Among tied coded stages, the one with the higher mean confidence across its agreeing raters wins. `tie_broken_by_confidence` is set to `True` in this case.

4. **Confidence.** `primary_confidence` is the mean confidence of the agreeing raters—not the winning ballot's confidence alone, but the average across everyone who voted for the winner.

5. **Secondary stage.** The most common secondary stage among agreeing raters is collected alongside the primary. This captures common "runner-up" impressions.

6. **Needs review.** Any segment with a `split` or `none` agreement level is marked `needs_review = True`. These are automatically included in the human review worksheet.

**Fields populated on each `Segment` object after Stage 3:**
- `primary_stage` (int 0–3, or None if ABSTAIN/split)
- `secondary_stage` (int or None)
- `llm_confidence_primary` (float, mean confidence of agreeing raters)
- `llm_confidence_secondary` (float or None)
- `llm_justification` (string, the first agreeing rater's rationale)
- `llm_run_consistency` (int, number of raters who voted for the winning stage)
- `rater_ids` (list of rater identifiers)
- `rater_votes` (list of per-rater ballot dicts with full vote details)
- `agreement_level` (`unanimous`/`majority`/`split`/`none`)
- `agreement_fraction` (float, n_agree / n_raters)
- `needs_review` (bool)
- `consensus_vote` (int or `ABSTAIN` or None)
- `tie_broken_by_confidence` (bool)

---

### Stage 3b: Codebook Classification (Optional)

**What it does:** Applies the 59-code VCE phenomenology codebook to each participant segment using a *two-method ensemble*: embedding similarity and LLM classification. The VCE codes describe specific phenomenological experiences (e.g., "Meta-Cognition," "Boundary Changes," "Pain") that can co-occur within a single segment. Unlike the VA-MR stage classification (which assigns one primary label), this is **multi-label** classification.

#### The VCE Codebook

The codebook (`codebook/phenomenology_codebook.py`) contains 59 codes across 7 domains:

| Domain | Code count | Example codes |
|--------|-----------|---------------|
| Affective | 11 | Affective Flattening, Lability, Agitation, Fear, Positive Affect |
| Cognitive | 11 | Meta-Cognition, Clarity, Delusions, Mental Stillness, Derealization |
| Conative | 8 | Motivation, Effort, Anhedonia, Goal Changes |
| Perceptual | 10 | Hypersensitivity, Visual Changes, Boundary Changes, Time Distortion |
| Sense of Self | 8 | Identity Changes, Agency Loss, Embodiment, Self-Other Boundaries |
| Social | 6 | Relationship Changes, Withdrawal, Compassion |
| Somatic | 5 | Pressure/Tension, Energy, Pain, Sleep, Appetite |

Each `CodeDefinition` has:
- `code_id`: URL-safe slug (e.g., `meta-cognition`, `affective-flattening`)
- `category`: human-readable name
- `domain`: one of the 7 domains
- `description`: what this code means
- `subcodes`: more specific variants within the code
- `inclusive_criteria`: what language and experiences qualify for this code
- `exclusive_criteria`: what to avoid including (boundary cases)

#### Method 1: Embedding-Based Classification

`EmbeddingCodebookClassifier` (`codebook/embedding_classifier.py`):

1. **Asymmetric encoding.** Segments are encoded as *query* embeddings (using the query-role instruction prefix built into Qwen3 and similar retrieval-optimized models). Codebook code texts (description + criteria + exemplars) are encoded as *passage* embeddings. This asymmetry is intentional—it matches the intended use case of the embedding model where a short query is being matched to longer reference passages.

2. **Composite score.** For each segment × code pair, a weighted score is computed:
   ```
   score = cosine_sim(seg, definition)
           + criteria_weight * cosine_sim(seg, inclusive_criteria)
           + exemplar_weight * cosine_sim(seg, exemplars)
   ```
   Default weights: `criteria_weight = 0.3`, `exemplar_weight = 0.5`. The exemplar component gets the highest weight because exemplar utterances are the most semantically proximate signal to what the segment text is actually doing.

3. **Two-pass classification.** In the first pass, codes above a confidence threshold are identified. In the second pass, the high-confidence codes from pass 1 are merged into the exemplar pool and the entire segment is re-scored. This improves recall on subtle phenomenology—if a segment clearly shows Metacognition, the enriched exemplar set in pass 2 makes it easier to also identify a secondary code like Mental Stillness.

4. **Threshold.** A code is assigned if its composite score exceeds `exemplar_confidence_threshold` (default 0.7). This threshold is in cosine similarity units.

#### Method 2: LLM-Based Classification

`LLMCodebookClassifier` uses the same LLM backend as Stage 3 but with a different multi-label prompt:
- All 59 codes and their descriptions are presented
- The LLM is asked which codes apply to the segment
- Multiple runs are executed and aggregated via `vote_multi_label()`

For multi-label voting, a code is included in the consensus only if a **strict majority** of raters applied it (count > n_raters / 2). This is deliberately stricter than the single-label threshold to compensate for the higher false-positive rate of multi-label prompts.

#### Method 3: Ensemble Reconciliation

`CodebookEnsemble` (`codebook/ensemble.py`) merges the two method outputs:
- Final code list = codes present in either method above threshold
- Embedding and LLM confidence scores for each code are preserved separately
- Codes present in one method but not the other are recorded in `codebook_disagreements`
- By default, embedding scores are weighted 0.6 and LLM scores 0.4 (embedding is preferred because it is deterministic and faster)

**Fields populated on each `Segment` object after Stage 3b:**
- `codebook_labels_embedding`: list of code IDs from embedding method
- `codebook_labels_llm`: list of code IDs from LLM method
- `codebook_labels_ensemble`: final merged list
- `codebook_disagreements`: codes in only one method (disagreements)
- `codebook_confidence`: dict of `{code_id → float}` confidence scores

---

### Stage 4: Cross-Validation (Theme ↔ Codebook)

**What it does:** Tests whether the VA-MR stages and VCE codes are empirically consistent—i.e., whether the phenomenological phenomena described by each VA-MR stage actually tend to co-occur with the VCE codes that theory predicts they should.

**This is the computational mutual constraints check.** If the two frameworks are independently tracking the same underlying phenomenological reality, their outputs should correlate in theoretically predictable ways. If they don't, it may indicate prompt engineering artifacts, construct underspecification, or genuine theoretical misalignment.

**Computation** (`process/cross_validation.py`):

1. **Filter** to segments that have both a VA-MR label and at least one VCE code.

2. **Base rates.** For each VCE code, compute its overall frequency across all dually-labeled segments:
   ```
   base_rate(code) = count(code) / total_segments
   ```

3. **Theme-specific rates.** For each VA-MR stage, compute how often each code appears within that stage's segments:
   ```
   rate(code | theme) = count(code in theme) / count(theme)
   ```

4. **Lift.** For each theme × code pair:
   ```
   lift = rate(code | theme) / base_rate(code)
   ```
   A lift of 1.0 means the code appears at exactly the base rate within this theme—no association. A lift of 2.0 means the code is twice as frequent in this theme as in the corpus overall. A lift of 0.5 means the code is half as likely—an anti-association.

5. **Thresholds.** Only associations with `lift ≥ 1.5` AND `count ≥ 3` are reported as meaningful. The count threshold suppresses spurious high-lift values from very rare codes.

**Theoretical predictions being tested:**
- Vigilance → high lift for Fear, Anxiety, Pain, Agitation
- Avoidance → high lift for Affective Flattening, Emotional Detachment
- Metacognition → high lift for Meta-Cognition, Clarity, Narrative Self Changes
- Reappraisal → high lift for Positive Affect, Worldview Changes, Self-Other Boundaries

**Outputs:**
- `05_validation/cross_validation/cross_validation_results.json`: Full matrix of all theme × code lift statistics
- `05_validation/cross_validation/top_theme_code_associations.json`: Filtered to only the highest-lift, most-frequent associations per theme

---

### Stage 5: Preparing the Human Validation Set

**What it does:** Creates a balanced, stratified random sample of segments that will be given to human coders for blind review. This set is the foundation for computing human–machine interrater reliability (Krippendorff's alpha ≥ 0.60 is the target).

**Balanced evaluation set** (`classification_tools/validation.py`, `create_balanced_evaluation_set()`):
- Only participant segments with a primary label are eligible
- Equal numbers are sampled from each VA-MR stage (stratified)
- Cross-session sampling: the set avoids taking two segments from the same session (distributes the coding burden evenly across the corpus)
- Default size: `n_per_class` items per stage

**Output:**
- `05_validation/human_coding_evaluation_set.csv`: Each row is one segment with its text, segment ID, session ID, and participant ID. The AI-assigned label is **not** included, ensuring blind coding conditions.

---

### Stage 6: Dataset Assembly

**What it does:** Takes the fully annotated list of `Segment` objects and produces the master dataset—a flat, serializable record for every segment in the corpus.

**Final label resolution** (`process/assembly/master_dataset.py`):

Final labels follow a three-tier priority:
1. `adjudicated`: A human expert has reviewed and confirmed or corrected the label
2. `human_consensus`: A human coder independently assigned the same label as the LLM
3. `llm_zero_shot`: No human coding yet—the LLM consensus is used directly

**Confidence tiers:**

Each segment is assigned a `label_confidence_tier` based on the LLM voting record:

| Tier | Conditions |
|------|-----------|
| `high` | `llm_run_consistency == n_runs` (unanimous) AND `llm_confidence_primary > 0.8` |
| `medium` | `llm_run_consistency >= 2` (majority) AND `llm_confidence_primary > 0.6` |
| `low` | Everything else (split votes, low confidence) |

These tiers allow downstream consumers to filter to only the most reliable labels for initial analyses.

**Outputs:**
- `06_training_data/master_segments.jsonl`: One JSON object per segment, one line per segment. All fields from the `Segment` dataclass are included, plus the derived `final_label`, `final_label_source`, and `label_confidence_tier` fields.
- `06_training_data/master_segments.csv`: Same data in CSV format.

**Schema of each row in `master_segments.jsonl`:**

```
segment_id              — globally unique segment identifier
trial_id                — study/trial identifier
participant_id          — anonymized participant ID
session_id              — session identifier (e.g., "c1s3")
session_number          — integer session number
cohort_id               — cohort number
session_variant         — '' or 'a'/'b' for split sessions
segment_index           — position in session (combined participant + therapist)
start_time_ms           — segment start timestamp in milliseconds
end_time_ms             — segment end timestamp in milliseconds
total_segments_in_session — segment count for this session
speaker                 — 'participant' or 'therapist'
text                    — full segment text
word_count              — word count
primary_stage           — VA-MR stage (0–3) or null
secondary_stage         — runner-up stage or null
llm_confidence_primary  — mean confidence of agreeing raters (0.0–1.0)
llm_confidence_secondary — secondary stage confidence or null
llm_justification       — first agreeing rater's explanation text
llm_run_consistency     — count of raters who voted for the winning stage
rater_ids               — JSON array of rater identifiers
rater_votes             — JSON array of per-rater ballot records
agreement_level         — 'unanimous' | 'majority' | 'split' | 'none'
agreement_fraction      — n_agree / n_raters
needs_review            — true if split or none
consensus_vote          — winning stage int, 'ABSTAIN', or null
tie_broken_by_confidence — true if tie was resolved by confidence
codebook_labels_embedding — array of VCE code IDs from embedding method
codebook_labels_llm       — array of VCE code IDs from LLM method
codebook_labels_ensemble  — final merged array of VCE code IDs
codebook_disagreements    — codes present in only one method
human_label             — human-assigned label (int or null)
human_secondary_label   — human secondary label or null
adjudicated_label       — expert-adjudicated label or null
in_human_coded_subset   — true if this segment was in the human eval set
label_status            — 'llm_only' | 'human_coded' | 'adjudicated'
final_label             — the authoritative label (priority: adjudicated > human_consensus > llm)
final_label_source      — which source the final_label came from
label_confidence_tier   — 'high' | 'medium' | 'low'
```

---

### Stage 7: Report Generation

**What it does:** Takes the master dataset and all intermediate results and writes every human-readable and machine-readable output file into the output directory structure.

#### 7a. Coded Transcripts (`01_transcripts/coded/`)

**File:** `coded_transcript_{session_id}.txt`

**What it is:** A human-readable rendering of the full session with every classification decision visible. This is the primary document a researcher uses to review the pipeline's output.

**Contents:**

*Session header:*
- Session ID, participant IDs, trial, duration, total segment count
- Rater roster (identifiers for each LLM used)
- Theme distribution table: for each VA-MR stage, count of segments assigned to it, percentage of classified segments, and mean confidence score
- Codebook distribution table (if VCE enabled): for each code, total applications and mean confidence
- Interrater reliability summary: percent agreement (unanimous), percent agreement (pairwise), Fleiss' kappa, Krippendorff's alpha, count of flagged segments

*Per-segment detail (for every segment):*
- Segment number, timestamp range, word count, speakers present
- The full segment text (wrapped at 76 chars)
- **Rater ballots section**: one line per rater showing their vote, the stage they chose, their confidence, and their justification text
- **Consensus line**: `CLASSIFIED as [Stage Name]  (unanimous, 3/3)` or `UNCLASSIFIED — SPLIT VOTE` etc.
  - Mean confidence
  - Secondary stage if present
  - Rationale: if multiple raters provided justifications, these are synthesized by a final LLM call; if only one rater had a justification, it is used verbatim
  - `↳ Tie broken by confidence` flag if applicable
  - `↳ FLAGGED FOR HUMAN REVIEW` flag for split/none segments
- VCE codes applied: `CODES: [meta-cognition, clarity, pain]`

**How confidence is derived in this report:** The mean confidence shown in the session header is the average of `llm_confidence_primary` across all classified segments for that stage. The mean confidence shown per segment is the average confidence of the raters who voted for the winning stage (not across all raters—only the agreeing ones).

#### 7b. Human Classification Forms (`05_validation/`)

**File:** `human_classification_{session_id}.txt`

**What it is:** A blind-coding worksheet for a human researcher. Contains the full text of every participant segment in the session, numbered and timestamped, with empty response fields. The AI-assigned labels are completely absent. The researcher reads each segment and writes in their own label.

**File:** `human_coding_evaluation_set.csv`

**What it is:** The stratified random sample generated in Stage 5. This is the specific subset used for formal IRR calculation against the AI labels.

#### 7c. Flagged for Review (`05_validation/flagged_for_review.txt`)

**What it is:** A list of all segments where the AI raters could not agree (split or none agreement). Each entry shows the segment text, the rater votes (who voted for what, with confidence), and the session context. This is the prioritized review queue for human expert attention.

#### 7d. Validation Test Sets (`05_validation/testsets/`)

**Files:**
- `human_classification_testset_worksheet_N.txt`: Blind-coding worksheet (no AI labels)
- `AI_classification_testset_worksheet_N.txt`: Same segments with AI labels shown (for side-by-side comparison)

**How generated:** A stratified random sample of `fraction_per_set` (default 10%) of participant segments, balanced across VA-MR stages. Multiple independent sets can be generated. These are used for formal reliability studies where a second researcher codes independently and then the two codings are compared against each other and against the AI.

#### 7e. Per-Transcript Statistics (`04_analysis_data/session_stats/`)

**File:** `stats_{session_id}.json`

**Contents:**
- `session_id`, `participant_id`, `trial_id`
- `n_segments_total`, `n_participant_segments`, `n_therapist_segments`
- `n_classified`, `n_unclassified`, `n_needs_review`
- `theme_distribution`: for each stage, `{count, percentage, mean_confidence, exemplar_text}`
- `top_codebook_codes`: if VCE enabled, most frequent codes with counts and mean confidence
- `irr_metrics`: `{percent_agreement, fleiss_kappa, krippendorff_alpha}`
- `duration_seconds`

**How derived:** Computed directly from the assembled DataFrame for that session. All counts are straightforward aggregations. Percentages are `count / n_classified * 100` (unclassified segments are excluded from the denominator so percentages sum to 100 across stages). Mean confidence uses only the classified segments for each stage. The exemplar text is the single classified segment with the highest `llm_confidence_primary` for that stage.

#### 7f. Cumulative Report (`04_analysis_data/cumulative_report.json`)

**What it is:** Dataset-wide aggregated statistics across all sessions and all participants.

**Contents:**
- `n_sessions`, `n_participants`, `n_segments_total`
- Global theme distribution (counts and percentages across entire corpus)
- Global IRR summary
- Top 10 codebook codes by frequency
- Per-participant segment counts
- Confidence tier distribution: how many segments fall into high/medium/low tiers

#### 7g. Training Data (`06_training_data/`)

**File:** `theme_classification.jsonl`

**What it is:** A supervised training dataset for fine-tuning a text classifier on the VA-MR task. Only includes segments with `label_confidence_tier` of `high` or `medium` (low-confidence labels are excluded). Format: `{"text": "...", "label": 2, "label_name": "Metacognition"}`.

**File:** `codebook_multilabel.jsonl`

**What it is:** A multi-label supervised training dataset for VCE code prediction. Each row: `{"text": "...", "labels": ["meta-cognition", "clarity"]}`. Only includes segments where the ensemble agreed (no lone-rater labels).

#### 7h. Metadata (`07_meta/`)

**File:** `qra_config.json`
The full `PipelineConfig` serialized to JSON. All API keys and tokens are blanked. This file is sufficient to reproduce the run exactly, using `python qra.py run --config ./output/07_meta/qra_config.json`.

**File:** `speaker_anonymization_key.json`
Mapping from real speaker names to anonymized IDs and roles:
```json
{
  "John Smith": {"role": "participant", "anonymized_id": "participant_1"},
  "Dr. Therapist": {"role": "therapist", "anonymized_id": "therapist_1"}
}
```

**File:** `theme_definitions.json`
The full VA-MR framework as JSON. Used by `qra analyze` to reconstruct the framework object without the Python source.

**File:** `codebook_definitions.json`
The full VCE codebook as JSON (if codebook classification was run).

**File:** `process_log.txt` (if `--verbose-segmentation` was set)
A complete timestamped log of every LLM prompt and response during segmentation refinement. Each boundary decision is recorded with the similarity score, the window values, and the LLM's judgment if called. Used for debugging segmentation failures.

**File:** `07_meta/llm_raw/`
Checkpoint files for theme classification. One JSONL per session, one line per segment, containing the raw LLM responses before parsing. Used to resume a run that was interrupted mid-session (`--resume-from checkpoint`).

**File:** `07_meta/codebook_raw/`
Checkpoint files for codebook classification. Includes `found_exemplar_utterances.json`—the actual segment texts that achieved high embedding scores during two-pass classification, exported for manual review.

---

### Stage 8: Results Analysis (Optional Post-Pipeline)

Stage 8 is invoked by `python qra.py analyze --output-dir ...` or automatically if `config.auto_analyze = True`. It reads the `master_segments.jsonl` and `theme_definitions.json` files produced by Stages 1–7 and generates a suite of analytical reports for longitudinal and comparative research.

#### 8a. Per-Session Analysis (`02_human_reports/per_session/`)

**File:** `session_{session_id}.json`

Builds on the Stage 7 session stats with deeper analysis:
- Stage distribution with confidence breakdowns
- Exemplar segments for each stage (highest-confidence examples)
- Within-session stage transitions (which stages appear sequentially—does Vigilance tend to precede Metacognition?)
- Codebook code frequency within each stage

#### 8b. Per-Participant Reports (`02_human_reports/per_participant/`)

**File:** `participant_{participant_id}.json`

Longitudinal trajectory for one participant across all their sessions:
- Session-by-session stage distribution (how does the proportion of Reappraisal change over time?)
- Dominant stage per session (the stage with the highest segment count)
- Stage progression trend (is the participant moving toward more advanced stages?)
- Codebook code frequency evolution across sessions

#### 8c. Per-Construct Reports (`02_human_reports/per_construct/`)

**File:** `construct_{stage_key}.json`

For each VA-MR stage:
- All segments across the full corpus that were assigned this stage, sorted by confidence
- The top 10 most characteristic segments (highest confidence)
- The most common codebook co-occurrences
- Distribution across participants and sessions

**File:** `codebook_exemplars.txt`

Human-readable summary of the VCE codes observed in the corpus, with the highest-confidence segment example for each code.

#### 8d. Graph-Ready Datasets (`04_analysis_data/graphing/`)

These CSVs are formatted for direct import into visualization tools (matplotlib, R, ggplot2, etc.):

**File:** `session_stage_counts.csv`

One row per session × stage combination. Columns: `session_id`, `session_number`, `participant_id`, `stage_id`, `stage_name`, `count`, `percentage`.

**Use:** Primary data source for bar charts and stacked area charts showing stage distribution per session.

**File:** `participant_stage_trajectories.csv`

One row per participant × session × stage. Columns: `participant_id`, `session_number`, `stage_name`, `count`, `percentage`, `cumulative_reappraisal`.

**Use:** Line charts showing individual participant trajectories across sessions.

**File:** `session_confidence_distributions.csv`

One row per classified segment. Columns: `session_id`, `participant_id`, `session_number`, `stage_name`, `llm_confidence_primary`, `agreement_level`, `label_confidence_tier`.

**Use:** Box plots of confidence distributions by stage and session; scatter plots of confidence vs. agreement.

**File:** `codebook_cooccurrence_matrix.csv`

One row per VA-MR stage × VCE code. Columns: `stage_name`, `code_id`, `count`, `rate`, `base_rate`, `lift`.

**Use:** Heatmaps of the cross-framework cooccurrence pattern.

#### 8e. Longitudinal Summary (`04_analysis_data/longitudinal_summary.json`)

A cohort-level summary aggregated across all participants and all sessions:
- Average stage distribution at each session number (session 1, session 2, etc.)
- Trend in Reappraisal prevalence (the target outcome stage for mindfulness intervention)
- Cross-participant variance in stage distributions
- Participants showing forward progression (increasing Reappraisal over time) vs. stable or regressing

**How "progression" is calculated:** For each participant, the Reappraisal percentage is computed for each session. A participant is marked as showing forward progression if their Reappraisal percentage in the second half of their sessions is higher than in the first half. This is a simple split-half comparison—no regression modeling.

#### 8f. Session Stage Progression (`04_analysis_data/session_stage_progression.csv`)

**What it is:** A transition matrix capturing how VA-MR stage labels flow within and between sessions.

**How derived:** For each pair of consecutive classified participant segments within a session, the transition is recorded: what stage came before, what stage came after. The output is a count and rate matrix (4×4 for VA-MR) plus:
- `forward_transitions`: percentage of transitions that move to a higher-numbered stage (Vigilance→Avoidance, Avoidance→Metacognition, Metacognition→Reappraisal)
- `backward_transitions`: percentage moving to a lower stage
- `lateral_transitions`: percentage staying at the same stage
- `adjacency_index`: mean absolute difference between consecutive stages (0 = all same, 3 = maximum jumps)

#### 8g. Figures (`03_figures/`)

PNG figures generated for the report:
- Stage frequency bar charts per session
- Participant trajectory line plots
- Confidence distribution box plots
- Stage transition heatmaps (if sufficient data)

#### 8h. Text Reports (`02_human_reports/`)

**File:** `session_report.txt`

A comprehensive session-by-session narrative report. For each session: stage distribution in prose form, notable theme transitions, exemplar quotes for each stage observed, interrater reliability summary, and codebook code highlights.

**File:** `stage_report_{stage_key}.txt`

For each VA-MR stage: all exemplar quotes from across the corpus, organized by session and participant, with confidence scores. The primary reference document for construct validity review.

**File:** `transition_explanation.txt`

A narrative account of every notable stage transition in the corpus. For each transition where the participant moved from one VA-MR stage to another, the therapist's intervening dialogue is surfaced. This answers: *what was the therapist saying in the moments before the participant shifted to Metacognition?* If the therapist cue is long (above a configurable token threshold), an LLM summarizes it. The resulting document is the primary artifact for analyzing therapist technique.

**File:** `therapist_cues_report.txt`

A structured analysis of therapist cues organized by the participant stage transition they preceded. Groups cues by type (e.g., all cues preceding Vigilance→Metacognition transitions together) to surface patterns in therapeutic language that are associated with stage advancement.

**File:** `longitudinal_report.txt`

A cohort-level narrative covering: mean stage distributions across all sessions, individual participant trajectories with commentary on notable patterns, whether the intervention appears to be associated with increased Reappraisal prevalence over sessions, and participants who showed atypical trajectories.

---

## Output Directory Reference

```
{output_dir}/
│
├── 00_index.txt                              # File manifest: every output with description
│
├── 01_transcripts/
│   ├── diarized/                             # Copies of raw input files (provenance)
│   └── coded/
│       └── coded_transcript_{session}.txt   # Human-readable coded session (see Stage 7a)
│
├── 02_human_reports/
│   ├── per_session/
│   │   └── session_{session_id}.json        # Deep session analysis (see Stage 8a)
│   ├── per_participant/
│   │   └── participant_{pid}.json           # Longitudinal trajectory (see Stage 8b)
│   ├── per_construct/
│   │   ├── construct_{stage_key}.json       # Per-stage exemplar corpus (see Stage 8c)
│   │   └── codebook_exemplars.txt           # VCE code examples (see Stage 8c)
│   ├── session_report.txt                   # Narrative session-by-session report (Stage 8h)
│   ├── stage_report_{stage_key}.txt         # Per-stage exemplar text report (Stage 8h)
│   ├── transition_explanation.txt           # Stage transitions with therapist cues (Stage 8h)
│   ├── therapist_cues_report.txt            # Cue analysis by transition type (Stage 8h)
│   └── longitudinal_report.txt             # Cohort-level longitudinal narrative (Stage 8h)
│
├── 03_figures/
│   └── *.png                               # Stage frequency, trajectory, heatmap figures
│
├── 04_analysis_data/
│   ├── session_stats/
│   │   └── stats_{session_id}.json         # Per-session statistics (Stage 7e)
│   ├── graphing/
│   │   ├── session_stage_counts.csv        # Stage counts per session (Stage 8d)
│   │   ├── participant_stage_trajectories.csv  # Longitudinal per-participant (Stage 8d)
│   │   ├── session_confidence_distributions.csv # Confidence by stage/session (Stage 8d)
│   │   └── codebook_cooccurrence_matrix.csv # Theme×code lift matrix (Stage 8d)
│   ├── cumulative_report.json              # Dataset-wide statistics (Stage 7f)
│   ├── longitudinal_summary.json           # Cohort longitudinal summary (Stage 8e)
│   └── session_stage_progression.csv       # Stage transition matrix (Stage 8f)
│
├── 05_validation/
│   ├── content_validity_test_set.jsonl     # Exemplar/subtle/adversarial items (Stage 2)
│   ├── human_coding_evaluation_set.csv     # Stratified blind-coding sample (Stage 5)
│   ├── flagged_for_review.txt              # Split-vote segments (Stage 7c)
│   ├── human_classification_{session}.txt  # Blind-coding worksheets (Stage 7b)
│   ├── testsets/
│   │   ├── human_classification_testset_worksheet_N.txt  # Blind set N (Stage 7d)
│   │   └── AI_classification_testset_worksheet_N.txt     # AI-labeled set N (Stage 7d)
│   └── cross_validation/
│       ├── cross_validation_results.json   # Full theme×code lift matrix (Stage 4)
│       └── top_theme_code_associations.json # Filtered top associations (Stage 4)
│
├── 06_training_data/
│   ├── master_segments.jsonl               # Master dataset (Stage 6)
│   ├── master_segments.csv                 # Same, CSV format
│   ├── theme_classification.jsonl          # VA-MR training data (Stage 7g)
│   └── codebook_multilabel.jsonl           # VCE multi-label training data (Stage 7g)
│
└── 07_meta/
    ├── qra_config.json                     # Reproducible pipeline config (Stage 7h)
    ├── speaker_anonymization_key.json      # Real name → anonymized ID map (Stage 7h)
    ├── theme_definitions.json             # Framework definitions as JSON (Stage 2)
    ├── codebook_definitions.json          # VCE codebook as JSON (Stage 3b)
    ├── process_log.txt                    # Verbose LLM I/O log (if enabled)
    ├── llm_raw/                           # Theme classification checkpoints
    └── codebook_raw/
        └── found_exemplar_utterances.json # High-confidence embedding hits (Stage 3b)
```

---

## How Measures Are Derived from the Data

This section provides a direct reference for the specific numeric values that appear in reports and what computation produced them.

### `primary_stage` (int 0–3)
The VA-MR stage assigned by majority vote across all LLM runs. Value 0 = Vigilance, 1 = Avoidance, 2 = Metacognition, 3 = Reappraisal. Null if all raters abstained or voted in a split.

### `llm_confidence_primary` (float 0–1)
The **mean** confidence reported by the LLM raters who voted for the winning stage. If three raters were used and two voted for Metacognition with confidences 0.88 and 0.91, `llm_confidence_primary = (0.88 + 0.91) / 2 = 0.895`. Raters who voted for a different stage or abstained do not contribute to this average.

### `agreement_level`
Derived from vote counts vs. rater count:
- `unanimous`: every rater cast the same ballot AND no parse errors occurred
- `majority`: winning ballot count > (total raters / 2), but not unanimous
- `split`: no ballot reached majority; `primary_stage` will be null
- `none`: all raters produced parse errors; no valid ballots

### `agreement_fraction`
`n_agree / n_raters`. The count of raters who voted for the winning stage divided by total raters (including those who errored—errors count against agreement).

### `label_confidence_tier`
A categorical summary of combined agreement + confidence:
- `high`: unanimous AND confidence > 0.8
- `medium`: majority (≥2/3) AND confidence > 0.6
- `low`: everything else (including all split-vote segments regardless of confidence)

### `lift` (cross-validation)
The ratio of observed code frequency within a theme to the code's base frequency in the corpus. `lift = 2.0` means the code appears twice as often in that theme as in the dataset overall. Computed separately for each theme × code pair from the dually-labeled subset of segments.

### `final_label`
The label used for all downstream analysis. Source priority: adjudicated > human consensus > LLM. At the start of a study (before any human coding), all final labels are `llm_zero_shot`. As human coding is completed and entered, the labels for coded segments upgrade to `human_consensus` (if the human agreed with the LLM) or `adjudicated` (if a human expert resolved a disagreement).

### Stage distribution percentages
`percentage = count(stage) / n_classified * 100`. The denominator excludes segments where `final_label` is null (ABSTAIN or split). So percentages always sum to 100% across the four stages.

### Progression trend
A participant's "forward progression" score is: mean(Reappraisal fraction in sessions 2+) − mean(Reappraisal fraction in sessions 1+). Positive values indicate increasing Reappraisal engagement over the course of the intervention. This is descriptive only—the pipeline produces no inferential statistics.

---

## LLM Backends

The pipeline supports five backends, selected by `config.theme_classification.backend`:

| Backend | Use case |
|---------|---------|
| `lmstudio` | Local LM Studio server (OpenAI-compatible API at `http://127.0.0.1:1234/v1`) |
| `openrouter` | Cloud API aggregator (GPT-4o, Claude, Gemini via `openrouter.ai`) |
| `ollama` | Local Ollama server |
| `replicate` | Cloud GPU inference (Replicate Python SDK) |
| `huggingface` | Direct local GPU via transformers library |

The LLM client (`classification_tools/llm_client.py`) handles:
- Automatic context-length detection (queried from the server, cached per model)
- Retry logic with exponential backoff (up to 3 retries, 2s → 4s → 8s)
- Reasoning model handling: models like DeepSeek-R1 emit a `reasoning_content` field before the final `content`; if `content` is empty, the client falls back to parsing JSON from the reasoning field
- `no_reasoning=True` mode for segmentation refinement calls (suppresses chain-of-thought tokens)

---

## Reproducibility

Every run is fully reproducible from its `qra_config.json`:
```
python qra.py run --config ./output/07_meta/qra_config.json
```

The config file captures all model names, backend configurations, segmentation parameters, confidence thresholds, and framework/codebook selections. API keys and tokens are blanked in the saved config and must be re-supplied via environment variables (`OPENROUTER_API_KEY`, `REPLICATE_API_TOKEN`).

Checkpoint files in `07_meta/llm_raw/` allow a run to resume mid-corpus if interrupted. Re-running with `--resume-from` will skip segments that already have saved responses.

---

## Design Decisions and Their Rationale

**Why are therapist segments collected but not classified?**
Therapist segments are interleaved into the timeline so the analysis module can reconstruct the exact dialogue immediately before any participant stage transition. The PURER framework (a therapist-side operationalization of guided inquiry technique) is planned for a future pipeline phase but has not yet been operationalized.

**Why are ABSTAIN votes valid ballots?**
Dismissing a segment as irrelevant to the framework is a substantive judgment, not a failure to respond. Including ABSTAIN in the vote count means that a unanimous ABSTAIN result (all three raters agreed the segment is irrelevant) is treated as high-confidence unclassifiable, whereas a split between ABSTAIN and a stage label correctly flags the segment for human review.

**Why is the Avoidance stage often less represented?**
VA-MR stage 1 (Avoidance) represents a transitional state—participants have developed attentional control but are still using it for suppression rather than investigation. This state may be briefer or less verbally elaborated than Vigilance (which involves much distress talk) or Metacognition (which involves much reflective observation). Lower Avoidance counts in a corpus may be clinically meaningful, not a classification artifact.

**Why use lift rather than raw co-occurrence for cross-validation?**
Raw co-occurrence counts are dominated by frequent codes and mask associations with rarer phenomena. Lift normalizes by base rate, so a code that appears in 5% of the corpus and in 15% of Reappraisal segments shows a lift of 3.0—a strong signal that would be invisible in raw counts.

**Why prefer embedding over LLM for codebook classification?**
Embedding similarity is deterministic, requires no LLM API calls, and scales to large corpora cheaply. LLM-based multi-label classification is expensive and more prone to false positives when the code list is long (59 codes). The ensemble approach retains LLM classification for recall—it catches cases that semantics-based similarity may miss—while using embedding scores to anchor the confidence values.
