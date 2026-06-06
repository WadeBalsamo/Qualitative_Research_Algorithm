# tests/testdata/

## Overview

All files in this directory are **fully synthetic** — no real participant health information (PHI) or personally identifiable information (PII) is present anywhere. Speakers, utterances, and session content are entirely invented for testing purposes.

## Files

| File | Description |
|------|-------------|
| `session1.vtt` | Synthetic session — early-arc: Vigilance, Avoidance, onset of Attention Regulation |
| `session2.vtt` | Synthetic session — mid-arc: Attention Regulation, Metacognition, early Reappraisal |
| `session3.vtt` | Synthetic session — late-arc: full Reappraisal, Metacognition consolidation, arc retrospective |

## WebVTT Format

Files mirror the format of the real Move-MORE `.vtt` transcripts (sourced from `data/MMORE/c1s1.vtt` and siblings):

- **Header**: `WEBVTT` on the first line, no trailing metadata
- **Cue structure**: integer index line → timestamp line → speaker-text line → blank line
- **Timestamp format**: `HH:MM:SS,mmm --> HH:MM:SS,mmm` (comma as decimal separator, matching the Zoom/diarizer export convention)
- **Speaker prefix**: `Speaker Name: utterance text` (colon-space separator; the parser splits on the first colon)
- Timestamps are monotonically increasing and non-overlapping within each file

## Speakers

| Name | Role |
|------|------|
| `Dr. Rivera` | Therapist / group facilitator (PURER-classified) |
| `Participant A` | Participant (VAAMR-classified) |
| `Participant B` | Participant (VAAMR-classified) |

## Embedded Signals

### VAAMR (participant arc)

| Stage | Example utterance (paraphrased) |
|-------|--------------------------------|
| Vigilance (0) | "I can't stop thinking about it — it's always there, pulling my attention" |
| Avoidance (1) | "When the flare started I just breathed really hard to push the feeling away" |
| Attention Regulation (2) | "I stayed with the sensation and kept bringing my attention back" |
| Metacognition (3) | "I could see the anxiety as a separate thing — I was watching myself get anxious" |
| Reappraisal (4) | "When I really sit with what I used to call pain I can see it's many different sensations that are always changing" |

### PURER (therapist moves)

| Move | Example utterance (paraphrased) |
|------|--------------------------------|
| Phenomenological (P) | "What did you notice during your home practice this week?" |
| Utilization (U) | "How will you carry that into this coming week?" |
| Reframing (R) | "What you're describing — that IS the beginning of the practice" |
| Educate/Expectancy (E) | "The mind wanders and reacts — that's how attention works" |
| Reinforcement (R2) | "That's a really important observation" |

## Purpose

These fixtures exist for the **integration test tier** (`tests/integration/`). They are designed to be passed directly to the pipeline's VTT ingestion path (`process/transcript_ingestion.py:load_vtt_session`) without any real data, allowing full end-to-end ingest → segment → classify → assemble smoke tests in CI environments that have no access to the real corpus.
