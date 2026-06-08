# 06 — VCE Codebook Embedding: Content Validity Check

## What this is

A self-retrieval content-validity check on the VCE (Varieties of Contemplative
Experience) phenomenology codebook's own exemplar utterances, using the
sentence-transformer embedding classifier (`EmbeddingCodebookClassifier`).

**The question asked:** For each exemplar utterance in the codebook, does the
embedding classifier assign the utterance's own code as the top-1 prediction?
Does it appear in the top-3?

This is a **closed-retrieval** ("does the codebook know itself?") validity check,
not a generalization check. It is a necessary but not sufficient condition for
real-transcript validity: if the classifier cannot retrieve a code from the
codebook's own exemplars, it will reliably fail on novel transcript text.

## Relationship to other validation axes

| Axis | What it tests | n_codes | Test corpus source |
|------|--------------|---------|-------------------|
| `cv_vaamr_v1` (VAAMR CV testset) | Single-label VAAMR stage classification | 5 stages | Curated exemplar/subtle/adversarial utterances in `qra.db` |
| **This experiment** | Multi-label VCE codebook embedding classifier | 54 codes | Codebook's own `exemplar_utterances` from `PHENOMENOLOGY_CODEBOOK.md` |

These axes are **orthogonal**: cv_vaamr_v1 is single-label ordinal; this is
multi-label nominal. Do not use `common/cv_scoring` here — a local scorer
is implemented in `run.py:_score()`.

## Provenance

This experiment recreates the VCE triple-veto embedding classifier introduced
across three commits:

- **f84d38b** (2026-03-15) — Initial codebook multi-classification alongside
  theme-construct classification; introduced `codebook/embedding_classifier.py`
  with triple-veto similarity logic.
- **cdbfc87** (2026-04-11) — Upgraded default model to `Qwen/Qwen3-Embedding-8B`
  (4096-dim); added IRR measures with multiple LLMs; rewrote scoring formula
  (definition + criteria + exemplar weighted cosine similarity with two-pass
  exemplar accumulation).
- **002757e** (2026-04-21) — Fixed codebook application bug; stable state of
  the embedding classifier before it was subsumed into the ensemble layer.

## Original vs this approximation

The original pipeline (commits above) used:
- **Three large causal-LLM embedding models** (Llama-4, Mixtral-8x7B,
  Qwen3-70B) with `cosine + euclidean + cosine-distance` triple-veto: a code
  was only assigned if all three models agreed it exceeded threshold.
- 4096-dim embeddings from `Qwen/Qwen3-Embedding-8B`.
- Asymmetric encoding with a `query` instruction prefix for segment texts.

This experiment uses **`all-MiniLM-L6-v2` (384-dim, symmetric encoding)**
in place of the Qwen3-Embedding-8B, because the repo's environment pins
`transformers==4.42.4`, which cannot load Qwen3 architecture weights
(see memory note: `project_gnn_embedding_transformers_pin.md`). The triple-veto
multi-model structure is approximated by the three arms below (varying
threshold and two-pass settings), not replicated exactly.

**This is honestly a downgrade in embedding quality.** Results from this
experiment reflect `all-MiniLM-L6-v2` capability, not the original
Qwen3-Embedding-8B results. Scores here set a floor; the real pipeline used
substantially richer embeddings.

## Arms

| Arm | `similarity_threshold` | `two_pass` | Notes |
|-----|----------------------|------------|-------|
| `triple_veto_default` | 1.375 (default) | True | Replicates default pipeline config |
| `relaxed_threshold` | 1.1 | True | Lower bar — more codes assigned; higher recall, lower precision |
| `no_two_pass` | 1.375 | False | Single-pass only; ablates the exemplar accumulation loop |

All arms use `embedding_model='all-MiniLM-L6-v2'` and `exemplar_weight=0.5`,
`use_query_prefix=False` (all-MiniLM has no query prompt).

## Metrics

- **coverage**: fraction of exemplar items assigned ≥1 code (any code)
- **top1 (covered)**: among covered items, gold code is the highest-confidence
  assignment
- **top3 (covered)**: gold code appears in the top-3 assignments by confidence
- **strict_top1 / strict_top3**: same, computed over ALL items (unanswered = 0;
  exposes threshold sensitivity)

## Run command

```bash
# From the repo root:
cd /home/wisgood/qra/Qualitative_Research_Algorithm

# Inspect corpus — no model load, offline-safe:
.venv/bin/python experiments/classification_methods/06_vce_codebook_embedding/run.py --dry-run

# Run all arms (requires all-MiniLM-L6-v2 cached or internet):
.venv/bin/python experiments/classification_methods/06_vce_codebook_embedding/run.py \
    --output-dir experiments/classification_methods/06_vce_codebook_embedding/results

# Run one arm:
.venv/bin/python experiments/classification_methods/06_vce_codebook_embedding/run.py \
    --arm relaxed_threshold \
    --output-dir experiments/classification_methods/06_vce_codebook_embedding/results
```

## Output files

```
results/
├── results.csv          — arm × metric summary table
├── results.md           — same as markdown
├── items_triple_veto_default.csv   — per-item hit/miss detail
├── items_relaxed_threshold.csv
└── items_no_two_pass.csv
```

## Current codebook state

As of the build date, `PHENOMENOLOGY_CODEBOOK.md` defines the `### Exemplar
Utterances` parser contract (blockquote format) but **none of the 54 code
definitions contain populated exemplar utterances** (no `> ` blockquote lines).
The `--dry-run` therefore reports `n_exemplar_items: 0`.

This means the content-validity check cannot run until exemplar utterances are
added to the codebook markdown. The experiment harness is fully wired and will
run automatically once exemplars are added — `--dry-run` will print a non-zero
`n_exemplar_items` count confirming they were parsed, and then the real run
path will work without code changes.

To add exemplars, edit `frameworks/PHENOMENOLOGY_CODEBOOK.md` and add
`### Exemplar Utterances` sections with blockquote lines (`> text`) under
each code block, following the parser contract described in the file header.

## Status

**NOT executed in this build. Run-ready.**

The `--dry-run` path is verified offline (no model load). The real
classification path (`EmbeddingCodebookClassifier.classify_segments`) is
wired correctly but has not been invoked.

**Blocker for real run:** `n_exemplar_items = 0` because no exemplar utterances
are populated in `PHENOMENOLOGY_CODEBOOK.md` yet. Add exemplars to unblock.
