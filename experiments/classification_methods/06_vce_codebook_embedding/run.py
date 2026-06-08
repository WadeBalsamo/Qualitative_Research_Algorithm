"""
experiments/classification_methods/06_vce_codebook_embedding/run.py
--------------------------------------------------------------------
Content-validity check: does each VCE codebook exemplar utterance get
assigned its own code by the embedding classifier?

This is a self-retrieval ("closed-retrieval") check — we build a test set
from the codebook's own exemplar_utterances, ask the classifier which codes
it assigns, and measure whether the gold code appears in the top-1 or top-3
assignments.

This is SEPARATE from the VAAMR cv_vaamr_v1 testset:
  - Multi-label: a single utterance can receive multiple codes
  - 54 codes / 6 domains (not 5 VAAMR stages)
  - No qra.db dependency — the test corpus is built directly from the
    codebook markdown (frameworks/PHENOMENOLOGY_CODEBOOK.md)

Provenance commits: f84d38b (Mar 2026 codebook multi-classification),
                    cdbfc87 (Apr 2026 Qwen3-Embedding-8B + IRR),
                    002757e (Apr 2026 codebook application bug fix).

Arms
----
triple_veto_default   — EmbeddingClassifierConfig defaults (similarity_threshold=1.375,
                         two_pass=True, exemplar_weight=0.5)
relaxed_threshold     — similarity_threshold=1.1  (lower bar, more codes assigned)
no_two_pass           — two_pass=False (single-pass only, no exemplar accumulation)

Usage
-----
    # Print corpus stats, no model load:
    python run.py --dry-run

    # Run all arms and write results/ (requires model download):
    python run.py --output-dir ./results

    # Run one arm only:
    python run.py --arm relaxed_threshold --output-dir ./results

NOTE: NOT executed during this build. Run-ready; model not loaded.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# sys.path bootstrap — mirrors experiments/gnn_reliability/harness.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(os.path.dirname(_HERE))          # experiments/
_ROOT = os.path.dirname(_EXPERIMENTS)                            # repo root
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Lightweight imports (no heavy deps at module level)
# ---------------------------------------------------------------------------
from constructs.codebook.phenomenology_codebook import get_phenomenology_codebook  # noqa: E402
from constructs.codebook.codebook_schema import Codebook, CodeDefinition           # noqa: E402
from classification_tools.data_structures import Segment                           # noqa: E402

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

ALL_ARMS = ("triple_veto_default", "relaxed_threshold", "no_two_pass")


def _make_config(arm: str):
    """
    Return an EmbeddingClassifierConfig for the given arm.
    Heavy import is deferred — only called on real (non-dry-run) execution.
    """
    from classification_tools.codebook_multilabel.config import EmbeddingClassifierConfig

    if arm == "triple_veto_default":
        return EmbeddingClassifierConfig(
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=1.375,
            two_pass=True,
            exemplar_weight=0.5,
            use_query_prefix=False,   # all-MiniLM has no 'query' prompt
        )
    elif arm == "relaxed_threshold":
        return EmbeddingClassifierConfig(
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=1.1,
            two_pass=True,
            exemplar_weight=0.5,
            use_query_prefix=False,
        )
    elif arm == "no_two_pass":
        return EmbeddingClassifierConfig(
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=1.375,
            two_pass=False,
            exemplar_weight=0.5,
            use_query_prefix=False,
        )
    else:
        raise ValueError(f"Unknown arm: {arm!r}. Choose from {ALL_ARMS}")


# ---------------------------------------------------------------------------
# Content-validity corpus builder
# ---------------------------------------------------------------------------

def build_cv_corpus(codebook: Codebook) -> List[Tuple[Segment, str]]:
    """
    Build the content-validity test set from codebook exemplar utterances.

    For each CodeDefinition with at least one exemplar_utterance, create one
    Segment per utterance.  The gold label is the code's code_id.

    Returns list of (Segment, gold_code_id).
    The segment_id encodes position: "<code_id>__ex<n>" for reproducibility.
    """
    items: List[Tuple[Segment, str]] = []
    for code in codebook.codes:
        for n, utterance in enumerate(code.exemplar_utterances):
            seg = Segment(
                segment_id=f"{code.code_id}__ex{n}",
                trial_id="vce_cv",
                participant_id="cv_participant",
                session_id="vce_cv_session",
                session_number=1,
                segment_index=len(items),
                start_time_ms=0,
                end_time_ms=0,
                total_segments_in_session=0,
                speaker="participant",
                text=utterance,
                word_count=len(utterance.split()),
                session_file="",
            )
            items.append((seg, code.code_id))
    return items


# ---------------------------------------------------------------------------
# Local scorer (multi-label, does NOT use common/cv_scoring)
# ---------------------------------------------------------------------------

def _score(
    results: Dict[str, List],   # segment_id -> list[CodeAssignment]
    items: List[Tuple[Segment, str]],
    arm: str,
    output_dir: Path,
) -> Dict:
    """
    Compute content-validity metrics and write results.csv + results.md.

    Metrics (per item, then aggregated):
      top1      — gold code_id is the highest-confidence assigned code
      top3      — gold code_id appears among the top-3 by confidence
      coverage  — fraction of items that received >=1 assigned code

    'top1' / 'top3' are computed only over items that received >=1 code
    (partial coverage does not penalise retrieval quality).  A second
    'strict_top1' / 'strict_top3' is computed over ALL items (unanswered
    items score 0) so that threshold effects are visible.
    """
    n_items = len(items)
    n_covered = 0
    n_top1 = 0
    n_top3 = 0
    n_strict_top1 = 0
    n_strict_top3 = 0

    per_item_rows = []
    for seg, gold_code in items:
        assignments = results.get(seg.segment_id, [])
        # Sort by confidence descending
        sorted_asgn = sorted(assignments, key=lambda a: a.confidence, reverse=True)
        covered = len(sorted_asgn) > 0
        code_ids_top3 = [a.code_id for a in sorted_asgn[:3]]
        top1_hit = bool(sorted_asgn) and sorted_asgn[0].code_id == gold_code
        top3_hit = gold_code in code_ids_top3

        if covered:
            n_covered += 1
        if top1_hit:
            n_top1 += 1
        if top3_hit:
            n_top3 += 1
        if top1_hit:
            n_strict_top1 += 1
        if top3_hit:
            n_strict_top3 += 1

        per_item_rows.append({
            "segment_id": seg.segment_id,
            "gold_code": gold_code,
            "covered": covered,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "n_assigned": len(sorted_asgn),
            "top1_pred": sorted_asgn[0].code_id if sorted_asgn else "",
            "top3_preds": "|".join(code_ids_top3),
        })

    # Aggregate
    coverage = n_covered / n_items if n_items else 0.0
    # Retrieval quality over covered items
    top1_acc = n_top1 / n_covered if n_covered else 0.0
    top3_acc = n_top3 / n_covered if n_covered else 0.0
    # Strict (over all items)
    strict_top1 = n_strict_top1 / n_items if n_items else 0.0
    strict_top3 = n_strict_top3 / n_items if n_items else 0.0

    summary = {
        "arm": arm,
        "n_items": n_items,
        "n_covered": n_covered,
        "coverage": round(coverage, 4),
        "top1_acc_covered": round(top1_acc, 4),
        "top3_acc_covered": round(top3_acc, 4),
        "strict_top1": round(strict_top1, 4),
        "strict_top3": round(strict_top3, 4),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-item CSV (one per arm)
    item_csv = output_dir / f"items_{arm}.csv"
    with open(item_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_item_rows[0].keys()) if per_item_rows else [])
        writer.writeheader()
        writer.writerows(per_item_rows)

    print(f"  [{arm}] coverage={coverage:.3f}  top1(cov)={top1_acc:.3f}  "
          f"top3(cov)={top3_acc:.3f}  strict_top1={strict_top1:.3f}  "
          f"strict_top3={strict_top3:.3f}  (n={n_items}, covered={n_covered})")
    return summary


def _write_results_files(summaries: List[Dict], output_dir: Path) -> None:
    """Write results.csv and results.md from aggregated summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # results.csv
    csv_path = output_dir / "results.csv"
    fieldnames = [
        "arm", "n_items", "n_covered", "coverage",
        "top1_acc_covered", "top3_acc_covered",
        "strict_top1", "strict_top3",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nResults written to: {csv_path}")

    # results.md — markdown table
    md_path = output_dir / "results.md"
    header = (
        "| arm | n_items | coverage | top1 (covered) | top3 (covered) "
        "| strict_top1 | strict_top3 |\n"
        "|-----|---------|----------|----------------|----------------"
        "|-------------|-------------|\n"
    )
    rows = []
    for s in summaries:
        rows.append(
            f"| {s['arm']} | {s['n_items']} | {s['coverage']:.3f} "
            f"| {s['top1_acc_covered']:.3f} | {s['top3_acc_covered']:.3f} "
            f"| {s['strict_top1']:.3f} | {s['strict_top3']:.3f} |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# VCE Codebook Embedding — Content Validity Results\n\n")
        f.write(
            "Self-retrieval check: does each exemplar utterance get assigned "
            "its own code?\n\n"
        )
        f.write(header)
        f.write("\n".join(rows) + "\n")
        f.write(
            "\n**top1/top3 (covered)**: accuracy restricted to items that "
            "received >=1 code assignment.\n"
            "**strict_top1/top3**: accuracy over ALL items "
            "(unanswered = 0; shows threshold effects).\n"
        )
    print(f"Results table:      {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Content-validity check for the VCE codebook embedding classifier. "
            "Builds a test set from codebook exemplar utterances and measures "
            "whether each utterance retrieves its own code (top-1 / top-3 / coverage). "
            "Use --dry-run to inspect the corpus WITHOUT loading any model."
        )
    )
    p.add_argument(
        "--output-dir",
        default="./results",
        help="Directory for results.csv, results.md, and per-arm item CSVs "
             "(default: ./results).",
    )
    p.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model ID (default: all-MiniLM-L6-v2). "
             "Override for experimentation; model must be sentence-transformers compatible.",
    )
    p.add_argument(
        "--arm",
        default="all",
        choices=list(ALL_ARMS) + ["all"],
        help="Which arm to run (default: all).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print corpus statistics without loading any embedding model. "
             "Safe to run offline.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Always safe: load codebook (pure-python markdown parse, no model)
    codebook = get_phenomenology_codebook()
    items = build_cv_corpus(codebook)

    n_codes = len(codebook.codes)
    n_exemplar_items = len(items)
    codes_with_exemplars = sum(1 for c in codebook.codes if c.exemplar_utterances)
    codes_without_exemplars = n_codes - codes_with_exemplars
    domains = codebook.domain_names

    print("=" * 60)
    print("VCE Codebook Embedding — Content Validity Check")
    print("=" * 60)
    print(f"  Codebook:              {codebook.name} v{codebook.version}")
    print(f"  n_codes:               {n_codes}")
    print(f"  Domains ({len(domains)}):           {', '.join(domains)}")
    print(f"  codes_with_exemplars:  {codes_with_exemplars}")
    print(f"  codes_without_exemplars: {codes_without_exemplars}")
    print(f"  n_exemplar_items:      {n_exemplar_items}  "
          f"(one Segment per exemplar utterance)")

    if args.dry_run:
        print()
        print("--dry-run: corpus built. No model loaded. Exiting.")
        print()
        print("To run the full evaluation:")
        print(
            f"  python run.py --output-dir {args.output_dir} "
            f"[--arm triple_veto_default|relaxed_threshold|no_two_pass|all]"
        )
        return

    # Real run: lazy-import heavy deps here, not at module level
    from classification_tools.codebook_multilabel.embedding_classifier import (
        EmbeddingCodebookClassifier,
    )

    arms_to_run = list(ALL_ARMS) if args.arm == "all" else [args.arm]
    output_dir = Path(args.output_dir)
    segments = [seg for seg, _ in items]

    summaries = []
    for arm in arms_to_run:
        print(f"\n--- Arm: {arm} ---")
        cfg = _make_config(arm)
        # Override the model from CLI if provided
        if args.model != "all-MiniLM-L6-v2":
            cfg.embedding_model = args.model

        classifier = EmbeddingCodebookClassifier(cfg)
        results = classifier.classify_segments(segments, codebook)
        summary = _score(results, items, arm, output_dir)
        summaries.append(summary)

    _write_results_files(summaries, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
