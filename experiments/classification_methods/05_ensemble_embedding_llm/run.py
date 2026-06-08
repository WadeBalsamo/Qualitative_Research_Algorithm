"""
experiments/classification_methods/05_ensemble_embedding_llm/run.py
-------------------------------------------------------------------
Embedding-similarity + single-LLM ensemble for VAAMR single-label
classification.  Adapted from the codebook-era ensemble (commit f84d38b)
to the VAAMR single-label task.

Arms
----
  llm_only         — single LLM pass (no embedding; baseline within this exp)
  embedding_only   — argmax cosine similarity to VAAMR theme anchors (no LLM)
  agree_or_llm     — use agreement; on disagreement keep LLM prediction
  agree_or_embedding — use agreement; on disagreement keep embedding prediction
  union_flag       — on disagreement: pred=LLM, marks needs_review in meta

Usage
-----
  # Live run (needs LM Studio at 10.0.0.58:1234 + sentence-transformers):
  python experiments/classification_methods/05_ensemble_embedding_llm/run.py \\
      --output-dir ./data/output

  # Dry run (no model/network — prints arm list + config and exits):
  python experiments/classification_methods/05_ensemble_embedding_llm/run.py --dry-run

  # Override LLM model:
  python experiments/classification_methods/05_ensemble_embedding_llm/run.py \\
      --model qwen/qwen3-8b --output-dir ./data/output

NOT executed in this build; run-ready; needs live LM Studio backend at 10.0.0.58:1234.
"""

import argparse
import pathlib
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# sys.path: make common/ importable from any cwd
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent
_METHODS_ROOT = _HERE.parent
if str(_METHODS_ROOT) not in sys.path:
    sys.path.insert(0, str(_METHODS_ROOT))

from common import data, cv_scoring, prompt_harness  # noqa: E402
from common.data import CvItem                        # noqa: E402


# ---------------------------------------------------------------------------
# Embedding-similarity arm (lazy-loaded)
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def _build_theme_anchors(framework) -> Dict[int, str]:
    """
    Build a text anchor string for each VAAMR theme.

    Concatenates the theme short_name, description, and any exemplar text
    so the embedding captures the full semantic space of each stage.
    """
    anchors = {}
    for theme in framework.themes:
        parts = [theme.short_name]
        if theme.description:
            parts.append(theme.description)
        if hasattr(theme, 'exemplars') and theme.exemplars:
            # exemplars may be a list of strings or a list of dicts
            for ex in theme.exemplars[:3]:   # limit to 3 for anchor quality
                if isinstance(ex, str):
                    parts.append(ex)
                elif isinstance(ex, dict):
                    parts.append(ex.get('text', '') or ex.get('utterance', ''))
        anchors[theme.theme_id] = ' '.join(p for p in parts if p).strip()
    return anchors


def _embedding_preds(items: List[CvItem]) -> Dict[str, Optional[int]]:
    """
    Classify items by argmax cosine similarity to VAAMR theme anchors.

    Lazy-imports sentence_transformers so the module stays import-clean.
    All-MiniLM-L6-v2 is the same embedding backbone used by the VCE
    codebook classifier (src/codebook/embedding_classifier.py), and is
    already cached in the repo environment.

    Parameters
    ----------
    items : list of CvItem

    Returns
    -------
    dict mapping item_id (str) -> predicted stage (int), never None
    (embedding always produces a winner — abstain is not possible).
    """
    # Lazy import — NOT executed during --dry-run or import-only passes.
    from sentence_transformers import SentenceTransformer  # noqa: F401

    import numpy as np

    framework = data.load_vaamr()
    anchors = _build_theme_anchors(framework)

    stage_ids = sorted(anchors.keys())
    anchor_texts = [anchors[s] for s in stage_ids]
    item_texts   = [item.text for item in items]

    model = SentenceTransformer(_EMBEDDING_MODEL)

    anchor_embs = model.encode(anchor_texts, normalize_embeddings=True)
    item_embs   = model.encode(item_texts,   normalize_embeddings=True)

    # cosine similarity = dot product when both are L2-normalised
    sims = item_embs @ anchor_embs.T      # shape: (n_items, n_stages)

    preds: Dict[str, Optional[int]] = {}
    for i, item in enumerate(items):
        winner_idx = int(np.argmax(sims[i]))
        preds[item.item_id] = stage_ids[winner_idx]

    return preds


# ---------------------------------------------------------------------------
# Ensemble combination
# ---------------------------------------------------------------------------

def _combine(
    llm_preds:   Dict[str, Optional[int]],
    emb_preds:   Dict[str, Optional[int]],
    strategy:    str,
    item_ids:    List[str],
) -> Dict[str, Optional[int]]:
    """
    Combine LLM and embedding predictions according to strategy.

    Strategies
    ----------
    agree_or_llm       : agree → that prediction; disagree → LLM
    agree_or_embedding : agree → that prediction; disagree → embedding
    union_flag         : agree → that prediction; disagree → LLM
                         (disagreement count tracked externally in meta)
    """
    result: Dict[str, Optional[int]] = {}
    for iid in item_ids:
        l = llm_preds.get(iid)
        e = emb_preds.get(iid)
        if l == e:
            result[iid] = l
        else:
            if strategy in ('agree_or_llm', 'union_flag'):
                result[iid] = l
            else:  # agree_or_embedding
                result[iid] = e
    return result


def _disagreement_count(
    llm_preds: Dict[str, Optional[int]],
    emb_preds: Dict[str, Optional[int]],
    item_ids:  List[str],
) -> int:
    return sum(
        1 for iid in item_ids
        if llm_preds.get(iid) != emb_preds.get(iid)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='05_ensemble_embedding_llm',
        description=(
            'Embedding-similarity + single-LLM ensemble for VAAMR '
            '(adapted from codebook-era f84d38b).'
        ),
    )
    p.add_argument(
        '--output-dir', default=None,
        help='Pipeline output dir containing qra.db.  Default: data/Meta/qra.db.',
    )
    p.add_argument(
        '--model', default='nvidia/nemotron-3-super',
        help='LM Studio model identifier.  Default: nvidia/nemotron-3-super.',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Print arm list + config and exit — no model loads, no network.',
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    prompt  = prompt_harness.PromptSpec()
    harness = prompt_harness.HarnessSpec(
        n_runs=1,
        merge='first',
        model=args.model,
    )

    framework = data.load_vaamr()
    n_anchors = len(framework.themes)

    _results_dir = _HERE
    results_csv = _results_dir / 'results.csv'
    results_md  = _results_dir / 'results.md'

    _arm_names = [
        'llm_only',
        'embedding_only',
        'agree_or_llm',
        'agree_or_embedding',
        'union_flag',
    ]

    if args.dry_run:
        print('=== 05_ensemble_embedding_llm — DRY RUN ===')
        print()
        print('  Arms:')
        for a in _arm_names:
            print(f'    {a}')
        print()
        print(f'  Would embed {n_anchors} VAAMR theme anchors with {_EMBEDDING_MODEL}')
        print(f'  Would call LLM ({args.model}) for items')
        print()
        print('  PromptSpec:')
        print(f'    context_window     = {prompt.context_window}')
        print(f'    randomize          = {prompt.randomize}')
        print(f'    zero_shot          = {prompt.zero_shot}')
        print(f'    n_exemplars        = {prompt.n_exemplars}')
        print(f'    include_subtle     = {prompt.include_subtle}')
        print(f'    include_adversarial= {prompt.include_adversarial}')
        print(f'    strict_json        = {prompt.strict_json}')
        print()
        print('  HarnessSpec:')
        print(f'    n_runs             = {harness.n_runs}')
        print(f'    merge              = {harness.merge!r}')
        print(f'    backend            = {harness.backend!r}')
        print(f'    base_url           = {harness.base_url}')
        print(f'    model              = {harness.model}')
        print()
        print(f'  output_dir  : {args.output_dir or "(default) data/Meta/qra.db"}')
        print(f'  results_csv : {results_csv}')
        print(f'  results_md  : {results_md}')
        print()
        print('  [DRY RUN] No model loads, no LLM calls.')
        return

    # -----------------------------------------------------------------------
    # Live run
    # -----------------------------------------------------------------------
    print(f'Loading CV items from {args.output_dir or "default DB"}...')
    items = data.load_cv_items(output_dir=args.output_dir)
    item_ids = [item.item_id for item in items]
    print(f'  {len(items)} items loaded.')

    # --- LLM preds (single-run) ---
    print(f'Running LLM arm ({args.model}, n_runs=1)...')
    llm_preds = prompt_harness.run_llm_arm(items, prompt, harness)

    # --- Embedding preds ---
    print(f'Running embedding arm ({_EMBEDDING_MODEL})...')
    emb_preds = _embedding_preds(items)

    # --- Score all arms ---
    disagree_n = _disagreement_count(llm_preds, emb_preds, item_ids)
    print(f'  LLM/embedding disagreement on {disagree_n}/{len(items)} items.')

    arm_configs = [
        ('llm_only',           llm_preds,                                              {}),
        ('embedding_only',     emb_preds,                                              {}),
        ('agree_or_llm',       _combine(llm_preds, emb_preds, 'agree_or_llm',   item_ids), {}),
        ('agree_or_embedding', _combine(llm_preds, emb_preds, 'agree_or_embedding', item_ids), {}),
        ('union_flag',         _combine(llm_preds, emb_preds, 'union_flag',      item_ids),
                               {'disagree_n': disagree_n, 'disagree_frac': round(disagree_n / max(len(items), 1), 4)}),
    ]

    print()
    print('=== Results ===')
    for arm_name, preds, meta in arm_configs:
        metrics = cv_scoring.score_arm(
            arm_name,
            preds,
            output_dir=args.output_dir,
            results_csv=results_csv,
            results_md=results_md,
            meta=meta if meta else None,
        )
        print(f'  [{arm_name}]')
        print(f'    acc_overall    : {metrics["acc_overall"]}')
        print(f'    acc_secondary  : {metrics["acc_secondary"]}')
        print(f'    acc_clear      : {metrics["acc_clear"]}')
        print(f'    acc_subtle     : {metrics["acc_subtle"]}')
        print(f'    acc_adversarial: {metrics["acc_adversarial"]}')

    print()
    print(f'  results_csv : {results_csv}')
    print(f'  results_md  : {results_md}')


if __name__ == '__main__':
    main()
