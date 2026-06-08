"""
experiments/classification_methods/common/prompt_harness.py
------------------------------------------------------------
Parametrized spec objects + builder used by downstream LLM battery scripts.

These objects are defined and import-clean.  Actual LLM calls only happen
when run_llm_arm() is explicitly invoked — nothing here triggers a model
load at import time.

Public API
----------
PromptSpec   — controls prompt construction (exemplars, context window, etc.)
    .to_theme_config() -> ThemeClassificationConfig

HarnessSpec  — controls run orchestration (n_runs, models, merge strategy)
    .to_client() -> LLMClient
    .merge_fn()  -> callable

run_llm_arm(items, prompt, harness) -> dict[item_id -> int | None]
    Wires classify_segments_zero_shot with the specs.
    NOTE: Makes real LLM calls.  Do NOT invoke in import-only passes.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# sys.path bootstrap — same pattern as data.py / cv_scoring.py
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Lightweight imports only at module level — no torch, no sentence-transformers.
from constructs.config import ThemeClassificationConfig           # noqa: E402
from classification_tools.llm_client import LLMClient, LLMClientConfig  # noqa: E402
from classification_tools.majority_vote import vote_single_label, ABSTAIN  # noqa: E402
from classification_tools.data_structures import Segment          # noqa: E402
from common.data import CvItem                                    # noqa: E402


# ---------------------------------------------------------------------------
# PromptSpec
# ---------------------------------------------------------------------------

@dataclass
class PromptSpec:
    """
    Controls how the LLM prompt is constructed.

    Maps onto ThemeClassificationConfig fields (see .to_theme_config()):
        context_window     -> context_window_segments
        randomize          -> randomize_codebook
        zero_shot          -> zero_shot_prompt
        n_exemplars        -> prompt_n_exemplars
        include_subtle     -> prompt_include_subtle
        include_adversarial-> prompt_include_adversarial
        with_criteria      -> (no direct field; reflected by zero_shot_prompt=False)
        strict_json        -> (prompt template always strict JSON; field reserved)
    """
    context_window: int = 2
    randomize: bool = True
    zero_shot: bool = False
    n_exemplars: Optional[int] = None
    include_subtle: bool = True
    include_adversarial: bool = True
    with_criteria: bool = True      # reserved — prompt template always includes criteria
    strict_json: bool = True        # reserved — template always strict JSON

    def to_theme_config(self, **overrides) -> 'ThemeClassificationConfig':
        """
        Build a ThemeClassificationConfig from this PromptSpec.

        Field mapping:
            context_window      -> context_window_segments
            randomize           -> randomize_codebook
            zero_shot           -> zero_shot_prompt
            n_exemplars         -> prompt_n_exemplars
            include_subtle      -> prompt_include_subtle
            include_adversarial -> prompt_include_adversarial
            (with_criteria / strict_json reserved; not forwarded)

        Callers may pass keyword overrides for any ThemeClassificationConfig
        field (e.g. model, backend, lmstudio_base_url, n_runs, etc.).
        """
        cfg = ThemeClassificationConfig(
            context_window_segments=self.context_window,
            randomize_codebook=self.randomize,
            zero_shot_prompt=self.zero_shot,
            prompt_n_exemplars=self.n_exemplars,
            prompt_include_subtle=self.include_subtle,
            prompt_include_adversarial=self.include_adversarial,
            # disable short-segment merging for CV items (each item is atomic)
            min_classifiable_words=0,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# HarnessSpec
# ---------------------------------------------------------------------------

@dataclass
class HarnessSpec:
    """
    Controls run orchestration.

    merge strategies:
        'first'           — use the first run's vote unconditionally
        'majority'        — vote_single_label (standard pipeline)
        'confidence'      — vote_single_label (uses confidence tiebreak)
        'triplicate_flag' — vote_single_label + flag when all 3 raters diverge
    """
    n_runs: int = 1
    per_run_models: Optional[List[str]] = None
    merge: str = 'majority'          # 'first'|'majority'|'confidence'|'triplicate_flag'
    secondary_weight: float = 0.6
    presence_threshold: float = 0.5
    abstain_as_ballot: bool = True
    backend: str = 'lmstudio'
    base_url: str = 'http://10.0.0.58:1234/v1'
    model: str = 'nvidia/nemotron-3-super'

    def to_client(self) -> LLMClient:
        """Build an LLMClient from this spec."""
        cfg = LLMClientConfig(
            backend=self.backend,
            model=self.model,
            lmstudio_base_url=self.base_url,
            models=self.per_run_models or [self.model],
        )
        return LLMClient(cfg)

    def merge_fn(self) -> Callable[[List[Optional[Dict]]], Dict]:
        """
        Return the merge callable suitable for classify_segments' merge_runs arg.

        All strategies ultimately delegate to vote_single_label (which is the
        canonical pipeline aggregator); the 'first' and 'triplicate_flag'
        variants wrap it with additional pre/post processing.
        """
        sw = self.secondary_weight
        pt = self.presence_threshold

        if self.merge == 'first':
            # Take the first non-None run and treat it as unanimous.
            def _first(parsed_runs):
                for run in parsed_runs:
                    if run is not None:
                        return vote_single_label([run], secondary_weight=sw,
                                                 presence_threshold=pt)
                return vote_single_label([None], secondary_weight=sw,
                                         presence_threshold=pt)
            return _first

        elif self.merge == 'triplicate_flag':
            # Standard majority vote; sets needs_review=True when all 3 raters
            # disagree (agreement_level == 'split' with n_ballots == 3).
            def _triplicate(parsed_runs):
                result = vote_single_label(parsed_runs, secondary_weight=sw,
                                           presence_threshold=pt)
                if (result.get('agreement_level') == 'split' and
                        result.get('n_ballots', 0) == 3):
                    result['needs_review'] = True
                    result['triplicate_flag'] = True
                return result
            return _triplicate

        else:
            # 'majority' or 'confidence' — both use vote_single_label as-is;
            # confidence tiebreaking is always active inside vote_single_label.
            def _majority(parsed_runs):
                return vote_single_label(parsed_runs, secondary_weight=sw,
                                         presence_threshold=pt)
            return _majority


# ---------------------------------------------------------------------------
# run_llm_arm
# ---------------------------------------------------------------------------

def run_llm_arm(
    items: List[CvItem],
    prompt: PromptSpec,
    harness: HarnessSpec,
) -> Dict[str, Optional[int]]:
    """
    Classify CV items with a live LLM and return {item_id -> predicted_stage}.

    WARNING: This function makes real LLM calls.  It MUST NOT be invoked
    during import-only validation passes.  Battery scripts call this only
    when running in live mode (not --dry-run).

    Parameters
    ----------
    items   : list of CvItem (from common.data.load_cv_items)
    prompt  : PromptSpec controlling prompt construction
    harness : HarnessSpec controlling run/merge configuration

    Returns
    -------
    dict mapping item_id (str) -> predicted primary stage (int) or None (abstain).
    """
    # Inline the heavy imports so this module stays import-clean at the top level.
    from classification_tools.theme_llm.llm_classifier import classify_segments_zero_shot
    from common.data import load_vaamr

    framework = load_vaamr()
    segments = [item.to_segment() for item in items]

    # Build a ThemeClassificationConfig wired from both specs.
    cfg = prompt.to_theme_config(
        model=harness.model,
        backend=harness.backend,
        lmstudio_base_url=harness.base_url,
        n_runs=harness.n_runs,
        per_run_models=list(harness.per_run_models) if harness.per_run_models else [],
        evidence_secondary_weight=harness.secondary_weight,
        evidence_presence_threshold=harness.presence_threshold,
        # Write checkpoints to a temp location; callers may override output_dir later.
        output_dir=str(Path('/tmp') / 'qra_cv_harness'),
        save_interval=len(items) + 1,   # disable mid-run saves for CV items
    )

    results_all, _ = classify_segments_zero_shot(
        segments=segments,
        framework=framework,
        config=cfg,
    )

    # Extract the consensus primary stage from each result dict.
    predictions: Dict[str, Optional[int]] = {}
    for item in items:
        iid = item.item_id
        result = results_all.get(iid)
        if result is None:
            predictions[iid] = None
            continue
        consensus = result.get('consensus', {})
        vote = consensus.get('consensus_vote')
        if vote is None or vote == ABSTAIN:
            predictions[iid] = None
        else:
            try:
                predictions[iid] = int(vote)
            except (TypeError, ValueError):
                predictions[iid] = None

    return predictions
