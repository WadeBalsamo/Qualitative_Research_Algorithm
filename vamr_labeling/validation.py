"""
validation.py
-------------
Stage 5: Human Validation and Metrics.

Adapted from:
  - create_binary_dataset() (docs 3, 4, 6, 7, 11) -> create_balanced_evaluation_set():
    extended from binary to 4-class with trial stratification.
  - custom_classification_report() (docs 3, 14, 16) -> compute_validation_metrics():
    extended from binary sensitivity/specificity/precision/FNR/F1 to multi-class
    per-stage metrics.
  - Content validity evaluation pattern from classify_ctl.py
  - save_classification_performance() from metrics_report module
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable
from sklearn.metrics import (
    confusion_matrix, f1_score, cohen_kappa_score,
)


# ---------------------------------------------------------------------------
# Balanced evaluation set construction
# Adapted from create_binary_dataset() throughout the Text Psychometrics codebase
# ---------------------------------------------------------------------------

def create_balanced_evaluation_set(
    segments_df: pd.DataFrame,
    n_per_stage: int = 50,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Create a balanced evaluation set for human coding comparison.

    Adapted from create_binary_dataset() which appears throughout the
    Text Psychometrics codebase:

        df_metadata_tag_1 = df_metadata[df_metadata[dv]==1].sample(
            n=n_per_dv, random_state=123)
        df_metadata_tag_0 = df_metadata[df_metadata[dv]==0].sample(
            n=n_per_dv, random_state=123)
        df_metadata_tag = pd.concat([df_metadata_tag_1, df_metadata_tag_0])
            .sample(frac=1).reset_index(drop=True)

    Extended from binary to 4-class balanced sampling, additionally
    stratifying by trial_id to ensure representation across datasets.
    """
    balanced_segments = []

    for stage_id in range(4):
        stage_segments = segments_df[segments_df['primary_stage'] == stage_id]

        if len(stage_segments) < n_per_stage:
            print(
                f"Warning: only {len(stage_segments)} segments for stage "
                f"{stage_id}, using all available"
            )
            sampled = stage_segments
        else:
            # Stratify by trial_id within each stage
            n_trials = stage_segments['trial_id'].nunique()
            per_trial = max(1, n_per_stage // n_trials)

            sampled = stage_segments.groupby('trial_id', group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), per_trial),
                    random_state=random_state,
                )
            )
            # Fill from remainder if stratification left us short
            if len(sampled) < n_per_stage:
                remaining = stage_segments[~stage_segments.index.isin(sampled.index)]
                extra_n = min(len(remaining), n_per_stage - len(sampled))
                if extra_n > 0:
                    extra = remaining.sample(n=extra_n, random_state=random_state)
                    sampled = pd.concat([sampled, extra])

        balanced_segments.append(sampled)

    evaluation_set = pd.concat(balanced_segments).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return evaluation_set


# ---------------------------------------------------------------------------
# Metrics computation
# Adapted from custom_classification_report() in classify_ctl.py
# ---------------------------------------------------------------------------

def compute_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute the full suite of validation metrics for LLM vs. human labels.

    Adapted from custom_classification_report() in classify_ctl.py:

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fnr = fn / (fn + tp)
        f1 = f1_score(y_true, y_pred)
        sensitivity = metrics.recall_score(y_true, y_pred)
        specificity = tn / (tn + fp)
        precision = metrics.precision_score(y_true, y_pred)

    Extended from binary to multi-class, computing per-class and
    macro-averaged metrics.
    """
    if stage_names is None:
        stage_names = ['Vigilance', 'Avoidance', 'Metacognition', 'Reappraisal']

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    per_stage = {}
    for stage_id, stage_name in enumerate(stage_names):
        y_true_bin = (y_true == stage_id).astype(int)
        y_pred_bin = (y_pred == stage_id).astype(int)

        if y_true_bin.sum() == 0:
            per_stage[stage_name] = {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'specificity': 0.0, 'fnr': 0.0, 'support': 0,
            }
            continue

        tn = int(((1 - y_true_bin) * (1 - y_pred_bin)).sum())
        fp = int(((1 - y_true_bin) * y_pred_bin).sum())
        fn = int((y_true_bin * (1 - y_pred_bin)).sum())
        tp = int((y_true_bin * y_pred_bin).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        per_stage[stage_name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'specificity': round(specificity, 4),
            'fnr': round(fnr, 4),
            'support': int(y_true_bin.sum()),
        }

    return {
        'overall_cohens_kappa': round(kappa, 4),
        'macro_f1': round(macro_f1, 4),
        'per_stage_metrics': per_stage,
        'confusion_matrix': cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Content validity
# Adapted from classify_ctl.py's content validity evaluation pattern
# ---------------------------------------------------------------------------

def compute_content_validity(
    test_items: List[Dict],
    classifier_fn: Callable[[str], int],
    stage_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate content validity: does the classifier recognize prototypical
    expressions of each stage?

    Adapted from classify_ctl.py's content validity evaluation:

        X_test_3_dv = X_test_3[X_test_3['y_test']==dv][feature_names]
        y_pred_content_validity_3 = best_model.predict(X_test_3_dv)
        custom_cr_content_3, sklearn_cr, y_pred_df =
            metrics_report.save_classification_performance(...)

    The key metric is sensitivity (recall) on prototypical test items
    for each stage, computed separately by difficulty tier.
    """
    if stage_names is None:
        stage_names = ['Vigilance', 'Avoidance', 'Metacognition', 'Reappraisal']

    results = {}

    for stage_id, stage_name in enumerate(stage_names):
        stage_items = [item for item in test_items if item['expected_stage'] == stage_id]

        if not stage_items:
            results[stage_name] = {'sensitivity': None, 'n_items': 0}
            continue

        texts = [item['text'] for item in stage_items]
        predictions = [classifier_fn(text) for text in texts]

        correct = sum(1 for pred in predictions if pred == stage_id)
        sensitivity = correct / len(stage_items)

        # Breakdown by difficulty tier
        by_difficulty = {}
        for difficulty in ['clear', 'subtle', 'adversarial']:
            tier_items = [
                item for item in stage_items if item['difficulty'] == difficulty
            ]
            if tier_items:
                tier_preds = [classifier_fn(item['text']) for item in tier_items]
                tier_correct = sum(1 for pred in tier_preds if pred == stage_id)
                by_difficulty[difficulty] = round(tier_correct / len(tier_items), 4)

        results[stage_name] = {
            'sensitivity': round(sensitivity, 4),
            'n_items': len(stage_items),
            'by_difficulty': by_difficulty,
        }

    return results
