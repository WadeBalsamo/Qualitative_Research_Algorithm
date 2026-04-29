"""
validation.py
-------------
Balanced evaluation set construction for human coding comparison.

Works with any N-class classification scheme (not hardcoded to 4 stages).
"""

import pandas as pd
from typing import List


def create_balanced_evaluation_set(
    segments_df: pd.DataFrame,
    n_per_class: int = 50,
    label_column: str = 'primary_stage',
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Create a balanced evaluation set for human coding comparison.

    Samples ``n_per_class`` segments per unique label value, stratifying
    by ``trial_id`` to ensure cross-trial representation.

    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame of labeled segments.
    n_per_class : int
        Target number of segments per class.
    label_column : str
        Column containing class labels.
    random_state : int
        Random seed for reproducibility.
    """
    unique_classes = sorted(segments_df[label_column].dropna().unique())
    balanced_segments: List[pd.DataFrame] = []

    for class_id in unique_classes:
        class_segments = segments_df[segments_df[label_column] == class_id]

        if len(class_segments) < n_per_class:
            print(
                f"Warning: only {len(class_segments)} segments for class "
                f"{class_id}, using all available"
            )
            sampled = class_segments
        else:
            # Stratify by trial_id within each class
            if 'trial_id' in class_segments.columns:
                n_trials = class_segments['trial_id'].nunique()
                per_trial = max(1, n_per_class // n_trials)

                sampled = class_segments.groupby('trial_id', group_keys=False).apply(
                    lambda x: x.sample(
                        n=min(len(x), per_trial),
                        random_state=random_state,
                    )
                )
                # Fill from remainder if stratification left us short
                if len(sampled) < n_per_class:
                    remaining = class_segments[~class_segments.index.isin(sampled.index)]
                    extra_n = min(len(remaining), n_per_class - len(sampled))
                    if extra_n > 0:
                        extra = remaining.sample(n=extra_n, random_state=random_state)
                        sampled = pd.concat([sampled, extra])
            else:
                sampled = class_segments.sample(
                    n=n_per_class, random_state=random_state
                )

        balanced_segments.append(sampled)

    # Dual-coded stratum: segments with a secondary label are the most ambiguous
    # and highest-value candidates for human review.
    if 'secondary_stage' in segments_df.columns:
        dual_coded = segments_df[
            segments_df[label_column].notna() & segments_df['secondary_stage'].notna()
        ]
        if len(dual_coded) > 0:
            if len(dual_coded) > n_per_class:
                dual_sampled = dual_coded.sample(n=n_per_class, random_state=random_state)
            else:
                dual_sampled = dual_coded
            balanced_segments.append(dual_sampled)
            print(
                f"  Dual-coded stratum: {len(dual_sampled)} segments "
                f"(of {len(dual_coded)} available)"
            )

    evaluation_set = pd.concat(balanced_segments).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return evaluation_set
