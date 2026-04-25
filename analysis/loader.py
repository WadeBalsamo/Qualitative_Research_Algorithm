"""
analysis/loader.py
------------------
Load and clean pipeline output data for analysis.

Single entry point: load_segments() returns a clean DataFrame ready for
all other analysis modules.
"""

import ast
import json
import os
import re
from typing import Optional

import pandas as pd
from process import output_paths as _paths


def find_master_csv(output_dir: str) -> str:
    """Return path to master_segments.csv, searching new then legacy locations."""
    # New layout: 06_training_data/master_segments.csv
    new_path = os.path.join(_paths.master_segments_dir(output_dir), 'master_segments.csv')
    if os.path.isfile(new_path):
        return new_path
    # Legacy: any master_segment*.csv in output_dir root
    for filename in os.listdir(output_dir):
        if 'master_segment' in filename and filename.endswith('.csv'):
            return os.path.join(output_dir, filename)
    raise FileNotFoundError(
        f"master_segments.csv not found in {output_dir}\n"
        f"Run the pipeline first: python qra.py run --output-dir {output_dir}"
    )


def _parse_list_column(val) -> list:
    """Safely parse a list stored as a string repr in CSV (e.g. \"['a', 'b']\")."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def _derive_cohort_id(session_id: str) -> Optional[int]:
    """Derive cohort_id from session_id string (e.g. 'c2s5' → 2)."""
    m = re.match(r'c(\d+)s', str(session_id), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _sort_key(session_id: str):
    """Sort key for canonical session IDs like c1s3, c1s4a, c2s7."""
    m = re.fullmatch(r'c(\d+)s(\d+)([a-z]?)', str(session_id).strip(), re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)), m.group(3).lower())
    return (999, 999, str(session_id))


def sort_session_ids(session_ids: list) -> list:
    """Sort canonical session IDs (c1s3, c1s4a, c2s7 ...) in longitudinal order."""
    return sorted(session_ids, key=_sort_key)


def load_segments(
    output_dir: str,
    speaker_filter: str = 'participant',
    require_labeled: bool = True,
) -> pd.DataFrame:
    """Load master_segment_dataset.csv and return a clean analysis-ready DataFrame.

    Parameters
    ----------
    output_dir : str
        Pipeline output directory containing master_segment_dataset.csv.
    speaker_filter : str
        Keep only segments from this speaker role ('participant', 'therapist',
        or None to keep all).
    require_labeled : bool
        If True, drop rows where final_label is NaN.

    Returns
    -------
    pd.DataFrame with guaranteed columns:
        segment_id, participant_id, session_id, session_number, cohort_id,
        session_variant, segment_index, text, word_count, primary_stage,
        final_label, llm_confidence_primary, llm_run_consistency,
        label_confidence_tier, codebook_labels_ensemble (list),
        llm_justification
    """
    csv_path = find_master_csv(output_dir)
    df = pd.read_csv(csv_path, low_memory=False)

    # Parse list columns stored as string repr
    for col in ('codebook_labels_ensemble', 'codebook_labels_embedding',
                'codebook_labels_llm', 'codebook_disagreements'):
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_column)
        else:
            df[col] = [[] for _ in range(len(df))]

    # Derive cohort_id from session_id if column absent (backwards compat)
    if 'cohort_id' not in df.columns or df['cohort_id'].isna().all():
        df['cohort_id'] = df['session_id'].apply(_derive_cohort_id)
    else:
        df['cohort_id'] = pd.to_numeric(df['cohort_id'], errors='coerce')

    if 'session_variant' not in df.columns:
        df['session_variant'] = ''

    # Ensure numeric types
    df['session_number'] = pd.to_numeric(df['session_number'], errors='coerce').fillna(0).astype(int)
    df['segment_index'] = pd.to_numeric(df['segment_index'], errors='coerce').fillna(0).astype(int)
    df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce').fillna(0).astype(int)
    df['llm_confidence_primary'] = pd.to_numeric(df['llm_confidence_primary'], errors='coerce')
    df['llm_run_consistency'] = pd.to_numeric(df['llm_run_consistency'], errors='coerce')
    df['start_time_ms'] = pd.to_numeric(df['start_time_ms'], errors='coerce').fillna(0).astype(int)
    df['end_time_ms'] = pd.to_numeric(df['end_time_ms'], errors='coerce').fillna(0).astype(int)

    # Ensure final_label and primary_stage exist
    if 'final_label' not in df.columns:
        df['final_label'] = df.get('primary_stage', pd.Series(dtype=float))
    if 'primary_stage' not in df.columns:
        df['primary_stage'] = df['final_label']

    if 'label_confidence_tier' not in df.columns:
        df['label_confidence_tier'] = 'low'
    if 'llm_justification' not in df.columns:
        df['llm_justification'] = ''
    if 'in_human_coded_subset' not in df.columns:
        df['in_human_coded_subset'] = False

    # Speaker filter
    if speaker_filter and 'speaker' in df.columns:
        df = df[df['speaker'] == speaker_filter].copy()

    # Drop unlabeled rows
    if require_labeled:
        df = df[df['final_label'].notna()].copy()

    # Cast labels to int now that NaNs are removed
    if require_labeled and len(df) > 0:
        df['final_label'] = df['final_label'].astype(int)
        df['primary_stage'] = df['primary_stage'].fillna(df['final_label']).astype(int)

    df = df.reset_index(drop=True)
    return df


def load_framework(output_dir: str) -> dict:
    """Load theme_definitions.json, searching new then legacy locations.

    Returns dict mapping int stage_id → {id, key, name, short_name, definition}.
    Raises FileNotFoundError if theme_definitions.json is absent.
    """
    for candidate in (
        os.path.join(_paths.meta_dir(output_dir), 'theme_definitions.json'),
        os.path.join(output_dir, 'meta', 'theme_definitions.json'),
        os.path.join(output_dir, 'theme_definitions.json'),
    ):
        if os.path.isfile(candidate):
            path = candidate
            break
    else:
        raise FileNotFoundError(
            f"theme_definitions.json not found in {output_dir}\n"
            f"This file is created by the pipeline. Re-run to regenerate it."
        )
    with open(path) as f:
        data = json.load(f)

    themes = data.get('themes', [])
    framework = {}
    for t in themes:
        tid = int(t.get('theme_id', t.get('id', 0)))
        framework[tid] = {
            'id': tid,
            'key': t.get('key', ''),
            'name': t.get('name', f'Stage {tid}'),
            'short_name': t.get('short_name', t.get('name', f'Stage {tid}')),
            'definition': t.get('definition', ''),
        }
    return framework
