"""
process/assembly/splits.py
--------------------------
Frozen, leakage-safe train/eval split manifest (contract P0).

Every model that trains on QRA's exports — and QRA's own GNN reliability gate — must use
*the same* participant-grouped folds, or "held-out" numbers are not comparable and a
participant can leak across train/eval. This module emits one canonical, deterministic
split owned by QRA so the thesis numbers are reproducible.

Output: ``02_meta/training_data/splits.json``
  {"strategy": "grouped_by_participant", "k": 5, "seed": 42,
   "assignment": {"P03": 0, "P07": 1, ...}, "holdout_participants": ["P15", "P16"]}

The assignment is grouped by ``participant_id`` (a participant never appears in two folds)
and is a pure function of the sorted participant set + seed, so re-running reproduces it
byte-for-byte.
"""

import json
import os
import random
from typing import Iterable, List, Optional

from .. import output_paths as _paths


def _participant_ids(segments) -> List[str]:
    """Sorted, de-duplicated participant ids (deterministic order before shuffling)."""
    ids = set()
    for seg in segments:
        pid = getattr(seg, 'participant_id', None) if not isinstance(seg, dict) else seg.get('participant_id')
        if pid is not None and str(pid).strip():
            ids.add(str(pid))
    return sorted(ids)


def export_split_manifest(
    segments,
    run_dir: str,
    k: int = 5,
    seed: int = 42,
    holdout_participants: Optional[Iterable[str]] = None,
) -> str:
    """Write the frozen grouped-by-participant split manifest. Returns the file path.

    ``holdout_participants`` (optional) are carved out of all folds as a final test set;
    the remaining participants are round-robin assigned to ``k`` folds after a seeded
    shuffle. With fewer than ``k`` non-holdout participants the effective fold count is
    capped at the participant count so every fold is non-empty.
    """
    holdout = sorted({str(p) for p in (holdout_participants or []) if str(p).strip()})
    all_ids = _participant_ids(segments)
    assignable = [p for p in all_ids if p not in set(holdout)]

    eff_k = max(1, min(int(k), len(assignable))) if assignable else int(k)
    order = list(assignable)
    random.Random(seed).shuffle(order)
    assignment = {pid: (i % eff_k) for i, pid in enumerate(order)}

    manifest = {
        'strategy': 'grouped_by_participant',
        'k': eff_k,
        'seed': int(seed),
        'assignment': {pid: assignment[pid] for pid in sorted(assignment)},
        'holdout_participants': holdout,
    }

    tdir = _paths.training_data_dir(run_dir)
    os.makedirs(tdir, exist_ok=True)
    path = os.path.join(tdir, 'splits.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path
