"""
experiments/classification_methods/common/data.py
--------------------------------------------------
Shared data-loading helpers for the classification_methods experiment campaign.

Public API
----------
repo_root() -> Path
resolve_db(output_dir=None) -> Path
CvItem  (dataclass)  — fields: item_id, ord, text, expected_stage, difficulty, source_field
    .to_segment() -> Segment
load_cv_items(output_dir=None, testset='cv_vaamr_v1') -> list[CvItem]
load_cv_segments(output_dir=None, testset='cv_vaamr_v1') -> list[Segment]
load_vaamr() -> ThemeFramework
stage_names() -> dict[int, str]

sys.path bootstrap: src/ and repo root are inserted so that
``import constructs.*`` and ``import classification_tools.*`` resolve
whether this module is imported or run as a script.
"""

import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# sys.path bootstrap — mirrors experiments/gnn_reliability/harness.py ~40-45
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))       # .../common/
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # repo root (3 levels up)
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Lazy heavy-import guard: sentence-transformers / torch are NOT imported at
# module level. Any function that needs them imports inside the function body.
# ---------------------------------------------------------------------------

# Framework imports are lightweight (pure-python dataclasses + markdown parse)
# and are safe at module level.
from constructs.registry import load as _registry_load      # noqa: E402
from constructs.theme_schema import ThemeFramework          # noqa: E402
from classification_tools.data_structures import Segment    # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Absolute path to the repo root (contains src/, experiments/, etc.)."""
    return Path(_ROOT)


def resolve_db(output_dir: Optional[str] = None) -> Path:
    """
    Return the path to the project SQLite database.

    Default: <repo>/data/Meta/qra.db
    If output_dir is given: <output_dir>/qra.db
    """
    if output_dir is not None:
        return Path(output_dir) / 'qra.db'
    return repo_root() / 'data' / 'Meta' / 'qra.db'


# ---------------------------------------------------------------------------
# CvItem dataclass
# ---------------------------------------------------------------------------

@dataclass
class CvItem:
    """One row from cv_testset_items, enriched with a .to_segment() helper."""
    item_id: str
    ord: int
    text: str
    expected_stage: int
    difficulty: str
    source_field: str

    def to_segment(self) -> Segment:
        """
        Return a valid Segment for this CV item.

        speaker='participant', role inferred from speaker field default.
        segment_id = str(item_id) (already a str; kept consistent).
        """
        return Segment(
            segment_id=str(self.item_id),
            trial_id='cv_test',
            participant_id='cv_participant',
            session_id='cv_test',
            session_number=1,
            segment_index=self.ord,
            start_time_ms=0,
            end_time_ms=0,
            total_segments_in_session=0,   # unknown at item level
            speaker='participant',
            text=self.text,
            word_count=len(self.text.split()),
            session_file='',
        )


# ---------------------------------------------------------------------------
# DB readers
# ---------------------------------------------------------------------------

def load_cv_items(
    output_dir: Optional[str] = None,
    testset: str = 'cv_vaamr_v1',
) -> List[CvItem]:
    """
    Load content-validity items from qra.db for the given testset.

    Returns list of CvItem ordered by ord (ascending).
    Raises FileNotFoundError if the DB does not exist.
    Raises KeyError if the testset is absent from the DB.
    """
    db_path = resolve_db(output_dir)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Project DB not found at {db_path}. "
            "Pass output_dir pointing to the directory containing qra.db, "
            "or ensure data/Meta/qra.db exists."
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ts_row = conn.execute(
            "SELECT 1 FROM cv_testsets WHERE name = ?", (testset,)
        ).fetchone()
        if ts_row is None:
            raise KeyError(
                f"Content-validity testset {testset!r} not found in {db_path}. "
                "Run `qra cv list` to see available testsets."
            )
        rows = conn.execute(
            "SELECT item_id, ord, text, expected_stage, difficulty, source_field "
            "FROM cv_testset_items WHERE testset_name = ? ORDER BY ord",
            (testset,),
        ).fetchall()
    finally:
        conn.close()

    return [
        CvItem(
            item_id=row['item_id'],
            ord=row['ord'],
            text=row['text'],
            expected_stage=row['expected_stage'],
            difficulty=row['difficulty'] or 'clear',
            source_field=row['source_field'] or '',
        )
        for row in rows
    ]


def load_cv_segments(
    output_dir: Optional[str] = None,
    testset: str = 'cv_vaamr_v1',
) -> List[Segment]:
    """
    Load CV items as Segment objects (ready for classification).

    Delegates to load_cv_items; converts each via CvItem.to_segment().
    """
    return [item.to_segment() for item in load_cv_items(output_dir, testset)]


# ---------------------------------------------------------------------------
# Framework helpers
# ---------------------------------------------------------------------------

def load_vaamr() -> ThemeFramework:
    """Return the VAAMR ThemeFramework (cached by constructs.registry)."""
    return _registry_load('vaamr')


def stage_names() -> Dict[int, str]:
    """
    Return {theme_id: short_name} for all VAAMR stages.

    Example: {0: 'Vigilance', 1: 'Avoidance', 2: 'Attention Regulation',
              3: 'Metacognition', 4: 'Reappraisal'}
    """
    fw = load_vaamr()
    return {t.theme_id: t.short_name for t in fw.themes}
