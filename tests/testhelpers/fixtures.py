"""
tests.testhelpers.fixtures
--------------------------
Shared builders for the QRA test suite. Consolidates the synthetic-data
patterns that were previously duplicated across individual test modules so
new tests can construct realistic inputs without re-deriving the schema.

Nothing here touches the network or downloads model weights:
``embedding_patch`` monkeypatches the GNN encoder with a deterministic RNG.
"""
from __future__ import annotations

import contextlib
from typing import List, Optional

import numpy as np
import pandas as pd

from classification_tools.data_structures import Segment


# ---------------------------------------------------------------------------
# Master-dataframe builders (GNN / analysis / report tests)
# ---------------------------------------------------------------------------
def synthetic_df(n_sessions: int = 3, dim: int = 16) -> pd.DataFrame:
    """A small interleaved participant/therapist master frame.

    Mirrors the columns the GNN layer and analysis modules read:
    final_label/primary_stage + rater_votes (participants), purer_primary
    (therapists), codebook ensemble labels, human-coded
    subset flags. Stages cycle 0..4 so every VAAMR stage appears.

    (This is the canonical version of the former ``test_gnn_layer._synthetic_df``.)
    """
    rows = []
    for s in range(n_sessions):
        sess = f'c1s{s + 1}'
        t = 0
        for j in range(6):
            spk = 'participant' if j % 2 == 0 else 'therapist'
            t += 1000
            stage = (j // 2) % 5
            rv = [{'stage': stage, 'confidence': 0.8},
                  {'stage': (stage + 1) % 5, 'confidence': 0.4, 'secondary_stage': stage}]
            rows.append(dict(
                segment_id=f'{sess}_{j}', session_id=sess, speaker=spk,
                text=f'utterance {sess} {j}', start_time_ms=t, end_time_ms=t + 800,
                final_label=(stage if spk == 'participant' else np.nan),
                primary_stage=(stage if spk == 'participant' else np.nan),
                rater_votes=(rv if spk == 'participant' else None),
                codebook_labels_ensemble=(['affect_x', 'somatic_y'] if spk == 'participant' else []),
                purer_primary=((j // 2) % 5 if spk == 'therapist' else np.nan),
                in_human_coded_subset=False, human_label=np.nan,
            ))
    return pd.DataFrame(rows)


def make_master_df(n_sessions: int = 3, n_participants: int = 2,
                   with_human_subset: bool = True) -> pd.DataFrame:
    """A richer master_segments-style frame for analysis / report generators.

    Adds the columns the analysis/reports modules expect: participant_id,
    session_number, confidence tiers, final_label_source, purer_final, and a
    deliberately rare Reappraisal (stage 4) cluster so rare-stage code paths
    (validation notes, motif purity) are exercised. Includes a human-coded
    subset for triangulation/validation tests.
    """
    stages_cycle = [0, 1, 2, 3, 2, 1, 0, 2, 3, 4]  # stage 4 (Reappraisal) is rare
    purer_cycle = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    rows = []
    for p in range(n_participants):
        pid = f'P{p + 1:02d}'
        cohort = 1
        for s in range(n_sessions):
            sess = f'c{cohort}p{p + 1}s{s + 1}'
            t = 0
            for j in range(8):
                spk = 'participant' if j % 2 == 0 else 'therapist'
                t += 1000
                seg = dict(
                    segment_id=f'{sess}_{j}',
                    trial_id='standard',
                    participant_id=pid,
                    session_id=sess,
                    session_number=s + 1,
                    cohort_id=cohort,
                    segment_index=j,
                    speaker=spk,
                    text=f'segment {sess} {j} text content',
                    start_time_ms=t, end_time_ms=t + 800,
                    word_count=5,
                )
                if spk == 'participant':
                    stage = stages_cycle[(s * 4 + j // 2) % len(stages_cycle)]
                    conf = 0.9 if stage != 4 else 0.55
                    seg.update(
                        primary_stage=stage,
                        secondary_stage=(stage + 1) % 5,
                        llm_confidence_primary=conf,
                        final_label=stage,
                        final_label_source='llm_zero_shot',
                        confidence_tier=('high' if conf >= 0.8 else 'low'),
                        rater_votes=[{'stage': stage, 'confidence': conf},
                                     {'stage': stage, 'confidence': conf - 0.1}],
                        codebook_labels_ensemble=['body_awareness', 'present_moment'],
                        purer_primary=np.nan,
                        in_human_coded_subset=with_human_subset and (j == 0),
                        human_label=(stage if (with_human_subset and j == 0) else np.nan),
                    )
                else:
                    move = purer_cycle[(s * 4 + j // 2) % len(purer_cycle)]
                    seg.update(
                        primary_stage=np.nan,
                        final_label=np.nan,
                        final_label_source=np.nan,
                        purer_primary=move,
                        purer_secondary=(move + 1) % 5,
                        purer_confidence_primary=0.8,
                        purer_final=move,
                        purer_final_source='llm_zero_shot',
                        in_human_coded_subset=False,
                        human_label=np.nan,
                    )
                rows.append(seg)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Segment builders (overlay / classification / freeze tests)
# ---------------------------------------------------------------------------
def make_segment(segment_id: str = 's1', speaker: str = 'participant',
                 session_id: str = 'c1s1', text: str = 'hello world',
                 **overrides) -> Segment:
    """A minimal Segment with sensible identity/temporal defaults."""
    seg = Segment(
        segment_id=segment_id,
        session_id=session_id,
        speaker=speaker,
        text=text,
        start_time_ms=overrides.pop('start_time_ms', 1000),
        end_time_ms=overrides.pop('end_time_ms', 2000),
    )
    for k, v in overrides.items():
        setattr(seg, k, v)
    return seg


def classified_segment(segment_id: str = 's1', speaker: str = 'participant',
                       primary_stage: Optional[int] = 2,
                       purer_primary: Optional[int] = None,
                       confidence: float = 0.85, **overrides) -> Segment:
    """A Segment carrying classification overlay fields, like a post-Stage-3 segment."""
    seg = make_segment(segment_id=segment_id, speaker=speaker, **overrides)
    if speaker == 'participant':
        seg.primary_stage = primary_stage
        seg.llm_confidence_primary = confidence
        seg.llm_run_consistency = 3
    else:
        seg.purer_primary = purer_primary if purer_primary is not None else 0
        seg.purer_confidence_primary = confidence
    return seg


class MockSegment:
    """Duck-typed stand-in for Segment used by import-safety / legacy tests.

    Accepts arbitrary kwargs and exposes them as attributes; mirrors the
    lightweight stub previously defined inline in test_legacy_import_safety.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f'MockSegment({self.__dict__!r})'


# ---------------------------------------------------------------------------
# Embedding monkeypatch
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def embedding_patch(dim: int = 16, seed: int = 7):
    """Replace the GNN encoder with a deterministic RNG so no weights download.

    Patches ``gnn_layer.embeddings.embed_segment_texts`` to return a
    ``(len(texts), dim)`` float32 array. Restores the original on exit.

        with embedding_patch(dim=16):
            runner.run_gnn_analysis(df, out_dir, config=cfg)
    """
    import gnn_layer.embeddings as emb
    orig = emb.embed_segment_texts
    rng = np.random.default_rng(seed)

    def _fake(texts: List[str], config):
        return rng.standard_normal((len(texts), dim)).astype('float32')

    emb.embed_segment_texts = _fake
    try:
        yield _fake
    finally:
        emb.embed_segment_texts = orig
