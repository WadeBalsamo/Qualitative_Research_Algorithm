"""Assembles the master segment dataset from classified segments."""

import json
from typing import List, Dict, Optional

import pandas as pd

from classification_tools.data_structures import Segment
from .. import output_paths as _paths
from ._shared import _ms_to_hms, _fmt_conf, _theme_name_from, _summarize_rationales


def _progression_for_segment(seg):
    """Soft stage-mixture + E[stage] progression coordinate for a participant segment.

    Reuses the numpy-only ballot math in ``gnn_layer.soft_labels`` (no trained GNN
    required): the multi-run ``rater_votes`` mixture, falling back to a one-hot of the
    final/primary label when ballots are unusable. Returns ``(mixture_list, coord)`` for
    participant segments and ``(None, None)`` for therapist segments (PURER, not VAAMR).
    """
    if getattr(seg, 'speaker', None) != 'participant':
        return None, None
    import numpy as np
    from gnn_layer.soft_labels import (
        ballots_to_mixture, mixture_to_progression, one_hot, N_VAAMR_STAGES,
    )
    votes = getattr(seg, 'rater_votes', None)
    mix = ballots_to_mixture(votes)
    if not votes or np.allclose(mix, 1.0 / N_VAAMR_STAGES):
        stage = getattr(seg, 'final_label', None)
        if stage is None:
            stage = getattr(seg, 'primary_stage', None)
        if stage is not None:
            try:
                mix = one_hot(int(stage))
            except (TypeError, ValueError):
                return None, None
        else:
            return None, None
    return [round(float(p), 6) for p in mix], round(mixture_to_progression(mix), 6)


def assemble_master_dataset(
    segments: List[Segment],
    output_path: str,
    confidence_tiers: Optional[Dict] = None,
    probe_ready: bool = False,
    gnn_ready: bool = False,
) -> pd.DataFrame:
    """
    Produce the master segment dataset.

    final_label provenance priority (highest → lowest):
        adjudicated > human_consensus > llm_zero_shot > probe_consensus > gnn_consensus

    The two CHEAP scalers — the probe (LLM-free per-rater ensemble; methodology §8.6) and
    the demoted GNN classifier — only ever FILL segments the LLM/human left unlabeled; they
    rank BELOW ``llm_zero_shot`` and can NEVER override it. Each is engaged only when its
    per-project reliability gate has passed (``probe_ready`` / ``gnn_ready``, resolved by the
    orchestrator from ``probe_gate.json`` / ``gnn_gate.json``), the segment carries that
    classifier's prediction, AND the classifier did not abstain on it. Abstention (probe
    "No code" / sub-floor confidence, or GNN deferral) keeps the segment unlabeled rather
    than forcing a confident-wrong guess into the corpus. Probe/GNN-filled rows are tagged
    ``label_confidence_tier`` 'probe' / 'gnn' (distinct from and lower than the LLM tiers) so
    downstream consumers (analysis, MindfulBERT) can down-weight or exclude them.

    With both cheap tiers off (the default), labels of record are byte-identical to the
    LLM-consensus pipeline. The raw LLM/probe/GNN predictions and the per-rater ballots stay
    in their columns (``primary_stage``, ``rater_votes``, ``probe_pred``, ``gnn_vaamr_pred``,
    …), so every fill is auditable. Note: unlike the retired ``gnn_authoritative`` path, NO
    classifier can override an existing LLM/human label here.
    """
    ct = confidence_tiers or {}
    high_consistency = ct.get('high_consistency', 3)
    high_confidence = ct.get('high_confidence', 0.8)
    medium_min_consistency = ct.get('medium_min_consistency', 2)
    medium_min_confidence = ct.get('medium_min_confidence', 0.6)

    rows = []
    for seg in segments:
        is_participant = seg.speaker == 'participant'
        # Compute final_label (VAAMR participant stage label of record). The cheap scalers
        # (probe, then demoted GNN) FILL below the LLM and never override it.
        _probe_pred = getattr(seg, 'probe_pred', None)
        _gnn_vaamr = getattr(seg, 'gnn_vaamr_pred', None)
        if seg.adjudicated_label is not None:
            final_label = seg.adjudicated_label
            final_label_source = 'adjudicated'
        elif seg.human_label is not None and seg.human_label == seg.primary_stage:
            final_label = seg.human_label
            final_label_source = 'human_consensus'
        elif seg.primary_stage is not None:
            final_label = seg.primary_stage
            final_label_source = 'llm_zero_shot'
        elif (probe_ready and _probe_pred is not None and is_participant
              and not getattr(seg, 'probe_abstain', False)):
            final_label = _probe_pred
            final_label_source = 'probe_consensus'
        elif (gnn_ready and _gnn_vaamr is not None and is_participant
              and not getattr(seg, 'gnn_vaamr_abstain', False)):
            final_label = _gnn_vaamr
            final_label_source = 'gnn_consensus'
        else:
            final_label = None
            final_label_source = None

        # Compute purer_final (PURER therapist move label of record). PURER has no
        # human/adjudicated/probe tier; the hierarchy is llm_zero_shot > gnn_consensus (fill).
        _gnn_purer = getattr(seg, 'gnn_purer_pred', None)
        _purer_llm = getattr(seg, 'purer_primary', None)
        if _purer_llm is not None:
            purer_final = _purer_llm
            purer_final_source = 'llm_zero_shot'
        elif (gnn_ready and _gnn_purer is not None and seg.speaker == 'therapist'
              and not getattr(seg, 'gnn_purer_abstain', False)):
            purer_final = _gnn_purer
            purer_final_source = 'gnn_consensus'
        else:
            purer_final = None
            purer_final_source = None

        # Compute confidence tier
        if (
            seg.llm_run_consistency == high_consistency
            and seg.llm_confidence_primary is not None
            and seg.llm_confidence_primary > high_confidence
        ):
            confidence_tier = 'high'
        elif (
            seg.llm_run_consistency is not None
            and seg.llm_run_consistency >= medium_min_consistency
            and seg.llm_confidence_primary is not None
            and seg.llm_confidence_primary > medium_min_confidence
        ):
            confidence_tier = 'medium'
        else:
            confidence_tier = 'low'

        # Cheap-tier fills carry a distinct, lower confidence tier so downstream consumers
        # can weight or exclude them (they are noisier than LLM/human labels).
        if final_label_source == 'probe_consensus':
            confidence_tier = 'probe'
        elif final_label_source == 'gnn_consensus':
            confidence_tier = 'gnn'

        # Soft stage mixture + continuous progression coordinate (participant only).
        # Note: computed from the raw ballots/primary label, BEFORE any GNN promotion,
        # so the coordinate is stable regardless of whether the GNN layer ran.
        stage_mixture, progression_coord = _progression_for_segment(seg)

        row = {
            'segment_id': seg.segment_id,
            'trial_id': seg.trial_id,
            'participant_id': seg.participant_id,
            'session_id': seg.session_id,
            'session_number': seg.session_number,
            'cohort_id': seg.cohort_id,
            'session_variant': seg.session_variant,
            'segment_index': seg.segment_index,
            'start_time_ms': seg.start_time_ms,
            'end_time_ms': seg.end_time_ms,
            'total_segments_in_session': seg.total_segments_in_session,
            'speaker': seg.speaker,
            'text': seg.text,
            'word_count': seg.word_count,
            # Theme labels
            'primary_stage': seg.primary_stage,
            'secondary_stage': seg.secondary_stage,
            'llm_confidence_primary': seg.llm_confidence_primary,
            'llm_confidence_secondary': seg.llm_confidence_secondary,
            'llm_justification': seg.llm_justification,
            'llm_run_consistency': seg.llm_run_consistency,
            # Interrater-reliability fields (getattr guards against older Segment schemas)
            'rater_ids': (json.dumps(v) if (v := getattr(seg, 'rater_ids', None)) else None),
            'rater_votes': (json.dumps(v) if (v := getattr(seg, 'rater_votes', None)) else None),
            'agreement_level': getattr(seg, 'agreement_level', None),
            'agreement_fraction': getattr(seg, 'agreement_fraction', None),
            'needs_review': getattr(seg, 'needs_review', False),
            'consensus_vote': (
                json.dumps(cv) if isinstance(cv := getattr(seg, 'consensus_vote', None), str)
                else cv
            ),
            'tie_broken_by_confidence': getattr(seg, 'tie_broken_by_confidence', False),
            # Codebook labels (if populated)
            'codebook_labels_embedding': seg.codebook_labels_embedding,
            'codebook_labels_llm': seg.codebook_labels_llm,
            'codebook_labels_ensemble': seg.codebook_labels_ensemble,
            'codebook_disagreements': seg.codebook_disagreements,
            # PURER labels (therapist segments only; None for participant segments)
            'purer_primary': getattr(seg, 'purer_primary', None),
            'purer_secondary': getattr(seg, 'purer_secondary', None),
            'purer_confidence_primary': getattr(seg, 'purer_confidence_primary', None),
            'purer_confidence_secondary': getattr(seg, 'purer_confidence_secondary', None),
            'purer_justification': getattr(seg, 'purer_justification', None),
            'purer_run_consistency': getattr(seg, 'purer_run_consistency', None),
            'purer_agreement_level': getattr(seg, 'purer_agreement_level', None),
            'purer_agreement_fraction': getattr(seg, 'purer_agreement_fraction', None),
            'purer_needs_review': getattr(seg, 'purer_needs_review', False),
            'purer_rater_ids': (
                json.dumps(v) if (v := getattr(seg, 'purer_rater_ids', None)) else None
            ),
            'purer_rater_votes': (
                json.dumps(v) if (v := getattr(seg, 'purer_rater_votes', None)) else None
            ),
            'purer_final': purer_final,
            'purer_final_source': purer_final_source,
            # GNN consensus-distillation predictions (graph-distilled). Demoted to a
            # non-authoritative FILL tier (gnn_consensus, below the LLM) — see final_label.
            'gnn_vaamr_pred': getattr(seg, 'gnn_vaamr_pred', None),
            'gnn_vaamr_conf': getattr(seg, 'gnn_vaamr_conf', None),
            'gnn_vaamr_abstain': getattr(seg, 'gnn_vaamr_abstain', None),
            'gnn_purer_pred': getattr(seg, 'gnn_purer_pred', None),
            'gnn_purer_conf': getattr(seg, 'gnn_purer_conf', None),
            'gnn_purer_abstain': getattr(seg, 'gnn_purer_abstain', None),
            'gnn_label_source': getattr(seg, 'gnn_label_source', None),
            # Probe scaler predictions (LLM-free per-rater ensemble; provenance tier
            # 'probe_consensus', ranked below the LLM — see final_label above).
            'probe_pred': getattr(seg, 'probe_pred', None),
            'probe_conf': getattr(seg, 'probe_conf', None),
            'probe_abstain': getattr(seg, 'probe_abstain', None),
            # Validation
            'human_label': seg.human_label,
            'human_secondary_label': seg.human_secondary_label,
            'adjudicated_label': seg.adjudicated_label,
            'in_human_coded_subset': seg.in_human_coded_subset,
            'label_status': seg.label_status,
            # Final
            'final_label': final_label,
            'final_label_source': final_label_source,
            'label_confidence_tier': confidence_tier,
            # Continuous progression target + soft stage mixture (participant only;
            # JSON-encoded list column). Consumed by the MindfulBERT export + workshop.
            'progression_coord': progression_coord,
            'stage_mixture': (json.dumps(stage_mixture) if stage_mixture is not None else None),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV only (the caller passes the .csv path as output_path)
    df.to_csv(output_path, index=False)

    # Print class distribution
    if not df.empty and 'speaker' in df.columns:
        participant_labeled = df[
            (df['speaker'] == 'participant') & (df['final_label'].notna())
        ]
    else:
        participant_labeled = pd.DataFrame()
    print("\nFinal label distribution:")
    if len(participant_labeled) > 0:
        print(participant_labeled['final_label'].value_counts().sort_index())
    print(f"\nTotal segments: {len(df)}")
    print(f"Participant segments with labels: {len(participant_labeled)}")

    return df
