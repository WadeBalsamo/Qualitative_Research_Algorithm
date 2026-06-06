"""
tests/unit/test_language_atlas.py
----------------------------------
Tests for analysis/reports/language_atlas.py.

generate_language_atlas(df, df_all, framework, output_dir) → Optional[str]

Key behaviors to cover:
  - Returns None when df_all yields no cue blocks (or enriched blocks are empty)
  - Returns None when enriched blocks are empty (no progression_coord)
  - Writes 06_reports/02_mechanism/language_atlas.txt when at least one valid block
  - Section headers present in output: 1a FORWARD, 1b BACKWARD, 2. EMERGENT MOTIFS,
    3. ALLIANCE / COUPLING FACTORS
  - Mechanism CSV drives the ranked forward/backward specs list
  - Missing GNN motif/coupling CSVs yield graceful fallback text
  - framework dict used for stage name rendering

Strategy:
  - Monkeypatch build_cue_blocks_with_segments to return controlled blocks
  - Monkeypatch _seg_lookup, _load_block_motifs, _enrich_blocks so no real data needed
  - Build a minimal df_all with progression_coord + mixture
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from process import output_paths as _paths


# ── minimal framework dict ────────────────────────────────────────────────────

FRAMEWORK = {
    0: {'short_name': 'Vigilance'},
    1: {'short_name': 'Avoidance'},
    2: {'short_name': 'AttnReg'},
    3: {'short_name': 'Metacog'},
    4: {'short_name': 'Reappraisal'},
}


def _make_df_all(n=6):
    """Minimal df_all with all columns needed by language_atlas helpers."""
    rows = []
    for i in range(n):
        spk = 'participant' if i % 2 == 0 else 'therapist'
        rows.append({
            'segment_id': f'seg_{i}',
            'session_id': 'c1s1',
            'speaker': spk,
            'text': f'Some {"participant" if spk == "participant" else "therapist"} text {i}',
            'start_time_ms': i * 2000,
            'end_time_ms': i * 2000 + 1500,
            'final_label': (i % 3) if spk == 'participant' else np.nan,
            'purer_primary': np.nan if spk == 'participant' else (i % 5),
            'mixture': [0.6, 0.3, 0.05, 0.03, 0.02] if spk == 'participant' else None,
            'progression_coord': (0.5 + i * 0.1) if spk == 'participant' else np.nan,
            'mixture_entropy': 0.4 if spk == 'participant' else np.nan,
        })
    return pd.DataFrame(rows)


def _make_minimal_blocks():
    """Two cue blocks: one forward, one backward."""
    return [
        {
            'session_id': 'c1s1',
            'from_seg_id': 'seg_0',
            'to_seg_id': 'seg_2',
            'from_stage': 0,
            'to_stage': 1,
            'transition_type': 'forward',
            'therapist_seg_ids': ['seg_1'],
        },
        {
            'session_id': 'c1s1',
            'from_seg_id': 'seg_2',
            'to_seg_id': 'seg_4',
            'from_stage': 1,
            'to_stage': 0,
            'transition_type': 'backward',
            'therapist_seg_ids': ['seg_3'],
        },
    ]


def _make_enriched_blocks():
    """Return two enriched blocks directly."""
    return [
        {
            'session_id': 'c1s1',
            'from_seg_id': 'seg_0',
            'to_seg_id': 'seg_2',
            'from_stage': 0,
            'to_stage': 1,
            'transition_type': 'forward',
            'therapist_seg_ids': ['seg_1'],
            'participant_id': 'P01',
            'delta_prog': 0.45,
            'from_entropy': 0.3,
            'from_mixture': [0.7, 0.2, 0.05, 0.03, 0.02],
            'dominant_purer': 'Reframing',
            'cue_motif': None,
            'n_therapist_segments': 1,
            'delta_direction': 'progress',
        },
        {
            'session_id': 'c1s1',
            'from_seg_id': 'seg_2',
            'to_seg_id': 'seg_4',
            'from_stage': 1,
            'to_stage': 0,
            'transition_type': 'backward',
            'therapist_seg_ids': ['seg_3'],
            'participant_id': 'P01',
            'delta_prog': -0.30,
            'from_entropy': 0.4,
            'from_mixture': [0.3, 0.6, 0.05, 0.03, 0.02],
            'dominant_purer': 'Education',
            'cue_motif': None,
            'n_therapist_segments': 1,
            'delta_direction': 'regress',
        },
    ]


def _minimal_seg_lookup(df_all):
    """Simple lookup with progression_coord and mixture for the test segments."""
    return {
        'seg_0': {'progression_coord': 0.5, 'mixture': [0.6, 0.3, 0.05, 0.03, 0.02],
                  'mixture_entropy': 0.3, 'purer': None, 'participant_id': 'P01'},
        'seg_1': {'progression_coord': None, 'mixture': None, 'mixture_entropy': None,
                  'purer': 2, 'participant_id': None},
        'seg_2': {'progression_coord': 0.95, 'mixture': [0.2, 0.7, 0.05, 0.03, 0.02],
                  'mixture_entropy': 0.4, 'purer': None, 'participant_id': 'P01'},
        'seg_3': {'progression_coord': None, 'mixture': None, 'mixture_entropy': None,
                  'purer': 3, 'participant_id': None},
        'seg_4': {'progression_coord': 0.65, 'mixture': [0.5, 0.4, 0.05, 0.03, 0.02],
                  'mixture_entropy': 0.5, 'purer': None, 'participant_id': 'P01'},
        'seg_5': {'progression_coord': None, 'mixture': None, 'mixture_entropy': None,
                  'purer': 4, 'participant_id': None},
    }


class TestLanguageAtlasNoBlocks(unittest.TestCase):
    """Returns None when build_cue_blocks_with_segments yields nothing."""

    def test_returns_none_when_no_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            df_all = _make_df_all()
            with patch('gnn_layer.inference.build_cue_blocks_with_segments', return_value=[]):
                from analysis.reports.language_atlas import generate_language_atlas
                result = generate_language_atlas(df_all, df_all, FRAMEWORK, tmp)
            self.assertIsNone(result)

    def test_returns_none_when_enriched_empty(self):
        """If enriched blocks are empty (no progression_coord match), returns None."""
        with tempfile.TemporaryDirectory() as tmp:
            df_all = _make_df_all()
            blocks = _make_minimal_blocks()
            with patch('gnn_layer.inference.build_cue_blocks_with_segments', return_value=blocks), \
                 patch('analysis.mechanism._seg_lookup', return_value={}), \
                 patch('analysis.mechanism._load_block_motifs', return_value={}), \
                 patch('analysis.mechanism._enrich_blocks', return_value=[]):
                from analysis.reports.language_atlas import generate_language_atlas
                result = generate_language_atlas(df_all, df_all, FRAMEWORK, tmp)
            self.assertIsNone(result)


class TestLanguageAtlasWrites(unittest.TestCase):
    """With valid enriched blocks, a file is written to 02_mechanism/language_atlas.txt."""

    def _run(self, tmp, df_all=None):
        if df_all is None:
            df_all = _make_df_all()
        blocks = _make_minimal_blocks()
        enriched = _make_enriched_blocks()
        lookup = _minimal_seg_lookup(df_all)
        with patch('gnn_layer.inference.build_cue_blocks_with_segments', return_value=blocks), \
             patch('analysis.mechanism._seg_lookup', return_value=lookup), \
             patch('analysis.mechanism._load_block_motifs', return_value={}), \
             patch('analysis.mechanism._enrich_blocks', return_value=enriched):
            from analysis.reports.language_atlas import generate_language_atlas
            return generate_language_atlas(df_all, df_all, FRAMEWORK, tmp)

    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            self.assertIsNotNone(path)
            self.assertTrue(os.path.isfile(path))

    def test_path_ends_with_language_atlas_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            self.assertTrue(path.endswith('language_atlas.txt'))

    def test_path_in_mechanism_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            self.assertIn('02_mechanism', path)

    def test_section_headers_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('THERAPEUTIC LANGUAGE ATLAS', content)
            self.assertIn('1a. TOP FORWARD-MOVING', content)
            self.assertIn('1b. BACKWARD-MOVING', content)
            self.assertIn('2. EMERGENT MOTIFS', content)
            self.assertIn('3. ALLIANCE / COUPLING FACTORS', content)

    def test_gnn_not_available_fallback(self):
        """Without GNN CSVs, sections 2 and 3 show fallback text."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            # No cue_motifs.csv → section 2 fallback
            self.assertIn('GNN motif discovery not available', content)
            # No coupling_factors.csv → section 3 fallback
            self.assertIn('GNN coupling not available', content)

    def test_forward_mover_run_instruction_when_no_mech_csv(self):
        """Without mechanism CSV, section 1a shows a run-first message."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            # No mechanism_delta_progression.csv in tmp → empty forward_specs
            self.assertIn('run the mechanism analysis first', content)


class TestLanguageAtlasMechanismCSV(unittest.TestCase):
    """When mechanism_delta_progression.csv exists, ranked specs drive section 1."""

    def _run_with_mech(self, tmp):
        """Seed the mechanism CSV then run atlas."""
        mech_dir = _paths.mechanism_dir(tmp)
        os.makedirs(mech_dir, exist_ok=True)
        pd.DataFrame([
            {'grouping': 'purer', 'from_stage': 1, 'from_stage_name': 'Avoidance',
             'behavior': 'Reframing', 'n': 10, 'mean_delta_prog': 0.42, 'fdr_significant': True},
            {'grouping': 'purer', 'from_stage': 0, 'from_stage_name': 'Vigilance',
             'behavior': 'Education', 'n': 6, 'mean_delta_prog': -0.25, 'fdr_significant': True},
        ]).to_csv(os.path.join(mech_dir, 'mechanism_delta_progression.csv'), index=False)

        df_all = _make_df_all()
        blocks = _make_minimal_blocks()
        enriched = _make_enriched_blocks()
        # Patch dominant_purer to 'Reframing' in forward block so example shows
        enriched[0]['dominant_purer'] = 'Reframing'
        enriched[1]['dominant_purer'] = 'Reframing'
        lookup = _minimal_seg_lookup(df_all)
        with patch('gnn_layer.inference.build_cue_blocks_with_segments', return_value=blocks), \
             patch('analysis.mechanism._seg_lookup', return_value=lookup), \
             patch('analysis.mechanism._load_block_motifs', return_value={}), \
             patch('analysis.mechanism._enrich_blocks', return_value=enriched):
            from analysis.reports.language_atlas import generate_language_atlas
            return generate_language_atlas(df_all, df_all, FRAMEWORK, tmp)

    def test_mechanism_csv_populates_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run_with_mech(tmp)
            self.assertIsNotNone(path)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            # The mechanism CSV rows feed section 1; the from-stage names should appear
            self.assertIn('Avoidance', content)


class TestLanguageAtlasWithCouplingCSV(unittest.TestCase):
    """When coupling_factors.csv exists, section 3 renders it."""

    def test_coupling_csv_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            gnn_dir = _paths.gnn_data_dir(tmp)
            os.makedirs(gnn_dir, exist_ok=True)
            pd.DataFrame([
                {'factor': 0, 'nearest_cf_ic': 'Alliance', 'forward_corr': 0.65,
                 'explained_variance_ratio': 0.21},
            ]).to_csv(os.path.join(gnn_dir, 'coupling_factors.csv'), index=False)

            df_all = _make_df_all()
            blocks = _make_minimal_blocks()
            enriched = _make_enriched_blocks()
            lookup = _minimal_seg_lookup(df_all)
            with patch('gnn_layer.inference.build_cue_blocks_with_segments', return_value=blocks), \
                 patch('analysis.mechanism._seg_lookup', return_value=lookup), \
                 patch('analysis.mechanism._load_block_motifs', return_value={}), \
                 patch('analysis.mechanism._enrich_blocks', return_value=enriched):
                from analysis.reports.language_atlas import generate_language_atlas
                path = generate_language_atlas(df_all, df_all, FRAMEWORK, tmp)

            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('Alliance', content)
            self.assertIn('forward-corr=+0.650', content)


if __name__ == '__main__':
    unittest.main()
