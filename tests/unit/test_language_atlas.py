"""
tests/unit/test_language_atlas.py
----------------------------------
Tests for analysis/reports/language_atlas.py (transition-centric, extractive).

generate_language_atlas(df, df_all, framework, output_dir) → Optional[str]

Covers:
  - Returns None when no cue blocks / enriched blocks
  - Writes 06_reports/03_mechanism/language_atlas.txt
  - Section headers: 0. TRANSITION CUE PROTOTYPES, 1. TRANSITION INVENTORY
  - Same-participant stage changes drive both sections; no-VTT fallback text
  - Helper behaviors: timecode/gap formatting, embedded-speaker-line parsing,
    same-participant filtering
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

import numpy as np
import pandas as pd


FRAMEWORK = {
    0: {'short_name': 'Vigilance'},
    1: {'short_name': 'Avoidance'},
    2: {'short_name': 'AttnReg'},
    3: {'short_name': 'Metacog'},
    4: {'short_name': 'Reappraisal'},
}


def _make_df_all(n=6):
    rows = []
    for i in range(n):
        spk = 'participant' if i % 2 == 0 else 'therapist'
        rows.append({
            'segment_id': f'seg_{i}',
            'session_id': 'c1s1',
            'speaker': spk,
            'participant_id': 'P01' if spk == 'participant' else None,
            'cohort_id': 1,
            'session_number': 1,
            'text': f'Some {"participant" if spk == "participant" else "therapist"} text {i}.',
            'start_time_ms': i * 2000 + 1000,
            'end_time_ms': i * 2000 + 2500,
            'final_label': (i % 3) if spk == 'participant' else np.nan,
            'mixture': [0.6, 0.3, 0.05, 0.03, 0.02] if spk == 'participant' else None,
            'progression_coord': (0.5 + i * 0.1) if spk == 'participant' else np.nan,
            'mixture_entropy': 0.4 if spk == 'participant' else np.nan,
        })
    return pd.DataFrame(rows)


def _make_minimal_blocks():
    return [
        {'session_id': 'c1s1', 'from_seg_id': 'seg_0', 'to_seg_id': 'seg_2',
         'from_stage': 0, 'to_stage': 1, 'transition_type': 'forward',
         'therapist_seg_ids': ['seg_1']},
        {'session_id': 'c1s1', 'from_seg_id': 'seg_2', 'to_seg_id': 'seg_4',
         'from_stage': 1, 'to_stage': 0, 'transition_type': 'backward',
         'therapist_seg_ids': ['seg_3']},
    ]


def _make_enriched_blocks():
    common = {'session_id': 'c1s1', 'participant_id': 'P01',
              'from_participant': 'P01', 'to_participant': 'P01',
              'from_entropy': 0.3, 'dominant_purer': None, 'cue_motif': None,
              'n_therapist_segments': 1}
    return [
        {**common, 'from_seg_id': 'seg_0', 'to_seg_id': 'seg_2',
         'from_stage': 0, 'to_stage': 1, 'transition_type': 'forward',
         'therapist_seg_ids': ['seg_1'], 'delta_prog': 0.45,
         'from_mixture': [0.7, 0.2, 0.05, 0.03, 0.02], 'delta_direction': 'progress'},
        {**common, 'from_seg_id': 'seg_2', 'to_seg_id': 'seg_4',
         'from_stage': 1, 'to_stage': 0, 'transition_type': 'backward',
         'therapist_seg_ids': ['seg_3'], 'delta_prog': -0.30,
         'from_mixture': [0.3, 0.6, 0.05, 0.03, 0.02], 'delta_direction': 'regress'},
    ]


def _minimal_seg_lookup(df_all):
    return {f'seg_{i}': {'progression_coord': 0.5 + i * 0.1, 'mixture': None,
                         'mixture_entropy': 0.3, 'purer': None, 'participant_id': 'P01'}
            for i in range(6)}


class TestLanguageAtlasNoBlocks(unittest.TestCase):

    def test_returns_none_when_no_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            df_all = _make_df_all()
            with patch('gnn_layer.cue_features.build_cue_blocks_with_segments', return_value=[]):
                from analysis.reports.language_atlas import generate_language_atlas
                result = generate_language_atlas(df_all, df_all, FRAMEWORK, tmp)
            self.assertIsNone(result)

    def test_returns_none_when_enriched_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            df_all = _make_df_all()
            blocks = _make_minimal_blocks()
            with patch('gnn_layer.cue_features.build_cue_blocks_with_segments', return_value=blocks), \
                 patch('analysis.mechanism._seg_lookup', return_value={}), \
                 patch('analysis.mechanism._load_block_motifs', return_value={}), \
                 patch('analysis.mechanism._enrich_blocks', return_value=[]):
                from analysis.reports.language_atlas import generate_language_atlas
                result = generate_language_atlas(df_all, df_all, FRAMEWORK, tmp)
            self.assertIsNone(result)


class TestLanguageAtlasWrites(unittest.TestCase):
    """With valid enriched blocks (no VTT in tmp), file written; fallbacks shown."""

    def _run(self, tmp, df_all=None):
        if df_all is None:
            df_all = _make_df_all()
        blocks = _make_minimal_blocks()
        enriched = _make_enriched_blocks()
        lookup = _minimal_seg_lookup(df_all)
        with patch('gnn_layer.cue_features.build_cue_blocks_with_segments', return_value=blocks), \
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
            self.assertTrue(path.endswith('language_atlas.txt'))

    def test_path_in_mechanism_dir(self):
        from process import output_paths as _paths
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            self.assertIn(_paths.reports_mechanism_dir(tmp), path)

    def test_section_headers_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('THERAPEUTIC LANGUAGE ATLAS', content)
            self.assertIn('0. MOST DISCRIMINATIVE THERAPIST LANGUAGE', content)
            self.assertIn('1. TRANSITION CUE PROTOTYPES', content)
            self.assertIn('2. TRANSITION INVENTORY', content)
            self.assertIn('[distillation method: ', content)

    def test_transitions_and_no_vtt_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            # both transition types present, with stage names
            self.assertIn('Vigilance → Avoidance', content)
            self.assertIn('Avoidance → Vigilance', content)
            # no VTT in tmp → per-example cue fallback + prototype fallback
            self.assertIn('no VTT timing — cue unavailable', content)
            self.assertIn('no usable proximal cues', content)
            # 2 same-participant stage changes counted
            self.assertIn('2 same-participant stage changes total', content)

    def test_no_llm_prose_sections(self):
        """Motif glossary / movers / summaries / syntheses are gone."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._run(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertNotIn('MOTIF GLOSSARY', content)
            self.assertNotIn('[SUMMARY', content)
            self.assertNotIn('LLM SYNTHESIS', content)
            self.assertNotIn('EMERGENT MOTIFS', content)


class TestAtlasHelpers(unittest.TestCase):

    def test_fmt_timecode(self):
        from analysis.reports.language_atlas import _fmt_timecode
        self.assertEqual(_fmt_timecode(120610), '0:02:00')
        self.assertEqual(_fmt_timecode(3_723_000), '1:02:03')
        self.assertEqual(_fmt_timecode(0), '--:--:--')
        self.assertEqual(_fmt_timecode(None), '--:--:--')
        self.assertEqual(_fmt_timecode(float('nan')), '--:--:--')

    def test_fmt_gap(self):
        from analysis.reports.language_atlas import _fmt_gap
        self.assertEqual(_fmt_gap(60_000, 273_000), 'gap 3m 33s')
        self.assertEqual(_fmt_gap(0, 273_000), 'gap n/a')
        self.assertEqual(_fmt_gap(273_000, 60_000), 'gap n/a')
        self.assertEqual(_fmt_gap(None, 60_000), 'gap n/a')

    def test_speaker_lines_multi_speaker(self):
        from analysis.reports.language_atlas import _speaker_lines
        text = ("[therapist_2]: Any other challenges?\n"
                "[Participant_MM001]: I fell asleep.\n"
                "and then I woke up.")
        lines = _speaker_lines(text, 'participant')
        self.assertEqual(lines[0], ('therapist_2', 'Any other challenges?'))
        self.assertEqual(lines[1][0], 'Participant_MM001')
        self.assertIn('and then I woke up', lines[1][1])

    def test_speaker_lines_unprefixed_uses_default(self):
        from analysis.reports.language_atlas import _speaker_lines
        lines = _speaker_lines('Just plain text.', 'Participant_MM009')
        self.assertEqual(lines, [('Participant_MM009', 'Just plain text.')])

    def test_same_participant(self):
        from analysis.reports.language_atlas import _same_participant
        self.assertTrue(_same_participant({'from_participant': 'P01', 'to_participant': 'P01'}))
        self.assertFalse(_same_participant({'from_participant': 'P01', 'to_participant': 'P02'}))
        self.assertFalse(_same_participant({'from_participant': None, 'to_participant': 'P02'}))
        self.assertFalse(_same_participant({}))


if __name__ == '__main__':
    unittest.main()
