"""
Tests for process/anonymization_editor.py — the post-hoc speaker-key editor and
its downstream cascade.

Covers:
  * pure key edit ops + diff_keys
  * build_segid_map rewrites ONLY the fragment component of a segment_id
  * token remapper avoids chaining and substring collisions
  * apply_key_update keeps overlay↔segment and checkpoint↔segment joins intact
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from process import anonymization_editor as ae
from process import segments_io, classifications_io
from process import output_paths as _paths
from classification_tools.data_structures import Segment

QRA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'qra.py')


def _seg(segment_id, participant_id, session_id, idx, text, speaker='participant',
         trial_id='MoveMORE'):
    return Segment(
        segment_id=segment_id, trial_id=trial_id, participant_id=participant_id,
        session_id=session_id, session_number=1, cohort_id=1, session_variant='',
        segment_index=idx, start_time_ms=idx * 1000, end_time_ms=(idx + 1) * 1000,
        total_segments_in_session=2, speaker=speaker, text=text,
        word_count=len(text.split()), speakers_in_segment=[participant_id],
    )


# ---------------------------------------------------------------------------
# Pure key edit ops
# ---------------------------------------------------------------------------

class TestKeyEditOps(unittest.TestCase):
    def setUp(self):
        self.key = {
            'Jane Doe': {'role': 'participant', 'anonymized_id': 'Participant_MM016'},
            'Dr Smith': {'role': 'therapist', 'anonymized_id': 'therapist_4'},
        }

    def test_rename_anon_id(self):
        new = ae.rename_anon_id(self.key, 'therapist_4', 'therapist_9')
        self.assertEqual(new['Dr Smith']['anonymized_id'], 'therapist_9')
        # original is untouched (no mutation)
        self.assertEqual(self.key['Dr Smith']['anonymized_id'], 'therapist_4')

    def test_rename_anon_id_missing_raises(self):
        with self.assertRaises(KeyError):
            ae.rename_anon_id(self.key, 'therapist_99', 'therapist_1')

    def test_rename_raw_name(self):
        new = ae.rename_raw_name(self.key, 'Jane Doe', 'Jane A. Doe')
        self.assertIn('Jane A. Doe', new)
        self.assertNotIn('Jane Doe', new)
        self.assertEqual(new['Jane A. Doe']['anonymized_id'], 'Participant_MM016')

    def test_change_role_validates(self):
        with self.assertRaises(ValueError):
            ae.change_role(self.key, 'therapist_4', 'wizard')

    def test_merge_speakers(self):
        key = dict(self.key)
        key['Janie Doe'] = {'role': 'participant', 'anonymized_id': 'Participant_MM020'}
        new = ae.merge_speakers(key, ['Jane Doe', 'Janie Doe'], 'Participant_MM016')
        self.assertEqual(new['Janie Doe']['anonymized_id'], 'Participant_MM016')
        self.assertEqual(new['Jane Doe']['anonymized_id'], 'Participant_MM016')

    def test_diff_keys(self):
        new = ae.rename_anon_id(self.key, 'therapist_4', 'therapist_9')
        self.assertEqual(ae.diff_keys(self.key, new), {'therapist_4': 'therapist_9'})

    def test_diff_keys_empty_for_raw_rename(self):
        new = ae.rename_raw_name(self.key, 'Jane Doe', 'Jane A. Doe')
        self.assertEqual(ae.diff_keys(self.key, new), {})


# ---------------------------------------------------------------------------
# Token remapper
# ---------------------------------------------------------------------------

class TestTokenRemapper(unittest.TestCase):
    def test_no_chaining(self):
        # therapist_1 -> therapist_2 must NOT then become therapist_3
        remap = ae._make_token_remapper({'therapist_1': 'therapist_2', 'therapist_2': 'therapist_3'})
        self.assertEqual(remap('{therapist_1} and {therapist_2}'), '{therapist_2} and {therapist_3}')

    def test_substring_safety(self):
        # therapist_1 must not match inside {therapist_10}
        remap = ae._make_token_remapper({'therapist_1': 'therapist_9'})
        self.assertEqual(remap('{therapist_10}'), '{therapist_10}')
        self.assertEqual(remap('{therapist_1}'), '{therapist_9}')

    def test_bracket_label(self):
        remap = ae._make_token_remapper({'Participant_MM016': 'Participant_MM099'})
        self.assertEqual(remap('[Participant_MM016]: hello'), '[Participant_MM099]: hello')


# ---------------------------------------------------------------------------
# segid map
# ---------------------------------------------------------------------------

class TestBuildSegidMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        segs = [
            _seg('MoveMORE_c1s1_MM016_seg0000', 'Participant_MM016', 'c1s1', 0,
                 '[Participant_MM016]: I notice {Participant_MM016} pain at 4 or 5.'),
            _seg('MoveMORE_c1s1_4_thseg0000', 'therapist_4', 'c1s1', 1,
                 'Hello from {therapist_4}.', speaker='therapist'),
        ]
        segments_io.write_session_segments(self.tmp, 'c1s1', segs, 'h')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fragment_only_rewrite_therapist(self):
        m = ae.build_segid_map(self.tmp, {'therapist_4': 'therapist_9'})
        self.assertEqual(m, {'MoveMORE_c1s1_4_thseg0000': 'MoveMORE_c1s1_9_thseg0000'})

    def test_fragment_only_rewrite_participant(self):
        # The '0' digits inside seg0000 must NOT be touched.
        m = ae.build_segid_map(self.tmp, {'Participant_MM016': 'Participant_MM099'})
        self.assertEqual(m, {'MoveMORE_c1s1_MM016_seg0000': 'MoveMORE_c1s1_MM099_seg0000'})

    def test_empty_relabel(self):
        self.assertEqual(ae.build_segid_map(self.tmp, {}), {})


# ---------------------------------------------------------------------------
# Full cascade — join integrity
# ---------------------------------------------------------------------------

class TestApplyKeyUpdateCascade(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.seg_p = _seg(
            'MoveMORE_c1s1_MM016_seg0000', 'Participant_MM016', 'c1s1', 0,
            '[Participant_MM016]: My name token is {Participant_MM016} here.')
        self.seg_t = _seg(
            'MoveMORE_c1s1_4_thseg0000', 'therapist_4', 'c1s1', 1,
            'Therapist {therapist_4} speaking.', speaker='therapist')
        segments_io.write_session_segments(self.tmp, 'c1s1', [self.seg_p, self.seg_t], 'h')

        # key
        ae.save_key(self.tmp, {
            'Jane Doe': {'role': 'participant', 'anonymized_id': 'Participant_MM016'},
            'Dr Smith': {'role': 'therapist', 'anonymized_id': 'therapist_4'},
        })

        # theme overlay keyed by segment_id
        self.seg_p.primary_stage = 2
        self.seg_t.primary_stage = 3
        classifications_io.write_theme_overlay(self.tmp, [self.seg_p, self.seg_t])

        # a plain checkpoint + a model-first runs checkpoint
        ckpt_dir = _paths.llm_checkpoints_dir(self.tmp)
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, 'llm_results_x_20260101.json'), 'w') as f:
            json.dump({
                'MoveMORE_c1s1_MM016_seg0000': {'vote': 'CODED'},
                'MoveMORE_c1s1_4_thseg0000': {'vote': 'CODED'},
            }, f)
        with open(os.path.join(ckpt_dir, 'llm_results_x_20260101_runs.json'), 'w') as f:
            json.dump({
                '_meta': {'format': 'model_first_v1', 'n_runs': 1,
                          'per_run_models': ['m'], 'completed_runs': [0]},
                'run_results': {
                    'MoveMORE_c1s1_MM016_seg0000': {'0': {'stage': 2}},
                    'MoveMORE_c1s1_4_thseg0000': {'0': {'stage': 3}},
                },
            }, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_overlay_ids(self):
        path = classifications_io.overlay_path(self.tmp, 'theme')
        ids = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ids.append(json.loads(line)['segment_id'])
        return ids

    def test_cascade_rewrites_everything_consistently(self):
        old_key = ae.load_key(self.tmp)
        new_key = ae.rename_anon_id(old_key, 'therapist_4', 'therapist_9')

        stats = ae.apply_key_update(
            self.tmp, old_key, new_key,
            backup=False, regenerate=False, verbose=False,
        )
        self.assertEqual(stats['relabel_map'], {'therapist_4': 'therapist_9'})

        # key on disk updated
        self.assertEqual(ae.load_key(self.tmp)['Dr Smith']['anonymized_id'], 'therapist_9')

        # frozen segments: id + participant_id + text token all remapped
        segs = {s.session_id: s for s in segments_io.read_session_segments(self.tmp, 'c1s1')}
        by_id = {s.segment_id: s for s in segments_io.read_session_segments(self.tmp, 'c1s1')}
        self.assertIn('MoveMORE_c1s1_9_thseg0000', by_id)
        self.assertNotIn('MoveMORE_c1s1_4_thseg0000', by_id)
        t = by_id['MoveMORE_c1s1_9_thseg0000']
        self.assertEqual(t.participant_id, 'therapist_9')
        self.assertEqual(t.speakers_in_segment, ['therapist_9'])
        self.assertIn('{therapist_9}', t.text)
        self.assertNotIn('therapist_4', t.text)
        # participant segment untouched
        self.assertIn('MoveMORE_c1s1_MM016_seg0000', by_id)

        # overlay join integrity: every overlay id resolves to a frozen segment
        seg_ids = set(by_id)
        overlay_ids = self._read_overlay_ids()
        self.assertIn('MoveMORE_c1s1_9_thseg0000', overlay_ids)
        for oid in overlay_ids:
            self.assertIn(oid, seg_ids, f"overlay id {oid} has no matching segment")

        # checkpoint join integrity (both shapes)
        ckpt_dir = _paths.llm_checkpoints_dir(self.tmp)
        with open(os.path.join(ckpt_dir, 'llm_results_x_20260101.json')) as f:
            plain = json.load(f)
        self.assertIn('MoveMORE_c1s1_9_thseg0000', plain)
        self.assertNotIn('MoveMORE_c1s1_4_thseg0000', plain)
        with open(os.path.join(ckpt_dir, 'llm_results_x_20260101_runs.json')) as f:
            runs = json.load(f)
        self.assertIn('_meta', runs)  # meta preserved
        self.assertIn('MoveMORE_c1s1_9_thseg0000', runs['run_results'])
        self.assertNotIn('MoveMORE_c1s1_4_thseg0000', runs['run_results'])

    def test_dry_run_writes_nothing(self):
        old_key = ae.load_key(self.tmp)
        new_key = ae.rename_anon_id(old_key, 'therapist_4', 'therapist_9')
        prev = ae.apply_key_update(self.tmp, old_key, new_key, dry_run=True)
        self.assertEqual(prev['n_segment_ids'], 1)
        # key unchanged on disk
        self.assertEqual(ae.load_key(self.tmp)['Dr Smith']['anonymized_id'], 'therapist_4')
        # segment id unchanged
        ids = {s.segment_id for s in segments_io.read_session_segments(self.tmp, 'c1s1')}
        self.assertIn('MoveMORE_c1s1_4_thseg0000', ids)


class TestEditAnonHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        r = subprocess.run([sys.executable, QRA, 'edit-anonymization', '--help'],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn('anonymization', (r.stdout + r.stderr).lower())


if __name__ == '__main__':
    unittest.main()
