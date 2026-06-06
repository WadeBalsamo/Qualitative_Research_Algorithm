"""
tests/unit/test_classification_loop.py
---------------------------------------
Unit tests for classification_tools/classification_loop.py.

Covers:
  - filter_participant_segments: Critical Design Rule enforcement —
    only participant segments pass through
  - classify_segments: single-model path with FakeLLMClient (n_runs=1)
  - classify_segments: multi-run (n_runs=2, per_run_models) model-first path
  - checkpoint write + resume: segment is not re-requested after checkpoint load
  - returned dict keys / stage-id range sanity
  - _save_checkpoint / _load_runs_checkpoint round-trip
  - _write_status_entry does not crash on typical merged result
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.classification_loop import (
    filter_participant_segments,
    classify_segments,
    _save_checkpoint,
    _load_runs_checkpoint,
    _write_status_entry,
    _ms_to_timecode,
    _stage_name,
)
from tests.testhelpers import FakeLLMClient, make_segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_ballot(stage_id: int):
    return {
        'vote': 'CODED',
        'primary_stage': stage_id,
        'primary_confidence': 0.85,
        'secondary_stage': None,
        'secondary_confidence': None,
        'justification': 'test',
        'evidence_phrase': '',
    }


def _abstain_ballot():
    return {
        'vote': 'ABSTAIN',
        'primary_stage': None,
        'primary_confidence': 0.5,
        'secondary_stage': None,
        'secondary_confidence': None,
        'justification': '',
        'evidence_phrase': '',
    }


def _identity_parse(text):
    """Parse helper: text is already the parsed dict (injected via responder)."""
    try:
        import json as _json
        return _json.loads(text)
    except Exception:
        return None


def _make_merge(n_runs=1):
    """Returns a merge_runs that picks the first non-None ballot."""
    def merge_runs(run_list):
        for r in run_list:
            if r is not None:
                return r
        return {'vote': 'ABSTAIN', 'primary_stage': None, 'primary_confidence': 0.0,
                'secondary_stage': None, 'secondary_confidence': None,
                'justification': '', 'rater_votes': [], 'consensus': {}}
    return merge_runs


# ---------------------------------------------------------------------------
# filter_participant_segments
# ---------------------------------------------------------------------------

class TestFilterParticipantSegments(unittest.TestCase):
    """Critical Design Rule: VAAMR must only receive participant segments."""

    def test_keeps_participants(self):
        segs = [
            make_segment('p1', speaker='participant'),
            make_segment('p2', speaker='participant'),
        ]
        result = filter_participant_segments(segs)
        self.assertEqual([s.segment_id for s in result], ['p1', 'p2'])

    def test_removes_therapists(self):
        segs = [
            make_segment('t1', speaker='therapist'),
            make_segment('p1', speaker='participant'),
            make_segment('t2', speaker='therapist'),
        ]
        result = filter_participant_segments(segs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].segment_id, 'p1')

    def test_all_therapist_gives_empty(self):
        segs = [make_segment(f't{i}', speaker='therapist') for i in range(4)]
        result = filter_participant_segments(segs)
        self.assertEqual(result, [])

    def test_empty_input(self):
        self.assertEqual(filter_participant_segments([]), [])

    def test_mixed_session_ids_preserves_boundary(self):
        """Segments from multiple sessions are filtered purely by speaker."""
        segs = [
            make_segment('c1s1_0', speaker='participant', session_id='c1s1'),
            make_segment('c1s1_1', speaker='therapist', session_id='c1s1'),
            make_segment('c1s2_0', speaker='participant', session_id='c1s2'),
            make_segment('c1s2_1', speaker='therapist', session_id='c1s2'),
        ]
        result = filter_participant_segments(segs)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(s.speaker == 'participant' for s in result))

    def test_unknown_speaker_excluded(self):
        """A segment with speaker != 'participant' must be excluded."""
        segs = [
            make_segment('x1', speaker='unknown'),
            make_segment('p1', speaker='participant'),
        ]
        result = filter_participant_segments(segs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].segment_id, 'p1')


# ---------------------------------------------------------------------------
# classify_segments — single-model path (n_runs=1, no per_run_models)
# ---------------------------------------------------------------------------

class TestClassifySegmentsSingleModel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dict_keyed_by_segment_id(self):
        segs = [make_segment('seg_001'), make_segment('seg_002')]
        client = FakeLLMClient(responder=lambda p: {'vote': 'CODED',
                                                     'primary_stage': 2,
                                                     'primary_confidence': 0.8,
                                                     'secondary_stage': None,
                                                     'secondary_confidence': None,
                                                     'justification': 'j',
                                                     'evidence_phrase': ''})
        results = classify_segments(
            segments=segs,
            client=client,
            n_runs=1,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=_make_merge(1),
        )
        self.assertIn('seg_001', results)
        self.assertIn('seg_002', results)

    def test_stage_id_in_range(self):
        """Merged results must contain stage ids in 0..4 for VAAMR."""
        segs = [make_segment(f'seg_{i}') for i in range(5)]
        stage_cycle = [0, 1, 2, 3, 4]

        call_count = [0]
        def responder(prompt):
            stage = stage_cycle[call_count[0] % 5]
            call_count[0] += 1
            return {'vote': 'CODED', 'primary_stage': stage, 'primary_confidence': 0.9,
                    'secondary_stage': None, 'secondary_confidence': None,
                    'justification': '', 'evidence_phrase': ''}

        client = FakeLLMClient(responder=responder)
        results = classify_segments(
            segments=segs,
            client=client,
            n_runs=1,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=_make_merge(1),
        )
        self.assertEqual(len(results), 5)
        for seg_id, merged in results.items():
            if merged and merged.get('vote') == 'CODED':
                self.assertIn(merged.get('primary_stage'), range(5))

    def test_client_called_once_per_segment_single_run(self):
        segs = [make_segment(f's{i}') for i in range(3)]
        client = FakeLLMClient()
        classify_segments(
            segments=segs,
            client=client,
            n_runs=1,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=_make_merge(1),
        )
        self.assertEqual(len(client.calls), 3)

    def test_empty_segment_list_returns_empty_dict(self):
        client = FakeLLMClient()
        results = classify_segments(
            segments=[],
            client=client,
            n_runs=1,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=_make_merge(1),
        )
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])

    def test_parse_failure_yields_merge_called_with_none(self):
        """When parse_response returns None, merge_runs is still called with [None]."""
        segs = [make_segment('seg_fail')]
        client = FakeLLMClient(raw_text='not-valid-json-at-all!!!')
        merged_inputs = []

        def capture_merge(run_list):
            merged_inputs.append(list(run_list))
            return {'vote': 'ABSTAIN', 'primary_stage': None, 'primary_confidence': 0.0,
                    'secondary_stage': None, 'secondary_confidence': None,
                    'justification': '', 'rater_votes': [], 'consensus': {}}

        classify_segments(
            segments=segs,
            client=client,
            n_runs=1,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=lambda text: None,
            merge_runs=capture_merge,
        )
        self.assertEqual(len(merged_inputs), 1)
        self.assertEqual(merged_inputs[0], [None])

    def test_checkpoint_written_to_output_dir(self):
        segs = [make_segment('seg_ckpt')]
        client = FakeLLMClient()
        classify_segments(
            segments=segs,
            client=client,
            n_runs=1,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=_make_merge(1),
            output_dir=self.tmp,
            file_prefix='test_ckpt',
        )
        ckpt_dir = os.path.join(self.tmp, 'checkpoints')
        self.assertTrue(os.path.isdir(ckpt_dir))
        files = os.listdir(ckpt_dir)
        self.assertTrue(any('test_ckpt' in f for f in files),
                        f"No checkpoint file found in {ckpt_dir}: {files}")

    def test_resume_skips_already_classified(self):
        """Segments whose IDs are already in the checkpoint must not be re-queried."""
        # Pre-write a checkpoint containing seg_001 already classified.
        existing = {'seg_001': {'vote': 'CODED', 'primary_stage': 3}}
        ckpt_path = os.path.join(self.tmp, 'resume.json')
        with open(ckpt_path, 'w') as f:
            json.dump(existing, f)

        segs = [make_segment('seg_001'), make_segment('seg_002')]
        client = FakeLLMClient()
        results = classify_segments(
            segments=segs,
            client=client,
            n_runs=1,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=_make_merge(1),
            resume_from=ckpt_path,
        )
        # Only seg_002 should have been queried (seg_001 was in checkpoint).
        self.assertEqual(len(client.calls), 1)
        # Both segments must be in the result.
        self.assertIn('seg_001', results)
        self.assertIn('seg_002', results)
        # seg_001 must retain its pre-existing classification.
        self.assertEqual(results['seg_001']['primary_stage'], 3)

    def test_build_prompt_receives_segment_and_index(self):
        """build_prompt must be called with correct (segment, run, all_segs, idx)."""
        segs = [make_segment('seg_a'), make_segment('seg_b')]
        calls = []

        def capturing_build_prompt(seg, run, all_segs, idx):
            calls.append((seg.segment_id, run, idx))
            return seg.text

        client = FakeLLMClient()
        classify_segments(
            segments=segs,
            client=client,
            n_runs=1,
            build_prompt=capturing_build_prompt,
            parse_response=_identity_parse,
            merge_runs=_make_merge(1),
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ('seg_a', 0, 0))
        self.assertEqual(calls[1], ('seg_b', 0, 1))


# ---------------------------------------------------------------------------
# classify_segments — model-first path (per_run_models, n_runs=2)
# ---------------------------------------------------------------------------

class TestClassifySegmentsModelFirst(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_config_attr(self):
        """FakeLLMClient config that supports .model assignment."""
        class Cfg:
            model = 'model-a'
            backend = 'fake'
            models = ['model-a', 'model-b']
            temperature = 0.0
            process_logger = None
        return Cfg()

    def test_model_first_calls_each_model_once_per_segment(self):
        """With per_run_models=[m0, m1], every segment sees exactly 2 requests."""
        segs = [make_segment('seg_0'), make_segment('seg_1')]
        cfg = self._make_config_attr()
        client = FakeLLMClient(config=cfg)

        def responder(prompt):
            return {'vote': 'CODED', 'primary_stage': 1, 'primary_confidence': 0.75,
                    'secondary_stage': None, 'secondary_confidence': None,
                    'justification': '', 'evidence_phrase': ''}

        client._responder = responder

        from classification_tools.majority_vote import vote_single_label

        def merge_runs(run_list):
            rater_ids = ['model-a', 'model-b']
            return vote_single_label(run_list, rater_ids=rater_ids)

        results = classify_segments(
            segments=segs,
            client=client,
            n_runs=2,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=merge_runs,
            output_dir=self.tmp,
            per_run_models=['model-a', 'model-b'],
        )
        # 2 segments × 2 models = 4 total calls
        self.assertEqual(len(client.calls), 4)
        self.assertIn('seg_0', results)
        self.assertIn('seg_1', results)

    def test_model_first_writes_runs_checkpoint(self):
        """Model-first path must write a *_runs.json checkpoint file."""
        segs = [make_segment('seg_a')]
        cfg = self._make_config_attr()
        client = FakeLLMClient(config=cfg)

        from classification_tools.majority_vote import vote_single_label

        def merge_runs(run_list):
            return vote_single_label(run_list, rater_ids=['model-a', 'model-b'])

        classify_segments(
            segments=segs,
            client=client,
            n_runs=2,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=merge_runs,
            output_dir=self.tmp,
            per_run_models=['model-a', 'model-b'],
            file_prefix='mf_test',
        )
        ckpt_dir = os.path.join(self.tmp, 'checkpoints')
        runs_files = [f for f in os.listdir(ckpt_dir) if '_runs.json' in f]
        self.assertTrue(len(runs_files) >= 1, f"No *_runs.json found: {os.listdir(ckpt_dir)}")

    def test_model_first_resume_skips_completed_run(self):
        """If run 0 is already in completed_runs, only run 1 executes."""
        import datetime
        segs = [make_segment('seg_r')]
        cfg = self._make_config_attr()
        client = FakeLLMClient(config=cfg)

        # Write a fake runs checkpoint with run 0 already done.
        ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
        ckpt_dir = os.path.join(self.tmp, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        runs_ckpt = os.path.join(ckpt_dir, f'resume_test_{ts}_runs.json')
        payload = {
            '_meta': {
                'format': 'model_first_v1',
                'n_runs': 2,
                'per_run_models': ['model-a', 'model-b'],
                'completed_runs': [0],
            },
            'run_results': {
                'seg_r': {'0': {'vote': 'CODED', 'primary_stage': 2,
                                'primary_confidence': 0.9,
                                'secondary_stage': None,
                                'secondary_confidence': None,
                                'justification': 'pre-cached', 'evidence_phrase': ''}},
            },
        }
        with open(runs_ckpt, 'w') as f:
            json.dump(payload, f)

        from classification_tools.majority_vote import vote_single_label

        def merge_runs(run_list):
            return vote_single_label(run_list, rater_ids=['model-a', 'model-b'])

        results = classify_segments(
            segments=segs,
            client=client,
            n_runs=2,
            build_prompt=lambda seg, run, all_segs, idx: seg.text,
            parse_response=_identity_parse,
            merge_runs=merge_runs,
            output_dir=self.tmp,
            per_run_models=['model-a', 'model-b'],
            resume_from=runs_ckpt,
        )
        # Only run 1 (model-b) should have been requested.
        self.assertEqual(len(client.calls), 1,
                         f"Expected 1 call (run 1 only), got {len(client.calls)}")
        self.assertIn('seg_r', results)


# ---------------------------------------------------------------------------
# _save_checkpoint / _load_runs_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointHelpers(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_load_merged_checkpoint(self):
        results = {'seg_a': {'vote': 'CODED', 'primary_stage': 1},
                   'seg_b': {'vote': 'ABSTAIN', 'primary_stage': None}}
        _save_checkpoint(results, self.tmp, 'test', 'mymodel', '26-01-01T00-00-00')
        ckpt_dir = os.path.join(self.tmp, 'checkpoints')
        files = os.listdir(ckpt_dir)
        self.assertEqual(len(files), 1)
        with open(os.path.join(ckpt_dir, files[0])) as f:
            loaded = json.load(f)
        self.assertEqual(loaded['seg_a']['primary_stage'], 1)
        self.assertIsNone(loaded['seg_b']['primary_stage'])

    def test_save_checkpoint_applies_serialize_fn(self):
        results = {'seg_c': [1, 2, 3]}
        _save_checkpoint(results, self.tmp, 'serialized', None, '26-01-01T00-00-00',
                         serialize_fn=lambda x: {'list_len': len(x)})
        ckpt_dir = os.path.join(self.tmp, 'checkpoints')
        files = os.listdir(ckpt_dir)
        with open(os.path.join(ckpt_dir, files[0])) as f:
            loaded = json.load(f)
        self.assertEqual(loaded['seg_c']['list_len'], 3)

    def test_load_runs_checkpoint_returns_state(self):
        from classification_tools.classification_loop import _save_runs_checkpoint
        run_results = {
            'seg_x': {'0': {'vote': 'CODED', 'primary_stage': 0}, '1': None},
        }
        completed = {0}
        _save_runs_checkpoint(run_results, completed, 2,
                              ['m0', 'm1'], self.tmp, 'load_test', None, '26-01-01T00-00-00')
        ckpt_dir = os.path.join(self.tmp, 'checkpoints')
        files = os.listdir(ckpt_dir)
        runs_file = [f for f in files if '_runs.json' in f][0]
        loaded_results, loaded_completed = _load_runs_checkpoint(
            os.path.join(ckpt_dir, runs_file), 2
        )
        self.assertIn('seg_x', loaded_results)
        self.assertIn(0, loaded_completed)
        self.assertNotIn(1, loaded_completed)

    def test_load_legacy_merged_checkpoint_returns_empty(self):
        """A plain merged-format checkpoint passed to _load_runs_checkpoint
        returns ({}, set()) so the caller re-classifies from scratch."""
        legacy = {'seg_a': {'vote': 'CODED', 'primary_stage': 2}}
        path = os.path.join(self.tmp, 'legacy.json')
        with open(path, 'w') as f:
            json.dump(legacy, f)
        run_results, completed = _load_runs_checkpoint(path, 2)
        self.assertEqual(run_results, {})
        self.assertEqual(completed, set())


# ---------------------------------------------------------------------------
# _write_status_entry
# ---------------------------------------------------------------------------

class TestWriteStatusEntry(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_does_not_raise_on_typical_result(self):
        seg = make_segment('seg_status', text='Some text here.')
        merged = {
            'rater_ids': ['r1'],
            'rater_votes': [{'rater': 'r1', 'vote': 'CODED', 'stage': 2,
                             'confidence': 0.9, 'secondary_stage': None,
                             'secondary_confidence': None, 'justification': 'ok'}],
            'consensus': {'primary_stage': 2, 'primary_confidence': 0.9,
                          'consensus_vote': 2, 'agreement_level': 'unanimous',
                          'n_agree': 1, 'n_raters': 1, 'needs_review': False,
                          'tie_broken_by_confidence': False},
        }
        status_path = os.path.join(self.tmp, 'status.txt')
        _write_status_entry(status_path, seg, 0, 1, merged, [])
        self.assertTrue(os.path.isfile(status_path))
        with open(status_path) as f:
            content = f.read()
        self.assertIn('seg_status', content)

    def test_does_not_raise_on_abstain_result(self):
        seg = make_segment('seg_abstain')
        merged = {
            'rater_ids': ['r1'],
            'rater_votes': [],
            'consensus': {'primary_stage': None, 'primary_confidence': 0.0,
                          'consensus_vote': 'ABSTAIN', 'agreement_level': 'unanimous',
                          'n_agree': 1, 'n_raters': 1, 'needs_review': False,
                          'tie_broken_by_confidence': False},
        }
        status_path = os.path.join(self.tmp, 'status_abstain.txt')
        _write_status_entry(status_path, seg, 0, 1, merged, [])
        self.assertTrue(os.path.isfile(status_path))

    def test_does_not_raise_on_non_dict_merged(self):
        """Non-dict merged values must not crash the status writer."""
        seg = make_segment('seg_nondict')
        status_path = os.path.join(self.tmp, 'status_nd.txt')
        _write_status_entry(status_path, seg, 0, 1, 'raw string result', [])
        self.assertTrue(os.path.isfile(status_path))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestUtilityHelpers(unittest.TestCase):

    def test_ms_to_timecode_zero(self):
        self.assertEqual(_ms_to_timecode(0), '00:00:00.000')

    def test_ms_to_timecode_one_hour(self):
        self.assertEqual(_ms_to_timecode(3600000), '01:00:00.000')

    def test_ms_to_timecode_mixed(self):
        # 1h 23m 45s 678ms
        ms = (1 * 3600 + 23 * 60 + 45) * 1000 + 678
        self.assertEqual(_ms_to_timecode(ms), '01:23:45.678')

    def test_stage_name_known(self):
        self.assertEqual(_stage_name(0), 'Vigilance')
        self.assertEqual(_stage_name(4), 'Reappraisal')

    def test_stage_name_unknown_returns_str(self):
        self.assertEqual(_stage_name(99), '99')
        self.assertEqual(_stage_name('ABSTAIN'), 'ABSTAIN')


if __name__ == '__main__':
    unittest.main()
