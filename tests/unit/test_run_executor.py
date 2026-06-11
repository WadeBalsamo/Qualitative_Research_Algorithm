"""M3 tests — process/run_executor.

The headline gate (``test_queue_equals_inline_record_for_record``): a queued
3-run VAAMR sweep through the executor reproduces an inline 3-model
``stage_classify_theme`` overlay record-for-record, because both vote through
``llm_classifier.build_merge_result`` over byte-identical ballots.

Also exercised: cell-wise interrupt → resume completes (only missing cells
re-fetched); STOP sentinel honored between runs; retry → completed_with_errors;
persistent failure → failed; dead rater → failed without a retry storm; flock
exclusion (RunnerBusy); lmstudio pre-flight skip + force bypass; PURER cue_block
guard.

Hermetic: FakeLLMClient patched at the same seam as test_consensus_rebuild
(``classification_tools.theme_llm.llm_classifier.LLMClient``); no network, no
model downloads.
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import segments_io, classifications_io as cio, run_registry as rr
from process import run_executor as rx
from process.config import PipelineConfig
from process.orchestrator import stage_classify_theme
from tests.testhelpers import FakeLLMClient, tiny_vaamr_framework, tiny_purer_framework


_LLM_SEAM = 'classification_tools.theme_llm.llm_classifier.LLMClient'

_P_TEXT = ('I keep noticing the pain and try to avoid moving, but I am '
           'attending to my breath with metacognition and reappraisal.')


def _participant(seg_id, idx):
    return Segment(
        segment_id=seg_id, trial_id='t', participant_id='P1', session_id='c1s1',
        session_number=1, cohort_id=1, segment_index=idx, speaker='participant',
        text=_P_TEXT, word_count=len(_P_TEXT.split()),
        start_time_ms=idx * 1000, end_time_ms=idx * 1000 + 800,
    )


def _therapist(seg_id, idx, text='What did you notice in your body just now?'):
    return Segment(
        segment_id=seg_id, trial_id='t', participant_id='P1', session_id='c1s1',
        session_number=1, cohort_id=1, segment_index=idx, speaker='therapist',
        text=text, word_count=len(text.split()),
        start_time_ms=idx * 1000, end_time_ms=idx * 1000 + 800,
    )


def _theme_config(tmp, raters, vote_mode='majority'):
    cfg = PipelineConfig()
    cfg.output_dir = tmp
    tc = cfg.theme_classification
    tc.per_run_models = list(raters)
    tc.n_runs = len(raters)
    tc.backend = 'fake'
    tc.temperature = 0.0
    tc.vote_mode = vote_mode
    tc.save_interval = 2
    cfg.speaker_filter.mode = 'exclude'
    cfg.speaker_filter.speakers = ['therapist']
    return cfg


def _stage_mapping(mapping, default='Vigilance'):
    """A responder factory: echoes the per-model construct NAME off fake.config.model."""
    def make(fake):
        def responder(prompt):
            name = mapping.get(fake.config.model, default)
            return {'primary_stage': name, 'primary_confidence': 0.8,
                    'secondary_stage': None, 'secondary_confidence': None,
                    'justification': f'j-{fake.config.model}', 'evidence_phrase': 'e'}
        return responder
    return make


def _snapshot(tmp, key):
    return {r['segment_id']: r for r in cio.read_overlay(tmp, key)}


def _ballot(stage):
    """A canonical parsed-run CODED ballot (the shape on_progress receives)."""
    return {'vote': 'CODED', 'primary_stage': stage, 'primary_confidence': 0.8,
            'secondary_stage': None, 'secondary_confidence': None,
            'justification': 'j', 'evidence_phrase': 'e'}


def _write_partial_ckpt(run_dir, overlay, run_id, cells):
    """Write a model_first_v1 per-run checkpoint so a retry resumes cell-wise."""
    import json as _json
    path = rx._per_run_checkpoint_path(run_dir, overlay, run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        '_meta': {'format': 'model_first_v1', 'n_runs': 1,
                  'per_run_models': ['mA'], 'completed_runs': []},
        'run_results': {sid: {'0': cell} for sid, cell in cells.items()},
    }
    with open(path, 'w') as f:
        _json.dump(payload, f)


def _new_fake(make_responder, config_backend='fake'):
    """Construct a FakeLLMClient whose responder closes over itself."""
    class _Cfg:
        backend = config_backend
        model = 'unset'
        models = []
        temperature = 0.0
        no_reasoning = False
        process_logger = None
    fake = FakeLLMClient(config=_Cfg())
    fake._responder = make_responder(fake)
    return fake


# ---------------------------------------------------------------------------
# Equivalence gate
# ---------------------------------------------------------------------------

class TestQueueEqualsInline(unittest.TestCase):
    def setUp(self):
        self.raters = ['mA', 'mB', 'mC']
        self.mapping = {'mA': 'Avoidance', 'mB': 'Avoidance', 'mC': 'Metacognition'}
        self.fw = tiny_vaamr_framework()

    def _seed(self, tmp):
        raw = [_participant(f'p{i}', i) for i in range(4)] + [_therapist('th0', 4)]
        segments_io.write_session_segments(tmp, 'c1s1', raw, 'hash1')

    def tearDown(self):
        for d in getattr(self, '_dirs', []):
            shutil.rmtree(d, ignore_errors=True)

    def test_queue_equals_inline_record_for_record(self):
        self._dirs = []
        # --- Inline 3-model classify ---
        inline = tempfile.mkdtemp(); self._dirs.append(inline)
        self._seed(inline)
        cfg_i = _theme_config(inline, self.raters)
        segs_i = segments_io.load_segments_for_stage(inline, apply=())
        with mock.patch(_LLM_SEAM, return_value=_new_fake(_stage_mapping(self.mapping))):
            stage_classify_theme(cfg_i, self.fw, segments=segs_i, output_dir=inline)
        inline_snap = _snapshot(inline, 'theme')

        # --- Queued 3-run sweep through the executor ---
        queued = tempfile.mkdtemp(); self._dirs.append(queued)
        self._seed(queued)
        cfg_q = _theme_config(queued, self.raters)
        run_ids = [rr.create_run(queued, overlay='theme', model=m, rater_label=m)
                   for m in self.raters]
        self.assertEqual(run_ids, sorted(run_ids))  # ascending = slot order
        with mock.patch(_LLM_SEAM, return_value=_new_fake(_stage_mapping(self.mapping))):
            summary = rx.execute_queue(queued, cfg_q, overlays=('theme',))

        self.assertIn('theme', summary['overlays_rebuilt'])
        self.assertFalse(summary['stopped_early'])
        # Every run completed (no errors from the fake).
        for rid in run_ids:
            self.assertEqual(summary['per_run'][rid], 'completed')
        # Selection policy 'all' auto-selected the 3 completed runs.
        self.assertEqual(rr.selected_runs(queued, 'theme'), run_ids)

        queued_snap = _snapshot(queued, 'theme')
        # Record-for-record equality on the 4 participant rows (therapist row is
        # present-but-null in both; compare the full overlay).
        self.assertEqual(set(inline_snap), set(queued_snap))
        for sid in inline_snap:
            self.assertEqual(inline_snap[sid], queued_snap[sid],
                             f'overlay record for {sid!r} differs (inline vs queued)')
        # And the consensus is the hand-computed majority (Avoidance=stage 1).
        self.assertEqual(queued_snap['p0']['primary_stage'], 1)
        self.assertEqual(queued_snap['p0']['agreement_level'], 'majority')


# ---------------------------------------------------------------------------
# Interruption + cell-wise resume
# ---------------------------------------------------------------------------

class _InterruptingFake(FakeLLMClient):
    """Raises KeyboardInterrupt after ``fail_after`` successful requests, once."""
    def __init__(self, mapping, fail_after):
        class _Cfg:
            backend = 'fake'; model = 'unset'; models = []
            temperature = 0.0; no_reasoning = False; process_logger = None
        super().__init__(config=_Cfg())
        self._mapping = mapping
        self._fail_after = fail_after
        self._armed = True

    def request(self, prompt):
        if self._armed and len(self.calls) >= self._fail_after:
            self._armed = False
            raise KeyboardInterrupt('injected mid-sweep')
        self.calls.append(prompt)
        name = self._mapping.get(self.config.model, 'Vigilance')
        import json as _json
        text = _json.dumps({'primary_stage': name, 'primary_confidence': 0.8,
                            'secondary_stage': None, 'secondary_confidence': None,
                            'justification': 'j', 'evidence_phrase': 'e'})
        return text, {'choices': [{'finish_reason': 'stop',
                                   'message': {'content': text, 'reasoning_content': ''}}]}


class TestInterruptResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [_participant(f'p{i}', i) for i in range(5)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'hash1')
        self.cfg = _theme_config(self.tmp, ['mA'])
        self.fw = tiny_vaamr_framework()
        self.run_id = rr.create_run(self.tmp, overlay='theme', model='mA', rater_label='mA')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_interrupt_then_resume_completes_only_missing(self):
        # First pass: interrupt after 2 successful requests.
        fake1 = _InterruptingFake({'mA': 'Avoidance'}, fail_after=2)
        with mock.patch(_LLM_SEAM, return_value=fake1):
            with self.assertRaises(KeyboardInterrupt):
                rx.execute_queue(self.tmp, self.cfg, overlays=('theme',))
        # Run is still 'running' (resumable) and a per-run checkpoint exists.
        run = rr.get_run(self.tmp, self.run_id)
        self.assertEqual(run['status'], 'running')
        ckpt = rx._per_run_checkpoint_path(self.tmp, 'theme', self.run_id)
        self.assertTrue(os.path.exists(ckpt))
        n_first = len(fake1.calls)
        self.assertGreaterEqual(n_first, 2)
        self.assertLess(n_first, 5)  # did NOT finish all 5

        # Resume: a fresh fake completes the rest; only missing cells re-fetched.
        fake2 = _new_fake(_stage_mapping({'mA': 'Avoidance'}))
        with mock.patch(_LLM_SEAM, return_value=fake2):
            summary = rx.execute_queue(self.tmp, self.cfg, overlays=('theme',))
        self.assertEqual(summary['per_run'][self.run_id], 'completed')
        # Total requests across both passes ≈ 5 (the checkpointed cells were not
        # re-fetched); the resume fake issued only the remainder.
        self.assertEqual(len(fake2.calls), 5 - n_first)
        snap = _snapshot(self.tmp, 'theme')
        self.assertEqual(len([s for s in snap.values()
                              if s.get('primary_stage') is not None]), 5)


# ---------------------------------------------------------------------------
# STOP sentinel
# ---------------------------------------------------------------------------

class TestStopSentinel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [_participant(f'p{i}', i) for i in range(3)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'hash1')
        self.cfg = _theme_config(self.tmp, ['mA', 'mB'])
        self.fw = tiny_vaamr_framework()
        self.r1 = rr.create_run(self.tmp, overlay='theme', model='mA', rater_label='mA')
        self.r2 = rr.create_run(self.tmp, overlay='theme', model='mB', rater_label='mB')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stop_after_first_run_leaves_second_queued(self):
        # A fake that touches STOP_QRA_RUNS as soon as run 1 (mA) finishes.
        stop_path = os.path.join(self.tmp, 'STOP_QRA_RUNS')

        def make(fake):
            def responder(prompt):
                # After mA's last segment, drop the STOP sentinel.
                if fake.config.model == 'mA' and len(fake.calls) >= 3:
                    open(stop_path, 'w').close()
                return {'primary_stage': 'Avoidance', 'primary_confidence': 0.8,
                        'secondary_stage': None, 'secondary_confidence': None,
                        'justification': 'j', 'evidence_phrase': 'e'}
            return responder

        with mock.patch(_LLM_SEAM, return_value=_new_fake(make)):
            summary = rx.execute_queue(self.tmp, self.cfg, overlays=('theme',))
        self.assertTrue(summary['stopped_early'])
        self.assertEqual(rr.get_run(self.tmp, self.r1)['status'], 'completed')
        self.assertEqual(rr.get_run(self.tmp, self.r2)['status'], 'queued')


# ---------------------------------------------------------------------------
# Retry / failure / dead-rater
# ---------------------------------------------------------------------------

class _AlwaysGarbageFake(FakeLLMClient):
    """Every request returns unparseable text → all-ERROR ballots."""
    def __init__(self):
        class _Cfg:
            backend = 'fake'; model = 'unset'; models = []
            temperature = 0.0; no_reasoning = False; process_logger = None
        super().__init__(config=_Cfg())

    def request(self, prompt):
        self.calls.append(prompt)
        return 'not json at all', {'choices': [{'finish_reason': 'stop',
                                                 'message': {'content': 'not json at all',
                                                             'reasoning_content': ''}}]}


class TestRetryAndFailure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [_participant(f'p{i}', i) for i in range(3)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'hash1')
        self.cfg = _theme_config(self.tmp, ['mA'])
        self.fw = tiny_vaamr_framework()
        self.run_id = rr.create_run(self.tmp, overlay='theme', model='mA', rater_label='mA')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_flaky_sweep_then_ok_resumes_cell_wise(self):
        # The executor's retry loop fires on SWEEP-level exceptions (request-level
        # errors are absorbed into ERROR ballots by _request_and_parse).  Inject a
        # sweep that classifies a couple of cells (checkpointing them) then raises
        # on the first attempt, and succeeds on the retry — only the missing cells
        # are re-fetched because the retry resumes from the per-run checkpoint.
        real_build = rx._build_sweep
        state = {'attempt': 0}

        def flaky_build(cfg, sub, run_dir, overlay, checkpoint_path, flush):
            real_sweep, applies = real_build(cfg, sub, run_dir, overlay,
                                             checkpoint_path, flush)

            def wrapped():
                state['attempt'] += 1
                if state['attempt'] == 1:
                    # Flush 2 cells, persist a checkpoint, then crash the sweep.
                    flush({'p0': _ballot(1), 'p1': _ballot(1)})
                    _write_partial_ckpt(run_dir, overlay,
                                        rr.get_run(run_dir, self.run_id)['run_id'],
                                        {'p0': _ballot(1), 'p1': _ballot(1)})
                    raise RuntimeError('sweep crashed mid-pass')
                return real_sweep()
            return wrapped, applies

        with mock.patch.object(rx, '_build_sweep', flaky_build), \
             mock.patch(_LLM_SEAM, return_value=_new_fake(_stage_mapping({'mA': 'Avoidance'}))):
            status = rx.execute_single_run(self.tmp, self.cfg,
                                           rr.get_run(self.tmp, self.run_id), retries=2)
        self.assertEqual(status, 'completed')
        self.assertEqual(state['attempt'], 2)  # crashed once, succeeded on retry
        # All 3 cells are CODED ballots (execute_single_run upserts ballots +
        # counters; the overlay itself is rebuilt by execute_queue).
        run = rr.get_run(self.tmp, self.run_id)
        self.assertEqual(run['n_coded'], 3)
        self.assertEqual(run['n_total'], 3)

    def test_persistent_sweep_failure_fails_after_retries(self):
        # A sweep that raises on EVERY attempt → 'failed' after retries exhausted.
        real_build = rx._build_sweep

        def always_fail_build(cfg, sub, run_dir, overlay, checkpoint_path, flush):
            def boom():
                raise RuntimeError('always crashes')
            return boom, None

        with mock.patch.object(rx, '_build_sweep', always_fail_build):
            status = rx.execute_single_run(self.tmp, self.cfg,
                                           rr.get_run(self.tmp, self.run_id), retries=2)
        self.assertEqual(status, 'failed')
        self.assertEqual(rr.get_run(self.tmp, self.run_id)['status'], 'failed')

    def test_dead_rater_fails_fast_without_retry_storm(self):
        fake = _AlwaysGarbageFake()
        with mock.patch(_LLM_SEAM, return_value=fake):
            status = rx.execute_single_run(self.tmp, self.cfg,
                                           rr.get_run(self.tmp, self.run_id), retries=5)
        self.assertEqual(status, 'completed_with_errors')  # garbage parses → ERROR cells but sweep succeeds
        run = rr.get_run(self.tmp, self.run_id)
        self.assertEqual(run['n_error'], 3)
        self.assertEqual(run['n_coded'], 0)
        # No retry storm: 3 segments × 1 model × parse-retries only (bounded),
        # NOT 5 retries × 3. _PARSE_RETRY_ATTEMPTS=3 per cell → 9 calls max.
        self.assertLessEqual(len(fake.calls), 9)


# ---------------------------------------------------------------------------
# flock exclusion
# ---------------------------------------------------------------------------

class TestFlock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [_participant('p0', 0)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'hash1')
        self.cfg = _theme_config(self.tmp, ['mA'])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_second_executor_under_held_lock_raises_runner_busy(self):
        with rx.acquire_runner_lock(self.tmp):
            with self.assertRaises(rx.RunnerBusy):
                rx.execute_queue(self.tmp, self.cfg, overlays=('theme',))


# ---------------------------------------------------------------------------
# lmstudio pre-flight skip + force
# ---------------------------------------------------------------------------

class TestPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [_participant('p0', 0), _participant('p1', 1)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'hash1')
        self.cfg = _theme_config(self.tmp, ['mA'])
        self.cfg.theme_classification.backend = 'lmstudio'
        self.fw = tiny_vaamr_framework()
        self.run_id = rr.create_run(self.tmp, overlay='theme', model='mA', rater_label='mA')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_model_mismatch_leaves_run_queued(self):
        with mock.patch('classification_tools.llm_client.LLMClient.check_loaded_model',
                        return_value=False):
            status = rx.execute_single_run(self.tmp, self.cfg,
                                           rr.get_run(self.tmp, self.run_id), retries=1)
        self.assertEqual(status, 'queued')
        self.assertEqual(rr.get_run(self.tmp, self.run_id)['status'], 'queued')

    def test_force_bypasses_preflight(self):
        # check_loaded_model False, but force=True → proceeds. Patch the seam so
        # the actual sweep uses a fake (no network).
        with mock.patch('classification_tools.llm_client.LLMClient.check_loaded_model',
                        return_value=False), \
             mock.patch(_LLM_SEAM, return_value=_new_fake(
                 _stage_mapping({'mA': 'Avoidance'}), config_backend='lmstudio')):
            status = rx.execute_single_run(self.tmp, self.cfg,
                                           rr.get_run(self.tmp, self.run_id),
                                           retries=1, force=True)
        self.assertEqual(status, 'completed')


# ---------------------------------------------------------------------------
# PURER cue_block guard
# ---------------------------------------------------------------------------

class TestPurerCueBlockGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [_participant('p0', 0), _therapist('t0', 1), _participant('p1', 2)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'hash1')
        self.cfg = PipelineConfig()
        self.cfg.output_dir = self.tmp
        self.cfg.run_purer_labeler = True
        tc = self.cfg.theme_classification
        tc.per_run_models = ['pA', 'pB']
        tc.n_runs = 2
        tc.backend = 'fake'
        self.cfg.purer_cue.classification_unit = 'cue_block'  # NOT supported
        self.run_id = rr.create_run(self.tmp, overlay='purer', model='pA', rater_label='pA')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cue_block_run_fails_with_message(self):
        status = rx.execute_single_run(self.tmp, self.cfg,
                                       rr.get_run(self.tmp, self.run_id), retries=1)
        self.assertEqual(status, 'failed')
        self.assertEqual(rr.get_run(self.tmp, self.run_id)['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
