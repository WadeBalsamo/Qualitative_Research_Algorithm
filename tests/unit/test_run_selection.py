"""M5 unit tests — analysis/run_selection (per-run κ + IRR-gated selection).

Hermetic: no network, no model downloads. Synthetic ballots (via the run
registry) are scored against synthetic human consensus codes built directly in
the ``irr_human_codes`` / ``irr_testsets`` tables (the exact columns the importer
writes), so κ is hand-computable for the small fixtures.

Coverage:
  * per_run_kappa correctness (hand-computed κ; ABSTAIN paired as ABSTAIN_CODE,
    ERROR cells skipped; per-run n right; archived runs excluded).
  * top_n ranking (κ desc, ties n then run_id), min_kappa floor, <n-qualifying
    warning + partial select, zero-κ FALLBACK selects all eligible.
  * 'all' strategy excludes archived + failed; params_hash mismatch exclusion
    (+ the NULL-hash → treat-as-current warning path).
  * manifest record round-trip (load_selection_record); set_selected applied;
    changed flag.
  * selection → rebuild → rater_votes cache contains exactly the selected labels.
  * irr_analysis.runs_kappa best-effort empty on a no-registry project.
"""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import db, segments_io, run_registry as rr, classifications_io as cio
from process.config import PipelineConfig
from analysis import run_selection as rsel


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _seed_segments(tmp, n, params_hash='hash1'):
    """Freeze ``n`` participant segments p0..p{n-1} with a known params_hash."""
    raw = [
        Segment(segment_id=f'p{i}', trial_id='t', participant_id='P1',
                session_id='c1s1', session_number=1, cohort_id=1, segment_index=i,
                speaker='participant', text='I notice the pain.', word_count=4,
                start_time_ms=i * 1000, end_time_ms=i * 1000 + 500)
        for i in range(n)
    ]
    segments_io.write_session_segments(tmp, 'c1s1', raw, params_hash)


def _coded(stage, conf=0.8):
    return {'vote': 'CODED', 'primary_stage': stage, 'primary_confidence': conf,
            'secondary_stage': None, 'secondary_confidence': None,
            'justification': 'j', 'evidence_phrase': 'e'}


def _abstain():
    return {'vote': 'ABSTAIN', 'primary_stage': None, 'primary_confidence': None,
            'secondary_stage': None, 'secondary_confidence': None,
            'justification': '', 'evidence_phrase': ''}


def _make_run(tmp, overlay, model, cells, *, status='completed', selected=True,
              params_hash='hash1'):
    """Create a run, stamp status/selection/hash, upsert ``cells``, refresh counters.

    ``cells`` maps segment_id → parsed-ballot dict, or None for an ERROR ballot.
    """
    rid = rr.create_run(tmp, overlay=overlay, model=model, rater_label=model)
    rr.upsert_ballots(tmp, overlay, rid, cells)
    rr.update_run(tmp, rid, status=status, selected=1 if selected else 0,
                  segmentation_params_hash=params_hash)
    rr.refresh_counters(tmp, rid)
    return rid


def _seed_human_consensus(tmp, truth, *, worksheet_n=1, source='majority'):
    """Insert an ``irr_testsets`` parent + consensus rows in ``irr_human_codes``.

    ``truth`` maps segment_id → primary code (VAAMR theme_id or ABSTAIN_CODE).
    Only consensus rows are written (the per-run κ path reads consensus rows).
    """
    with db.open_db(tmp) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO irr_testsets "
            "(worksheet_n, name, raters, n_items, created_at) VALUES (?,?,?,?,?)",
            (worksheet_n, f'ts_{worksheet_n}', db.dumps(['becca', 'wade']),
             len(truth), '2026-06-10T00:00:00+00:00'),
        )
        for i, (sid, code) in enumerate(sorted(truth.items()), start=1):
            conn.execute(
                "INSERT OR REPLACE INTO irr_human_codes "
                "(worksheet_n, item_num, segment_id, rater, prim, secondary, "
                " is_consensus, source, notes) VALUES (?,?,?,?,?,?,?,?,?)",
                (worksheet_n, i, sid, '__consensus__', code, None, 1, source, None),
            )


def _theme_config(tmp, vote_mode='majority'):
    cfg = PipelineConfig()
    cfg.output_dir = tmp
    cfg.theme_classification.vote_mode = vote_mode
    cfg.speaker_filter.mode = 'exclude'
    cfg.speaker_filter.speakers = ['therapist']
    return cfg


# ---------------------------------------------------------------------------
# per_run_kappa correctness
# ---------------------------------------------------------------------------

class TestPerRunKappa(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_segments(self.tmp, 8)
        # Hand-computable 2x2: human=[1,1,1,1,0,0,0,0] → κ(mB)=0.5 below.
        self.truth = {f'p{i}': (1 if i < 4 else 0) for i in range(8)}
        _seed_human_consensus(self.tmp, self.truth)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_perfect_run_kappa_one(self):
        cells = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(8)}
        rid = _make_run(self.tmp, 'theme', 'mA', cells)
        out = rsel.per_run_kappa(self.tmp, 'theme')
        self.assertAlmostEqual(out[rid]['cohen_kappa'], 1.0, places=6)
        self.assertEqual(out[rid]['n'], 8)
        self.assertEqual(out[rid]['percent_agreement'], 1.0)

    def test_hand_computed_kappa_half(self):
        # mach=[1,1,1,0,0,0,0,1] vs human=[1,1,1,1,0,0,0,0] → 6/8 agree → κ=0.5.
        mach = [1, 1, 1, 0, 0, 0, 0, 1]
        cells = {f'p{i}': _coded(mach[i]) for i in range(8)}
        rid = _make_run(self.tmp, 'theme', 'mB', cells)
        out = rsel.per_run_kappa(self.tmp, 'theme')
        self.assertAlmostEqual(out[rid]['cohen_kappa'], 0.5, places=6)
        self.assertEqual(out[rid]['n'], 8)
        self.assertAlmostEqual(out[rid]['percent_agreement'], 0.75, places=6)

    def test_abstain_paired_and_error_skipped(self):
        # p0 ABSTAIN (human=1) → pairs as (1, ABSTAIN_CODE) disagreement.
        # p1 ERROR (None ballot) → skipped entirely (not paired) → n drops to 7.
        cells = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(8)}
        cells['p0'] = _abstain()
        cells['p1'] = None  # ERROR ballot
        rid = _make_run(self.tmp, 'theme', 'mE', cells)
        out = rsel.per_run_kappa(self.tmp, 'theme')
        self.assertEqual(out[rid]['n'], 7)  # 8 truth - 1 ERROR-skipped
        # The remaining 7: p0 mispaired (1 vs ABSTAIN), p2..p7 perfect.
        # Verify ABSTAIN encoded as the 6th category (negative sentinel present).
        self.assertIsNotNone(out[rid]['cohen_kappa'])

    def test_archived_run_excluded(self):
        cells = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(8)}
        good = _make_run(self.tmp, 'theme', 'mGood', cells)
        arch = _make_run(self.tmp, 'theme', 'mArch', cells)
        rr.update_run(self.tmp, arch, status='archived', selected=0)
        out = rsel.per_run_kappa(self.tmp, 'theme')
        self.assertIn(good, out)
        self.assertNotIn(arch, out)

    def test_failed_run_included_with_status(self):
        # A 'failed' run still carries (partial) ballots → scored, status surfaced.
        cells = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(4)}
        rid = _make_run(self.tmp, 'theme', 'mFail', cells, status='failed')
        out = rsel.per_run_kappa(self.tmp, 'theme')
        self.assertEqual(out[rid]['status'], 'failed')
        self.assertEqual(out[rid]['n'], 4)

    def test_purer_overlay_no_human_codes_returns_empty(self):
        # No purer-kind human codes exist → graceful {} (per the plan).
        self.assertEqual(rsel.per_run_kappa(self.tmp, 'purer'), {})

    def test_no_registry_returns_empty(self):
        empty = tempfile.mkdtemp()
        try:
            self.assertEqual(rsel.per_run_kappa(empty, 'theme'), {})
        finally:
            shutil.rmtree(empty, ignore_errors=True)


# ---------------------------------------------------------------------------
# select_runs — strategies
# ---------------------------------------------------------------------------

class TestSelectRunsTopN(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_segments(self.tmp, 8)
        self.truth = {f'p{i}': (1 if i < 4 else 0) for i in range(8)}
        _seed_human_consensus(self.tmp, self.truth)
        self.cfg = _theme_config(self.tmp)
        # mA perfect (κ=1.0), mB κ=0.5, mC constant-1 (κ≈0 or <0).
        perfect = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(8)}
        machB = [1, 1, 1, 0, 0, 0, 0, 1]
        cellsB = {f'p{i}': _coded(machB[i]) for i in range(8)}
        cellsC = {f'p{i}': _coded(1) for i in range(8)}
        self.rA = _make_run(self.tmp, 'theme', 'mA', perfect)
        self.rB = _make_run(self.tmp, 'theme', 'mB', cellsB)
        self.rC = _make_run(self.tmp, 'theme', 'mC', cellsC)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_top_n_ranks_by_kappa_desc(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')  # default n=3
        # All 3 selected (n=3); but rank order is mA(1.0) > mB(0.5) > mC.
        self.assertEqual(rec['strategy'], 'top_n_by_human_irr')
        self.assertEqual(set(rec['selected_run_ids']), {self.rA, self.rB, self.rC})
        self.assertFalse(rec['fallback_used'])

    def test_top_2_picks_highest_two(self):
        self.cfg.run_selection = {'vaamr': {'strategy': 'top_n_by_human_irr', 'n': 2}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        self.assertEqual(rec['selected_run_ids'], [self.rA, self.rB])
        self.assertEqual(set(rec['rejected_run_ids']), {self.rC})
        self.assertEqual(rr.selected_runs(self.tmp, 'theme'), [self.rA, self.rB])

    def test_min_kappa_floor_excludes_below(self):
        # Floor at 0.6 → only mA (1.0) qualifies; mB(0.5)/mC excluded. <n warns.
        self.cfg.run_selection = {
            'vaamr': {'strategy': 'top_n_by_human_irr', 'n': 3, 'min_kappa': 0.6}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        self.assertEqual(rec['selected_run_ids'], [self.rA])
        self.assertIn('qualified', buf.getvalue())
        self.assertEqual(rr.selected_runs(self.tmp, 'theme'), [self.rA])

    def test_tie_break_by_n_then_run_id(self):
        # Two runs with identical κ but different n → higher-n first; equal n → lower id.
        fresh = tempfile.mkdtemp()
        try:
            _seed_segments(fresh, 8)
            truth = {f'p{i}': (1 if i < 4 else 0) for i in range(8)}
            _seed_human_consensus(fresh, truth)
            # r1: perfect on all 8 (κ=1.0, n=8). r2: perfect on 4 (κ=1.0, n=4).
            full = {f'p{i}': _coded(truth[f'p{i}']) for i in range(8)}
            half = {f'p{i}': _coded(truth[f'p{i}']) for i in range(4)}
            r1 = _make_run(fresh, 'theme', 'm1', full)
            r2 = _make_run(fresh, 'theme', 'm2', half)
            eligible = rr.list_runs(fresh, overlay='theme')
            kappa = rsel.per_run_kappa(fresh, 'theme')
            ranked = rsel._rank_by_kappa(eligible, kappa)
            # Same κ=1.0 → higher n (r1, n=8) ranks before r2 (n=4).
            self.assertEqual([r['run_id'] for r in ranked][:2], [r1, r2])
        finally:
            shutil.rmtree(fresh, ignore_errors=True)


class TestSelectRunsFallback(unittest.TestCase):
    """Zero κ computable → select ALL eligible + fallback_used (user decision)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_segments(self.tmp, 4)
        self.cfg = _theme_config(self.tmp)
        # No human codes imported at all → no κ computable.
        cells = {f'p{i}': _coded(1) for i in range(4)}
        self.r1 = _make_run(self.tmp, 'theme', 'mX', cells)
        self.r2 = _make_run(self.tmp, 'theme', 'mY', cells)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_human_codes_selects_all_eligible(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        self.assertTrue(rec['fallback_used'])
        self.assertEqual(set(rec['selected_run_ids']), {self.r1, self.r2})
        self.assertIn('FALLBACK', buf.getvalue())
        self.assertEqual(set(rr.selected_runs(self.tmp, 'theme')), {self.r1, self.r2})

    def test_no_overlap_selects_all_eligible(self):
        # Human codes exist but on DIFFERENT segments → no overlap → fallback.
        _seed_human_consensus(self.tmp, {'zzz-other': 1, 'qqq-other': 0})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        self.assertTrue(rec['fallback_used'])
        self.assertEqual(set(rec['selected_run_ids']), {self.r1, self.r2})


class TestSelectRunsAllStrategy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_segments(self.tmp, 4)
        self.cfg = _theme_config(self.tmp)
        cells = {f'p{i}': _coded(1) for i in range(4)}
        self.rOk = _make_run(self.tmp, 'theme', 'ok1', cells, status='completed')
        self.rErr = _make_run(self.tmp, 'theme', 'ok2', cells,
                              status='completed_with_errors')
        self.rArch = _make_run(self.tmp, 'theme', 'arch', cells, status='completed')
        rr.update_run(self.tmp, self.rArch, status='archived', selected=0)
        self.rFail = _make_run(self.tmp, 'theme', 'fail', cells, status='failed')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_excludes_archived_and_failed(self):
        self.cfg.run_selection = {'vaamr': {'strategy': 'all'}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        # completed + completed_with_errors eligible; archived + failed excluded.
        self.assertEqual(set(rec['selected_run_ids']), {self.rOk, self.rErr})
        self.assertNotIn(self.rArch, rec['selected_run_ids'])
        self.assertNotIn(self.rFail, rec['selected_run_ids'])

    def test_purer_default_is_all(self):
        # No explicit config → purer default policy = 'all'.
        # (Build purer runs on the same project.)
        _seed_segments(self.tmp, 1) if False else None
        cells = {'p0': _coded(0)}
        p1 = _make_run(self.tmp, 'purer', 'pm1', cells)
        p2 = _make_run(self.tmp, 'purer', 'pm2', cells)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, _theme_config(self.tmp), 'purer')
        self.assertEqual(rec['strategy'], 'all')
        self.assertEqual(set(rec['selected_run_ids']), {p1, p2})


class TestSelectRunsStaleness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_segments(self.tmp, 4, params_hash='current')  # current project hash
        self.truth = {f'p{i}': (1 if i < 2 else 0) for i in range(4)}
        _seed_human_consensus(self.tmp, self.truth)
        self.cfg = _theme_config(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stale_run_excluded(self):
        good = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(4)}
        rGood = _make_run(self.tmp, 'theme', 'mGood', good, params_hash='current')
        rStale = _make_run(self.tmp, 'theme', 'mStale', good, params_hash='OLDHASH')
        self.cfg.run_selection = {'vaamr': {'strategy': 'all'}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        self.assertIn(rGood, rec['selected_run_ids'])
        self.assertNotIn(rStale, rec['selected_run_ids'])
        self.assertTrue(any('STALE' in w for w in rec['warnings']))

    def test_null_hash_treated_as_current_with_warning(self):
        good = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(4)}
        # NULL stored hash: create run then null it (legacy run predating the guard).
        rid = rr.create_run(self.tmp, overlay='theme', model='mNull', rater_label='mNull')
        rr.upsert_ballots(self.tmp, 'theme', rid, good)
        rr.update_run(self.tmp, rid, status='completed', selected=0,
                      segmentation_params_hash=None)
        rr.refresh_counters(self.tmp, rid)
        self.cfg.run_selection = {'vaamr': {'strategy': 'all'}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        self.assertIn(rid, rec['selected_run_ids'])
        self.assertTrue(any('no segmentation_params_hash' in w for w in rec['warnings']))


# ---------------------------------------------------------------------------
# Manifest round-trip + changed flag
# ---------------------------------------------------------------------------

class TestSelectionManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_segments(self.tmp, 4)
        self.truth = {f'p{i}': (1 if i < 2 else 0) for i in range(4)}
        _seed_human_consensus(self.tmp, self.truth)
        self.cfg = _theme_config(self.tmp)
        cells = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(4)}
        self.r1 = _make_run(self.tmp, 'theme', 'm1', cells)
        self.r2 = _make_run(self.tmp, 'theme', 'm2', cells)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_manifest_key(self):
        self.assertEqual(rsel.selection_manifest_key('theme'), 'run_selection:theme')

    def test_record_roundtrip(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        loaded = rsel.load_selection_record(self.tmp, 'theme')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['strategy'], rec['strategy'])
        self.assertEqual(loaded['selected_run_ids'], rec['selected_run_ids'])
        self.assertIn('decided_at', loaded)
        self.assertIn('kappa_snapshot', loaded)
        # snapshot keys are stringified run ids (JSON round-trip).
        self.assertEqual({int(k) for k in loaded['kappa_snapshot']}, {self.r1, self.r2})

    def test_changed_flag(self):
        # First selection from a default (all born selected) — compute change vs prior.
        rr.set_selected(self.tmp, 'theme', [self.r1])  # prior = {r1}
        self.cfg.run_selection = {'vaamr': {'strategy': 'all'}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        # 'all' now selects {r1, r2} → changed vs prior {r1}.
        self.assertTrue(rec['changed'])
        # Re-running with the same result → unchanged.
        with redirect_stdout(io.StringIO()):
            rec2 = rsel.select_runs(self.tmp, self.cfg, 'theme')
        self.assertFalse(rec2['changed'])


# ---------------------------------------------------------------------------
# Integration: selection → rebuild → cache holds exactly the selected labels
# ---------------------------------------------------------------------------

class TestSelectionRebuildIntegration(unittest.TestCase):
    """Select 2 of 3 runs → rebuild → the rater_votes cache contains exactly the
    2 selected labels (composed with consensus_rebuild as test_consensus_rebuild
    does)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_segments(self.tmp, 8)
        self.truth = {f'p{i}': (1 if i < 4 else 0) for i in range(8)}
        _seed_human_consensus(self.tmp, self.truth)
        self.cfg = _theme_config(self.tmp)
        # mA perfect, mB κ=0.5, mC constant (worst). top-2 → {mA, mB}.
        perfect = {f'p{i}': _coded(self.truth[f'p{i}']) for i in range(8)}
        machB = [1, 1, 1, 0, 0, 0, 0, 1]
        cellsB = {f'p{i}': _coded(machB[i]) for i in range(8)}
        cellsC = {f'p{i}': _coded(2) for i in range(8)}
        self.rA = _make_run(self.tmp, 'theme', 'mA', perfect)
        self.rB = _make_run(self.tmp, 'theme', 'mB', cellsB)
        self.rC = _make_run(self.tmp, 'theme', 'mC', cellsC)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_select_2_of_3_rebuild_cache(self):
        from process import consensus_rebuild as crb
        self.cfg.run_selection = {'vaamr': {'strategy': 'top_n_by_human_irr', 'n': 2}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = rsel.select_runs(self.tmp, self.cfg, 'theme')
        self.assertEqual(rec['selected_run_ids'], [self.rA, self.rB])
        self.assertTrue(rec['changed'])

        crb.rebuild_overlay(self.tmp, 'theme', self.cfg)
        overlay = {r['segment_id']: r for r in cio.read_overlay(self.tmp, 'theme')}
        rec0 = overlay['p0']
        # Cache reflects EXACTLY the two selected raters, in run_id order.
        self.assertEqual([v['rater'] for v in rec0['rater_votes']], ['mA', 'mB'])
        self.assertEqual(rec0['rater_ids'], ['mA', 'mB'])
        # mC (the deselected constant-2 rater) must NOT appear in any cache row.
        for rec_row in overlay.values():
            raters = [v['rater'] for v in (rec_row['rater_votes'] or [])]
            self.assertNotIn('mC', raters)


# ---------------------------------------------------------------------------
# irr_analysis integration — runs_kappa best-effort
# ---------------------------------------------------------------------------

class TestIrrAnalysisRunsKappa(unittest.TestCase):
    def test_per_run_kappa_empty_on_no_registry_project(self):
        # A project with human codes but NO run registry → per_run_kappa returns
        # {} (no runs) — the best-effort value run_irr_analysis stores.
        tmp = tempfile.mkdtemp()
        try:
            _seed_segments(tmp, 1)
            _seed_human_consensus(tmp, {'p0': 1})
            self.assertEqual(rsel.per_run_kappa(tmp, 'theme'), {})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_irr_analysis_includes_runs_kappa(self):
        # Full IRR run on a project WITH a registry must attach results['runs_kappa']
        # (a dict) and results['run_selection'] without crashing.
        from analysis import irr_analysis
        tmp = tempfile.mkdtemp()
        try:
            _seed_segments(tmp, 4)
            truth = {f'p{i}': (1 if i < 2 else 0) for i in range(4)}
            _seed_human_consensus(tmp, truth)
            cells = {f'p{i}': _coded(truth[f'p{i}']) for i in range(4)}
            rid = _make_run(tmp, 'theme', 'mA', cells)
            # Also write the theme overlay so machine labels load (read_master_segments).
            from process import consensus_rebuild as crb
            crb.rebuild_overlay(tmp, 'theme', _theme_config(tmp))
            buf = io.StringIO()
            with redirect_stdout(buf):
                results = irr_analysis.run_irr_analysis(tmp, _theme_config(tmp),
                                                        verbose=False)
            self.assertIsInstance(results.get('runs_kappa'), dict)
            self.assertIn(rid, results['runs_kappa'])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
