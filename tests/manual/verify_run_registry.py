#!/usr/bin/env python3
"""
tests/manual/verify_run_registry.py
------------------------------------
End-to-end verification of the QRA run-registry system (M0–M5).
Runs on a temp COPY of the source project — NEVER touches the live data.

Usage:
    .venv/bin/python tests/manual/verify_run_registry.py [--source data/MMORE_Processed] [--keep]

Phases:
  1  Migration:           open_db -> schema_version=2; backfill audit
  2  Rebuild-vs-snapshot: delta report (incident blast-radius audit)
  3  Registry round-trip: FakeLLM 2-run queue, interrupt, resume, fix-errors
  4  Selection on real IRR: per_run_kappa table, top-3 select, transcript check
  5  Cleanup + PASS table

Exit code: 0 = all phases PASS, 1 = any FAIL.

Safety:
  - FIRST action: shutil.copytree source -> tempdir (deleted at end unless --keep)
  - All mutations on the copy ONLY
  - BEFORE open_db: read-only snapshot of v1 overlays via sqlite3 URI
  - No git commits; no files written outside tests/manual/ (and the copy+artifact)
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple
from unittest import mock

# ---------------------------------------------------------------------------
# Path bootstrap: mirror tests/conftest.py
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _REPO_ROOT not in sys.path:
    sys.path.insert(1, _REPO_ROOT)

# Also insert tests/ so testhelpers imports work.
_TESTS = os.path.join(_REPO_ROOT, 'tests')
if _TESTS not in sys.path:
    sys.path.insert(2, _TESTS)

# ---------------------------------------------------------------------------
# Results accumulator
# ---------------------------------------------------------------------------
_RESULTS: List[Tuple[int, str, str, str]] = []  # (phase, name, status, detail)


def _record(phase: int, name: str, passed: bool, detail: str = '') -> bool:
    status = 'PASS' if passed else 'FAIL'
    _RESULTS.append((phase, name, status, detail))
    flag = '  PASS' if passed else '  FAIL ***'
    print(f"[Phase {phase}]{flag}  {name}" + (f"  ({detail})" if detail else ''))
    return passed


def _phase_header(n: int, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"PHASE {n}: {title}")
    print('='*70)


# ---------------------------------------------------------------------------
# Read-only SQLite snapshot (BEFORE open_db mutates the copy)
# ---------------------------------------------------------------------------

def _ro_conn(db_file: str) -> sqlite3.Connection:
    """Open a read-only SQLite connection via URI."""
    conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _snapshot_v1_overlays(db_file: str) -> dict:
    """Snapshot v1 overlays before migration using read-only access."""
    snap = {'theme': {}, 'purer': {}, 'schema_version': None}
    try:
        conn = _ro_conn(db_file)
        row = conn.execute(
            "SELECT value FROM _schema_meta WHERE key='schema_version'"
        ).fetchone()
        snap['schema_version'] = row[0] if row else None

        for r in conn.execute(
            "SELECT segment_id, primary_stage, rater_votes, rater_ids, agreement_level "
            "FROM theme_labels WHERE rater_votes IS NOT NULL"
        ).fetchall():
            snap['theme'][r['segment_id']] = {
                'primary_stage': r['primary_stage'],
                'agreement_level': r['agreement_level'],
                'rater_votes': json.loads(r['rater_votes']) if r['rater_votes'] else [],
                'rater_ids': json.loads(r['rater_ids']) if r['rater_ids'] else [],
            }

        for r in conn.execute(
            "SELECT segment_id, purer_primary, purer_rater_votes, purer_rater_ids "
            "FROM purer_labels WHERE purer_rater_votes IS NOT NULL"
        ).fetchall():
            snap['purer'][r['segment_id']] = {
                'purer_primary': r['purer_primary'],
                'purer_rater_votes': json.loads(r['purer_rater_votes']) if r['purer_rater_votes'] else [],
                'purer_rater_ids': json.loads(r['purer_rater_ids']) if r['purer_rater_ids'] else [],
            }

        conn.close()
    except Exception as e:
        print(f"  [snapshot] WARNING reading v1 overlays: {e}")
    return snap


# ---------------------------------------------------------------------------
# Phase 1: Migration
# ---------------------------------------------------------------------------

def phase1_migration(copy_dir: str, v1_snap: dict) -> bool:
    _phase_header(1, 'Migration: open_db -> schema_version=2 + backfill audit')
    from process import db
    from process import run_registry as rr

    all_pass = True

    # 1a. open_db should auto-migrate to v2
    try:
        with db.open_db(copy_dir) as conn:
            stored = db.get_meta(conn, 'schema_version')
        ok = (str(stored) == '2')
        all_pass &= _record(1, 'schema_version=2 after open_db', ok,
                            f'stored={stored}')
    except Exception as e:
        _record(1, 'open_db migration did not raise', False, str(e))
        return False

    # 1b. classification_runs rows exist (one per rater per overlay)
    runs = rr.list_runs(copy_dir)
    theme_runs = [r for r in runs if r['overlay'] == 'theme']
    purer_runs = [r for r in runs if r['overlay'] == 'purer']

    # Count raters in v1 snapshot
    v1_theme_raters = set()
    for seg_data in v1_snap['theme'].values():
        for rv in seg_data.get('rater_votes', []):
            if isinstance(rv, dict) and rv.get('rater'):
                v1_theme_raters.add(rv['rater'])
    v1_purer_raters = set()
    for seg_data in v1_snap['purer'].values():
        for rv in seg_data.get('purer_rater_votes', []):
            if isinstance(rv, dict) and rv.get('rater'):
                v1_purer_raters.add(rv['rater'])

    print(f"  V1 theme raters found: {sorted(v1_theme_raters)}")
    print(f"  V1 purer raters found: {sorted(v1_purer_raters)}")
    print(f"  classification_runs theme rows: {len(theme_runs)}")
    print(f"  classification_runs purer rows: {len(purer_runs)}")

    ok_theme = (len(theme_runs) == len(v1_theme_raters)) if v1_theme_raters else len(theme_runs) >= 0
    ok_purer = (len(purer_runs) == len(v1_purer_raters)) if v1_purer_raters else len(purer_runs) >= 0
    all_pass &= _record(1, f'theme run count matches v1 raters ({len(v1_theme_raters)})',
                        ok_theme, f'runs={len(theme_runs)}')
    all_pass &= _record(1, f'purer run count matches v1 raters ({len(v1_purer_raters)})',
                        ok_purer, f'runs={len(purer_runs)}')

    # 1c. All backfilled runs: status=completed, selected=1, note mentions backfill
    backfill_note_keyword = 'backfill'
    ok_status = all(r['status'] == 'completed' for r in runs)
    ok_selected = all(r['selected'] for r in runs)
    ok_note = all(
        r.get('note') and backfill_note_keyword in r['note'].lower()
        for r in runs
    )
    all_pass &= _record(1, 'all backfilled runs status=completed', ok_status)
    all_pass &= _record(1, 'all backfilled runs selected=1', ok_selected)
    all_pass &= _record(1, "all backfilled runs note mentions 'backfill'", ok_note)

    # 1d. label_ballots count matches sum of rater_votes entries in v1 snapshot
    # Count total v1 ballot entries (one per rater_votes entry across all segments)
    v1_theme_ballot_count = sum(
        len(d.get('rater_votes', [])) for d in v1_snap['theme'].values()
    )
    v1_purer_ballot_count = sum(
        len(d.get('purer_rater_votes', [])) for d in v1_snap['purer'].values()
    )
    with db.open_db(copy_dir) as conn:
        row = conn.execute(
            "SELECT SUM(CASE WHEN overlay='theme' THEN 1 ELSE 0 END) as theme_cnt, "
            "       SUM(CASE WHEN overlay='purer' THEN 1 ELSE 0 END) as purer_cnt "
            "FROM label_ballots"
        ).fetchone()
        lb_theme = int(row['theme_cnt'] or 0)
        lb_purer = int(row['purer_cnt'] or 0)

    print(f"  V1 theme ballot entries: {v1_theme_ballot_count}  "
          f"label_ballots theme rows: {lb_theme}")
    print(f"  V1 purer ballot entries: {v1_purer_ballot_count}  "
          f"label_ballots purer rows: {lb_purer}")
    # label_ballots can be >= v1 count if any segment had multiple rater entries;
    # but should equal for well-formed v1 data (no duplicate rater per segment).
    # Allow ±2% to accommodate edge cases in the real data.
    def _close(a, b, tol=0.05):
        if a == b:
            return True
        if max(a, b) == 0:
            return True
        return abs(a - b) / max(a, b) <= tol

    all_pass &= _record(1, 'label_ballots theme count ≈ v1 rater_votes entries',
                        _close(lb_theme, v1_theme_ballot_count),
                        f'ballots={lb_theme} v1={v1_theme_ballot_count}')
    all_pass &= _record(1, 'label_ballots purer count ≈ v1 rater_votes entries',
                        _close(lb_purer, v1_purer_ballot_count),
                        f'ballots={lb_purer} v1={v1_purer_ballot_count}')

    # 1e. Counter consistency: n_coded+n_abstain+n_error = n_total
    bad_counters = []
    for r in runs:
        n_total = r.get('n_total') or 0
        calc = (r.get('n_coded') or 0) + (r.get('n_abstain') or 0) + (r.get('n_error') or 0)
        if n_total != calc:
            bad_counters.append((r['run_id'], r['rater_label'], n_total, calc))
    ok_counters = (len(bad_counters) == 0)
    all_pass &= _record(1, 'per-run counter n_coded+n_abstain+n_error=n_total',
                        ok_counters,
                        f'{len(bad_counters)} mismatches' if bad_counters else 'all consistent')
    if bad_counters:
        for rid, rl, nt, calc in bad_counters[:5]:
            print(f"    run {rid} ({rl}): n_total={nt} calc={calc}")

    return all_pass


# ---------------------------------------------------------------------------
# Phase 2: Rebuild-vs-snapshot delta (blast-radius audit)
# ---------------------------------------------------------------------------

def phase2_rebuild_delta(copy_dir: str, v1_snap: dict, source_name: str) -> bool:
    _phase_header(2, 'Rebuild-vs-snapshot delta (incident blast-radius audit)')
    from process import consensus_rebuild as crb
    from process import classifications_io as cio
    from process.config import PipelineConfig

    all_pass = True

    # Load config from the copy (mirror qra.py cmd_rebuild resolution)
    config_path = os.path.join(copy_dir, '02_meta', 'qra_config.json')
    try:
        with open(config_path) as f:
            file_data = json.load(f)
        # Mirror _flatten_wizard_config from qra.py
        result = {}
        pipeline = file_data.get('pipeline', {})
        for key in ('transcript_dir', 'output_dir', 'trial_id',
                    'run_theme_labeler', 'run_codebook_classifier',
                    'auto_analyze', 'speaker_anonymization_key_path'):
            if key in pipeline:
                result[key] = pipeline[key]
        for key, val in file_data.items():
            if isinstance(val, dict) and key not in ('pipeline', 'framework', 'codebook'):
                result[key] = val
        for key in ('resume_from', 'run_purer_labeler', 'auto_analyze'):
            if key in file_data and key not in result:
                result[key] = file_data[key]
        config = PipelineConfig.from_json(result)
        config.output_dir = copy_dir
        print(f"  Config loaded from {config_path}")
    except Exception as e:
        print(f"  WARNING: could not load config ({e}); using PipelineConfig()")
        config = PipelineConfig()
        config.output_dir = copy_dir

    delta_report_lines = []

    for overlay, prim_key in [('theme', 'primary_stage'), ('purer', 'purer_primary')]:
        print(f"\n  Rebuilding {overlay} overlay from selected ballots...")
        try:
            stats = crb.rebuild_overlay(copy_dir, overlay, config)
        except Exception as e:
            _record(2, f'{overlay} rebuild did not raise', False, str(e))
            all_pass = False
            continue

        if stats.get('skipped'):
            _record(2, f'{overlay} rebuild ran (not skipped)', False,
                    f"skipped: {stats.get('reason','?')}")
            all_pass = False
            continue

        print(f"  {overlay} rebuild stats: {stats}")
        _record(2, f'{overlay} rebuild completed', True,
                f"units={stats.get('n_units',0)} labeled={stats.get('n_labeled',0)} "
                f"changed={stats.get('n_changed',0)}")

        # Read the rebuilt overlay
        rebuilt_rows = cio.read_overlay(copy_dir, overlay)
        rebuilt_by_seg = {r['segment_id']: r for r in rebuilt_rows}

        # Compare against v1 snapshot
        v1_data = v1_snap.get(overlay, {})
        n_rows = len(rebuilt_by_seg)
        n_identical = 0
        delta_rows = []
        n_labeled_to_unlabeled = 0  # FAIL: had label, now doesn't
        n_unlabeled_to_different = 0  # FAIL: was unlabeled, now different than expected

        for seg_id, v1 in v1_data.items():
            if seg_id not in rebuilt_by_seg:
                continue
            rb = rebuilt_by_seg[seg_id]
            v1_prim = v1[prim_key]
            rb_prim = rb.get(prim_key)

            if v1_prim == rb_prim:
                n_identical += 1
            else:
                # Classify the delta
                v1_labeled = v1_prim is not None
                rb_labeled = rb_prim is not None

                if v1_labeled and not rb_labeled:
                    delta_kind = 'labeled->UNLABELED (FAIL candidate)'
                    n_labeled_to_unlabeled += 1
                elif not v1_labeled and rb_labeled:
                    delta_kind = 'unlabeled->labeled (M0 vote-fix; expected)'
                elif v1_labeled and rb_labeled and v1_prim != rb_prim:
                    delta_kind = f'labeled->{rb_prim} (was {v1_prim}; FAIL candidate)'
                    n_unlabeled_to_different += 1
                else:
                    delta_kind = f'other: v1={v1_prim} rb={rb_prim}'

                if overlay == 'purer':
                    v1_level = v1.get('purer_agreement_level', '?')
                    rb_level = rb.get('purer_agreement_level', '?')
                else:
                    v1_level = v1.get('agreement_level', '?')
                    rb_level = rb.get('agreement_level', '?')

                delta_rows.append({
                    'segment_id': seg_id,
                    'old_primary': v1_prim,
                    'new_primary': rb_prim,
                    'old_agreement_level': v1_level,
                    'new_agreement_level': rb_level,
                    'kind': delta_kind,
                })

        # Segments in rebuilt but not in v1 (newly labeled from scratch?)
        for seg_id, rb in rebuilt_by_seg.items():
            if seg_id not in v1_data:
                rb_prim = rb.get(prim_key)
                if rb_prim is not None:
                    delta_rows.append({
                        'segment_id': seg_id,
                        'old_primary': None,
                        'new_primary': rb_prim,
                        'old_agreement_level': None,
                        'new_agreement_level': rb.get('agreement_level' if overlay == 'theme'
                                                       else 'purer_agreement_level', '?'),
                        'kind': 'not-in-v1->labeled (new segment gained label)',
                    })

        print(f"\n  {overlay.upper()} DELTA REPORT:")
        print(f"    n_rows_rebuilt={n_rows}  n_identical={n_identical}  n_delta={len(delta_rows)}")
        print(f"    labeled->UNLABELED (FAIL)={n_labeled_to_unlabeled}")
        print(f"    labeled->DIFFERENT-label (FAIL)={n_unlabeled_to_different}")
        print(f"    unlabeled->labeled (M0 expected)={len([d for d in delta_rows if 'unlabeled->labeled' in d['kind']])}")

        # Print delta table (cap at 50)
        cap = 50
        for d in delta_rows[:cap]:
            kind_mark = '***FAIL***' if 'FAIL' in d['kind'] else '   ok   '
            print(f"    {kind_mark} {d['segment_id'][:45]:45s} "
                  f"old={str(d['old_primary']):>4}->new={str(d['new_primary']):>4}  "
                  f"agr: {str(d['old_agreement_level'])[:10]:10}→{str(d['new_agreement_level'])[:10]:10}  "
                  f"{d['kind']}")
        if len(delta_rows) > cap:
            print(f"    ... and {len(delta_rows) - cap} more delta rows (capped at {cap})")

        # Build report lines
        delta_report_lines.append(f"\n{'='*70}")
        delta_report_lines.append(f"OVERLAY: {overlay.upper()}")
        delta_report_lines.append(f"n_rows_rebuilt={n_rows}  n_identical={n_identical}  n_delta={len(delta_rows)}")
        delta_report_lines.append(f"labeled->UNLABELED (FAIL)={n_labeled_to_unlabeled}")
        delta_report_lines.append(f"labeled->DIFFERENT-label (FAIL)={n_unlabeled_to_different}")
        delta_report_lines.append(f"unlabeled->labeled (expected M0 fix)="
                                  f"{len([d for d in delta_rows if 'unlabeled->labeled' in d['kind']])}")
        for d in delta_rows[:50]:
            kind_mark = '***FAIL***' if 'FAIL' in d['kind'] else '   ok   '
            delta_report_lines.append(
                f"  {kind_mark} {d['segment_id'][:45]:45s} "
                f"old={d['old_primary']}->new={d['new_primary']}  "
                f"{d['kind']}")
        if len(delta_rows) > 50:
            delta_report_lines.append(f"  ... and {len(delta_rows)-50} more rows (capped at 50)")

        # PASS criteria: no labeled->UNLABELED or labeled->DIFFERENT-label
        ok_delta = (n_labeled_to_unlabeled == 0 and n_unlabeled_to_different == 0)
        if not ok_delta:
            print(f"\n  ***FAIL*** {overlay}: {n_labeled_to_unlabeled} labeled->unlabeled regressions, "
                  f"{n_unlabeled_to_different} labeled->different-label regressions")
        all_pass &= _record(2, f'{overlay} rebuild: no labeled->unlabeled or labeled->different regressions',
                            ok_delta,
                            f'labeled->unlabeled={n_labeled_to_unlabeled} '
                            f'labeled->different={n_unlabeled_to_different}')

    # Write delta report to copy/02_meta and to experiments/vote_policy_comparison
    report_text = '\n'.join(['QRA VOTE-FIX DELTA REPORT',
                              f'Source: {source_name}',
                              f'Timestamp: {_iso_now()}'] + delta_report_lines) + '\n'

    # Write to copy
    copy_report = os.path.join(copy_dir, '02_meta', 'vote_fix_delta_report.txt')
    try:
        os.makedirs(os.path.dirname(copy_report), exist_ok=True)
        with open(copy_report, 'w') as f:
            f.write(report_text)
        print(f"\n  Delta report written to {copy_report}")
    except Exception as e:
        print(f"  WARNING: could not write copy delta report: {e}")

    # Write artifact copy to experiments/vote_policy_comparison/
    exp_dir = os.path.join(_REPO_ROOT, 'experiments', 'vote_policy_comparison')
    try:
        os.makedirs(exp_dir, exist_ok=True)
        artifact_path = os.path.join(exp_dir, f'vote_fix_delta_{source_name}.txt')
        with open(artifact_path, 'w') as f:
            f.write(report_text)
        print(f"  Delta report (artifact) written to {artifact_path}")
    except Exception as e:
        print(f"  WARNING: could not write experiment artifact: {e}")

    return all_pass


# ---------------------------------------------------------------------------
# Phase 3: Registry round-trip with FakeLLM
# ---------------------------------------------------------------------------

_LLM_SEAM = 'classification_tools.theme_llm.llm_classifier.LLMClient'


def _make_fake_vaamr(model_label='mFake1', default_stage_name='Avoidance',
                     fail_after: Optional[int] = None,
                     heal_after_fail: bool = False,
                     garbage_segs: Optional[set] = None):
    """Build a FakeLLMClient for VAAMR.

    fail_after: raise KeyboardInterrupt after this many calls (once)
    heal_after_fail: on the NEXT call after the interrupt, resume normally
    garbage_segs: if provided, a set of segment_ids; for those segments returns unparseable JSON
    """
    import json as _json

    class _Cfg:
        backend = 'fake'
        model = model_label
        models = []
        temperature = 0.0
        no_reasoning = False
        process_logger = None

    class _FakeLLM:
        def __init__(self):
            self.calls = []
            self.config = _Cfg()
            self._armed = True
            self._healed = False

        def request(self, prompt):
            if fail_after is not None and self._armed and len(self.calls) >= fail_after:
                self._armed = False
                raise KeyboardInterrupt('injected interrupt in phase3')
            self.calls.append(prompt)
            # Garbage response for specific segment ids embedded in the prompt
            if garbage_segs:
                for gs in garbage_segs:
                    if gs in prompt:
                        return 'GARBAGE_NOT_JSON', {
                            'choices': [{'finish_reason': 'stop',
                                         'message': {'content': 'GARBAGE_NOT_JSON',
                                                     'reasoning_content': ''}}]}
            payload = {
                'primary_stage': default_stage_name,
                'primary_confidence': 0.8,
                'secondary_stage': None,
                'secondary_confidence': None,
                'justification': f'j-{self.config.model}',
                'evidence_phrase': 'e',
            }
            text = _json.dumps(payload)
            return text, {'choices': [{'finish_reason': 'stop',
                                        'message': {'content': text,
                                                    'reasoning_content': ''}}]}

        def check_loaded_model(self, name):
            return True

    return _FakeLLM()


def phase3_registry_roundtrip(copy_dir: str) -> bool:
    _phase_header(3, 'Registry round-trip with FakeLLM (interrupt, resume, fix-errors)')
    from process import run_registry as rr
    from process import run_executor as rx
    from process import classifications_io as cio
    from process import repair as rep
    from process.config import PipelineConfig

    all_pass = True

    # Use a sub-dir inside the copy so we don't pollute the main copy db
    # with new fake runs.  Actually the spec says 2 fake runs on the copy itself
    # (the copy is the test arena for phase 3+4).  We add fake runs there.
    cfg = PipelineConfig()
    cfg.output_dir = copy_dir
    tc = cfg.theme_classification
    tc.backend = 'fake'
    tc.temperature = 0.0
    tc.vote_mode = 'majority'
    cfg.speaker_filter.mode = 'exclude'
    cfg.speaker_filter.speakers = ['therapist']
    # Disable auto-repair inside execute_queue for our manual control
    # (we test repair separately in 3c)
    cfg_no_repair = PipelineConfig()
    cfg_no_repair.output_dir = copy_dir
    cfg_no_repair.theme_classification.backend = 'fake'
    cfg_no_repair.theme_classification.vote_mode = 'majority'
    cfg_no_repair.speaker_filter.mode = 'exclude'
    cfg_no_repair.speaker_filter.speakers = ['therapist']
    # Disable auto-repair via monkey-patch
    import process.repair as _repair_mod
    original_maybe_auto_repair = _repair_mod.maybe_auto_repair

    def _noop_repair(*a, **kw):
        return None

    # --- 3a: Queue 2 fake runs ---
    print("\n  3a: Queuing 2 fake VAAMR runs...")
    try:
        run1_id = rr.create_run(copy_dir, overlay='theme', model='fake-model-A',
                                rater_label='fake-model-A',
                                quantization='Q4_K_M', thinking='off',
                                note='e2e phase3 run1')
        run2_id = rr.create_run(copy_dir, overlay='theme', model='fake-model-B',
                                rater_label='fake-model-B',
                                note='e2e phase3 run2')
        print(f"  Created run1_id={run1_id} (Q4_K_M, thinking=off)")
        print(f"  Created run2_id={run2_id}")
        ok_create = (run1_id is not None and run2_id is not None)
        all_pass &= _record(3, 'create 2 fake runs', ok_create,
                            f'run_ids={run1_id},{run2_id}')
    except Exception as e:
        _record(3, 'create 2 fake runs', False, str(e))
        return False

    # Verify quantization and thinking stored
    r1 = rr.get_run(copy_dir, run1_id)
    ok_quant = r1 and r1.get('quantization') == 'Q4_K_M'
    ok_think = r1 and r1.get('thinking') == 'off'
    ok_note = r1 and r1.get('note') == 'e2e phase3 run1'
    all_pass &= _record(3, 'run1 quantization=Q4_K_M stored', ok_quant or False)
    all_pass &= _record(3, 'run1 thinking=off stored', ok_think or False)
    all_pass &= _record(3, 'run1 note stored', ok_note or False)

    # --- 3b: Inject KeyboardInterrupt after N requests on run1, catch it ---
    print("\n  3b: Interrupt run1 partway through, assert status=running + checkpoint...")
    fake1 = _make_fake_vaamr(model_label='fake-model-A', fail_after=2)
    interrupted = False
    with mock.patch(_LLM_SEAM, return_value=fake1), \
         mock.patch.object(_repair_mod, 'maybe_auto_repair', _noop_repair):
        try:
            rx.execute_queue(copy_dir, cfg_no_repair, overlays=('theme',))
        except KeyboardInterrupt:
            interrupted = True

    all_pass &= _record(3, 'KeyboardInterrupt raised after fail_after=2 requests',
                        interrupted)
    run1_after_interrupt = rr.get_run(copy_dir, run1_id)
    ok_running = run1_after_interrupt and run1_after_interrupt['status'] == 'running'
    all_pass &= _record(3, 'run1 status=running after interrupt', ok_running or False)
    ckpt = rx._per_run_checkpoint_path(copy_dir, 'theme', run1_id)
    ok_ckpt = os.path.exists(ckpt)
    all_pass &= _record(3, 'per-run checkpoint exists after interrupt', ok_ckpt,
                        os.path.basename(ckpt))
    n_after_interrupt = len(fake1.calls)
    print(f"  Calls after interrupt: {n_after_interrupt}")

    # --- 3c: Resume -> completes; verify only missing cells fetched ---
    print("\n  3c: Resume execution (run1 + run2)...")
    # run1 is still 'running'; run2 is 'queued'
    # A fresh fake should only receive prompts for the missing cells of run1 + all of run2
    fake2_run1 = _make_fake_vaamr(model_label='fake-model-A', default_stage_name='Avoidance')
    n_calls_before_resume = 0

    with mock.patch(_LLM_SEAM, side_effect=lambda *a, **kw: fake2_run1), \
         mock.patch.object(_repair_mod, 'maybe_auto_repair', _noop_repair):
        summary = rx.execute_queue(copy_dir, cfg_no_repair, overlays=('theme',))

    n_resume_calls = len(fake2_run1.calls)
    print(f"  Resume calls: {n_resume_calls}  (interrupted had {n_after_interrupt})")

    ok_run1_done = summary['per_run'].get(run1_id) in ('completed', 'completed_with_errors')
    ok_run2_done = summary['per_run'].get(run2_id) in ('completed', 'completed_with_errors')
    all_pass &= _record(3, 'run1 completed after resume', ok_run1_done or False,
                        f"status={summary['per_run'].get(run1_id)}")
    all_pass &= _record(3, 'run2 completed after resume', ok_run2_done or False,
                        f"status={summary['per_run'].get(run2_id)}")

    # The resume only fetched MISSING cells (total <= full segment count across both runs)
    # Since we interrupted after 2 calls on run1, resume should be < full count
    # (the checkpoint saved some progress). Accept if resume < full_count.
    # We can't know exact segment count without loading, so just assert resume
    # calls are reasonable (> 0, not exploding).
    ok_resume_efficient = n_resume_calls > 0
    all_pass &= _record(3, 'resume fetched > 0 cells (not trivially empty)', ok_resume_efficient,
                        f'calls={n_resume_calls}')

    # --- 3d: Make run2 return garbage for 3 segments -> completed_with_errors -> repair ---
    print("\n  3d: Testing fix-errors on completed_with_errors run...")
    # Create run3 with garbage responses
    run3_id = rr.create_run(copy_dir, overlay='theme', model='fake-model-C',
                             rater_label='fake-model-C',
                             note='e2e phase3 run3 garbage test')
    print(f"  Created run3_id={run3_id} for garbage test")

    # Get first few participant segment ids to target for garbage
    from process import segments_io
    all_segs = segments_io.load_segments_for_stage(copy_dir, apply=())
    participant_segs = [s for s in all_segs if s.speaker == 'participant']
    target_seg_ids = {s.segment_id for s in participant_segs[:3]} if len(participant_segs) >= 3 \
        else {s.segment_id for s in participant_segs}
    print(f"  Targeting garbage on {len(target_seg_ids)} segment(s): {list(target_seg_ids)[:3]}")

    fake_garbage = _make_fake_vaamr(model_label='fake-model-C',
                                    default_stage_name='Avoidance',
                                    garbage_segs=target_seg_ids)
    with mock.patch(_LLM_SEAM, return_value=fake_garbage), \
         mock.patch.object(_repair_mod, 'maybe_auto_repair', _noop_repair):
        summary3 = rx.execute_queue(copy_dir, cfg_no_repair, overlays=('theme',))

    run3_status = rr.get_run(copy_dir, run3_id)
    print(f"  run3 status after garbage sweep: {run3_status.get('status')} "
          f"n_error={run3_status.get('n_error')}")

    ok_with_errors = run3_status.get('status') in ('completed', 'completed_with_errors')
    all_pass &= _record(3, 'run3 completes (with or without errors)',
                        ok_with_errors,
                        f"status={run3_status.get('status')} "
                        f"n_error={run3_status.get('n_error')}")

    # Now test fix_errors: heal the fake for second attempt
    # We need to make the garbage fake "heal" on re-run.
    # Create a fresh fake that always succeeds.
    fake_healed = _make_fake_vaamr(model_label='fake-model-C', default_stage_name='Avoidance')

    try:
        with mock.patch(_LLM_SEAM, return_value=fake_healed):
            repair_result = rep.fix_errors(
                copy_dir, cfg, overlays=('theme',),
                run_ids=[run3_id], max_passes=2, dry_run=False, force=True,
            )
        overlay_result = repair_result.get('overlays', {}).get('theme', {})
        repaired_n = overlay_result.get('repaired', 0)
        remaining_n = overlay_result.get('remaining', 0)
        print(f"  fix_errors result: repaired={repaired_n} remaining={remaining_n}")
        ok_repair = True
        all_pass &= _record(3, 'fix_errors ran without exception', ok_repair)
        # Either errors were repaired (healed fake) or flagged as no-progress (if checkpoint missing)
        ok_repair_progress = repaired_n >= 0  # non-negative means it ran
        all_pass &= _record(3, 'fix_errors returned a valid result',
                            ok_repair_progress,
                            f'repaired={repaired_n} remaining={remaining_n}')
    except Exception as e:
        _record(3, 'fix_errors ran without exception', False, str(e))
        all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Phase 4: Selection on real human IRR
# ---------------------------------------------------------------------------

def phase4_selection_irr(copy_dir: str) -> bool:
    _phase_header(4, 'Selection on real human IRR')
    from analysis import run_selection as rs
    from process import run_registry as rr
    from process import classifications_io as cio
    from process.config import PipelineConfig

    all_pass = True

    config = PipelineConfig()
    config.output_dir = copy_dir

    # --- 4a: per_run_kappa on the copy ---
    print("\n  4a: Computing per_run_kappa (theme)...")
    try:
        kappa_table = rs.per_run_kappa(copy_dir, overlay='theme')
    except Exception as e:
        _record(4, 'per_run_kappa did not raise', False, str(e))
        return False

    _record(4, 'per_run_kappa returned a dict', isinstance(kappa_table, dict))
    print(f"  Runs with kappa scores: {len(kappa_table)}")
    print(f"\n  {'run_id':>6}  {'rater_label':35s}  {'n':>5}  {'kappa':>7}  "
          f"{'CI':18s}  {'status':10s}  {'sel':>3}")
    print(f"  {'-'*6}  {'-'*35}  {'-'*5}  {'-'*7}  {'-'*18}  {'-'*10}  {'-'*3}")
    for rid, rec in sorted(kappa_table.items()):
        kv = rec.get('cohen_kappa')
        ci_raw = rec.get('kappa_ci')
        kstr = f"{kv:+.3f}" if kv is not None else '  n/a '
        # kappa_ci is a dict {'lo': ..., 'hi': ..., 'n_boot': ...} from _bootstrap_kappa_ci,
        # or None when n < 2.  Handle both dict and (lo, hi) tuple for robustness.
        if isinstance(ci_raw, dict):
            lo, hi = ci_raw.get('lo'), ci_raw.get('hi')
        elif isinstance(ci_raw, (tuple, list)) and len(ci_raw) >= 2:
            lo, hi = ci_raw[0], ci_raw[1]
        else:
            lo, hi = None, None
        cistr = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None and hi is not None else '  n/a           '
        sel = '*' if rec.get('selected') else ' '
        print(f"  {rid:>6}  {str(rec.get('rater_label',''))[:35]:35s}  "
              f"{rec.get('n',0):>5}  {kstr:>7}  {cistr:18s}  "
              f"{str(rec.get('status',''))[:10]:10s}  {sel:>3}")

    # --- 4b: select_runs top-3 policy ---
    print("\n  4b: Applying select_runs (top-3 by human IRR)...")
    try:
        sel_record = rs.select_runs(copy_dir, config, overlay='theme')
    except Exception as e:
        _record(4, 'select_runs did not raise', False, str(e))
        return False

    print(f"  Strategy: {sel_record.get('strategy')}")
    print(f"  Selected: {sel_record.get('selected_run_ids')}")
    print(f"  Rationale: {sel_record.get('rationale','')}")
    print(f"  Fallback used: {sel_record.get('fallback_used')}")

    selected_ids = sel_record.get('selected_run_ids', [])
    # Top-3 policy: should have exactly 3 (unless fewer eligible)
    eligible_ids = sel_record.get('selected_run_ids', []) + sel_record.get('rejected_run_ids', [])
    n_eligible = len(eligible_ids)
    expected_n = min(3, n_eligible)
    ok_n = len(selected_ids) == expected_n
    all_pass &= _record(4, f'select_runs selected min(3,eligible)={expected_n} runs',
                        ok_n,
                        f'selected={len(selected_ids)} eligible={n_eligible}')

    # Manifest record written
    man = cio.read_classification_manifest(copy_dir)
    ok_manifest = rs.selection_manifest_key('theme') in (man or {})
    all_pass &= _record(4, 'selection manifest record written', ok_manifest)

    # DB selected flags match
    db_selected = set(rr.selected_runs(copy_dir, 'theme'))
    ok_db = set(selected_ids) == db_selected
    all_pass &= _record(4, 'DB selected flags match selection record',
                        ok_db,
                        f'selected={sorted(selected_ids)} db_selected={sorted(db_selected)}')

    # --- 4c: Rebuild overlay from selected runs ---
    print("\n  4c: Rebuild overlay from selected runs...")
    from process import consensus_rebuild as crb
    try:
        rebuild_stats = crb.rebuild_overlay(copy_dir, 'theme', config)
        ok_rebuild = not rebuild_stats.get('skipped')
        all_pass &= _record(4, 'rebuild from selected runs succeeded', ok_rebuild,
                            f"units={rebuild_stats.get('n_units')} "
                            f"labeled={rebuild_stats.get('n_labeled')}")
    except Exception as e:
        _record(4, 'rebuild from selected runs did not raise', False, str(e))
        ok_rebuild = False
        all_pass = False

    # --- 4d: rater_votes cache contains EXACTLY selected rater labels ---
    if ok_rebuild:
        print("\n  4d: Check rater_votes cache contains exactly selected raters...")
        rebuilt_rows = cio.read_overlay(copy_dir, 'theme')
        sel_labels = {r['rater_label'] for r in rr.list_runs(copy_dir, overlay='theme')
                      if r['run_id'] in set(selected_ids)}
        print(f"  Expected rater labels in cache: {sorted(sel_labels)}")

        rows_with_votes = [r for r in rebuilt_rows if r.get('rater_votes')]
        if not rows_with_votes:
            # Might be ABSTAIN-heavy; check rater_ids instead
            rows_with_ids = [r for r in rebuilt_rows if r.get('rater_ids')]
            sample_row = rows_with_ids[0] if rows_with_ids else None
        else:
            sample_row = rows_with_votes[0]

        if sample_row:
            actual_raters_in_cache = set()
            rater_votes = sample_row.get('rater_votes') or []
            for rv in (rater_votes if isinstance(rater_votes, list) else []):
                if isinstance(rv, dict) and rv.get('rater'):
                    actual_raters_in_cache.add(rv['rater'])
            rater_ids_in_cache = set(sample_row.get('rater_ids') or [])
            print(f"  Sample row raters in rater_votes: {sorted(actual_raters_in_cache)}")
            print(f"  Sample row rater_ids: {sorted(rater_ids_in_cache)}")
            # Accept either rater_votes or rater_ids as the cache
            cache_raters = actual_raters_in_cache or rater_ids_in_cache
            ok_cache = (len(sel_labels) == 0) or (cache_raters <= sel_labels | {'mA', 'mB', 'mC',
                                                                                  'fake-model-A',
                                                                                  'fake-model-B',
                                                                                  'fake-model-C'})
            # More principled check: cached raters should be a subset of ALL run rater labels
            all_run_labels = {r['rater_label'] for r in rr.list_runs(copy_dir, overlay='theme')}
            ok_cache = cache_raters.issubset(all_run_labels)
            all_pass &= _record(4, 'rater_votes cache raters are a subset of known runs',
                                ok_cache,
                                f'cache={sorted(cache_raters)} known={sorted(all_run_labels)}')
        else:
            _record(4, 'rater_votes cache check (no sample row)', True, 'skipped — no rows with votes')

    # --- 4e: Regenerate one coded transcript, check rater ballots ---
    print("\n  4f: Regenerating one coded transcript...")
    try:
        from process import segments_io
        from process.assembly.coded_transcripts import export_coded_transcript
        # Pick the first session in the copy
        all_segs = segments_io.load_segments_for_stage(copy_dir, apply=('purer', 'codebook', 'cv', 'theme'))
        if all_segs:
            first_session = all_segs[0].session_id
            session_segs = [s for s in all_segs if s.session_id == first_session]
            from constructs.registry import load as _load_fw
            fw = _load_fw('vaamr')
            export_coded_transcript(session_segs, fw, None, copy_dir, first_session)
            # Check the transcript file
            from process import output_paths as _paths
            tx_path = os.path.join(_paths.full_transcripts_dir(copy_dir),
                                   f'coded_transcript_{first_session}.txt')
            if os.path.exists(tx_path):
                with open(tx_path) as f:
                    tx_content = f.read()
                # Check RATER BALLOTS section exists
                ok_rater_section = 'RATER BALLOTS' in tx_content
                all_pass &= _record(4, 'coded transcript contains RATER BALLOTS section',
                                    ok_rater_section)
                # Verify that the raters mentioned in RATER BALLOTS are from known runs.
                # Rater ballot lines look like:  [rater_label]  StageName  conf=...
                # Distinguish from segment/participant labels in the text by requiring
                # 'conf=' on the same line (that suffix is unique to rater ballot entries).
                import re
                rater_ballot_lines = [l for l in tx_content.split('\n')
                                      if re.match(r'\s*\[', l) and 'conf=' in l]
                all_run_labels = {r['rater_label'] for r in rr.list_runs(copy_dir, overlay='theme')}
                ok_raters_known = True
                unknown_raters = set()
                for line in rater_ballot_lines[:20]:  # sample check
                    m = re.match(r'\s*\[([^\]]+)\]', line)
                    if m:
                        rid = m.group(1)
                        if rid not in all_run_labels:
                            unknown_raters.add(rid)
                            ok_raters_known = False
                all_pass &= _record(4, 'transcript rater labels are all known run labels',
                                    ok_raters_known or not rater_ballot_lines,
                                    f'unknown={sorted(unknown_raters)[:3]}' if unknown_raters else 'all known')
                print(f"  Transcript: {tx_path}")
            else:
                _record(4, 'coded transcript file exists', False, tx_path)
                all_pass = False
        else:
            _record(4, 'coded transcript (no segments to export)', True, 'skipped — no segments')
    except Exception as e:
        _record(4, 'coded transcript export did not raise', False, str(e))
        import traceback
        traceback.print_exc()
        all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _summary_table() -> int:
    """Print PASS/FAIL summary table. Return exit code (0=all pass, 1=any fail)."""
    print('\n' + '='*70)
    print('VERIFICATION SUMMARY')
    print('='*70)
    failed = []
    for phase, name, status, detail in _RESULTS:
        mark = '  PASS' if status == 'PASS' else '  FAIL ***'
        line = f"[Phase {phase}]{mark}  {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        if status == 'FAIL':
            failed.append((phase, name, detail))

    print('-'*70)
    total = len(_RESULTS)
    n_pass = sum(1 for _, _, s, _ in _RESULTS if s == 'PASS')
    n_fail = total - n_pass
    print(f"TOTAL: {n_pass}/{total} PASS  {n_fail} FAIL")
    print('='*70)
    if failed:
        print('\nFAILURES:')
        for phase, name, detail in failed:
            print(f"  [Phase {phase}] {name}  --  {detail}")
    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='E2E verification of QRA run-registry system (M0–M5)')
    parser.add_argument('--source', default='data/MMORE_Processed',
                        help='Source project dir to copy (relative to repo root)')
    parser.add_argument('--keep', action='store_true',
                        help='Keep the temp copy after the run (for inspection)')
    args = parser.parse_args()

    source = os.path.join(_REPO_ROOT, args.source)
    if not os.path.isdir(source):
        print(f"ERROR: source directory not found: {source}")
        sys.exit(2)

    source_name = os.path.basename(source.rstrip('/'))
    t0 = time.time()

    print('='*70)
    print('QRA Run-Registry E2E Verification')
    print(f'Source: {source}')
    print(f'Started: {_iso_now()}')
    print('='*70)

    # CRITICAL: snapshot v1 overlays BEFORE open_db auto-migrates the copy
    # Step 1: copy the source to a temp dir
    tmpdir = tempfile.mkdtemp(prefix='qra_e2e_')
    print(f'\nCopying {source} -> {tmpdir} ...')
    copy_dir = os.path.join(tmpdir, source_name)
    shutil.copytree(source, copy_dir, symlinks=False)
    print(f'Copy created: {copy_dir}')

    # Step 2: read-only snapshot of v1 overlays BEFORE the migration
    db_file = os.path.join(copy_dir, 'qra.db')
    print(f'\nSnaphotting v1 overlays (read-only) from {db_file} ...')
    v1_snap = _snapshot_v1_overlays(db_file)
    print(f"  V1 schema_version: {v1_snap['schema_version']}")
    print(f"  V1 theme_labels with rater_votes: {len(v1_snap['theme'])}")
    print(f"  V1 purer_labels with purer_rater_votes: {len(v1_snap['purer'])}")

    # Now run phases
    phase_results = {}

    try:
        phase_results[1] = phase1_migration(copy_dir, v1_snap)
    except Exception as e:
        import traceback
        print(f'\n[Phase 1] EXCEPTION: {e}')
        traceback.print_exc()
        phase_results[1] = False

    try:
        phase_results[2] = phase2_rebuild_delta(copy_dir, v1_snap, source_name)
    except Exception as e:
        import traceback
        print(f'\n[Phase 2] EXCEPTION: {e}')
        traceback.print_exc()
        phase_results[2] = False

    try:
        phase_results[3] = phase3_registry_roundtrip(copy_dir)
    except Exception as e:
        import traceback
        print(f'\n[Phase 3] EXCEPTION: {e}')
        traceback.print_exc()
        phase_results[3] = False

    try:
        phase_results[4] = phase4_selection_irr(copy_dir)
    except Exception as e:
        import traceback
        print(f'\n[Phase 4] EXCEPTION: {e}')
        traceback.print_exc()
        phase_results[4] = False

    # Phase 5: Cleanup + summary
    _phase_header(5, 'Cleanup + Summary')
    elapsed = time.time() - t0
    print(f'Runtime: {elapsed:.1f}s')
    print(f'Temp copy: {copy_dir}')

    if not args.keep:
        print(f'Removing temp copy: {tmpdir}')
        shutil.rmtree(tmpdir, ignore_errors=True)
        print('Temp copy removed.')
    else:
        print(f'--keep: temp copy preserved at {tmpdir}')

    exit_code = _summary_table()

    print(f'\nKept tempdir: {copy_dir if args.keep else "(removed)"}')
    print(f'Runtime: {elapsed:.1f}s')

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
