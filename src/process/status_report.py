"""
process/status_report.py
-------------------------
``qra status`` — a per-stage health dashboard for a QRA project.

Surfaces, at a glance, what has run and whether it succeeded.  Its reason for
existing is the PURER total-rater-failure incident (2026-06-08): a classification
run that *looked* complete (a fresh manifest timestamp, an overlay table written)
but left every label NULL because one rater emitted 100% parse errors and the
other abstained on 80% of cue blocks.  The merge correctly produced NULL — but
nothing surfaced the per-rater collapse.

The critical element here that ``interactive_tui._detect_state`` does not provide
is ``checkpoint_health``: it reads the latest ``*_runs.json`` checkpoint for VAAMR
and PURER and counts CODED / ABSTAIN / ERROR ballots *per rater*, so a dead rater
is impossible to miss.

Public API:
  * ``gather_status(output_dir) -> dict``  — assemble the full status structure
  * ``format_status_text(status) -> str``  — terminal dashboard
  * ``format_status_json(status) -> str``  — machine-readable dump
"""
import glob
import json
import os
import sqlite3
from typing import List, Optional

from . import output_paths as _paths


# ---------------------------------------------------------------------------
# Checkpoint health — per-rater CODED / ABSTAIN / ERROR from *_runs.json
# ---------------------------------------------------------------------------

def _read_checkpoint_health(output_dir: str, prefix: str) -> List[dict]:
    """Per-rater ballot quality from the latest ``{prefix}_*_runs.json`` checkpoint.

    Returns ``[{model, coded, abstain, error, total}, ...]`` — one entry per rater
    slot, in slot order.  Returns ``[]`` when no model-first checkpoint exists.

    The model-first checkpoint stores ``run_results[seg_id][str(run_idx)]``:
      * ``None``                     → parse error (the rater produced nothing usable)
      * ``{'vote': 'ABSTAIN', ...}`` → a valid abstention ballot
      * ``{'vote': 'CODED', ...}``   → a concrete coded ballot
    A missing run-key (rater never attempted that segment) is not counted.
    """
    ckpt_dir = _paths.llm_checkpoints_dir(output_dir)
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, f'{prefix}_*_runs.json')))
    if not candidates:
        return []
    path = candidates[-1]  # latest by filename timestamp
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []

    meta = data.get('_meta', {})
    if meta.get('format') != 'model_first_v1':
        return []
    per_run_models = meta.get('per_run_models', [])
    n_runs = meta.get('n_runs', len(per_run_models))
    run_results = data.get('run_results', {})

    health: List[dict] = []
    for run_idx in range(n_runs):
        key = str(run_idx)
        coded = abstain = error = 0
        for seg_data in run_results.values():
            if key not in seg_data:
                continue
            val = seg_data[key]
            if val is None:
                error += 1
            elif isinstance(val, dict) and val.get('vote') == 'ABSTAIN':
                abstain += 1
            else:
                coded += 1
        model = per_run_models[run_idx] if run_idx < len(per_run_models) else f'run {run_idx + 1}'
        health.append({
            'model': model,
            'coded': coded,
            'abstain': abstain,
            'error': error,
            'total': coded + abstain + error,
        })
    return health


# ---------------------------------------------------------------------------
# Small SQLite helpers (read-only; status must never mutate the project)
# ---------------------------------------------------------------------------

def _scalar(conn: sqlite3.Connection, sql: str, default=0):
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.Error:
        return default
    if row is None or row[0] is None:
        return default
    return row[0]


def _agreement_breakdown(conn: sqlite3.Connection, table: str, col: str) -> dict:
    out = {'unanimous': 0, 'majority': 0, 'plurality_coded': 0, 'split': 0, 'none': 0}
    try:
        rows = conn.execute(
            f"SELECT {col}, COUNT(*) FROM {table} GROUP BY {col}"
        ).fetchall()
    except sqlite3.Error:
        return out
    for level, n in rows:
        if level in out:
            out[level] = n
    return out


# ---------------------------------------------------------------------------
# gather_status — the assembled structure
# ---------------------------------------------------------------------------

def gather_status(output_dir: str) -> dict:
    """Assemble a per-stage status dict for ``output_dir`` (see module docstring)."""
    db_file = _paths.db_path(output_dir)

    # Config discovery (mirror interactive_tui._detect_state)
    config_path = None
    for c in (os.path.join(_paths.meta_dir(output_dir), 'qra_config.json'),
              os.path.join(output_dir, 'qra_config.json')):
        if os.path.isfile(c):
            config_path = c
            break

    manifest = {}
    project = {
        'db_path': db_file,
        'config_path': config_path,
        'sessions': 0,
        'total_segments': 0,
        'participant_segments': 0,
        'therapist_segments': 0,
        'segmenter_version': None,
        'last_ingest': None,
    }
    vaamr = {'labeled': 0, 'null_count': 0, 'needs_review': 0, 'model': None,
             'n_runs': None, 'completed_at': None,
             'agreement_breakdown': {}, 'checkpoint_health': []}
    purer = {'labeled': 0, 'null_count': 0, 'total_rows': 0, 'model': None,
             'n_runs': None, 'completed_at': None,
             'agreement_breakdown': {}, 'checkpoint_health': []}
    codebook = {'labeled': 0, 'model': None, 'completed_at': None}
    irr = {'testset_count': 0, 'human_codes_count': 0}
    cv = {'testset_count': 0}

    if os.path.isfile(db_file):
        from . import classifications_io as _cio
        manifest = _cio.read_classification_manifest(output_dir) or {}
        conn = sqlite3.connect(db_file)
        try:
            project['total_segments'] = _scalar(conn, "SELECT COUNT(*) FROM segments")
            project['sessions'] = _scalar(conn, "SELECT COUNT(DISTINCT session_id) FROM segments")
            project['participant_segments'] = _scalar(
                conn, "SELECT COUNT(*) FROM segments WHERE speaker = 'participant'")
            project['therapist_segments'] = _scalar(
                conn, "SELECT COUNT(*) FROM segments WHERE speaker = 'therapist'")
            project['segmenter_version'] = _scalar(
                conn, "SELECT segmenter_version FROM segments LIMIT 1", default=None)
            project['last_ingest'] = _scalar(
                conn, "SELECT MAX(ingest_timestamp) FROM segments", default=None)

            # VAAMR
            vaamr['labeled'] = _scalar(
                conn, "SELECT COUNT(*) FROM theme_labels WHERE primary_stage IS NOT NULL")
            vaamr['null_count'] = max(0, project['participant_segments'] - vaamr['labeled'])
            vaamr['needs_review'] = _scalar(
                conn, "SELECT COUNT(*) FROM theme_labels WHERE needs_review = 1")
            vaamr['agreement_breakdown'] = _agreement_breakdown(
                conn, 'theme_labels', 'agreement_level')

            # PURER
            purer['total_rows'] = _scalar(conn, "SELECT COUNT(*) FROM purer_labels")
            purer['labeled'] = _scalar(
                conn, "SELECT COUNT(*) FROM purer_labels WHERE purer_primary IS NOT NULL")
            purer['null_count'] = max(0, purer['total_rows'] - purer['labeled'])
            purer['agreement_breakdown'] = _agreement_breakdown(
                conn, 'purer_labels', 'purer_agreement_level')

            # Codebook
            codebook['labeled'] = _scalar(
                conn,
                "SELECT COUNT(*) FROM codebook_labels "
                "WHERE codebook_labels_ensemble IS NOT NULL "
                "AND codebook_labels_ensemble NOT IN ('[]', 'null', '')")

            # IRR / CV testsets
            irr['testset_count'] = _scalar(conn, "SELECT COUNT(*) FROM irr_testsets")
            irr['human_codes_count'] = _scalar(conn, "SELECT COUNT(*) FROM irr_human_codes")
            cv['testset_count'] = _scalar(conn, "SELECT COUNT(*) FROM cv_testsets")
        finally:
            conn.close()

    # Manifest-sourced model/run/timestamp metadata
    for key, dest in (('theme', vaamr), ('purer', purer), ('codebook', codebook)):
        entry = manifest.get(key) or {}
        if 'model' in entry:
            dest['model'] = entry['model']
        if 'n_runs' in dest and entry.get('n_runs') is not None:
            dest['n_runs'] = entry['n_runs']
        if entry.get('completed_at'):
            dest['completed_at'] = entry['completed_at']

    # Per-rater checkpoint health (the headline diagnostic). Prefixes single-homed
    # in reclassify_ops.
    from .reclassify_ops import checkpoint_prefix as _ckpt_prefix
    vaamr['checkpoint_health'] = _read_checkpoint_health(output_dir, _ckpt_prefix('theme'))
    purer['checkpoint_health'] = _read_checkpoint_health(output_dir, _ckpt_prefix('purer'))

    # Schema-v2 run registry (per-overlay run table + ballot-derived counters).
    # Guarded: pre-migration DBs / missing tables -> empty lists, never crash.
    runs = _gather_runs(output_dir)

    # Probe / GNN reliability gates (reuse the TUI status probes)
    from .interactive_tui import _gnn_status, _probe_status
    gnn = {'status': _gnn_status(output_dir)}
    probe = {'status': _probe_status(output_dir),
             'kappa_human': None, 'kappa_llm': None, 'ready_for_scaling': False}
    try:
        from classification_tools.probe.probe_classifier import read_probe_gate
        verdict = read_probe_gate(output_dir)
        if verdict is not None:
            probe['kappa_human'] = verdict.get('probe_human_kappa')
            probe['kappa_llm'] = verdict.get('probe_llm_kappa')
            probe['ready_for_scaling'] = bool(verdict.get('ready_for_scaling'))
    except Exception:
        pass

    analysis = {
        'executive_summary_present': os.path.isfile(_paths.reports_results_path(output_dir)),
        'longitudinal_present': bool(
            glob.glob(os.path.join(_paths.reports_outcomes_dir(output_dir), 'longitudinal*'))),
    }

    return {
        'project': project,
        'vaamr': vaamr,
        'purer': purer,
        'codebook': codebook,
        'probe': probe,
        'gnn': gnn,
        'irr': irr,
        'cv': cv,
        'runs': runs,
        'analysis': analysis,
    }


def _gather_runs(output_dir: str) -> dict:
    """Per-overlay registry runs (schema v2): a compact list per overlay.

    Returns ``{overlay: [{run_id, rater_label, status, selected, n_total,
    n_coded, n_abstain, n_error}, ...]}``.  Pre-migration DBs or missing tables
    yield empty lists (``run_registry.list_runs`` already guards these) so this
    never crashes a v1 project's status.
    """
    out = {'theme': [], 'purer': [], 'codebook': []}
    try:
        from . import run_registry as _rr
    except Exception:
        return out
    for overlay in out:
        try:
            rows = _rr.list_runs(output_dir, overlay=overlay)
        except Exception:
            rows = []
        out[overlay] = [
            {
                'run_id': r['run_id'],
                'rater_label': r['rater_label'],
                'status': r['status'],
                'selected': bool(r['selected']),
                'n_total': r['n_total'],
                'n_coded': r['n_coded'],
                'n_abstain': r['n_abstain'],
                'n_error': r['n_error'],
            }
            for r in rows
        ]
    return out


# ---------------------------------------------------------------------------
# format_status_text — terminal dashboard
# ---------------------------------------------------------------------------

_W = 64


def _mark(ok: bool, partial: bool = False) -> str:
    if partial:
        return '✗'
    return '✓' if ok else '·'


def _short_date(iso: Optional[str]) -> str:
    return (iso or '')[:10] or '—'


def _checkpoint_lines(health: List[dict], indent: str = '                    ') -> List[str]:
    """Render per-rater CODED/ABSTAIN/ERROR, flagging a dead rater."""
    lines: List[str] = []
    n = len(health)
    for i, h in enumerate(health):
        branch = '└─' if i == n - 1 else '┌─' if i == 0 else '├─'
        total = h['total'] or 1
        flag = ''
        if h['error'] == h['total'] and h['total'] > 0:
            flag = '  ← ALL ERRORS'
        elif h['error'] / total >= 0.5:
            flag = '  ← mostly errors'
        elif h['abstain'] / total >= 0.8:
            flag = '  ← mostly abstains'
        lines.append(
            f"{indent}{branch} {h['model']:<28}  "
            f"coded {h['coded']}  abstain {h['abstain']}  error {h['error']}{flag}"
        )
    return lines


def _runs_lines(runs: dict) -> List[str]:
    """Render the schema-v2 registry runs, one compact table per overlay.

    Reuses the dead-rater flag thresholds from the checkpoint-health view but
    sourced from durable ballot counters: error_frac >= .5 -> 'DEAD?',
    abstain_frac >= .8 -> 'ABSTAINY'.  Returns ``[]`` when no overlay has runs.
    """
    overlays = [(k, runs.get(k) or []) for k in ('theme', 'purer', 'codebook')]
    if not any(rows for _, rows in overlays):
        return []

    label = {'theme': 'VAAMR', 'purer': 'PURER', 'codebook': 'CODEBOOK'}
    lines: List[str] = ['']
    lines.append(f"RUNS (registry)")
    for overlay, rows in overlays:
        if not rows:
            continue
        lines.append(f"  {label.get(overlay, overlay)}")
        n = len(rows)
        for i, r in enumerate(rows):
            branch = '└─' if i == n - 1 else '├─'
            total = r['n_total'] or 0
            flag = ''
            if total > 0:
                if (r['n_error'] or 0) / total >= 0.5:
                    flag = '  ← DEAD?'
                elif (r['n_abstain'] or 0) / total >= 0.8:
                    flag = '  ← ABSTAINY'
            sel = '*' if r['selected'] else ' '
            lines.append(
                f"    {branch} [{r['run_id']:>2}]{sel} {str(r['rater_label']):<26}  "
                f"{str(r['status']):<14}  "
                f"coded {r['n_coded'] or 0}  abstain {r['n_abstain'] or 0}  "
                f"error {r['n_error'] or 0}{flag}"
            )
    return lines


def format_status_text(status: dict) -> str:
    proj = status['project']
    vaamr = status['vaamr']
    purer = status['purer']
    cb = status['codebook']
    probe = status['probe']
    gnn = status['gnn']
    irr = status['irr']
    cv = status['cv']
    runs = status.get('runs') or {}
    analysis = status['analysis']

    L: List[str] = []
    L.append('')
    L.append(f"QRA PROJECT STATUS  {os.path.dirname(proj['db_path']) or '.'}")
    L.append('═' * _W)
    L.append('')

    if not os.path.isfile(proj['db_path']):
        L.append('  ✗  No qra.db found — this is not a QRA project directory.')
        L.append('')
        return '\n'.join(L)

    # INGEST
    L.append(
        f"INGEST         {_mark(proj['total_segments'] > 0)}  "
        f"{proj['sessions']} sessions  ·  {proj['total_segments']} segments  ·  "
        f"{_short_date(proj['last_ingest'])}"
    )

    # VAAMR
    has_v = vaamr['labeled'] > 0 or bool(vaamr['checkpoint_health'])
    failed_v = vaamr['labeled'] == 0 and bool(vaamr['checkpoint_health'])
    if has_v:
        meta = []
        if vaamr['model']:
            meta.append(str(vaamr['model']))
        if vaamr['n_runs']:
            meta.append(f"{vaamr['n_runs']} runs")
        meta.append(_short_date(vaamr['completed_at']))
        L.append(
            f"VAAMR          {_mark(vaamr['labeled'] > 0, partial=failed_v)}  "
            f"{vaamr['labeled']} of {proj['participant_segments']} participant "
            f"segments labeled  ·  {'  ·  '.join(meta)}"
        )
        ab = vaamr['agreement_breakdown']
        if any(ab.values()) or vaamr['needs_review']:
            _plur = (f"plurality {ab['plurality_coded']}  "
                     if ab.get('plurality_coded') else '')
            L.append(
                f"                    unanimous {ab.get('unanimous', 0)}  "
                f"majority {ab.get('majority', 0)}  {_plur}split {ab.get('split', 0)}  "
                f"none {ab.get('none', 0)}  needs_review {vaamr['needs_review']}"
            )
        L.extend(_checkpoint_lines(vaamr['checkpoint_health']))
    else:
        L.append("VAAMR          ·  not run")

    # PURER
    has_p = purer['total_rows'] > 0 or bool(purer['checkpoint_health'])
    failed_p = purer['labeled'] == 0 and has_p
    if has_p:
        denom = purer['total_rows'] or 0
        suffix = ''
        if failed_p and denom:
            suffix = f"  ({denom} DB rows, all NULL)"
        L.append(
            f"PURER          {_mark(purer['labeled'] > 0, partial=failed_p)}  "
            f"{purer['labeled']} of {denom} therapist segments labeled{suffix}"
        )
        ab = purer['agreement_breakdown']
        if not failed_p and any(ab.values()):
            _plur = (f"plurality {ab['plurality_coded']}  "
                     if ab.get('plurality_coded') else '')
            L.append(
                f"                    unanimous {ab.get('unanimous', 0)}  "
                f"majority {ab.get('majority', 0)}  {_plur}split {ab.get('split', 0)}  "
                f"none {ab.get('none', 0)}"
            )
        L.extend(_checkpoint_lines(purer['checkpoint_health']))
    else:
        L.append("PURER          ·  not run")

    # CODEBOOK
    if cb['labeled'] > 0:
        meta = []
        if cb['model']:
            meta.append(str(cb['model']))
        meta.append(_short_date(cb['completed_at']))
        L.append(f"CODEBOOK       ✓  {cb['labeled']} segments coded  ·  {'  ·  '.join(meta)}")
    else:
        L.append("CODEBOOK       ·  not run")

    # PROBE
    if probe['status'] == 'absent':
        L.append("PROBE          ·  not trained")
    else:
        ready = probe['status'] == 'ready'
        kh = probe['kappa_human']
        kl = probe['kappa_llm']
        khs = f"{kh:.2f}" if isinstance(kh, (int, float)) else 'n/a'
        kls = f"{kl:.2f}" if isinstance(kl, (int, float)) else 'n/a'
        verdict = 'gate PASSED — ready for LLM-free scaling' if ready else 'gate: below human band'
        L.append(f"PROBE          {_mark(ready)}  ↔human κ {khs}  ↔LLM κ {kls}  ·  {verdict}")

    # GNN
    gnn_label = {
        'ready': '✓  classifier gate passed — ready for LLM-free scaling',
        'not_ready': '·  classifier trained, gate not yet reliable',
        'trained': '·  classifier trained, gate not run',
        'discovery': '✓  discovery + mechanism reports present (classifier OFF)',
        'absent': '·  not run',
    }.get(gnn['status'], '·  not run')
    L.append(f"GNN            {gnn_label}")

    # IRR
    if irr['testset_count'] > 0:
        L.append(
            f"IRR            ✓  {irr['testset_count']} testset(s)  ·  "
            f"{irr['human_codes_count']} human codes")
    else:
        L.append("IRR            ·  0 testsets")

    # CV
    if cv['testset_count'] > 0:
        L.append(f"CONTENT-VALID  ✓  {cv['testset_count']} testset(s)")
    else:
        L.append("CONTENT-VALID  ·  0 testsets")

    # RUNS registry (schema v2) — durable per-rater run table, when present.
    L.extend(_runs_lines(runs))

    # ANALYSIS
    if analysis['executive_summary_present']:
        L.append("ANALYSIS       ✓  executive summary + reports present")
    elif analysis['longitudinal_present']:
        L.append("ANALYSIS       ·  partial (no executive summary yet)")
    else:
        L.append("ANALYSIS       ·  not run")

    L.append('')
    return '\n'.join(L)


def format_status_json(status: dict) -> str:
    return json.dumps(status, indent=2, default=str)
