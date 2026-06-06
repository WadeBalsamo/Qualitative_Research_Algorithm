"""
process/anonymization_editor.py
-------------------------------
Post-hoc editing of the speaker anonymization key with a full, referentially
consistent cascade across every downstream artifact.

The anonymization key (02_meta/speaker_anonymization_key.json) maps
  raw_name -> {role, anonymized_id}
and the anonymized_id propagates into many derived artifacts: the
``participant_id`` / ``speaker`` / ``speakers_in_segment`` fields and the
``{anonymized_id}`` text tokens of frozen segments, the ``segment_id`` of every
segment (which embeds ``anonymized_id.split('_')[-1]`` — e.g. ``Participant_MM016``
-> ``MM016``, ``therapist_4`` -> ``4``), classification overlays keyed by
segment_id, checkpoint dict keys, the master dataset, analysis JSON/reports,
per-participant filenames, and validation worksheets.

This module lets a researcher rename / merge / relabel speakers after the fact
and rewrite all of that consistently.

Design (see also the per-artifact table in the feature plan):

  Phase A — Surgical remap of source-of-truth artifacts using two maps derived
            from the key diff:
              relabel_map : {old_anon_id -> new_anon_id}  (field/token/filename)
              segid_map   : {old_segment_id -> new_segment_id}  (recomputed by
                            replacing only the fragment prefix of each segment_id)
            Optional ``remove_names`` additionally re-runs the NLP de-id pass
            (text_anonymization.scrub_segments) to catch newly-added names.

  Phase B — Regenerate derived artifacts (master dataset, analysis JSON, reports)
            by reusing the existing assemble/analyze stages, which is far more
            robust than hand-rewriting 100-column CSVs and exemplar JSON.

Audit logs (02_meta/auditable_logs/*.txt) are intentionally left untouched —
they hold only anonymized labels (never real names) and serve as an immutable
provenance trail.

Public API:
  load_key(output_dir) -> dict
  save_key(output_dir, key)
  rename_anon_id / rename_raw_name / change_role / add_entry / remove_entry /
    merge_speakers   -> new key dict
  diff_keys(old, new) -> relabel_map
  build_segid_map(output_dir, relabel_map) -> segid_map
  apply_key_update(output_dir, old_key, new_key, ...) -> stats dict
  run_anonymization_tui(output_dir)
"""
import glob
import json
import os
import re
import shutil
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from . import output_paths as _paths
from . import segments_io as _sio
from . import classifications_io as _cio
from . import db as _db
from ._freeze import write_frozen, sha256_text

VALID_ROLES = ('participant', 'therapist', 'staff')


# ---------------------------------------------------------------------------
# Key I/O
# ---------------------------------------------------------------------------

def _key_path(output_dir: str) -> str:
    return os.path.join(_paths.meta_dir(output_dir), 'speaker_anonymization_key.json')


def load_key(output_dir: str) -> dict:
    """Return the raw key dict {raw_name: {role, anonymized_id}} (or {} if absent)."""
    path = _key_path(output_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def save_key(output_dir: str, key: dict) -> str:
    """Atomically write the key JSON + regenerate the human-readable table. Returns JSON path."""
    path = _key_path(output_dir)
    write_frozen(path, lambda fh: json.dump(key, fh, indent=2), force=True)
    # Reuse the orchestrator's table writer so the .txt format stays identical.
    from .orchestrator import _write_anonymization_key_txt
    _write_anonymization_key_txt(key, _paths.anonymization_key_txt_path(output_dir))
    return path


# ---------------------------------------------------------------------------
# Key edit operations — each returns a NEW key dict (never mutates the input)
# ---------------------------------------------------------------------------

def _clone(key: dict) -> dict:
    return {raw: dict(entry) for raw, entry in key.items()}


def list_anon_ids(key: dict) -> List[str]:
    """Distinct anonymized_ids present in the key, sorted."""
    return sorted({e.get('anonymized_id', '') for e in key.values() if e.get('anonymized_id')})


def rename_anon_id(key: dict, old_id: str, new_id: str) -> dict:
    """Set every entry whose anonymized_id == old_id to new_id."""
    new = _clone(key)
    found = False
    for entry in new.values():
        if entry.get('anonymized_id') == old_id:
            entry['anonymized_id'] = new_id
            found = True
    if not found:
        raise KeyError(f"no key entry has anonymized_id {old_id!r}")
    return new


def rename_raw_name(key: dict, old_raw: str, new_raw: str) -> dict:
    """Rename the raw (original) speaker label; anonymized_id is unchanged."""
    if old_raw not in key:
        raise KeyError(f"raw name {old_raw!r} not in key")
    if new_raw in key and new_raw != old_raw:
        raise ValueError(f"raw name {new_raw!r} already exists in key")
    new = _clone(key)
    new[new_raw] = new.pop(old_raw)
    return new


def change_role(key: dict, anon_id: str, role: str) -> dict:
    """Change the role of every entry with the given anonymized_id."""
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
    new = _clone(key)
    found = False
    for entry in new.values():
        if entry.get('anonymized_id') == anon_id:
            entry['role'] = role
            found = True
    if not found:
        raise KeyError(f"no key entry has anonymized_id {anon_id!r}")
    return new


def add_entry(key: dict, raw_name: str, role: str, anon_id: str) -> dict:
    """Add a new raw_name -> {role, anonymized_id} mapping."""
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
    if raw_name in key:
        raise ValueError(f"raw name {raw_name!r} already exists in key")
    new = _clone(key)
    new[raw_name] = {'role': role, 'anonymized_id': anon_id}
    return new


def remove_entry(key: dict, raw_name: str) -> dict:
    """Remove a raw_name from the key."""
    if raw_name not in key:
        raise KeyError(f"raw name {raw_name!r} not in key")
    new = _clone(key)
    del new[raw_name]
    return new


def merge_speakers(key: dict, raw_names: List[str], target_anon_id: str) -> dict:
    """Point several raw labels at a single anonymized_id (collapse duplicates)."""
    new = _clone(key)
    for raw in raw_names:
        if raw not in new:
            raise KeyError(f"raw name {raw!r} not in key")
        new[raw]['anonymized_id'] = target_anon_id
    return new


# ---------------------------------------------------------------------------
# Map construction
# ---------------------------------------------------------------------------

def diff_keys(old_key: dict, new_key: dict) -> Dict[str, str]:
    """Return {old_anon_id -> new_anon_id} for raw names whose anon_id changed.

    Only raw names present in BOTH keys with a changed anonymized_id contribute.
    Added/removed raw names do not produce a relabel (no old/new pair to map).
    """
    relabel: Dict[str, str] = {}
    for raw, new_entry in new_key.items():
        old_entry = old_key.get(raw)
        if not old_entry:
            continue
        old_id = old_entry.get('anonymized_id')
        new_id = new_entry.get('anonymized_id')
        if old_id and new_id and old_id != new_id:
            if old_id in relabel and relabel[old_id] != new_id:
                raise ValueError(
                    f"inconsistent relabel: {old_id!r} maps to both "
                    f"{relabel[old_id]!r} and {new_id!r}"
                )
            relabel[old_id] = new_id
    return relabel


def _fragment(anon_id: str) -> str:
    """The segment_id fragment derived from an anonymized_id (matches ingestion)."""
    return anon_id.split('_')[-1]


def build_segid_map(output_dir: str, relabel_map: Dict[str, str]) -> Dict[str, str]:
    """Scan frozen segments and compute {old_segment_id -> new_segment_id}.

    For each segment whose participant_id changed, the new segment_id is built by
    replacing only the fragment portion of the id:
        old prefix = "{trial}_{session}_{old_fragment}_"
        new id     = "{trial}_{session}_{new_fragment}_{tail}"
    This is exact and never touches digits inside timestamps/offsets.

    Raises ValueError (before any writes) if a segment_id can't be parsed or if
    two distinct old ids would collide onto one new id.
    """
    segid_map: Dict[str, str] = {}
    if not relabel_map:
        return segid_map
    for sid in _sio.list_segmented_sessions(output_dir):
        for seg in _sio.read_session_segments(output_dir, sid):
            old_anon = seg.participant_id
            if old_anon not in relabel_map:
                continue
            new_anon = relabel_map[old_anon]
            prefix = f"{seg.trial_id}_{seg.session_id}_{_fragment(old_anon)}_"
            if not seg.segment_id.startswith(prefix):
                raise ValueError(
                    f"cannot parse segment_id {seg.segment_id!r}: expected prefix "
                    f"{prefix!r} (derived from participant_id {old_anon!r}). "
                    "Aborting before any writes."
                )
            tail = seg.segment_id[len(prefix):]
            new_segment_id = f"{seg.trial_id}_{seg.session_id}_{_fragment(new_anon)}_{tail}"
            if new_segment_id in segid_map.values() and segid_map.get(seg.segment_id) != new_segment_id:
                raise ValueError(
                    f"segment_id collision: two segments would both become "
                    f"{new_segment_id!r}. Aborting before any writes."
                )
            segid_map[seg.segment_id] = new_segment_id
    return segid_map


def _make_token_remapper(relabel_map: Dict[str, str]) -> Callable[[str], str]:
    """Single-pass remapper for {old}/[old] tokens in segment text.

    Wrapped in {}/[] delimiters so 'therapist_1' never matches inside
    '{therapist_10}'. Single pass (regex) so chained renames don't double-apply.
    """
    if not relabel_map:
        return lambda t: t
    alt = '|'.join(re.escape(i) for i in sorted(relabel_map, key=len, reverse=True))
    pat = re.compile(r'\{(' + alt + r')\}|\[(' + alt + r')\]')

    def _sub(m: re.Match) -> str:
        if m.group(1) is not None:
            return '{' + relabel_map[m.group(1)] + '}'
        return '[' + relabel_map[m.group(2)] + ']'

    return lambda t: pat.sub(_sub, t) if t else t


def _make_label_remapper(relabel_map: Dict[str, str]) -> Callable[[str], str]:
    """Single-pass whole-word remapper for bare anon_ids in human-readable text
    (worksheets show ``Participant: Participant_MM016``). Word boundaries keep
    'therapist_1' from matching 'therapist_10'.
    """
    if not relabel_map:
        return lambda t: t
    alt = '|'.join(re.escape(i) for i in sorted(relabel_map, key=len, reverse=True))
    pat = re.compile(r'\b(' + alt + r')\b')
    return lambda t: pat.sub(lambda m: relabel_map[m.group(1)], t) if t else t


# ---------------------------------------------------------------------------
# Per-artifact rewriters (Phase A)
# ---------------------------------------------------------------------------

def _remap_session_segments(
    output_dir: str,
    sid: str,
    relabel_map: Dict[str, str],
    segid_map: Dict[str, str],
    token_remap: Callable[[str], str],
    *,
    rescrub: bool = False,
    new_key: Optional[dict] = None,
    rescrub_opts: Optional[dict] = None,
) -> int:
    """Rewrite one session's frozen segments.jsonl in place. Returns segment count."""
    segs = _sio.read_session_segments(output_dir, sid)
    for seg in segs:
        if seg.participant_id in relabel_map:
            seg.participant_id = relabel_map[seg.participant_id]
        if seg.speakers_in_segment:
            seg.speakers_in_segment = [relabel_map.get(s, s) for s in seg.speakers_in_segment]
        if seg.segment_id in segid_map:
            seg.segment_id = segid_map[seg.segment_id]
        seg.text = token_remap(seg.text)
    if rescrub:
        from .text_anonymization import scrub_segments
        scrub_segments(segs, new_key or {}, **(rescrub_opts or {}))
        for seg in segs:
            seg.word_count = len(seg.text.split())
    _sio.overwrite_segment_texts(output_dir, sid, segs)
    return len(segs)


def _remap_dict_keys(d: dict, segid_map: Dict[str, str]) -> dict:
    """Return a new dict with keys remapped via segid_map. Keys starting with '_'
    (e.g. '_meta') are preserved untouched."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(k, str) and k.startswith('_'):
            out[k] = v
        else:
            out[segid_map.get(k, k)] = v
    return out


def _remap_checkpoint_file(path: str, segid_map: Dict[str, str]) -> int:
    """Remap segment_id dict keys in a classification checkpoint JSON. Returns keys changed."""
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0

    def _count_changes(keys) -> int:
        return sum(1 for k in keys if k in segid_map)

    if '_meta' in data and isinstance(data.get('run_results'), dict):
        changed = _count_changes(data['run_results'].keys())
        data['run_results'] = _remap_dict_keys(data['run_results'], segid_map)
    else:
        changed = _count_changes(data.keys())
        data = _remap_dict_keys(data, segid_map)

    if changed:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, default=str)
    return changed


def _remap_checkpoints(output_dir: str, segid_map: Dict[str, str]) -> int:
    total = 0
    dirs = [
        _paths.llm_checkpoints_dir(output_dir),
        os.path.join(_paths.codebook_raw_dir(output_dir), 'checkpoints'),
    ]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, '*.json'))):
            total += _remap_checkpoint_file(path, segid_map)
    return total


def _remap_text_files(output_dir: str, label_remap: Callable[[str], str]) -> int:
    """Whole-word remap bare anon_ids in validation worksheets (.txt). Returns files changed."""
    changed = 0
    roots = [_paths.testsets_dir(output_dir), _paths.content_validity_dir(output_dir)]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                if not fname.endswith('.txt'):
                    continue
                fpath = os.path.join(dirpath, fname)
                with open(fpath, encoding='utf-8') as fh:
                    content = fh.read()
                new_content = label_remap(content)
                if new_content != content:
                    with open(fpath, 'w', encoding='utf-8') as fh:
                        fh.write(new_content)
                    changed += 1
    return changed


def _remap_cv_items(output_dir: str, segid_map: Dict[str, str],
                    relabel_map: Dict[str, str], token_remap: Callable[[str], str],
                    label_remap: Callable[[str], str]) -> int:
    """Remap content-validity item text in the SQLite ``cv_testset_items`` table.

    CV item rows carry no segment_id/participant_id, so only the ``text`` column
    is rewritten via ``label_remap(token_remap(text))``. Returns rows changed.
    """
    if not _db.db_exists(output_dir):
        return 0
    changed = 0
    with _db.open_db(output_dir) as conn:
        names = [r['name'] for r in conn.execute('SELECT name FROM cv_testsets').fetchall()]
        for name in names:
            items = conn.execute(
                'SELECT item_id, text FROM cv_testset_items WHERE testset_name = ?',
                (name,),
            ).fetchall()
            for item in items:
                text = item['text']
                if not isinstance(text, str):
                    continue
                new_text = label_remap(token_remap(text))
                if new_text != text:
                    conn.execute(
                        'UPDATE cv_testset_items SET text = ? '
                        'WHERE testset_name = ? AND item_id = ?',
                        (new_text, name, item['item_id']),
                    )
                    changed += 1
    return changed


def _refresh_testset_meta_shas(output_dir: str) -> int:
    """Recompute per-item sha256 in the SQLite ``testset_items`` table so
    drift-detection stays valid after segment text changed. Returns rows updated."""
    if not _db.db_exists(output_dir):
        return 0
    # Index current segment text by (session_id, segment_index).
    text_by_key: Dict[Tuple[str, int], str] = {}
    for sid in _sio.list_segmented_sessions(output_dir):
        for seg in _sio.read_session_segments(output_dir, sid):
            text_by_key[(seg.session_id, seg.segment_index)] = seg.text

    updated = 0
    with _db.open_db(output_dir) as conn:
        rows = conn.execute(
            'SELECT worksheet_n, item_num, session_id, seg_num, sha256 FROM testset_items'
        ).fetchall()
        for row in rows:
            sess = row['session_id']
            seg_num = row['seg_num']
            if sess is None or seg_num is None:
                continue
            text = text_by_key.get((sess, int(seg_num) - 1))  # seg_num is 1-based
            if text is None:
                continue
            new_sha = sha256_text(text)
            if row['sha256'] != new_sha:
                conn.execute(
                    'UPDATE testset_items SET sha256 = ? '
                    'WHERE worksheet_n = ? AND item_num = ?',
                    (new_sha, row['worksheet_n'], row['item_num']),
                )
                updated += 1
    return updated


# ---------------------------------------------------------------------------
# Backups + regeneration
# ---------------------------------------------------------------------------

_BACKUP_DIRS = (
    '02_meta', '01_transcripts', '03_analysis_data', '04_validation', '06_reports',
)


def _backup(output_dir: str) -> str:
    """Copy mutable artifact dirs to <output_dir>/_backups/<timestamp>/. Returns the backup path."""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest_root = os.path.join(output_dir, '_backups', stamp)
    os.makedirs(dest_root, exist_ok=True)
    # The SQLite store (qra.db) is the mutated source of truth but lives at the
    # output_dir root, outside the _BACKUP_DIRS subdirs, so back it up explicitly
    # (including any WAL/SHM sidecar files).
    db_main = _db.db_path(output_dir)
    for src in (db_main, db_main + '-wal', db_main + '-shm'):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest_root, os.path.basename(src)))
    for sub in _BACKUP_DIRS:
        src = os.path.join(output_dir, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest_root, sub))
    return dest_root


def _clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def _regenerate_derived(output_dir: str, config, *, verbose: bool = True) -> Dict[str, int]:
    """Rebuild master dataset + analysis JSON/reports from the remapped source-of-truth.

    Deletes stale per_participant/per_session/per_theme outputs first so files
    renamed by the relabel don't linger under their old names.
    """
    from .orchestrator import stage_assemble
    from analysis.runner import run_analysis

    stats = {'master': 0, 'analysis_files': 0}

    has_overlay = any(
        _cio.overlay_exists(output_dir, key)
        for key in ('theme', 'purer', 'codebook', 'cv')
    )
    if has_overlay and _sio.list_segmented_sessions(output_dir):
        master_df = stage_assemble(config, output_dir=output_dir)
        stats['master'] = len(master_df)

    # Drop stale per-entity outputs so renamed ids don't leave orphan files.
    for d in (
        _paths.participants_json_dir(output_dir),
        _paths.sessions_json_dir(output_dir),
        _paths.themes_json_dir(output_dir),
        _paths.reports_per_participant_dir(output_dir),
        _paths.reports_per_session_dir(output_dir),
        _paths.themes_dir(output_dir),
    ):
        _clear_dir(d)

    if os.path.isdir(_paths.analysis_data_dir(output_dir)) and stats['master']:
        result = run_analysis(output_dir, verbose=verbose)
        stats['analysis_files'] = len(result.get('files_generated', []))
    return stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def preview_key_update(output_dir: str, old_key: dict, new_key: dict) -> dict:
    """Compute the relabel/segid maps and affected-artifact counts WITHOUT writing."""
    relabel_map = diff_keys(old_key, new_key)
    segid_map = build_segid_map(output_dir, relabel_map)
    n_overlays = sum(
        1 for key in ('theme', 'purer', 'codebook', 'cv')
        if _cio.overlay_exists(output_dir, key)
    )
    n_checkpoints = 0
    for d in (_paths.llm_checkpoints_dir(output_dir),
              os.path.join(_paths.codebook_raw_dir(output_dir), 'checkpoints')):
        if os.path.isdir(d):
            n_checkpoints += len(glob.glob(os.path.join(d, '*.json')))
    return {
        'relabel_map': relabel_map,
        'n_renamed_ids': len(relabel_map),
        'n_segment_ids': len(segid_map),
        'n_sessions': len(_sio.list_segmented_sessions(output_dir)),
        'n_overlays': n_overlays,
        'n_checkpoints': n_checkpoints,
    }


def apply_key_update(
    output_dir: str,
    old_key: dict,
    new_key: dict,
    *,
    remove_names: bool = False,
    include_locked: bool = True,
    rescrub_opts: Optional[dict] = None,
    backup: bool = True,
    dry_run: bool = False,
    config=None,
    regenerate: bool = True,
    verbose: bool = True,
) -> dict:
    """Apply a key edit and cascade it across every derived artifact.

    ``include_locked`` documents that frozen segments are rewritten (they always
    are — ``overwrite_segment_texts`` bypasses the freeze guard); it is accepted
    for symmetry / future gating.

    Returns a stats dict. With ``dry_run=True`` nothing is written and the return
    value is the preview from :func:`preview_key_update`.
    """
    # Build + validate ALL maps before touching anything (may raise).
    relabel_map = diff_keys(old_key, new_key)
    segid_map = build_segid_map(output_dir, relabel_map)

    if dry_run:
        return preview_key_update(output_dir, old_key, new_key)

    text_changes = bool(relabel_map) or remove_names
    if backup and (text_changes or new_key != old_key):
        backup_path = _backup(output_dir)
    else:
        backup_path = None

    token_remap = _make_token_remapper(relabel_map)
    label_remap = _make_label_remapper(relabel_map)

    stats: dict = {
        'backup': backup_path,
        'relabel_map': relabel_map,
        'n_segment_ids': len(segid_map),
        'sessions_rewritten': 0,
        'overlay_rows': 0,
        'checkpoint_keys': 0,
        'testset_txt_files': 0,
        'cv_item_rows': 0,
        'testset_meta_files': 0,
    }

    # --- Always persist the new key ---
    save_key(output_dir, new_key)

    # --- Frozen segments (fields + segment_id + text tokens, optional re-scrub) ---
    if text_changes or segid_map:
        for sid in _sio.list_segmented_sessions(output_dir):
            _remap_session_segments(
                output_dir, sid, relabel_map, segid_map, token_remap,
                rescrub=remove_names, new_key=new_key, rescrub_opts=rescrub_opts,
            )
            stats['sessions_rewritten'] += 1

    # --- Classification overlays (segment_id field) ---
    if segid_map:
        for key in ('theme', 'purer', 'codebook', 'cv'):
            stats['overlay_rows'] += _cio.remap_overlay_segment_ids(output_dir, key, segid_map)
        # --- Classification checkpoints (segment_id dict keys) ---
        stats['checkpoint_keys'] += _remap_checkpoints(output_dir, segid_map)

    # --- Validation worksheets + content-validity items ---
    if relabel_map:
        stats['testset_txt_files'] += _remap_text_files(output_dir, label_remap)
        stats['cv_item_rows'] += _remap_cv_items(
            output_dir, segid_map, relabel_map, token_remap, label_remap)

    # --- Recompute testset meta SHAs if segment text changed ---
    if text_changes:
        stats['testset_meta_files'] += _refresh_testset_meta_shas(output_dir)

    # --- Phase B: regenerate master dataset + analysis/reports ---
    if regenerate and text_changes:
        if config is None:
            from .config import PipelineConfig
            config = PipelineConfig()
            config.output_dir = output_dir
        stats['regenerated'] = _regenerate_derived(output_dir, config, verbose=verbose)

    return stats


# ---------------------------------------------------------------------------
# TUI driver — reuses the interactive_tui display/prompt helpers
# ---------------------------------------------------------------------------

def _roster(key: dict) -> List[Tuple[str, str, str]]:
    """Speakers as (raw_name, role, anonymized_id), sorted by anonymized_id."""
    return sorted(
        ((raw, e.get('role', ''), e.get('anonymized_id', '')) for raw, e in key.items()),
        key=lambda t: (t[2], t[0]),
    )


def _shared_raws(key: dict, anon_id: str) -> List[str]:
    """All raw labels currently pointing at anon_id (>1 means merged speaker)."""
    return [raw for raw, e in key.items() if e.get('anonymized_id') == anon_id]


def _print_roster(working: dict, _info) -> List[Tuple[str, str, str]]:
    roster = _roster(working)
    if not roster:
        _info('(speaker key is empty — choose "a" to add a speaker)')
        return roster
    width = max((len(anon) for _r, _ro, anon in roster), default=16)
    _info('Speakers in this project (anonymized ID  ⟵  original name):')
    _info('')
    for i, (raw, role, anon) in enumerate(roster, 1):
        shared = _shared_raws(working, anon)
        tag = f'   [shared by {len(shared)}]' if len(shared) > 1 else ''
        _info(f'[{i:>2}]  {anon:<{width}}  ⟵  {raw}   ({role}){tag}')
    return roster


def _prompt_new_anon_id(working: dict, role: str, current_id: str, helpers) -> str:
    """Walk-through prompt for a speaker's new anonymized ID.

    Offers the next free ID as a suggestion, lets the user type a custom ID,
    press Enter to keep the current one, or 'list' to merge into an existing ID.
    """
    _menu, _ask, _confirm, _section, _info, _ok, _warn, _err, _pause = helpers
    from .speaker_walkthrough import _next_id
    try:
        suggested = _next_id(working, role) if role in VALID_ROLES else current_id
    except ValueError:
        suggested = current_id

    _info(f'Current anonymized ID : {current_id}')
    _info(f'Next free ID for role : {suggested}')
    _info("Enter a new ID, type 's' to accept the suggested ID, 'list' to merge")
    _info("into an existing ID, or just press Enter to keep the current one.")
    ans = _ask('New anonymized ID', current_id)
    if ans == current_id:
        return current_id
    if ans.lower() == 's':
        return suggested
    if ans.lower() == 'list':
        existing = [a for a in list_anon_ids(working) if a != current_id]
        if not existing:
            _warn('No other IDs to merge into.')
            return current_id
        for i, a in enumerate(existing, 1):
            _info(f'  [{i}] {a}')
        pick = _ask('Pick a number (Enter to cancel)')
        try:
            idx = int(pick) - 1
            if 0 <= idx < len(existing):
                return existing[idx]
        except ValueError:
            pass
        _warn('Cancelled.')
        return current_id
    return ans


def _edit_speaker(working: dict, raw: str, helpers) -> dict:
    """Per-speaker action submenu. Returns the updated working key."""
    _menu, _ask, _confirm, _section, _info, _ok, _warn, _err, _pause = helpers
    while raw in working:
        entry = working[raw]
        role = entry.get('role', '')
        anon = entry.get('anonymized_id', '')
        shared = _shared_raws(working, anon)
        opts = [
            (f'Rename anonymized ID  (current: {anon})',
             'Cascades to segment_ids, {tokens}, overlays, checkpoints, reports.'
             + (f'\nNOTE: {len(shared)} raw labels share this ID — all move together.'
                if len(shared) > 1 else '')),
            (f'Rename original/raw name  (current: {raw!r})',
             'Fix the real-name spelling; anonymized ID is unchanged.'),
            (f'Change role  (current: {role})', 'participant / therapist / staff'),
            ('Merge into another speaker', 'Point this label at an existing anonymized ID.'),
            ('Remove this speaker', 'Delete this raw label from the key.'),
        ]
        c = _menu(f'Speaker:  {raw}  →  {anon}  ({role})', opts, back_label='Back to roster')
        try:
            if c == 0:
                return working
            elif c == 1:
                new_id = _prompt_new_anon_id(working, role, anon, helpers)
                if new_id and new_id != anon:
                    working = rename_anon_id(working, anon, new_id)
                    _ok(f'{anon} -> {new_id}')
            elif c == 2:
                new_raw = _ask('New original/raw name', raw)
                if new_raw and new_raw != raw:
                    working = rename_raw_name(working, raw, new_raw)
                    _ok(f'{raw!r} -> {new_raw!r}')
                    raw = new_raw
            elif c == 3:
                new_role = _ask('Role (participant/therapist/staff)', role)
                if new_role and new_role != role:
                    working = change_role(working, anon, new_role)
                    _ok(f'{anon} role -> {new_role}')
            elif c == 4:
                others = [a for a in list_anon_ids(working) if a != anon]
                if not others:
                    _warn('No other anonymized IDs to merge into.')
                    _pause()
                    continue
                for i, a in enumerate(others, 1):
                    _info(f'  [{i}] {a}')
                pick = _ask('Merge into which ID? (number, Enter to cancel)')
                try:
                    idx = int(pick) - 1
                    if 0 <= idx < len(others):
                        working = merge_speakers(working, [raw], others[idx])
                        _ok(f'{raw!r} now -> {others[idx]}')
                except ValueError:
                    _warn('Cancelled.')
            elif c == 5:
                if _confirm(f'Remove {raw!r} from the key?', default=False):
                    working = remove_entry(working, raw)
                    _ok(f'removed {raw!r}')
                    return working
        except (KeyError, ValueError) as exc:
            _err(str(exc))
            _pause()
    return working


def _walk_all_speakers(working: dict, helpers) -> dict:
    """Sequentially walk every speaker, prompting for a new anonymized ID."""
    _menu, _ask, _confirm, _section, _info, _ok, _warn, _err, _pause = helpers
    _section('Walk through all speakers')
    _info('For each speaker: enter a new anonymized ID, type "s" for the suggested')
    _info('next free ID, or press Enter to keep the current ID. Ctrl-C to stop.')
    for raw, role, _anon in _roster(working):
        if raw not in working:        # may have been merged away mid-walk
            continue
        cur = working[raw].get('anonymized_id', '')
        print()
        _info(f'Speaker:  {raw}   role={role}   current ID={cur}')
        try:
            new_id = _prompt_new_anon_id(working, role, cur, helpers)
        except KeyboardInterrupt:
            print()
            _warn('Walk-through stopped.')
            break
        if new_id and new_id != cur:
            working = rename_anon_id(working, cur, new_id)
            _ok(f'{cur} -> {new_id}')
    return working


def run_anonymization_tui(output_dir: str, config=None) -> None:
    """Roster-driven interactive editor for the speaker anonymization key + cascade."""
    from .interactive_tui import (
        _menu, _ask, _confirm, _section, _info, _ok, _warn, _err, _pause, _hr,
    )
    helpers = (_menu, _ask, _confirm, _section, _info, _ok, _warn, _err, _pause)

    old_key = load_key(output_dir)
    if not old_key:
        _warn(f"No speaker_anonymization_key.json found in {output_dir}/02_meta/.")
        _warn("Run `qra ingest` first, or choose 'a' to add a speaker.")
    working = _clone(old_key)
    remove_names = False

    while True:
        _section('Edit Speaker Anonymization Key')
        roster = _print_roster(working, _info)
        pending = working != old_key
        print()
        _hr()
        _info('Select a speaker NUMBER to rename / edit, or an action:')
        _info('  r  Walk through ALL speakers in sequence')
        _info('  a  Add a new speaker')
        _info(f'  n  Toggle re-run name removal (--remove-names): {"ON" if remove_names else "OFF"}')
        _info('  p  Preview cascade (dry-run — writes nothing)')
        _info(f'  w  APPLY changes & cascade{"   *pending edits*" if pending else ""}')
        _info('  0  Discard & exit')
        _hr()
        sel = _ask('Choice').strip().lower()

        try:
            if sel in ('0', 'q', ''):
                if pending and not _confirm('Discard unsaved edits and exit?', default=False):
                    continue
                if not pending:
                    return
                return
            elif sel == 'r':
                working = _walk_all_speakers(working, helpers)
            elif sel == 'a':
                raw = _ask('New original/raw name')
                if not raw:
                    continue
                role = _ask('Role (participant/therapist/staff)', 'participant')
                from .speaker_walkthrough import _next_id
                try:
                    suggested = _next_id(working, role)
                except ValueError:
                    suggested = ''
                anon_id = _ask('Anonymized ID', suggested)
                working = add_entry(working, raw, role, anon_id)
                _ok(f'added {raw!r} -> {anon_id} ({role})')
            elif sel == 'n':
                remove_names = not remove_names
            elif sel == 'p':
                prev = preview_key_update(output_dir, old_key, working)
                print()
                _info(f"Renamed anonymized IDs : {prev['n_renamed_ids']}  {prev['relabel_map'] or ''}")
                _info(f"Segment IDs to rewrite : {prev['n_segment_ids']}")
                _info(f"Sessions               : {prev['n_sessions']}")
                _info(f"Overlays / checkpoints : {prev['n_overlays']} / {prev['n_checkpoints']}")
                _info(f"Re-run name removal    : {'yes' if remove_names else 'no'}")
                _pause()
            elif sel == 'w':
                if not pending and not remove_names:
                    _warn('No pending edits and --remove-names is off — nothing to apply.')
                    _pause()
                    continue
                prev = preview_key_update(output_dir, old_key, working)
                print()
                _info(f"About to rename {prev['n_renamed_ids']} ID(s), rewrite "
                      f"{prev['n_segment_ids']} segment_id(s) across {prev['n_sessions']} session(s),")
                _info(f"remap {prev['n_overlays']} overlay(s) + {prev['n_checkpoints']} checkpoint(s), "
                      f"then regenerate master + analysis.")
                _info('A timestamped backup is written to _backups/ first.')
                if not _confirm('Apply and cascade now?', default=False):
                    continue
                stats = apply_key_update(
                    output_dir, old_key, working,
                    remove_names=remove_names, backup=True, config=config, verbose=True,
                )
                _ok(f"Done. Backup: {stats.get('backup')}")
                _ok(f"Sessions rewritten: {stats['sessions_rewritten']}, "
                    f"overlay rows: {stats['overlay_rows']}, checkpoint keys: {stats['checkpoint_keys']}")
                if stats.get('regenerated'):
                    _ok(f"Regenerated master ({stats['regenerated']['master']} segments) + "
                        f"{stats['regenerated']['analysis_files']} analysis files")
                old_key = load_key(output_dir)
                working = _clone(old_key)
                remove_names = False
                _pause()
            elif sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(roster):
                    working = _edit_speaker(working, roster[idx][0], helpers)
                else:
                    _warn(f'No speaker numbered {sel}.')
                    _pause()
            else:
                _warn(f'Unrecognized choice: {sel!r}')
        except (KeyError, ValueError) as exc:
            _err(str(exc))
            _pause()
