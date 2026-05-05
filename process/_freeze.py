"""
process/_freeze.py
------------------
Generic freeze enforcement primitives.

Used by segments_io (frozen per-session segmentation) and
human_forms (frozen validation testsets) to guarantee that
once-written artifacts are never silently overwritten.
"""
import hashlib
import json
import os
from typing import Callable, Dict, IO, TYPE_CHECKING

if TYPE_CHECKING:
    from classification_tools.data_structures import Segment


class FrozenArtifactError(RuntimeError):
    """Raised when a frozen artifact would be overwritten or has drifted."""


def write_frozen(path: str, write_fn: Callable[[IO], None], *, force: bool = False) -> None:
    """
    Write a file via write_fn(open_handle).

    Raises FrozenArtifactError if path already exists unless force=True.
    Atomic: writes to path + '.tmp' then renames.
    """
    if not force and os.path.exists(path):
        raise FrozenArtifactError(
            f"Frozen artifact already exists: {path}  "
            f"(pass force=True to overwrite)"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            write_fn(fh)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def sha256_text(text: str) -> str:
    """Return hex SHA-256 of text encoded as UTF-8."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def verify_content_sha(
    snapshot_path: str,
    segments_by_id: Dict[str, 'Segment'],
) -> Dict[str, bool]:
    """
    For each segment_id in the snapshot JSONL, return True if the current
    segment's text SHA-256 matches the stored content_sha256.

    Used by testset refresh to detect drift.
    """
    results: Dict[str, bool] = {}
    with open(snapshot_path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec['segment_id']
            stored_sha = rec.get('content_sha256', '')
            seg = segments_by_id.get(sid)
            if seg is None:
                results[sid] = False
            else:
                results[sid] = sha256_text(seg.text) == stored_sha
    return results
