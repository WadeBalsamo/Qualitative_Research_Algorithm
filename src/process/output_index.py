"""
process/output_index.py
-----------------------
Walk the run directory at pipeline completion and write 00_index.txt —
one line per artifact, grouped by folder, with file size.
"""
import os

from . import output_paths as _paths

_FOLDER_LABELS = {
    '01_transcripts':   'Coded transcripts',
    '02_meta':          'Provenance & configuration',
    '03_analysis_data': 'Machine-readable data',
    '04_validation':    'Validation artifacts',
    '05_figures':       'Analysis figures (PNG)',
    '06_reports':       'Human-readable reports',
}

_SKIP_NAMES = {'00_index.txt'}
_SKIP_DIRS  = {'__pycache__', 'auditable_logs', 'codebook_raw'}


def _fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1024 ** 2:.1f} MB"


def write_index(run_dir: str) -> str:
    """Walk run_dir and write 00_index.txt. Returns the written path."""
    entries: dict[str, list[tuple[str, int]]] = {}

    for root, dirs, files in os.walk(run_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)

        rel_root = os.path.relpath(root, run_dir)
        top_folder = rel_root.split(os.sep)[0] if rel_root != '.' else '.'

        for fname in sorted(files):
            if fname in _SKIP_NAMES:
                continue
            full = os.path.join(root, fname)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            rel_path = os.path.relpath(full, run_dir)
            entries.setdefault(top_folder, []).append((rel_path, size))

    lines = [f"QRA Output Index — {os.path.basename(run_dir)}", ""]

    for folder in sorted(entries):
        label = _FOLDER_LABELS.get(folder, folder)
        lines.append(f"── {label} ({'/' + folder + '/' if folder != '.' else run_dir})")
        for rel_path, size in entries[folder]:
            lines.append(f"   {rel_path:<60}  {_fmt_size(size):>9}")
        lines.append("")

    index_path = os.path.join(run_dir, '00_index.txt')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return index_path
