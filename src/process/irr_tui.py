"""
process/irr_tui.py
------------------
Interactive menu for the inter-rater-reliability feature (``qra irr`` with no
subcommand).  Reuses the shared TUI primitives from ``interactive_tui``.

Menu:
  [1] Import human-coded CSV
  [2] Run IRR analysis (pull live LLM + GNN, compute, write report + figures)
  [3] View summary (headline of the last report)
  [4] List imported test-sets
"""

import os

from .interactive_tui import _section, _menu, _ask, _ok, _warn, _err, _info, _pause
from . import output_paths as _paths


def _default_csv() -> str:
    """Best-effort default path to the committed human-coded CSV."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, 'data', 'irr', 'human_coded_testsets.csv')


def run_irr_tui(output_dir: str = None) -> None:
    _section('Inter-Rater Reliability (IRR)')
    if not output_dir:
        output_dir = _ask('Project output directory', './data/output')
    output_dir = os.path.expanduser(output_dir)
    if not os.path.isdir(output_dir):
        _err(f'No such directory: {output_dir}')
        return

    while True:
        choice = _menu(
            f'IRR — {output_dir}',
            [
                ('Import human-coded CSV',
                 'Parse the wide CSV, normalize labels, resolve segments, persist to qra.db.'),
                ('Run IRR analysis',
                 'Pull current LLM + GNN labels, compute all three families,\n'
                 'write report + figures + data files.'),
                ('View summary', 'Show the headline of the last IRR report.'),
                ('List imported test-sets', 'Show imported worksheets + rosters.'),
            ],
            back_label='Back to main menu / exit',
        )
        if choice == 0:
            return
        if choice == 1:
            _do_import(output_dir)
        elif choice == 2:
            _do_run(output_dir)
        elif choice == 3:
            _do_view(output_dir)
        elif choice == 4:
            _do_list(output_dir)


def _do_import(output_dir: str) -> None:
    from . import irr_import
    csv_path = _ask('CSV path', _default_csv())
    csv_path = os.path.expanduser(csv_path)
    if not os.path.isfile(csv_path):
        _err(f'No such file: {csv_path}')
        return
    try:
        summary = irr_import.import_irr_csv(output_dir, csv_path, verbose=False)
    except Exception as e:  # noqa: BLE001 - surface to the user
        _err(f'Import failed: {e}')
        return
    _ok(f"Imported worksheets {summary['worksheets']} "
        f"({summary['n_items']} items, {summary['n_codes']} code rows).")
    if summary['warnings']:
        _warn(f"{len(summary['warnings'])} warning(s):")
        for w in summary['warnings']:
            _info(f'  - {w}')
    _pause()


def _do_run(output_dir: str) -> None:
    from analysis import irr_analysis, irr_figures
    from analysis.reports import irr_report, irr_items
    try:
        results = irr_analysis.run_irr_analysis(output_dir, verbose=False)
        report_path = irr_report.generate_irr_report(results, output_dir)
        item_files = irr_items.write_irr_item_details(results, output_dir)
        figs = irr_figures.write_irr_figures(results, output_dir)
    except Exception as e:  # noqa: BLE001
        _err(f'IRR analysis failed: {e}')
        return
    _ok(f"Report: {report_path}")
    _ok(f"Per-item detail: {len(item_files)} test-set file(s)")
    _ok(f"Data + figures: {_paths.irr_validation_dir(output_dir)} ({len(figs)} figure(s))")
    if not results.get('have_machine_labels'):
        _warn('No frozen segments / machine labels found — only Human↔Human computed.')
    _pause()


def _do_view(output_dir: str) -> None:
    path = _paths.reports_irr_path(output_dir)
    if not os.path.isfile(path):
        _warn('No IRR report yet — run the analysis first (option 2).')
        _pause()
        return
    with open(path, encoding='utf-8') as f:
        text = f.read()
    # Show through the end of the HEADLINE block.
    marker = '\n==='
    idx = text.find(marker, text.find('HEADLINE'))
    print()
    print(text[:idx] if idx > 0 else text[:1500])
    _pause()


def _do_list(output_dir: str) -> None:
    from . import irr_import
    testsets = irr_import.list_imported_testsets(output_dir)
    if not testsets:
        _warn('No imported IRR test-sets yet.')
        _pause()
        return
    print()
    for t in testsets:
        _info(f"worksheet {t['worksheet_n']}: {t['name']}  "
              f"({t['n_items']} items; raters: {', '.join(t['raters'])})")
    _pause()
