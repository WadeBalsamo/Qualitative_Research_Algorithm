"""
classification_tools/zeroshot_reporting.py
------------------------------------------
Human-readable graded report for the zero-shot content-validity test
(`qra run --test-zeroshot`).  Extracted from qra.py to keep the CLI entry
point thin; the format mirrors the 04_validation/ artifacts.
"""
import os


def write_zeroshot_report(
    test_items: list,
    results_all: dict,
    rater_ids: list,
    framework,
    output_dir: str,
) -> str:
    """
    Write a human-readable graded report for the zero-shot content validity test.

    Format mirrors 04_validation/ artifacts (78-char separators, same header style).
    Returns the written file path.
    """
    import datetime as _dt
    import textwrap

    from process import output_paths as _paths

    _W = 78
    id_to_name = {t.theme_id: t.short_name for t in framework.themes} if framework else {}

    def _sname(stage_id):
        if stage_id is None:
            return 'ABSTAIN'
        return id_to_name.get(stage_id, str(stage_id))

    def _fconf(c):
        if c is None:
            return '?'
        return f'{float(c):.2f}'

    # ---------------------------------------------------------------------------
    # Pass-type helper: 'primary', 'secondary', or None.
    # A secondary match still counts as correct — if a rater assigned two
    # stages and the expected one is among them, that should not penalise the
    # model.  This mirrors pipeline behaviour where both primary and secondary
    # labels are recorded and considered meaningful.
    # ---------------------------------------------------------------------------
    def _pass_type(primary, secondary, expected):
        if primary is not None and primary == expected:
            return 'primary'
        if secondary is not None and secondary == expected:
            return 'secondary'
        return None

    # Build per-item result rows
    rows = []
    for item in test_items:
        iid = item['test_item_id']
        expected = item['expected_stage']
        difficulty = item['difficulty']
        text = item['text']

        raw = results_all.get(iid, {})
        rater_votes = raw.get('rater_votes', [])
        consensus = raw.get('consensus', {})

        # vote_single_label: top-level keys are 'primary_stage', 'secondary_stage',
        # 'consensus_vote' (not 'stage'/'vote')
        cons_primary = consensus.get('primary_stage')
        cons_secondary = consensus.get('secondary_stage')
        cons_cv = consensus.get('consensus_vote')  # int | 'ABSTAIN' | None (split)
        cons_pt = _pass_type(cons_primary, cons_secondary, expected)

        # Per-rater votes indexed by rater id
        votes_by_rater = {rv.get('rater', ''): rv for rv in rater_votes}

        rows.append({
            'item_id': iid,
            'expected': expected,
            'difficulty': difficulty,
            'text': text,
            'rater_votes': votes_by_rater,
            'cons_primary': cons_primary,
            'cons_secondary': cons_secondary,
            'cons_cv': cons_cv,
            'cons_pt': cons_pt,           # 'primary' | 'secondary' | None
            'cons_pass': cons_pt is not None,
        })

    # ---------------------------------------------------------------------------
    # Score helpers: count passes (primary OR secondary) and secondary-only passes
    # ---------------------------------------------------------------------------
    tiers = ['clear', 'subtle', 'adversarial']

    def _score(rows_subset, rater=None):
        """Returns (n_pass, n_secondary_only, n_total)."""
        n_pass = 0
        n_sec = 0
        total = len(rows_subset)
        for r in rows_subset:
            if rater is not None:
                rv = r['rater_votes'].get(rater, {})
                vote = rv.get('vote', 'ERROR')
                pt = _pass_type(
                    rv.get('stage') if vote == 'CODED' else None,
                    rv.get('secondary_stage') if vote == 'CODED' else None,
                    r['expected'],
                )
            else:
                pt = r['cons_pt']
            if pt == 'primary':
                n_pass += 1
            elif pt == 'secondary':
                n_pass += 1
                n_sec += 1
        return n_pass, n_sec, total

    def _cell(n_pass, n_sec, total):
        if total == 0:
            return 'N/A'
        pct = 100 * n_pass / total
        base = f'{n_pass}/{total} ({pct:.0f}%)'
        return base + (f' ~{n_sec}' if n_sec else '')

    # Pad rater ids to consistent display width
    max_rater_len = max((len(r) for r in rater_ids), default=10)
    rater_col_w = max(max_rater_len, len('CONSENSUS'))

    vdir = _paths.validation_dir(output_dir)
    os.makedirs(vdir, exist_ok=True)
    out_path = os.path.join(vdir, 'content_validity_zeroshot_results.txt')

    with open(out_path, 'w', encoding='utf-8') as fh:
        # ---- Header ----
        fh.write('=' * _W + '\n')
        fh.write('CONTENT VALIDITY ZERO-SHOT CLASSIFICATION RESULTS\n')
        fh.write('=' * _W + '\n')
        fh.write(f'Framework: {framework.name}   Version: {framework.version}   '
                 f'Themes: {len(framework.themes)}\n')
        fh.write(f'Generated: {_dt.date.today().isoformat()}\n')
        fh.write('=' * _W + '\n\n')

        # ---- Rater lineup ----
        fh.write('RATER LINEUP\n')
        fh.write('-' * _W + '\n')
        if len(rater_ids) == 1:
            fh.write(f'  Single rater: {rater_ids[0]}\n')
        else:
            for i, rid in enumerate(rater_ids, start=1):
                fh.write(f'  Run {i}: {rid}\n')
        fh.write('\n')

        # ---- Score summary table ----
        fh.write('SCORE SUMMARY\n')
        fh.write('-' * _W + '\n')
        fh.write(
            '  Scoring: [PASS] = expected stage is primary label; '
            '[PASS~] = expected stage\n'
            '  is secondary label (both count as correct). '
            '~N in a cell = secondary-match count.\n\n'
        )
        header_cols = ['Overall'] + tiers
        col_w = 16
        header_line = f'  {"Rater":{rater_col_w + 2}}'
        for h in header_cols:
            header_line += f'  {h:<{col_w}}'
        fh.write(header_line.rstrip() + '\n')
        fh.write('  ' + '-' * (rater_col_w + 2 + (col_w + 2) * len(header_cols)) + '\n')

        all_raters_for_summary = rater_ids + ['CONSENSUS']
        for rater in all_raters_for_summary:
            is_cons = (rater == 'CONSENSUS')
            label = rater if not is_cons else 'CONSENSUS'
            n_p, n_s, t = _score(rows, None if is_cons else rater)
            row_line = f'  {label:{rater_col_w + 2}}  {_cell(n_p, n_s, t):<{col_w}}'
            for tier in tiers:
                tier_rows = [r for r in rows if r['difficulty'] == tier]
                n_p2, n_s2, t2 = _score(tier_rows, None if is_cons else rater)
                row_line += f'  {_cell(n_p2, n_s2, t2):<{col_w}}'
            fh.write(row_line.rstrip() + '\n')

        fh.write('\n')
        fh.write('  By stage (consensus):\n')
        for theme in sorted(framework.themes, key=lambda t: t.theme_id):
            stage_rows = [r for r in rows if r['expected'] == theme.theme_id]
            n_p, n_s, t = _score(stage_rows, None)
            pct = f'{100*n_p/t:.0f}%' if t else '?'
            sec_note = f'  (~{n_s} secondary)' if n_s else ''
            fh.write(
                f'    Stage {theme.theme_id} {theme.short_name:<16} '
                f'{n_p}/{t} ({pct}){sec_note}\n'
            )
        fh.write('\n')

        # ---- Item-by-item ----
        fh.write('=' * _W + '\n')
        fh.write('ITEM-BY-ITEM RESULTS\n')
        fh.write('=' * _W + '\n\n')

        for row in rows:
            exp_name = _sname(row['expected'])
            if row['cons_pt'] == 'primary':
                cons_label = '[PASS]'
            elif row['cons_pt'] == 'secondary':
                cons_label = '[PASS~]'
            else:
                cons_label = '[FAIL]'

            fh.write('=' * _W + '\n')
            fh.write(
                f"[{row['item_id']}]  Tier: {row['difficulty']:<12}  "
                f"Expected: {exp_name}   Consensus: {cons_label}\n"
            )
            fh.write('-' * _W + '\n')
            for line in textwrap.wrap(
                f'"{row["text"]}"', width=_W - 2,
                initial_indent='  ', subsequent_indent='  ',
            ) or ['  ']:
                fh.write(line + '\n')
            fh.write('\n')

            for rid in rater_ids:
                rv = row['rater_votes'].get(rid, {})
                vote = rv.get('vote', 'ERROR')
                primary = rv.get('stage') if vote == 'CODED' else None
                secondary = rv.get('secondary_stage') if vote == 'CODED' else None
                conf = rv.get('confidence')
                sec_conf = rv.get('secondary_confidence')
                just = (rv.get('justification') or '').strip()

                pt = _pass_type(primary, secondary, row['expected'])
                result_tag = '[PASS]' if pt == 'primary' else ('[PASS~]' if pt == 'secondary' else '[FAIL]')

                if vote == 'CODED':
                    stage_str = _sname(primary)
                    if secondary is not None:
                        stage_str += f' / {_sname(secondary)}'
                    conf_str = f'  conf={_fconf(conf)}'
                    if secondary is not None and sec_conf is not None:
                        conf_str += f'/{_fconf(sec_conf)}'
                else:
                    stage_str = vote
                    conf_str = ''

                label_col = f'[{rid}]'
                fh.write(
                    f'  {label_col:{rater_col_w + 2}}  '
                    f'{stage_str:<22}{conf_str}  {result_tag}\n'
                )
                # Show justification when this item has any failure (rater or consensus)
                if just and (pt != 'primary' or not row['cons_pass']):
                    for line in textwrap.wrap(
                        just, width=_W - 8,
                        initial_indent='      → ', subsequent_indent='        ',
                    ):
                        fh.write(line + '\n')

            # Consensus line with primary + secondary
            if row['cons_primary'] is not None:
                cons_stage_str = _sname(row['cons_primary'])
                if row['cons_secondary'] is not None:
                    cons_stage_str += f' / {_sname(row["cons_secondary"])}'
            elif row['cons_cv'] == 'ABSTAIN':
                cons_stage_str = 'ABSTAIN'
            elif row['cons_cv'] is None:
                cons_stage_str = 'SPLIT'
            else:
                cons_stage_str = 'ERROR'

            fh.write(
                f'  {"CONSENSUS":{rater_col_w + 2}}  '
                f'{cons_stage_str:<22}              {cons_label}\n'
            )
            fh.write('\n')

    return out_path

