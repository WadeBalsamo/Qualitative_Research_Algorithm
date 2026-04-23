"""
analysis/construct.py
---------------------
Per-construct reports: one per VA-MR stage and one per codebook code.

Stage reports show prevalence trends, per-participant breakdowns, and
the most prototypical exemplars. Codebook code reports show co-occurrence
with VA-MR stages and longitudinal prevalence.
"""

import json
import os
import re

import pandas as pd

from .exemplars import select_prototypical_exemplars, format_exemplar
from .loader import sort_session_ids


def _compute_lift(n_code_in_stage: int, n_stage: int, n_code: int, n_total: int) -> float:
    """Compute lift = observed co-occurrence rate / expected by independence.

    lift = (n_code_in_stage / n_stage) / (n_code / n_total)
    Returns 0.0 if any denominator is zero.
    """
    if n_stage == 0 or n_total == 0 or n_code == 0:
        return 0.0
    observed = n_code_in_stage / n_stage
    expected = n_code / n_total
    return round(observed / expected, 4)


def _longitudinal_slope(values_by_session: dict, session_order: list) -> float:
    """Linear slope of a per-session metric across ordered sessions."""
    x, y = [], []
    for i, sid in enumerate(session_order):
        if str(sid) in values_by_session:
            x.append(i)
            y.append(values_by_session[str(sid)])
    if len(x) < 2:
        return 0.0
    try:
        from scipy.stats import linregress
        slope, *_ = linregress(x, y)
        return round(float(slope), 4)
    except ImportError:
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den = sum((xi - mx) ** 2 for xi in x)
        return round(num / den, 4) if den != 0 else 0.0


def generate_stage_report(
    df: pd.DataFrame,
    stage_id: int,
    framework: dict,
    output_dir: str,
) -> dict:
    """Generate a construct report for a single VA-MR stage.

    Parameters
    ----------
    df : DataFrame
        Full participant dataset.
    stage_id : int
        VA-MR stage ID (0–3).
    framework : dict
        From loader.load_framework().
    output_dir : str
        Root pipeline output directory.

    Returns
    -------
    Report dict (also written to reports/analysis/constructs/stage_{id}_{name}.json).
    """
    stage_info = framework.get(stage_id, {'name': f'Stage {stage_id}', 'short_name': f'Stage {stage_id}'})
    stage_name_slug = re.sub(r'[^a-z0-9]+', '_', stage_info.get('short_name', '').lower()).strip('_')

    n_total = len(df)
    n_stage = int((df['final_label'] == stage_id).sum())
    overall_prevalence = round(n_stage / n_total, 4) if n_total > 0 else 0.0

    # By session number
    session_ids = sort_session_ids(df['session_id'].unique().tolist())
    prev_by_session = {}
    for sid in session_ids:
        sdf = df[df['session_id'] == sid]
        n_s = len(sdf)
        cnt = int((sdf['final_label'] == stage_id).sum())
        snum = str(int(sdf['session_number'].iloc[0])) if n_s > 0 else str(sid)
        prev_by_session[snum] = round(cnt / n_s, 4) if n_s > 0 else 0.0

    longitudinal_trend = _longitudinal_slope(
        prev_by_session,
        [str(int(df[df['session_id'] == s]['session_number'].iloc[0]))
         for s in session_ids if len(df[df['session_id'] == s]) > 0],
    )

    # By participant
    prev_by_participant = {}
    for pid in sorted(df['participant_id'].unique()):
        pdf = df[df['participant_id'] == pid]
        n_p = len(pdf)
        cnt = int((pdf['final_label'] == stage_id).sum())
        prev_by_participant[pid] = round(cnt / n_p, 4) if n_p > 0 else 0.0

    # By cohort
    prev_by_cohort = {}
    if 'cohort_id' in df.columns:
        for cid in sorted(df['cohort_id'].dropna().unique()):
            cdf = df[df['cohort_id'] == cid]
            n_c = len(cdf)
            cnt = int((cdf['final_label'] == stage_id).sum())
            prev_by_cohort[str(int(cid))] = round(cnt / n_c, 4) if n_c > 0 else 0.0

    # Top 5 exemplars overall
    stage_df = df[df['final_label'] == stage_id]
    ex_rows = select_prototypical_exemplars(stage_df, n=5)
    top_exemplars = [format_exemplar(ex_rows.iloc[i]) for i in range(len(ex_rows))]

    # Co-occurring codebook codes with lift
    co_occurring_codes = []
    if 'codebook_labels_ensemble' in stage_df.columns:
        exploded_stage = stage_df.explode('codebook_labels_ensemble')
        exploded_stage = exploded_stage[
            exploded_stage['codebook_labels_ensemble'].notna() &
            (exploded_stage['codebook_labels_ensemble'] != '')
        ]
        if not exploded_stage.empty:
            # Global code counts for lift denominator
            all_exploded = df.explode('codebook_labels_ensemble')
            all_exploded = all_exploded[
                all_exploded['codebook_labels_ensemble'].notna() &
                (all_exploded['codebook_labels_ensemble'] != '')
            ]
            global_code_counts = all_exploded['codebook_labels_ensemble'].value_counts().to_dict()

            stage_code_counts = exploded_stage['codebook_labels_ensemble'].value_counts()
            for code_id, cnt in stage_code_counts.head(15).items():
                n_code = global_code_counts.get(code_id, 0)
                lift = _compute_lift(int(cnt), n_stage, n_code, n_total)
                co_occurring_codes.append({
                    'code_id': str(code_id),
                    'count': int(cnt),
                    'lift': lift,
                })
            co_occurring_codes.sort(key=lambda x: x['lift'], reverse=True)
            co_occurring_codes = co_occurring_codes[:10]

    report = {
        'stage_id': stage_id,
        'stage_name': stage_info.get('name', ''),
        'stage_short_name': stage_info.get('short_name', ''),
        'overall_prevalence': overall_prevalence,
        'n_segments_total': n_total,
        'n_segments_this_stage': n_stage,
        'prevalence_by_session_number': prev_by_session,
        'prevalence_by_participant': prev_by_participant,
        'prevalence_by_cohort': prev_by_cohort,
        'longitudinal_trend': longitudinal_trend,
        'top_exemplars': top_exemplars,
        'co_occurring_codes': co_occurring_codes,
    }

    out_dir = os.path.join(output_dir, 'reports', 'analysis', 'constructs', 'json')
    os.makedirs(out_dir, exist_ok=True)
    fname = f'stage_{stage_id}_{stage_name_slug}.json'
    with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def generate_codebook_code_report(
    df: pd.DataFrame,
    code_id: str,
    framework: dict,
    output_dir: str,
) -> dict:
    """Generate a construct report for a single codebook code.

    Parameters
    ----------
    df : DataFrame
        Full participant dataset.
    code_id : str
        Codebook code identifier.
    framework : dict
        From loader.load_framework().
    output_dir : str
        Root pipeline output directory.

    Returns
    -------
    Report dict (also written to reports/analysis/constructs/codebook_{code_id}.json).
    """
    # Rows that contain this code
    code_df = df[df['codebook_labels_ensemble'].apply(
        lambda codes: code_id in (codes or [])
    )].copy()

    n_total = len(df)
    n_coded = len(code_df)
    overall_prevalence = round(n_coded / n_total, 4) if n_total > 0 else 0.0

    # By session number
    session_ids = sort_session_ids(df['session_id'].unique().tolist())
    prev_by_session = {}
    for sid in session_ids:
        sdf = df[df['session_id'] == sid]
        n_s = len(sdf)
        cnt = sdf['codebook_labels_ensemble'].apply(
            lambda codes: code_id in (codes or [])
        ).sum()
        snum = str(int(sdf['session_number'].iloc[0])) if n_s > 0 else str(sid)
        prev_by_session[snum] = round(int(cnt) / n_s, 4) if n_s > 0 else 0.0

    # Stage co-occurrence with lift
    stage_ids = sorted(framework.keys())
    stage_co_occurrence = {}
    for st in stage_ids:
        st_df = df[df['final_label'] == st]
        n_st = len(st_df)
        cnt_code_in_stage = st_df['codebook_labels_ensemble'].apply(
            lambda codes: code_id in (codes or [])
        ).sum()
        lift = _compute_lift(int(cnt_code_in_stage), n_st, n_coded, n_total)
        stage_co_occurrence[str(st)] = {
            'stage_name': framework.get(st, {}).get('short_name', f'Stage {st}'),
            'count': int(cnt_code_in_stage),
            'lift': lift,
        }

    # Top 3 exemplars
    ex_rows = select_prototypical_exemplars(code_df, n=3)
    top_exemplars = [format_exemplar(ex_rows.iloc[i]) for i in range(len(ex_rows))]

    report = {
        'code_id': code_id,
        'overall_prevalence': overall_prevalence,
        'n_segments_total': n_total,
        'n_segments_coded': n_coded,
        'prevalence_by_session_number': prev_by_session,
        'stage_co_occurrence': stage_co_occurrence,
        'top_exemplars': top_exemplars,
    }

    out_dir = os.path.join(output_dir, 'reports', 'analysis', 'constructs', 'json')
    os.makedirs(out_dir, exist_ok=True)
    safe_code_id = re.sub(r'[^a-z0-9_\-]', '_', code_id.lower())
    fname = f'codebook_{safe_code_id}.json'
    with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def generate_all_construct_reports(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> list:
    """Generate all stage reports and all codebook code reports.

    Returns list of stage report dicts.
    Skips codebook code reports if no codebook_labels_ensemble data is present.
    """
    stage_reports = []
    # VA-MR stage reports
    for stage_id in sorted(framework.keys()):
        try:
            report = generate_stage_report(df, stage_id, framework, output_dir)
            stage_reports.append(report)
        except Exception as e:
            print(f"  Warning: stage report failed for stage {stage_id}: {e}")

    # Codebook code reports — only if data present
    if 'codebook_labels_ensemble' not in df.columns:
        return stage_reports

    # Collect all unique code IDs
    all_codes = set()
    for codes in df['codebook_labels_ensemble']:
        if isinstance(codes, list):
            all_codes.update(codes)

    if not all_codes:
        return stage_reports

    for code_id in sorted(all_codes):
        if not code_id:
            continue
        try:
            generate_codebook_code_report(df, code_id, framework, output_dir)
        except Exception as e:
            print(f"  Warning: codebook report failed for {code_id}: {e}")

    return stage_reports


def generate_codebook_text_report(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
):
    """
    Produce a human-readable codebook exemplar report at
    reports/analysis/constructs/codebook_exemplars.txt.

    Computes all statistics and exemplars directly from ``df`` (the master
    segment DataFrame) — not from the pre-written per-code JSON files.
    Code definitions (description, subcodes, criteria) are loaded from
    meta/codebook_definitions.json when available.

    Returns the output path, or None if no codebook labels are present.
    """
    import datetime as _dt
    import textwrap

    if 'codebook_labels_ensemble' not in df.columns:
        return None

    # Collect all code IDs that appear in the data
    seen_codes: set = set()
    for codes in df['codebook_labels_ensemble']:
        if isinstance(codes, list):
            seen_codes.update(c for c in codes if c)
    if not seen_codes:
        return None

    n_total = len(df)

    # Load codebook definitions from meta/
    definitions: dict = {}   # code_id -> raw dict
    cb_meta: dict = {}
    for candidate in (
        os.path.join(output_dir, 'meta', 'codebook_definitions.json'),
        os.path.join(output_dir, 'codebook_definitions.json'),
    ):
        if os.path.exists(candidate):
            with open(candidate, encoding='utf-8') as f:
                cb_meta = json.load(f)
            for c in cb_meta.get('codes', []):
                definitions[c['code_id']] = c
            break

    # Determine domain ordering from definitions; fall back to alphabetical
    domains_ordered: list = []
    domain_to_codes: dict = {}
    if definitions:
        for c in cb_meta.get('codes', []):
            cid = c['code_id']
            if cid not in seen_codes:
                continue   # skip codes not present in dataset
            d = c.get('domain', 'Other')
            if d not in domain_to_codes:
                domains_ordered.append(d)
                domain_to_codes[d] = []
            domain_to_codes[d].append(cid)
    else:
        domain_to_codes['All Codes'] = sorted(seen_codes)
        domains_ordered = ['All Codes']

    # Pre-compute per-code DataFrames and lift against all stages
    stage_ids = sorted(framework.keys())

    constructs_dir = os.path.join(output_dir, 'reports', 'analysis', 'constructs')
    os.makedirs(constructs_dir, exist_ok=True)
    out_path = os.path.join(constructs_dir, 'codebook_exemplars.txt')

    def _wrap(text, width=72, indent='  ', subsequent='  ') -> list:
        return textwrap.wrap(str(text or ''), width=width,
                             initial_indent=indent,
                             subsequent_indent=subsequent) or [indent]

    with open(out_path, 'w', encoding='utf-8') as fh:
        cb_name = cb_meta.get('name', 'Codebook')
        cb_ver  = cb_meta.get('version', '')
        cb_desc = cb_meta.get('description', '')
        n_codes_with_data = len(seen_codes)
        n_total_defs = len(definitions) if definitions else n_codes_with_data

        fh.write("=" * 78 + "\n")
        fh.write("CODEBOOK EXEMPLARS\n")
        fh.write(f"{cb_name}  v{cb_ver}\n" if cb_ver else f"{cb_name}\n")
        if cb_desc:
            for line in _wrap(cb_desc, width=76, indent='', subsequent=''):
                fh.write(line + "\n")
        fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}\n")
        fh.write(
            f"Codebook codes: {n_total_defs}  |  "
            f"Codes present in dataset: {n_codes_with_data}  |  "
            f"Total segments: {n_total}\n"
        )
        fh.write("=" * 78 + "\n\n")

        for domain in domains_ordered:
            code_ids = domain_to_codes.get(domain, [])
            if not code_ids:
                continue
            fh.write(f"DOMAIN: {domain}  ({len(code_ids)} codes)\n")
            fh.write("=" * 78 + "\n\n")

            for code_id in code_ids:
                defn = definitions.get(code_id, {})
                category = defn.get('category', code_id)

                fh.write(f"[{code_id}]\n")
                fh.write(f"  {category}\n")
                fh.write("-" * 78 + "\n")

                desc = defn.get('description', '')
                if desc:
                    for line in _wrap(desc, width=74, indent='  Description:   ',
                                      subsequent='                 '):
                        fh.write(line + "\n")

                subcodes = defn.get('subcodes', [])
                if subcodes:
                    sc_str = ', '.join(subcodes)
                    for line in _wrap(sc_str, width=74, indent='  Subcodes:      ',
                                      subsequent='                 '):
                        fh.write(line + "\n")

                incl = defn.get('inclusive_criteria', '')
                if incl:
                    for line in _wrap(incl, width=74, indent='  Include when:  ',
                                      subsequent='                 '):
                        fh.write(line + "\n")

                excl = defn.get('exclusive_criteria', '')
                if excl:
                    for line in _wrap(excl, width=74, indent='  Exclude when:  ',
                                      subsequent='                 '):
                        fh.write(line + "\n")

                exemplar_utts = defn.get('exemplar_utterances', [])
                if exemplar_utts:
                    fh.write("  Canonical exemplars:\n")
                    for utt in exemplar_utts:
                        for line in _wrap(utt, width=72, indent='    • ',
                                          subsequent='      '):
                            fh.write(line + "\n")

                # ---- Compute stats directly from df ----
                code_mask = df['codebook_labels_ensemble'].apply(
                    lambda codes: code_id in (codes or [])
                )
                code_df = df[code_mask].copy()
                n_coded = len(code_df)
                pct = round(100.0 * n_coded / n_total, 1) if n_total > 0 else 0.0

                fh.write(f"\n  DATASET STATS  (N={n_coded} of {n_total} segments, {pct}%)\n")

                # Stage co-occurrence by lift
                if stage_ids and n_coded > 0:
                    lift_rows = []
                    for st in stage_ids:
                        st_name = framework.get(st, {}).get('short_name', f'Stage {st}')
                        n_st = int((df['final_label'] == st).sum())
                        n_code_in_st = int(
                            (code_df['final_label'] == st).sum()
                        )
                        lift = _compute_lift(n_code_in_st, n_st, n_coded, n_total)
                        if n_code_in_st > 0:
                            lift_rows.append((st_name, n_code_in_st, lift))
                    lift_rows.sort(key=lambda x: x[2], reverse=True)
                    if lift_rows:
                        parts = [f"{name} ×{lift:.2f} (n={cnt})"
                                 for name, cnt, lift in lift_rows]
                        for line in _wrap(', '.join(parts), width=72,
                                          indent='    Stage co-occurrence: ',
                                          subsequent='                        '):
                            fh.write(line + "\n")

                # Exemplars directly from df using the same selector used elsewhere
                ex_rows = select_prototypical_exemplars(code_df, n=5)
                if not ex_rows.empty:
                    fh.write("  Dataset exemplars:\n")
                    for i in range(len(ex_rows)):
                        ex = ex_rows.iloc[i]
                        sid  = str(ex.get('session_id', '?'))
                        pid  = str(ex.get('participant_id', '?'))
                        conf = ex.get('llm_confidence_primary')
                        cons = ex.get('llm_run_consistency')
                        conf_str = f"conf={conf:.2f}" if isinstance(conf, (int, float)) else ''
                        cons_str = f"consistency={int(cons)}" if pd.notna(cons) else ''
                        meta = '  '.join(filter(None, [conf_str, cons_str]))
                        fh.write(f"    [{i+1}]  {sid}  {pid}  {meta}\n")
                        text = str(ex.get('text', ''))
                        for tl in textwrap.wrap(text, width=70,
                                                initial_indent='         ',
                                                subsequent_indent='         '):
                            fh.write(tl + "\n")
                        just = str(ex.get('llm_justification') or '').strip()
                        if just:
                            for line in _wrap(just, width=70,
                                              indent='         → ',
                                              subsequent='           '):
                                fh.write(line + "\n")
                else:
                    fh.write("  (No high-confidence exemplars found for this code)\n")

                fh.write("\n")

            fh.write("\n")

    return out_path
