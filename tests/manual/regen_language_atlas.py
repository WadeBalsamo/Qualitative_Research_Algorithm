"""Manual driver: re-run GNN discovery (persists motif exemplars) + regenerate
the language atlas for a project, WITHOUT running the full `qra analyze`.

Usage:  python tests/manual/regen_language_atlas.py <output_dir> [--skip-gnn]
"""

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, 'data', 'MMORE_Processed')
    skip_gnn = '--skip-gnn' in sys.argv

    from analysis.loader import load_segments, load_framework
    from process import output_paths as _paths

    df = load_segments(out, speaker_filter='participant', require_labeled=True)
    df_all = load_segments(out, speaker_filter=None, require_labeled=False)
    framework = load_framework(out)
    print(f"loaded {len(df_all)} segments ({len(df)} labeled participant)")

    pcfg = None
    cfg_path = os.path.join(_paths.meta_dir(out), 'qra_config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding='utf-8') as f:
                raw = json.load(f)
            from process.setup_wizard import build_config_from_wizard_data
            pcfg = build_config_from_wizard_data(raw)
        except Exception as e:
            print(f"config load failed ({e}); using GNN defaults")

    # Attach stage mixtures exactly as analysis/runner.py does before mechanism.
    from analysis.superposition import attach_superposition
    sup_cfg = getattr(pcfg, 'superposition', None) if pcfg is not None else None
    attach_superposition(df, out, config=sup_cfg, n_stages=len(framework) or 5)
    attach_superposition(df_all, out, config=sup_cfg, n_stages=len(framework) or 5)
    print(f"mixture coverage: {df_all['mixture'].notna().sum()}/{len(df_all)}")

    if not skip_gnn:
        from gnn_layer.runner import run_gnn_analysis
        gcfg = getattr(pcfg, 'gnn_layer', None) if pcfg is not None else None
        mech_cfg = getattr(pcfg, 'mechanism', None) if pcfg is not None else None
        within = bool(getattr(mech_cfg, 'cue_within_participant_only', False))
        res = run_gnn_analysis(df_all, out, framework=framework, config=gcfg,
                               cue_within_participant_only=within)
        print(f"GNN status: {res.get('status')}; {len(res.get('files_written', []))} files")

    from analysis.mechanism import run_mechanism_analysis
    mech_cfg = getattr(pcfg, 'mechanism', None) if pcfg is not None else None
    mres = run_mechanism_analysis(df, df_all, out, framework, config=mech_cfg)
    print(f"mechanism: {mres.get('n_blocks')} blocks")

    # LLM client for long-cue summarization (graceful None if unconfigured/offline).
    llm_client = None
    try:
        from analysis.runner import _load_analysis_context
        llm_client = _load_analysis_context(out, None, print).llm_client
    except Exception as e:
        print(f"no LLM client for cue summaries ({e}); long cues stay verbatim")
    print(f"LLM client: {'yes' if llm_client is not None else 'no'}")

    from analysis.reports.language_atlas import generate_language_atlas
    path = generate_language_atlas(df, df_all, framework, out, llm_client=llm_client)
    print(f"atlas: {path}")


if __name__ == '__main__':
    main()
