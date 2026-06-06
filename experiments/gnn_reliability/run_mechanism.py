"""
experiments/gnn_reliability/run_mechanism.py
--------------------------------------------
PRIMARY deliverable (design_decisions.md §1A / §9): therapist-cue (PURER) →
participant-VAAMR mechanism via model-counterfactual sensitivity, triangulated
against the independent observed-Δprogression analysis (analysis/mechanism.py).

For each candidate 5-class Qwen GNN (precipitates edges ON — the typed therapist→
participant handle the counterfactual needs):
  1. compute the model's OWN grouped-CV gate κ (LLM axis) → the reported TRUST CONTEXT;
  2. train the full model, run the counterfactual influence + §1A triangulation
     (Spearman ρ over (from_stage × move) cells + participant-clustered bootstrap CI);
  3. report ρ / CI / converges beside the gate κ, with the FDR-null caveat.

The progression coordinate E[stage]=Σk·pₖ is a 5-stage concept, so the mechanism
uses 5-class models (the 6-class No-code head is the classifier, not the regressor).
NO causal claims (n≈32 observational + elicitation confound).

Run:  python experiments/gnn_reliability/run_mechanism.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

OUT = os.environ.get('QRA_BATTERY_OUTPUT', 'data/Meta')
MECH_CSV = os.path.join(OUT, '03_analysis_data', 'mechanism', 'mechanism_delta_progression.csv')

QWEN = dict(embedding_backend='openai', embedding_base_url='http://10.0.0.58:1234/v1',
            embedding_model='text-embedding-qwen3-embedding-8b',
            use_query_prefix=True, embedding_batch_size=8)


def _cfg(**kw):
    from gnn_layer.config import GnnLayerConfig
    return GnnLayerConfig(**{**QWEN, 'vaamr_n_classes': 5, 'precipitates_edges': True, **kw})


# Candidate mechanism models (all 5-class + precipitates). The best-converging one is
# the reported mechanism read; all are logged.
CANDIDATES = {
    'plain':    _cfg(),
    'balanced': _cfg(vaamr_class_balance=True, vaamr_focal_gamma=2.0),
}


def _train_full(df, emb, cfg):
    from gnn_layer import graph_builder as gb, train as T, soft_labels as SL
    from gnn_layer.runner import _vocabs
    g = gb.build_graph(df, emb, cfg)
    vce = _vocabs(cfg)
    soft = SL.build_soft_targets(df, cfg.label_mode, n_stages=5)
    tg = T.assemble_targets(g, soft, cfg, df_all=df, vce_codes=vce or None)
    model, _ = T.train_model(g, tg, cfg, n_vce=len(vce))
    return model, g


def main():
    from experiments.gnn_reliability import harness as H
    from gnn_layer.influence import (run_counterfactual_experiment, triangulate,
                                     write_influence_csv, write_influence_report)

    df = H.load_corpus(OUT)
    folds = H.build_folds(df, verbose=False)
    emb = H.get_embeddings(df, 'qwen', OUT)
    print(f"Qwen embeddings n={len(emb)} dim={len(next(iter(emb.values())))}\n")

    best = None
    for name, cfg in CANDIDATES.items():
        print(f">>> mechanism candidate: {name} (precipitates=ON, 5-class)")
        # (1) this model's gate κ (trust context) under grouped CV
        oof = H.run_gnn_arm(df, emb, folds, cfg)
        gate = H.score_arm(f'MECH_{name}', oof, df, OUT, 5, write_ledger=False)
        gate_k = gate['llm_axis']['cohen_kappa_205']
        # (2) full model + counterfactual + triangulation
        model, g = _train_full(df, emb, cfg)
        res = run_counterfactual_experiment(
            model, g, df, cfg,
            gate_kappa=(round(float(gate_k), 4) if gate_k is not None else None),
            mechanism_csv=MECH_CSV)
        infl, tri = res['influence'], res['triangulation']
        print(f"    gate κ(LLM,205)={gate_k}  cue_blocks={infl.get('n_blocks')}")
        print("    per-move counterfactual influence:")
        for r in infl.get('per_move', []):
            print(f"      {r['move_name']:<16} {r['mean_influence']:+.4f} "
                  f"CI[{r['ci_lo']},{r['ci_hi']}] nblk={r['n_blocks']}")
        if tri:
            print(f"    §1A: n_cells={tri['n_cells']} ρ={tri['spearman_rho']} "
                  f"CI=[{tri['ci_lo']},{tri['ci_hi']}] excludes0={tri['ci_excludes_zero']} "
                  f"sign(FDR)={tri['sign_agreement']} n_fdr={tri['n_fdr_significant']} "
                  f"converges={tri['converges']}")
            rho = tri.get('spearman_rho')
            score = (rho if (rho is not None and tri.get('ci_excludes_zero')) else
                     (rho if rho is not None else -2))
            if best is None or (score is not None and score > best[0]):
                best = (score, name, infl, tri, gate_k)
        print()

    if best is not None:
        _, name, infl, tri, gate_k = best
        print("=" * 72)
        print(f"REPORTED MECHANISM READ: candidate '{name}'  (best ρ-convergence)")
        print(f"  Spearman ρ={tri['spearman_rho']}  CI=[{tri['ci_lo']},{tri['ci_hi']}]  "
              f"excludes0={tri['ci_excludes_zero']}  trust-context gate κ={gate_k}")
        print(f"  §1A converges (strict FDR)={tri['converges']}  "
              f"(FDR-significant cells={tri['n_fdr_significant']} → see §9A adapted read)")
        print("=" * 72)
        # persist the reported read's artifacts
        write_influence_csv(infl, OUT)
        write_influence_report(infl, triangulate(infl, OUT), OUT)
        print("wrote 06_reports/06_gnn/influence.txt + 03_analysis_data/gnn/gnn_counterfactual_influence.csv")


if __name__ == '__main__':
    main()
