"""
experiments/gnn_reliability/run_mechanism.py — RETIRED (superseded)
===================================================================
This harness ran the FIRST mechanism instrument: a model-counterfactual "influence"
read on a per-segment 5-class GNN classifier (the former ``gnn_layer/influence.py``).
For each (from_stage x move) cell it swapped the therapist node's feature with each
PURER-move centroid, re-ran the forward pass, and triangulated the predicted shift
against the independent observed-Delta-progression ranking (``analysis/mechanism.py``).

WHY IT WAS RETIRED (pilot, n~=32 — see ``gnn_reliability/RESULTS.md`` ("mechanism") and
``docs/methodology.md`` Section 8.5, Track B). The per-segment classifier-counterfactual
was mis-specified for a *process* question three ways: (1) kNN-similarity edges are
content noise on a transition question; (2) the model was never trained on transitions;
(3) the swapped cue reached the participant node through a single diluted edge
(~0.03 shift). On the pilot it *inverted* the observed ranking (Spearman rho = -0.13).

REPLACEMENT (current, default-on at ``qra analyze``): the dyadic FROM->CUE->TO
transition model ``src/gnn_layer/transition.py`` —
    TO_mixture ~= f(FROM_mixture, FROM_stage, pooled raw-Qwen cue)
with NO kNN and FROM-stage conditioning. Its learned counterfactual triangulates
POSITIVELY with the observed ranking (Spearman rho ~= +0.34, versus the retired
classifier-counterfactual's -0.13) and ships with the confound-localization map
(``src/gnn_layer/confound.py``). ``gnn_layer/influence.py`` and the old run_mechanism
body were removed in the GNN repositioning (classifier -> default OFF).

This file is kept as a CATALOG RECORD of the retired approach; it no longer imports the
deleted ``gnn_layer.influence`` module. To reproduce the *current* mechanism read:

    qra analyze -o ./data/Meta   # writes 06_reports/06_gnn/{transition_model,confound_localization}.txt
"""

import sys


def main():
    print(__doc__)
    print("RETIRED harness — run `qra analyze` for the current transition-model "
          "mechanism read (gnn_layer/transition.py + confound.py).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
