"""GNN consensus-distillation classifier (GraphSAGE over a content-similarity graph).

A SEPARATE CONCERN from the discovery/mechanism work-streams in gnn_layer/. Default OFF
(GnnLayerConfig.gnn_classifier_enabled): the pilot refuted its scaler role (H5 — grouped-CV
kappa 0.05-0.14, below human-human; a probe ties/beats it; see docs/graph_experiments.md
and methodology Section 8.5). Kept as the documented reliability-gate / distillation
instrument and re-adjudicable at Cohorts 3-4 scale.

This is the GNN member of QRA's classifier family (alongside the SINGLE-LABEL
classification_tools/theme_llm and probe, and the MULTI-LABEL
classification_tools/codebook_multilabel). It lives here rather than under
classification_tools/ because it shares the gnn_layer embedding/graph substrate
(embeddings, soft_labels, cue_features, figures, reports)."""
