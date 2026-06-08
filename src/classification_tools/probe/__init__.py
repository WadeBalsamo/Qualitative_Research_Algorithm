"""
classification_tools.probe
--------------------------
The LLM-free, gated, abstention-aware **single-label** VAAMR scaler: a per-rater
ensemble of L2-regularized logistic-regression probes over Qwen embeddings
(methodology Section 8.6). Ranks BELOW the LLM label of record — fills unlabeled
participant segments and abstains where unsure.
"""
