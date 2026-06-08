"""
classification_tools.theme_llm
------------------------------
The PRIMARY classification path: zero-shot LLM **single-label** theme
classification for any ThemeFramework (VAAMR participant stages, PURER therapist
moves), plus PURER cue-unit classification.

`llm_classifier.py` also hosts the LLM arm of the multi-label codebook
(`LLMCodebookClassifier`); the embedding arm + ensemble reconciliation live in
the sibling `classification_tools.codebook_multilabel` package.
"""
