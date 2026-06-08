"""
classification_tools.codebook_multilabel
----------------------------------------
The **multi-label** codebook classifier (VCE phenomenology codes): embedding
similarity (`embedding_classifier`) + LLM reconciliation (`ensemble`), with the
classifier configs in `config`.

This is the multi-label counterpart to the single-label theme path in
`classification_tools.theme_llm`. The codebook *definitions* it classifies into
live in `constructs.codebook` (schema + markdown loader + the phenomenology
codebook preset).
"""
