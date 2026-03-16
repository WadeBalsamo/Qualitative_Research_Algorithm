"""
pipeline_hooks.py
-----------------
Observer interface for pipeline events.

Three implementations:
- PipelineObserver: base class with no-op methods
- SilentObserver: minimal output (stage headers + summaries)
- GuidedModeObserver: verbose educational narration with interactive prompts
"""


class PipelineObserver:
    """Base observer with no-op methods for all pipeline events."""

    def on_stage_start(self, stage_name: str, stage_number: str, **kwargs):
        """Called when a pipeline stage begins."""

    def on_stage_progress(self, stage_name: str, message: str, **kwargs):
        """Called for incremental progress updates within a stage."""

    def on_stage_complete(self, stage_name: str, summary: str, **kwargs):
        """Called when a pipeline stage finishes."""

    def on_pipeline_complete(self, output_dir: str, **kwargs):
        """Called when the entire pipeline finishes."""


class SilentObserver(PipelineObserver):
    """Minimal output: stage headers and summaries for autonomous mode."""

    def on_stage_start(self, stage_name: str, stage_number: str, **kwargs):
        print(f"\n{'=' * 60}")
        print(f"STAGE {stage_number}: {stage_name}")
        print(f"{'=' * 60}")

    def on_stage_progress(self, stage_name: str, message: str, **kwargs):
        print(f"  {message}")

    def on_stage_complete(self, stage_name: str, summary: str, **kwargs):
        print(f"  {summary}")

    def on_pipeline_complete(self, output_dir: str, **kwargs):
        print(f"\n{'=' * 60}")
        print("PIPELINE COMPLETE")
        print(f"{'=' * 60}")
        print(f"All outputs in: {output_dir}")


STAGE_EXPLANATIONS = {
    'ingestion': (
        "This stage reads diarized therapy transcripts and segments them into\n"
        "analyzable units. Each transcript is split using semantic similarity\n"
        "(via the '{embedding_model}' embedding model) so that each segment\n"
        "captures a coherent thought or exchange.\n"
        "\n"
        "Segmentation parameters:\n"
        "  - Min words per segment: {min_words}\n"
        "  - Max words per segment: {max_words}\n"
        "  - Silence threshold: {silence_ms}ms\n"
        "  - Semantic shift percentile: {shift_pct}th"
    ),
    'operationalization': (
        "The framework defines the theoretical constructs used for classification.\n"
        "Each theme has an operational definition, prototypical features,\n"
        "distinguishing criteria, and exemplar utterances.\n"
        "\n"
        "Framework: {framework_name} ({num_themes} themes)\n"
        "Themes: {theme_list}\n"
        "\n"
        "This stage exports the theme definitions and a content validity test\n"
        "set that can be used to verify the LLM understands the constructs."
    ),
    'theme_classification': (
        "Each segment is classified by an LLM using zero-shot prompting.\n"
        "The model reads the segment text and the theme definitions, then\n"
        "assigns a primary theme label.\n"
        "\n"
        "To ensure reliability, classification is run {n_runs} times per segment.\n"
        "Consistency across runs and confidence scores are tracked.\n"
        "\n"
        "Model: {model}\n"
        "Backend: {backend}"
    ),
    'response_parsing': (
        "LLM outputs are parsed from free-text responses into structured labels.\n"
        "The parser uses an 8-category error taxonomy to handle edge cases\n"
        "(e.g., the LLM returning a theme name instead of an ID, or providing\n"
        "multiple labels when only one was requested)."
    ),
    'codebook_classification': (
        "Codebook classification assigns multiple fine-grained codes to each\n"
        "segment using two independent methods:\n"
        "\n"
        "  1. Embedding-based: Computes semantic similarity between the segment\n"
        "     and each code's definition/exemplars using transformer embeddings.\n"
        "     Uses a triple-veto scoring system (cosine, Euclidean, tertiary).\n"
        "\n"
        "  2. LLM-based: The language model reads the segment and codebook,\n"
        "     then selects applicable codes with confidence scores.\n"
        "\n"
        "An ensemble step reconciles the two methods, flagging disagreements\n"
        "for human review when the methods assign different codes."
    ),
    'cross_validation': (
        "Cross-validation checks whether the observed co-occurrence of themes\n"
        "and codebook codes matches the hypothesized mapping.\n"
        "\n"
        "For each theme, the pipeline computes:\n"
        "  - Rate: How often each code appears with this theme\n"
        "  - Base rate: How often the code appears across all themes\n"
        "  - Lift: Rate / base rate (>1 means stronger-than-chance association)\n"
        "\n"
        "Codes with lift >= {min_lift} confirm the hypothesis.\n"
        "Unexpected strong associations are flagged for investigation."
    ),
    'human_validation_set': (
        "A balanced evaluation set is created for human coders to independently\n"
        "validate a sample of the LLM's classifications.\n"
        "\n"
        "The set includes {n_per_class} segments per theme, sampled to ensure\n"
        "each theme is equally represented. This enables computing inter-rater\n"
        "reliability metrics like Cohen's kappa."
    ),
    'dataset_assembly': (
        "The final stage assembles all results into a master dataset:\n"
        "\n"
        "  - master_segments.jsonl: Every segment with all labels and metadata\n"
        "  - session_adjacency.jsonl: Tracks which themes appear next to each\n"
        "    other within sessions (for sequential pattern analysis)\n"
        "  - session_stage_progression.csv: Forward/backward transition counts\n"
        "    per session (do participants move through stages linearly?)\n"
        "\n"
        "Each segment receives a confidence tier: high, medium, or low,\n"
        "based on consistency ({high_consistency}/3 runs) and confidence\n"
        "thresholds ({high_confidence} for high, {medium_confidence} for medium)."
    ),
}


class GuidedModeObserver(PipelineObserver):
    """Verbose educational narration with interactive prompts at each stage."""

    def __init__(self, config=None, framework=None):
        self.config = config
        self.framework = framework

    def on_stage_start(self, stage_name: str, stage_number: str, **kwargs):
        print(f"\n{'=' * 70}")
        print(f"  STAGE {stage_number}: {stage_name}")
        print(f"{'=' * 70}")

        explanation_key = kwargs.get('explanation_key', stage_name.lower().replace(' ', '_'))
        template = STAGE_EXPLANATIONS.get(explanation_key, '')
        if template:
            context = self._build_template_context()
            context.update(kwargs)
            try:
                explanation = template.format(**context)
            except KeyError:
                explanation = template
            print(f"\n{explanation}\n")

        input("  Press Enter to continue...")

    def on_stage_progress(self, stage_name: str, message: str, **kwargs):
        print(f"  {message}")

    def on_stage_complete(self, stage_name: str, summary: str, **kwargs):
        print(f"\n  Result: {summary}")
        print()

    def on_pipeline_complete(self, output_dir: str, **kwargs):
        print(f"\n{'=' * 70}")
        print("  PIPELINE COMPLETE")
        print(f"{'=' * 70}")
        print(f"\n  All outputs have been written to: {output_dir}")
        print()
        print("  Next steps:")
        print("    1. Review the master_segments JSONL file for your classified data")
        print("    2. Complete human coding of the evaluation set")
        print("    3. Run validation report generation")
        print("    4. Feed master_segments to downstream analysis tools")
        print()

    def _build_template_context(self) -> dict:
        """Build template context from config and framework."""
        ctx = {}
        if self.config:
            seg = self.config.segmentation
            ctx.update({
                'embedding_model': seg.embedding_model,
                'min_words': seg.min_segment_words,
                'max_words': seg.max_segment_words,
                'silence_ms': seg.silence_threshold_ms,
                'shift_pct': seg.semantic_shift_percentile,
                'model': self.config.theme_classification.model,
                'backend': self.config.theme_classification.backend,
                'n_runs': self.config.theme_classification.n_runs,
                'n_per_class': self.config.validation.n_per_class,
                'high_consistency': self.config.confidence_tiers.high_consistency,
                'high_confidence': self.config.confidence_tiers.high_confidence,
                'medium_confidence': self.config.confidence_tiers.medium_min_confidence,
                'min_lift': 1.5,
            })
        if self.framework:
            ctx.update({
                'framework_name': self.framework.name,
                'num_themes': self.framework.num_themes,
                'theme_list': ', '.join(
                    t.short_name for t in self.framework.themes
                ),
            })
        return ctx
