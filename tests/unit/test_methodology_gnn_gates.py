"""
tests/unit/test_methodology_gnn_gates.py
-----------------------------------------
Conceptual conformance: the validation-gated scaling story and GNN defaults.

Methodology references:
  §5.3 / §8.5 — GNN is a distilled student of the LLM consensus.
                Before it labels new segments independently, it must pass an
                out-of-sample reliability gate (κ ≥ irr_target=0.70) without
                collapsing rare stages (Reappraisal especially).
  §H3a / Capability D — VCE excluded by default (include_vce_nodes=False):
                VCE must earn its place; the ablation test is opt-in.
  §H1, §6.3 — Avoidance barrier: barrier_from=1 (Avoidance) → barrier_to=2
                (Attention Regulation), the rate-limiting transition.

Tests:
  1. GnnLayerConfig defaults: gnn_authoritative=False, include_vce_nodes=False,
     irr_target=0.70, produce_consensus_labels=True.
  2. PipelineConfig: run_codebook_classifier=False by default (VCE optional).
  3. gnn_layer/validation.py: RARE_STAGES contains stage IDs 3 and 4
     (Metacognition and Reappraisal — the over-smoothing risk group; the task
     description says "targets Reappraisal"; the code covers both 3 and 4;
     we assert what the code actually has, as the task instructs).
  4. write_validation_report: the report includes a clear LLM-FREE SCALING?
     YES/NO verdict line and uses irr_target=0.70 in the body.
  5. EfficacyConfig: barrier_from=1 (Avoidance) and barrier_to=2
     (Attention Regulation) — the documented avoidance barrier (§H1, §6.3).
"""

import os
import sys
import tempfile
import shutil
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from gnn_layer.config import GnnLayerConfig
from gnn_layer.classifier.validation import RARE_STAGES, RARE_STAGE_FLOOR, write_validation_report
from process.config import PipelineConfig, EfficacyConfig


# ---------------------------------------------------------------------------
# 1. GnnLayerConfig defaults
# ---------------------------------------------------------------------------

class TestGnnLayerConfigDefaults(unittest.TestCase):
    """
    §5.3 / §8.5: the default GNN configuration is conservative — it computes
    graph labels but does NOT make them authoritative, does NOT include VCE
    nodes (VCE must earn its place, §H3a), and targets κ ≥ 0.70.
    """

    def setUp(self):
        self.cfg = GnnLayerConfig()

    def test_gnn_authoritative_defaults_to_false(self):
        """
        §8.5: gnn_consensus tier is engaged ONLY when gnn_authoritative=True.
        Default must be False — analyst explicitly flips this after reviewing
        the per-stage reliability gate.
        """
        self.assertFalse(self.cfg.gnn_authoritative,
                         "gnn_authoritative must default to False (§8.5)")

    def test_include_vce_nodes_defaults_to_false(self):
        """
        §H3a / Capability D: VCE is excluded by default; it re-enters only when
        the ablation is explicitly requested (include_vce_nodes=True +
        run_gnn_ablation). VCE must earn its place empirically.
        """
        self.assertFalse(self.cfg.include_vce_nodes,
                         "include_vce_nodes must default to False (VCE excluded by default, §H3a)")

    def test_irr_target_defaults_to_070(self):
        """
        §5.3 / H5: the reliability gate threshold is κ ≥ 0.70 (irr_target).
        This matches the Text Psychometrics standard for inter-rater reliability.
        """
        self.assertAlmostEqual(self.cfg.irr_target, 0.70,
                               msg="irr_target must default to 0.70 (§5.3)")

    def test_produce_consensus_labels_defaults_to_true(self):
        """
        The GNN writes per-segment labels even in non-authoritative mode — the
        labels are stored alongside LLM runs for auditability and comparison.
        """
        self.assertTrue(self.cfg.produce_consensus_labels,
                        "produce_consensus_labels must default to True")


# ---------------------------------------------------------------------------
# 2. PipelineConfig: VCE optional by default
# ---------------------------------------------------------------------------

class TestPipelineConfigVCEDefault(unittest.TestCase):
    """
    §H3a: VCE classification (run_codebook_classifier) is OFF by default.
    The pipeline is correct without VCE; VCE is an optional research probe.
    """

    def test_run_codebook_classifier_defaults_to_false(self):
        cfg = PipelineConfig()
        self.assertFalse(cfg.run_codebook_classifier,
                         "run_codebook_classifier must default to False — "
                         "VCE is optional and must earn its place (§H3a)")


# ---------------------------------------------------------------------------
# 3. RARE_STAGES: the over-smoothing safeguard
# ---------------------------------------------------------------------------

class TestRareStagesSafeguard(unittest.TestCase):
    """
    §5.3: the per-stage breakdown is "what catches rare-but-critical Reappraisal
    erosion that an aggregate κ would hide." The code watches both Metacognition
    (3) and Reappraisal (4) as RARE_STAGES — asserted from the actual constant.

    The task description says "targets Reappraisal (read RARE_STAGES)": we
    assert what the code actually contains, which is (3, 4).
    """

    def test_rare_stages_contains_reappraisal(self):
        """
        Reappraisal (theme_id=4) is in RARE_STAGES — it is the stage most
        at risk of over-smoothing collapse in a GNN trained on imbalanced data.
        """
        self.assertIn(4, RARE_STAGES,
                      "RARE_STAGES must include stage 4 (Reappraisal) — §5.3")

    def test_rare_stages_contains_metacognition(self):
        """
        Metacognition (theme_id=3) is also in RARE_STAGES — the code watches
        both stages 3 and 4 as over-smoothing risks.
        """
        self.assertIn(3, RARE_STAGES,
                      "RARE_STAGES must include stage 3 (Metacognition)")

    def test_rare_stage_floor_is_050(self):
        """
        RARE_STAGE_FLOOR=0.50: a rare stage with held-out recall below this
        threshold triggers a warning and blocks the scaling verdict.
        """
        self.assertAlmostEqual(RARE_STAGE_FLOOR, 0.50,
                               msg="RARE_STAGE_FLOOR must be 0.50")


# ---------------------------------------------------------------------------
# 4. write_validation_report: LLM-FREE SCALING? YES/NO verdict + irr_target
# ---------------------------------------------------------------------------

class TestValidationReportVerdict(unittest.TestCase):
    """
    §5.3 / H5: the validation report must emit an explicit LLM-FREE SCALING?
    YES/NO verdict so analysts know when the gate has been passed.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _minimal_metrics(self, ready=True, kappa=0.75):
        """Build a minimal metrics dict that write_validation_report accepts."""
        return {
            'irr_target': 0.70,
            'vaamr_overall': {'n': 50, 'percent_agreement': 0.88, 'cohen_kappa': kappa},
            'vaamr_per_class': [],
            'purer_overall': {'n': 30, 'percent_agreement': 0.85, 'cohen_kappa': kappa},
            'purer_per_class': [],
            'vaamr_human': {'graph_vs_human': {'n': 0}, 'llm_vs_human': {'n': 0}},
            'ready_for_scaling': ready,
            'rare_stage_notes': [],
            'vaamr_ready': ready,
            'purer_ready': ready,
        }

    def test_report_contains_yes_no_verdict_line(self):
        """The report must have an explicit YES/NO on the scaling question."""
        metrics = self._minimal_metrics(ready=True)
        path = write_validation_report(metrics, self.tmpdir)
        with open(path, 'r') as f:
            content = f.read()
        self.assertIn('LLM-FREE SCALING', content,
                      "Report must contain 'LLM-FREE SCALING' verdict line")
        # Either YES or NO must appear
        self.assertTrue('YES' in content or 'NO' in content,
                        "Report must include YES or NO verdict")

    def test_report_says_yes_when_ready(self):
        """When ready_for_scaling=True, report says YES."""
        metrics = self._minimal_metrics(ready=True)
        path = write_validation_report(metrics, self.tmpdir)
        with open(path, 'r') as f:
            content = f.read()
        self.assertIn('YES', content)

    def test_report_says_no_when_not_ready(self):
        """When ready_for_scaling=False, report says NO."""
        metrics = self._minimal_metrics(ready=False, kappa=0.55)
        path = write_validation_report(metrics, self.tmpdir)
        with open(path, 'r') as f:
            content = f.read()
        self.assertIn('NO', content)

    def test_report_uses_irr_target_070(self):
        """
        §5.3: the gate threshold is explicitly mentioned in the report body
        so readers know what κ the graph must reach.
        """
        metrics = self._minimal_metrics(ready=True)
        path = write_validation_report(metrics, self.tmpdir)
        with open(path, 'r') as f:
            content = f.read()
        # The target appears as "0.70" in the report
        self.assertIn('0.70', content,
                      "Report must state the irr_target=0.70 threshold")


# ---------------------------------------------------------------------------
# 5. EfficacyConfig: avoidance barrier IDs
# ---------------------------------------------------------------------------

class TestEfficacyConfigBarrier(unittest.TestCase):
    """
    §H1, §6.3: the Avoidance→Attention-Regulation transition is the
    rate-limiting step (the avoidance barrier). EfficacyConfig must encode
    barrier_from=1 (Avoidance, theme_id) and barrier_to=2 (Attention Regulation).
    """

    def setUp(self):
        self.cfg = EfficacyConfig()

    def test_barrier_from_is_avoidance(self):
        """
        §H1: barrier_from=1 corresponds to Avoidance (VAAMR stage 1).
        The avoidance barrier is the Avoidance → Attention-Regulation transition.
        """
        self.assertEqual(self.cfg.barrier_from, 1,
                         "barrier_from must be 1 (Avoidance, VAAMR stage 1) — §H1")

    def test_barrier_to_is_attention_regulation(self):
        """
        §H1: barrier_to=2 corresponds to Attention Regulation (VAAMR stage 2).
        The Avoidance→Attention-Regulation crossing is the developmental barrier.
        """
        self.assertEqual(self.cfg.barrier_to, 2,
                         "barrier_to must be 2 (Attention Regulation, VAAMR stage 2) — §H1")

    def test_barrier_direction_is_forward(self):
        """barrier_from < barrier_to confirms this is a forward-progression barrier."""
        self.assertLess(self.cfg.barrier_from, self.cfg.barrier_to,
                        "Avoidance barrier must be a forward-direction transition "
                        "(barrier_from < barrier_to)")

    def test_pipeline_config_efficacy_barrier_defaults_match(self):
        """PipelineConfig.efficacy has the same barrier defaults."""
        pc = PipelineConfig()
        self.assertEqual(pc.efficacy.barrier_from, 1)
        self.assertEqual(pc.efficacy.barrier_to, 2)


if __name__ == '__main__':
    unittest.main()
