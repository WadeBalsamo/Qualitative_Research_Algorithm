"""Guard: the experiments CATALOG mirror (``src/experiments/gnn_reliability/``) must stay
byte-identical to the LIVE apparatus (``experiments/gnn_reliability/``) for every shared
module.

Why this matters: ``tests/conftest.py`` puts ``src/`` on ``sys.path``, so
``import experiments.gnn_reliability`` can resolve to EITHER tree depending on path order.
If the catalog mirror drifts from the live apparatus (e.g. a module the live copy now
imports from ``gnn_layer.classifier`` is still imported from ``gnn_layer`` in the stale
mirror), collection of the GNN-reliability tests fails. Keeping the shared files identical
makes the two copies interchangeable. See ``src/experiments/README.md`` ("Relationship to
the live tree") and ``src/experiments/WORKFLOW.md``.

Catalog-only extras (``capacity_scaler.py`` and the ``*.md`` write-ups) are intentionally
NOT mirrored and are not checked here.
"""
import os
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIVE = os.path.join(_ROOT, 'experiments', 'gnn_reliability')
MIRROR = os.path.join(_ROOT, 'src', 'experiments', 'gnn_reliability')

# Apparatus files that BOTH trees must carry identically.
SHARED = ['__init__.py', 'anchors_arm.py', 'baselines.py', 'harness.py',
          'run_battery.py', 'run_mechanism.py']


class TestExperimentsCatalogSync(unittest.TestCase):
    def test_mirror_is_byte_identical_to_live_apparatus(self):
        for name in SHARED:
            live = os.path.join(LIVE, name)
            mirror = os.path.join(MIRROR, name)
            self.assertTrue(os.path.isfile(live), f"missing live apparatus file: {live}")
            self.assertTrue(os.path.isfile(mirror), f"missing catalog mirror file: {mirror}")
            with open(live, 'rb') as fh:
                live_bytes = fh.read()
            with open(mirror, 'rb') as fh:
                mirror_bytes = fh.read()
            self.assertEqual(
                live_bytes, mirror_bytes,
                f"\n{name}: the catalog mirror src/experiments/gnn_reliability/{name} has "
                f"DRIFTED from the live experiments/gnn_reliability/{name}.\n"
                f"Re-sync it (cp experiments/gnn_reliability/{name} "
                f"src/experiments/gnn_reliability/{name}) so the unit tests and the discovery "
                f"layer never import a stale apparatus. See src/experiments/README.md.")


if __name__ == '__main__':
    unittest.main()
