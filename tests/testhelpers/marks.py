"""
tests.testhelpers.marks
-----------------------
Speed gating for the test suite.

The UNIT tier must stay fast: no multi-epoch GNN training, no full
``run_gnn_analysis``, no real model weights. Tests that need a heavy/real run
(full GNN training, end-to-end pipeline, real embeddings) are decorated with
``@slow_test`` and are SKIPPED unless ``QRA_RUN_SLOW=1`` is set in the
environment.

  * ``tests/run_unit_tests.py``  leaves QRA_RUN_SLOW unset  -> slow tests skip.
  * ``tests/run_integration_tests.py`` sets QRA_RUN_SLOW=1   -> slow tests run.

Rule of thumb for unit tests: a single GNN test method should finish in well
under a second. Use epochs<=3, knn_k<=2, n_sessions=1, hidden_dim<=8 for any
hermetic GNN forward/backward check; anything heavier belongs behind
``@slow_test``.
"""
import os
import unittest

#: True when slow/real-model tests should run (integration tier).
RUN_SLOW = os.environ.get("QRA_RUN_SLOW", "").strip() not in ("", "0", "false", "False")

#: True when integration/real-model tests should run (alias; set by the
#: integration runner). Kept separate so a future split is possible.
RUN_INTEGRATION = os.environ.get("QRA_RUN_INTEGRATION", "").strip() not in ("", "0", "false", "False") or RUN_SLOW

slow_test = unittest.skipUnless(
    RUN_SLOW, "slow/full-model test — set QRA_RUN_SLOW=1 to run (integration tier)"
)

integration_test = unittest.skipUnless(
    RUN_INTEGRATION, "integration test — set QRA_RUN_INTEGRATION=1 (or QRA_RUN_SLOW=1) to run"
)
