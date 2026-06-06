"""
tests.testhelpers — shared test infrastructure.

Re-exports the most commonly used builders/helpers so test modules can do::

    from tests.testhelpers import synthetic_df, FakeLLMClient, embedding_patch

Importing this package also bootstraps the repository root onto sys.path.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # tests/testhelpers/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .fixtures import (  # noqa: E402
    synthetic_df,
    make_segment,
    classified_segment,
    MockSegment,
    embedding_patch,
    make_master_df,
)
from .fake_llm import FakeLLMClient  # noqa: E402
from .marks import slow_test, integration_test, RUN_SLOW, RUN_INTEGRATION  # noqa: E402
from .tiny_models import (  # noqa: E402
    TINY_EMBED,
    build_tiny_config,
    tiny_vaamr_framework,
    tiny_purer_framework,
    load_real_framework_or_skip,
)

__all__ = [
    "synthetic_df",
    "make_segment",
    "classified_segment",
    "MockSegment",
    "embedding_patch",
    "make_master_df",
    "FakeLLMClient",
    "TINY_EMBED",
    "build_tiny_config",
    "tiny_vaamr_framework",
    "tiny_purer_framework",
    "load_real_framework_or_skip",
    "slow_test",
    "integration_test",
    "RUN_SLOW",
    "RUN_INTEGRATION",
]
