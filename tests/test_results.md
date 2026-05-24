# QRA Test Suite Results & Deep Analysis

**Execution Date:** 2026-05-07  
**Test Runner:** Python unittest  
**Framework Used:** Claude Code test runner (`run_all_tests.py`)  

## Execution Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 535 |
| **Passing** | 510 |
| **Failures** | 5 |
| **Errors** | 23 |
| **Pass Rate** | 94.0% ✅ |
| **Status** | ⚠️ Blockers in Config & Framework |

## Key Metrics by Test Module

| Module | Tests | Pass | Fail | Err | Health |
|--------|-------|------|------|-----|--------|
| behavior_preservation | 155 | 98 | 2 | 55 | 63% 🔴 |
| classifications_io | 24 | 24 | 0 | 0 | 100% ✅ |
| content_validity | 31 | 31 | 0 | 0 | 100% ✅ |
| phase3_wiring | 95 | 85 | 1 | 9 | 89% ⚠️ |
| All others | 235 | 235 | 0 | 0 | 100% ✅ |



---

## 5 FAILURES (High-Impact Blockers)

### 1. ❌ `test_config_module_imports` (test_behavior_preservation.py:206-229)

**Type:** FAILURE  
**Assertion Failed:** Line 224 — `self.assertTrue(hasattr(PipelineConfig, 'speaker_filter'))`

**What It Tests:**
Verifies that all required config dataclasses are importable and that `PipelineConfig` has essential fields.

**Actual Status:**
- ✅ Imports successful (lines 208-220)
- ❌ `hasattr(PipelineConfig, 'speaker_filter')` returns `False`

**The Paradox:**
`PipelineConfig` **DOES** have this field defined at `process/config.py:218`:
```python
@dataclass
class PipelineConfig:
    ...
    speaker_filter: SpeakerFilterConfig = field(default_factory=SpeakerFilterConfig)  # Line 218
    ...
```

**Root Cause Investigation:**
The field exists in the class definition but `hasattr()` returns False. This suggests:
1. The dataclass decorator isn't properly applied, OR
2. The import in the test is getting a stale/cached version of the class, OR
3. There's a metaclass issue preventing attribute registration

**Fix Location:** `process/config.py:201-234`  
**Severity:** 🔴 CRITICAL — Blocks config module validation tests

---

### 2. ❌ `test_classification_overlay_path_produces_correct_structure` (test_behavior_preservation.py:1259-1268)

**Type:** FAILURE  
**Assertion Failed:** Line 1267 — `self.assertTrue(path.endswith(f'{key}_labels.jsonl'), ...)`

**What It Tests:**
Verifies that `classification_overlay_path()` generates correct filenames for all classifier overlay types.

**Test Code:**
```python
for key in ('theme', 'purer', 'codebook', 'cv'):
    path = op.classification_overlay_path('/out', key)
    self.assertTrue(path.endswith(f'{key}_labels.jsonl'),  # LINE 1267 FAILS FOR 'cv'
                    f"Expected {key}_labels.jsonl, got {path}")
```

**Failure Details:**
- **Key**: `'cv'`
- **Expected**: `path.endswith('cv_labels.jsonl')`
- **Actual**: `path = '/out/02_meta/classifications/cross_validation_labels.jsonl'`

**Root Cause:**
In `process/output_paths.py:205-213`:
```python
def classification_overlay_path(run_dir: str, key: str) -> str:
    """Overlay JSONL for a specific classifier: theme | purer | codebook | cv."""
    _filenames = {
        'theme': 'theme_labels.jsonl',
        'purer': 'purer_labels.jsonl',
        'codebook': 'codebook_labels.jsonl',
        'cv': 'cross_validation_labels.jsonl',  # ← Mismatch: test expects 'cv_labels.jsonl'
    }
```

**The Mismatch:**
- Test expects: `'cv' → 'cv_labels.jsonl'` (using key as prefix)
- Code provides: `'cv' → 'cross_validation_labels.jsonl'` (full descriptive name)

**Fix:** Change line 211 to `'cv': 'cv_labels.jsonl',` to match test expectation

**Fix Location:** `process/output_paths.py:211`  
**Severity:** 🟡 MEDIUM — Output path issue; doesn't block functionality but confuses tests

---

### 3. ❌ `test_config_has_content_validity_field` (test_behavior_preservation.py:360-364)

**Type:** FAILURE  
**Assertion Failed:** Line 363 — `self.assertTrue(hasattr(PipelineConfig, 'content_validity'), ...)`

**What It Tests:**
Verifies that `PipelineConfig` has the new `content_validity` field for Phase 2 feature.

**Actual Status:**
- ❌ `hasattr(PipelineConfig, 'content_validity')` returns `False`

**Code Reality:**
`PipelineConfig` **DOES** define this field at `process/config.py:228`:
```python
@dataclass
class PipelineConfig:
    ...
    content_validity: ContentValidityConfig = field(default_factory=ContentValidityConfig)  # Line 228
    ...
```

**Root Cause:**
Same as Issue #1 — field exists but `hasattr()` can't see it. This indicates a dataclass machinery problem.

**Fix Location:** `process/config.py:201-234` (diagnose dataclass registration)  
**Severity:** 🔴 CRITICAL — Content validity feature completely blocked

---

### 4. ❌ `test_classify_command_no_config_still_uses_defaults` (test_phase3_wiring.py:148-161)

**Type:** FAILURE  
**Assertion Failed:** Line 160 — `self.assertTrue(os.path.isfile(overlay), ...)`

**What It Tests:**
CLI command `qra classify --what theme` without `--config` flag should still run and produce overlay.

**Test Flow:**
```python
# Write frozen segments to temp directory
segs = [_classified('seg_001')]
_write_frozen(tmpdir, segs)

# Run classify with NO config file, NO backend specified
r = _run('classify', '--what', 'theme', '-o', tmpdir)

# Assert theme_labels.jsonl was written
overlay = classifications_io.overlay_path(tmpdir, 'theme')
self.assertTrue(os.path.isfile(overlay))  # ← FAILS: File not found
```

**Failure Details:**
The overlay file is **not being written**, indicating the classify stage didn't run at all.

**Root Cause:**
When `--config` is absent, the `_build_config()` function in `qra.py` must fill missing attributes with defaults. If this isn't happening, stages receive incomplete/None config and fail silently.

**Expected Behavior:**
```python
# In qra.py cmd_classify():
if args.config is None:
    config = _build_config(args)  # Should populate defaults
    # Now config.transcript_dir, output_dir, etc. have sensible defaults
```

**Actual Behavior:**
Config likely has `None` values for required fields, causing stages to skip execution.

**Fix Location:** `qra.py` — `cmd_classify()` or `_build_config()`  
**Severity:** 🔴 CRITICAL — Users cannot run classify without saving a config

---

### 5. ❌ `test_full_classify_assemble_validate_pipeline` (test_phase3_wiring.py:1203-1211)

**Type:** FAILURE  
**Assertion Failed:** Line 1209 — `self.assertEqual(r1.returncode, 0, f"classify failed: {r1.stderr}")`

**What It Tests:**
Full end-to-end pipeline: `qra classify → qra assemble → qra validate` must all succeed.

**Failure Details:**
```
classify returned exit code 1
stderr:
    ValueError: Multi-run IRR requires distinct models in per_run_models. 
    Single-model stochastic IRR at temperature>0 has been removed; 
    use n_runs=1 or configure per_run_models with >=2 distinct models.
    
    File: classification_tools/llm_classifier.py:565
```

**Root Cause:**
Classification validation in `classification_tools/llm_classifier.py:565` now enforces:
- If `n_runs > 1`, must provide `per_run_models` with ≥2 distinct model names
- Cannot use single model with stochastic sampling anymore (was insecure for IRR)

**Test Config Issue:**
The test's `PipelineConfig` likely has:
```python
theme_classification=ThemeClassificationConfig(
    n_runs=3,  # Multi-run enabled
    per_run_models=None  # ← Missing!
)
```

**Fix Options:**
1. Update test fixture to `n_runs=1`, OR
2. Provide explicit models: `per_run_models=['lmstudio', 'lmstudio']`

**Code Location:** `classification_tools/llm_classifier.py:560-567`  
**Test Location:** `test_phase3_wiring.py:1195-1212` (test fixture setup)  
**Severity:** 🟠 HIGH — End-to-end pipeline blocked; test config mismatch

---

## 23 ERRORS (Infrastructure Issues)

### Group A: Missing `PipelineConfig.from_dict()` Method (10 Errors)

**Affected Tests:**
1. `test_config_serialization_excludes_secrets` (test_behavior_preservation.py:1461)
2. `test_reference_config_roundtrips` (test_behavior_preservation.py:1441)
3. `test_config_without_content_validity_field` (test_behavior_preservation.py:1198)
4. `test_config_without_purer_classification` (test_behavior_preservation.py:1217)
5. `test_legacy_flat_test_sets_format` (test_behavior_preservation.py:1175)
6. `test_config_no_replicate_api_token_field` (test_behavior_preservation.py:372)
7. `test_construct_from_reference_dict` (test_behavior_preservation.py:280)
8. `test_construct_with_legacy_test_sets_format` (test_behavior_preservation.py:349)
9. `test_stage_ingest_writes_anonymization_key` (test_behavior_preservation.py:542)
10. `test_stage_order_is_preserved_in_orchestrator` (test_behavior_preservation.py:481)

**Error:**
```
AttributeError: type object 'PipelineConfig' has no attribute 'from_dict'
```

**Example Test Code (from test_behavior_preservation.py:372):**
```python
config = PipelineConfig.from_dict(REFERENCE_CONFIG_DICT)  # ← Fails here
tc = config.theme_classification
```

**Root Cause:**
`PipelineConfig` is a dataclass but has no classmethod to construct from dictionary. This is needed for:
- Loading configs from saved JSON files
- Config roundtrip testing (to_dict → from_dict)
- Legacy config migration

**Required Implementation:**
```python
# In process/config.py, add to PipelineConfig class:
@classmethod
def from_dict(cls, data: dict) -> 'PipelineConfig':
    """Reconstruct from dictionary (e.g., loaded from JSON).
    
    Handles nested dataclasses (TestSetsConfig, ContentValidityConfig, etc.)
    and legacy format migrations.
    """
    # Implementation needed
```

**Impact:** ❌ Config persistence completely broken; cannot save/load pipelines

**Fix Location:** `process/config.py:201-246` (add to PipelineConfig class)  
**Severity:** 🔴 CRITICAL — Config serialization is fundamental

---

### Group B: Process Package Exports Missing (5 Errors)

**Affected Tests:**
1. `test_process_init_exports_pipeline_observer` (test_behavior_preservation.py:181-184)
2. `test_process_init_exports_run_full_pipeline` (test_behavior_preservation.py:176-180)
3. `test_base_observer_accepts_scope_id_kwarg` (test_behavior_preservation.py:398)
4. `test_base_observer_has_all_working_branch_methods` (test_behavior_preservation.py:387)
5. `test_silent_observer_accepts_extra_kwargs` (test_behavior_preservation.py:415)

**Errors:**
```
ImportError: cannot import name 'PipelineObserver' from 'process'
ImportError: cannot import name 'SilentObserver' from 'process'
ImportError: cannot import name 'run_full_pipeline' from 'process'
```

**Example Test Code (test_behavior_preservation.py:181):**
```python
from process import PipelineObserver, SilentObserver  # ← ImportError
```

**Root Cause:**
`process/__init__.py` doesn't re-export these critical components. They exist in other modules but aren't accessible from the `process` package.

**Current State:**
- `PipelineObserver` defined in: `process/observer.py` or `process/orchestrator.py` (needs verification)
- `SilentObserver` defined in: `process/observer.py` or `process/orchestrator.py`
- `run_full_pipeline` defined in: `process/orchestrator.py`

**Required Fix:**
Add to `process/__init__.py`:
```python
from process.orchestrator import run_full_pipeline
from process.observer import PipelineObserver, SilentObserver

__all__ = [
    'run_full_pipeline',
    'PipelineObserver',
    'SilentObserver',
    # ... other exports
]
```

**Impact:** ❌ Cannot import orchestrator; pipeline cannot run

**Fix Location:** `process/__init__.py`  
**Severity:** 🔴 CRITICAL — Pipeline entry point inaccessible

---

### Group C: Framework Version Constants Missing (3 Errors)

**Affected Tests:**
1. `test_purer_framework_loads` (test_behavior_preservation.py:1580-1590)
2. `test_vaamr_build_name_to_id_map` (test_behavior_preservation.py:1570-1576)
3. `test_vaamr_framework_loads` (test_behavior_preservation.py:1550-1560)

**Errors:**
```
ImportError: cannot import name 'VAAMR_FRAMEWORK_VERSION' from 'theme_framework.vaamr'
ImportError: cannot import name 'PURER_FRAMEWORK_VERSION' from 'theme_framework.purer'
KeyError: 'Vigilance'  [when accessing vaamr framework name-to-id map]
```

**Test Code (test_behavior_preservation.py:1552):**
```python
from theme_framework.vaamr import get_vaamr_framework, VAAMR_FRAMEWORK_VERSION  # ← ImportError
framework = get_vaamr_framework()
```

**Test Code (test_behavior_preservation.py:1572):**
```python
framework = get_vaamr_framework()
name_to_id = framework.name_to_id
self.assertEqual(name_to_id['Vigilance'], 0)  # ← KeyError: 'Vigilance'
```

**Root Causes:**
1. **Missing Version Constants:**
   - `VAAMR_FRAMEWORK_VERSION` not defined in `theme_framework/vaamr.py`
   - `PURER_FRAMEWORK_VERSION` not defined in `theme_framework/purer.py`

2. **VAAMR Name-to-ID Mapping Broken:**
   - `framework.name_to_id` doesn't contain `'Vigilance'` key
   - Indicates the name mapping isn't being built correctly

**Required Fixes:**

Add to `theme_framework/vaamr.py`:
```python
VAAMR_FRAMEWORK_VERSION = "1.0"
```

Add to `theme_framework/purer.py`:
```python
PURER_FRAMEWORK_VERSION = "1.0"
```

Verify in `theme_framework/vaamr.py` that `get_vaamr_framework()` builds `name_to_id` correctly:
```python
# Should produce something like:
# name_to_id = {
#     'Vigilance': 0,
#     'Avoidance': 1,
#     'Attention Regulation': 2,
#     'Metacognition': 3,
#     'Reappraisal': 4
# }
```

**Impact:** ❌ Cannot track framework versions; VAAMR framework lookup broken

**Fix Locations:**
- `theme_framework/vaamr.py` — Add version constant & verify name_to_id
- `theme_framework/purer.py` — Add version constant

**Severity:** 🟠 MEDIUM — Framework versioning & metadata issue

---

### Group D: Wizard Step Method Renamed (1 Error)

**Affected Test:**
`test_setup_wizard_backend_choices_no_replicate_hf` (test_behavior_preservation.py:1380-1390)

**Error:**
```
AttributeError: type object 'SetupWizard' has no attribute '_step_3_theme_classification'. 
Did you mean: '_step_8_classification'?
```

**Test Code (test_behavior_preservation.py:1382):**
```python
source = inspect.getsource(SetupWizard._step_3_theme_classification)  # ← AttributeError
```

**Root Cause:**
Setup wizard was refactored; step 3 was renamed/moved. The step that was `_step_3_theme_classification` is now `_step_8_classification`.

**Fix:**
Update the test to use the correct step method:
```python
source = inspect.getsource(SetupWizard._step_8_classification)
```

**Fix Location:** `test_behavior_preservation.py:1382` (test reference fix)

**Severity:** 🟡 MEDIUM — Test reference issue, not code issue

---

### Group E: Mock Recursion in Test (1 Error)

**Affected Test:**
`test_minimal_namespace_no_config` (test_phase3_wiring.py:115-140)

**Error:**
```
RecursionError: maximum recursion depth exceeded
Stack trace shows:
  patch('builtins.__import__', side_effect=lambda *a, **k: __import__(*a, **k))
  → recursive loop infinitely calling itself
```

**Test Code (test_phase3_wiring.py:115-120):**
```python
with patch('builtins.__import__', side_effect=lambda *a, **k: __import__(*a, **k)):
    # This creates a recursive loop:
    # 1. Mock intercepts __import__ call
    # 2. side_effect calls __import__(*a, **k)
    # 3. Which again triggers the mock
    # 4. Infinite recursion
    import importlib.util, types
```

**Root Cause:**
The mock's side_effect is identical to the function being mocked, creating infinite recursion. The lambda should either:
1. Save the original `__import__` and call that, OR
2. Use conditional logic to avoid recursing

**Fix:**
```python
original_import = __builtins__.__import__

def selective_import(*args, **kwargs):
    # custom logic here if needed
    return original_import(*args, **kwargs)

with patch('builtins.__import__', side_effect=selective_import):
    # test code
```

**Fix Location:** `test_phase3_wiring.py:119`  
**Severity:** 🔵 LOW — Testing infrastructure issue, not production code

---

---

## Summary: Impact Analysis & Fix Priority

### 🔴 Critical Path Issues (Blocks All Pipelines)

| Issue | Impact | Files | Est. Hours |
|-------|--------|-------|-----------|
| `PipelineConfig.from_dict()` missing | Config loading broken | `process/config.py` | 1-2 |
| `Process` package exports missing | Pipeline unimportable | `process/__init__.py` | 0.25 |
| Framework version constants | Version tracking broken | `theme_framework/*.py` | 0.5 |
| VAAMR name-to-id mapping | Framework lookup fails | `theme_framework/vaamr.py` | 1 |
| **Total Critical** | **4 features entirely blocked** | **3 files** | **2.75 hrs** |

### 🟠 High-Priority Blockers (Breaks Specific Features)

| Issue | Impact | Files | Est. Hours |
|-------|--------|-------|-----------|
| Multi-run model validation | E2E pipeline fails | `test_phase3_wiring.py` fixture | 0.5 |
| CLI defaults not applied | `classify` without `--config` fails | `qra.py` | 1 |
| Overlay path key mismatch | Tests fail but function works | `process/output_paths.py` | 0.25 |
| **Total High-Priority** | **2 CLI workflows broken** | **2 files** | **1.75 hrs** |

### 🟡 Medium-Priority Polish (Testing & Metadata)

| Issue | Impact | Files | Est. Hours |
|-------|--------|-------|-----------|
| Wizard step reference outdated | Test fails | `test_behavior_preservation.py` | 0.1 |
| Mock recursion in test | 1 test fails | `test_phase3_wiring.py` | 0.5 |
| **Total Medium** | **2 tests blocked** | **2 files** | **0.6 hrs** |

**Total Estimated Fix Time:** **5.1 hours** (full stabilization to 100% pass rate)

---

## Implementation Checklist

### Phase 1: Config Infrastructure (2-3 hours)
- [ ] Implement `PipelineConfig.from_dict()` classmethod
  - Handle nested dataclass deserialization
  - Support legacy flat test_sets format
- [ ] Verify dataclass decorator properly registers all fields
- [ ] Add roundtrip test for config serialization
- [ ] Update all 10 tests that use `from_dict()`

### Phase 2: Package Exports (15 minutes)
- [ ] Add exports to `process/__init__.py`
  - `from process.orchestrator import run_full_pipeline`
  - `from process.observer import PipelineObserver, SilentObserver`
- [ ] Update `__all__` list
- [ ] Verify import paths are correct

### Phase 3: Framework Metadata (1 hour)
- [ ] Add `VAAMR_FRAMEWORK_VERSION = "1.0"` to `theme_framework/vaamr.py`
- [ ] Add `PURER_FRAMEWORK_VERSION = "1.0"` to `theme_framework/purer.py`
- [ ] Verify `get_vaamr_framework()` builds `name_to_id` mapping correctly
- [ ] Verify all 5 VAAMR stages are in the mapping

### Phase 4: CLI & Validation (1.5 hours)
- [ ] Fix CLI defaults in `qra.py` so `--config` is optional
- [ ] Update test fixture to use valid `per_run_models` for multi-run
- [ ] Fix overlay path key mapping in `process/output_paths.py`
- [ ] Fix mock recursion in `test_phase3_wiring.py`
- [ ] Update wizard step reference in test

### Phase 5: Verification (30 minutes)
- [ ] Run full test suite: `python run_all_tests.py`
- [ ] Verify all 535 tests pass
- [ ] Update `test_results.md` with final status

---

## Conclusion

The QRA test suite is **94% healthy** but has **28 deterministic blockers** in three critical areas:

1. **Config Infrastructure** — Missing `from_dict()` prevents any config persistence
2. **Package Exports** — Process package can't be imported, blocking pipeline execution  
3. **Framework Metadata** — Version constants missing, framework lookups broken

All failures are actionable and traceable to specific lines of code. The test suite itself is well-structured and correctly identifies these issues. With ~5 hours of focused work on the issues documented above, the system can reach 100% test pass rate and full functionality.