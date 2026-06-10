"""
experiments/mechanism/_common.py
================================
Shared, defensive helpers for the corroboration experiments E3–E9
(planning doc, now docs/ROADMAP.md). Mirrors the direct-file-load pattern of
run_interaction_model.py (E1+E2) so we NEVER import the heavy
src/{process,analysis,gnn_layer} packages (their __init__ pulls
sentence_transformers/transformers, pinned to numpy<2, which would crash).

Only stdlib + numpy/pandas at module load; sentence-transformers and
statsmodels are imported lazily inside the functions that need them.

Everything here is observational, n≈26 participants / ~150 cue blocks —
hypothesis-generating, never causal.
"""
from __future__ import annotations
import os, sys, json, hashlib, ast
import importlib.util
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CSV = os.path.join(ROOT, "data", "Meta", "02_meta", "training_data", "master_segments.csv")
DESIGN_CSV = os.path.join(HERE, "_design.csv")

LABELS = [0, 1, 2, 3, 4]
PURER = {0: "Phenomenology", 1: "Utilization", 2: "Reframing", 3: "Education", 4: "Reinforcement"}
VAAMR = {0: "Vigilance", 1: "Avoidance", 2: "AttnReg", 3: "Metacog", 4: "Reappraisal"}


# ------------------------------------------------------------------ module load
def load_module(name: str, relpath: str):
    """Direct file-path import — bypasses src/<pkg>/__init__.py (numpy-pin-sensitive).
    Idempotent: returns the already-registered module if present."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "src", relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cue_blocks_mod():
    return load_module("qra_cue_blocks_standalone", "process/cue_blocks.py")


def stats_mod():
    """analysis/stats.py — cluster_bootstrap_ci, permutation_test, mixedlm_trend, …"""
    return load_module("qra_stats_standalone", "analysis/stats.py")


# ------------------------------------------------------------------ data
def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def is_participant(df: pd.DataFrame) -> pd.Series:
    return df["speaker"].astype(str).str.lower().str.contains("participant", na=False)


def participant_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """The VAAMR-labeled participant segments (final_label not null), int label."""
    pp = df[is_participant(df) & df["final_label"].notna()].copy()
    pp["final_label"] = pp["final_label"].astype(float).astype(int)
    pp["segment_id"] = pp["segment_id"].astype(str)
    return pp


# ------------------------------------------------------------------ design frame
def build_design(df: pd.DataFrame | None = None, with_text: bool = False) -> pd.DataFrame:
    """FROM→CUE→TO triples via the canonical cue-block builder (process/cue_blocks.py).

    Columns: participant_id, session_id, session_number, from_stage, to_stage,
    move (dominant PURER 0–4 or NaN), delta_prog, transition_type, n_therapist
    [+ cue_text, from_text when with_text]. Matches run_interaction_model.build_design
    (same stage_key='final_label', require_stage=True) so it reproduces _design.csv."""
    if df is None:
        df = load_df()
    specs = cue_blocks_mod().cue_blocks_from_records(
        df.to_dict("records"), stage_key="final_label", require_stage=True)
    rows = []
    for sp in specs:
        fi, ti = sp.from_item, sp.to_item
        purers = [int(t["purer_primary"]) for t in sp.therapist_items
                  if pd.notna(t.get("purer_primary"))]
        move = Counter(purers).most_common(1)[0][0] if purers else None
        fc, tc = fi.get("progression_coord"), ti.get("progression_coord")
        delta = (tc - fc) if (pd.notna(fc) and pd.notna(tc)) else np.nan
        row = dict(
            participant_id=str(fi.get("participant_id")),
            session_id=str(sp.session_id),
            session_number=fi.get("session_number"),
            from_stage=int(sp.from_stage), to_stage=int(sp.to_stage),
            move=move, delta_prog=delta, transition_type=sp.transition_type,
            n_therapist=len(sp.therapist_items),
        )
        if with_text:
            ctext = " ".join(str(t.get("text", "")) for t in sp.therapist_items).strip()
            row["cue_text"] = ctext
            row["from_text"] = str(fi.get("text", ""))
            row["from_segment_id"] = str(fi.get("segment_id", ""))
        rows.append(row)
    return pd.DataFrame(rows)


def load_design() -> pd.DataFrame:
    """Reuse the frozen _design.csv (E1/E2 artifact) when present; else rebuild.
    Used by arms that need only the base columns (E5, E8, E9)."""
    try:
        if os.path.exists(DESIGN_CSV):
            d = pd.read_csv(DESIGN_CSV)
            d["participant_id"] = d["participant_id"].astype(str)
            return d
    except Exception:
        pass
    return build_design()


# ------------------------------------------------------------------ embeddings
def embed_minilm(texts, cache_path: str | None = None, ids=None) -> np.ndarray:
    """all-MiniLM-L6-v2 sentence embeddings (384-d), OFFLINE (model is locally cached).

    Optional npz disk cache keyed by an id-list hash so E3/E6 don't re-encode.
    Raises on failure — callers wrap in try/except and degrade."""
    texts = [("" if t is None else str(t)) for t in texts]
    key = None
    if cache_path and ids is not None:
        key = hashlib.sha1(("␟".join(map(str, ids))).encode()).hexdigest()
        if os.path.exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=True)
                if str(z.get("key")) == key and int(z["vecs"].shape[0]) == len(texts):
                    return z["vecs"]
            except Exception:
                pass
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = np.asarray(model.encode(texts, show_progress_bar=False, batch_size=64),
                      dtype=np.float32)
    if cache_path and key is not None:
        try:
            np.savez(cache_path, vecs=vecs, key=key)
        except Exception:
            pass
    return vecs


# ------------------------------------------------------------------ kappa + CI
def cohen_kappa(a, b) -> float:
    from sklearn.metrics import cohen_kappa_score
    if len(a) < 1:
        return float("nan")
    try:
        return float(cohen_kappa_score(a, b))
    except Exception:
        return float("nan")


def kappa_cluster_ci(a, b, clusters, seed=42, n_boot=2000) -> dict:
    """Participant-clustered bootstrap 95% CI for Cohen's κ between aligned label
    lists. Packs each (a,b) pair into one finite float so resampling WHOLE
    participants preserves the pairing (mirrors gnn_reliability.harness._kappa_cluster_ci).
    Labels assumed in [-1, 8]; pack as (a+1)*10 + (b+1)."""
    S = stats_mod()
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    point = cohen_kappa(a_arr.tolist(), b_arr.tolist())
    if len(a_arr) < 2:
        return {"point": point, "lo": None, "hi": None,
                "n": int(len(a_arr)), "n_clusters": len(set(clusters))}
    packed = ((a_arr + 1) * 10 + (b_arr + 1)).astype(float)

    def _stat(arr: np.ndarray) -> float:
        codes = arr.astype(int)
        aa = (codes // 10) - 1
        bb = (codes % 10) - 1
        k = cohen_kappa(aa.tolist(), bb.tolist())
        return float("nan") if k != k else float(k)

    res = S.cluster_bootstrap_ci(packed, list(clusters), statistic=_stat,
                                 n_boot=n_boot, seed=seed)
    return {"point": point, "lo": res["lo"], "hi": res["hi"],
            "n": int(len(a_arr)), "n_clusters": int(res["n_clusters"])}


# ------------------------------------------------------------------ misc
def parse_codes(cell) -> list:
    """codebook_labels_ensemble cell -> list[str] (safe literal_eval of "['a','b']")."""
    if cell is None or (isinstance(cell, float) and cell != cell):
        return []
    s = str(cell).strip()
    if not s or s in ("[]", "nan", "None"):
        return []
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else []
    except Exception:
        return []


def write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def stratified_group_folds(y, groups, n_splits=5, seed=42):
    """sklearn StratifiedGroupKFold split list; n_splits clipped to #groups."""
    from sklearn.model_selection import StratifiedGroupKFold
    n = min(n_splits, len(np.unique(groups)))
    n = max(2, n)
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups)), n
