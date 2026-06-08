"""
experiments/mechanism/run_e7_lift_controls.py
=============================================
E7 — lift controls (masterplan §4; Q4/Q5; methodology R6).

H3 claims VAAMR×VCE co-occurrence lift is *construct* evidence ("VCE sharpens the
arc"). A reviewer's objection (R6): lift between two LLM-derived label sets could be
shared-lexicon co-dependency, not construct validity. The named control (§8.2) is a
SHUFFLED-STAGE PERMUTATION null:

  1. Observed lift(stage, code) = P(code | stage) / P(code) on participant segments
     carrying both a VAAMR final_label and >=1 VCE ensemble code.
  2. Permute the stage labels across those segments, recompute lift, K times.
  3. Report which (stage, code) associations EXCEED the permutation null (perm_p<.05),
     i.e., lift not explainable by base-rate co-dependency. BH-FDR across the family.

H3a off-graph Δκ is optional/heavy and is documented-and-skipped here.

Observational; only ~39 segments carry both labels — under-identified, as expected.
Run:  .venv/bin/python experiments/mechanism/run_e7_lift_controls.py
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import _common as C

SEED = 42
K = 2000
MIN_CODE_N = 3   # ignore codes appearing in <3 segments (lift undefined/unstable)


def build_stage_code(df: pd.DataFrame):
    pp = C.participant_labeled(df).copy()
    pp["codes"] = pp["codebook_labels_ensemble"].apply(C.parse_codes)
    pp = pp[pp["codes"].map(len) > 0].copy()
    return pp


def lift_table(stages: np.ndarray, code_sets, codes_vocab, n_stages=5) -> dict:
    """lift[(stage, code)] = P(code|stage)/P(code). Returns dict + counts."""
    n = len(stages)
    base = {c: np.mean([c in s for s in code_sets]) for c in codes_vocab}
    out = {}
    for st in range(n_stages):
        idx = np.where(stages == st)[0]
        if len(idx) == 0:
            continue
        for c in codes_vocab:
            p_cond = np.mean([c in code_sets[i] for i in idx])
            if base[c] > 0:
                out[(st, c)] = p_cond / base[c]
    return out, base


def main() -> int:
    print("=" * 78)
    print("E7 — VAAMR×VCE lift + shuffled-stage permutation control")
    print("=" * 78)
    df = C.load_df()
    pp = build_stage_code(df)
    stages = pp["final_label"].to_numpy()
    code_sets = pp["codes"].tolist()
    print(f"\nparticipant segments with VAAMR label AND >=1 VCE code: {len(pp)}  "
          f"participants: {pp['participant_id'].nunique()}")
    if len(pp) < 5:
        out = {"error": "too few co-labeled segments", "n": int(len(pp))}
        p = C.write_json(out, os.path.join(os.path.dirname(__file__), "_e7_results.json"))
        print("INFEASIBLE:", out); print(f"wrote {p}"); return 0

    # vocab: codes appearing >= MIN_CODE_N times
    from collections import Counter
    cnt = Counter(c for s in code_sets for c in s)
    vocab = sorted([c for c, k in cnt.items() if k >= MIN_CODE_N])
    print(f"VCE codes total={len(cnt)}; tested (appearing >={MIN_CODE_N}x)={len(vocab)}")

    obs, base = lift_table(stages, code_sets, vocab)
    # permutation null: shuffle stage labels, recompute lift
    rng = np.random.default_rng(SEED)
    ge = {k: 0 for k in obs}
    for _ in range(K):
        perm = rng.permutation(stages)
        plift, _ = lift_table(perm, code_sets, vocab)
        for k, v in obs.items():
            if plift.get(k, 0.0) >= v:
                ge[k] += 1

    rows = []
    for (st, c), v in obs.items():
        # observed co-occurrence count for context
        n_cell = int(sum((stages[i] == st) and (c in code_sets[i]) for i in range(len(stages))))
        perm_p = (ge[(st, c)] + 1) / (K + 1)
        rows.append(dict(stage=int(st), stage_name=C.VAAMR[st], code=c,
                         lift=round(float(v), 3), n_cell=n_cell,
                         code_base_rate=round(float(base[c]), 3),
                         perm_p=round(float(perm_p), 4)))
    # BH-FDR across the family
    S = C.stats_mod()
    bh = S.benjamini_hochberg([r["perm_p"] for r in rows], alpha=0.05)
    for r, q, rej in zip(rows, bh["qvalues"], bh["reject"]):
        r["q_value"] = round(float(q), 4) if q == q else None
        r["fdr_reject"] = bool(rej)
    rows.sort(key=lambda r: (r["perm_p"], -r["lift"]))

    exceed_raw = [r for r in rows if r["perm_p"] < 0.05]
    exceed_fdr = [r for r in rows if r["fdr_reject"]]
    print(f"\n{len(rows)} (stage,code) cells tested; "
          f"{len(exceed_raw)} exceed the permutation null (raw p<.05); "
          f"{len(exceed_fdr)} survive BH-FDR")
    print("Top associations vs the shuffled-stage null:")
    for r in rows[:12]:
        flag = " *FDR" if r["fdr_reject"] else ""
        print(f"  {r['stage_name']:11} × {r['code'][:34]:34} lift={r['lift']:.2f} "
              f"n={r['n_cell']:2} perm_p={r['perm_p']:.3f}{flag}")

    out = dict(
        design=dict(n_segments=int(len(pp)),
                    n_participants=int(pp["participant_id"].nunique()),
                    n_codes_tested=len(vocab), K=K, seed=SEED, min_code_n=MIN_CODE_N),
        n_cells=len(rows), n_exceed_perm_raw=len(exceed_raw),
        n_exceed_fdr=len(exceed_fdr),
        cells=rows, exceed_perm_raw=exceed_raw, exceed_fdr=exceed_fdr,
        h3a_off_graph="documented-and-skipped (heavy; needs the probe/LLM held-out "
                      "Δκ-with/without-VCE harness — out of scope for this corroboration pass)",
        note="Lift = P(code|stage)/P(code); permutation null shuffles stage labels (holds "
             "code base rates fixed) so surviving cells are not pure base-rate co-dependency. "
             "At ~39 co-labeled segments the family is tiny and under-powered.")
    p = C.write_json(out, os.path.join(os.path.dirname(__file__), "_e7_results.json"))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
