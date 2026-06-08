"""
experiments/mechanism/run_e5_purer_noise.py
===========================================
E5 — PURER-label-noise robustness (masterplan §4, Q17).

The entire dyadic-mechanism story rests on therapist PURER cue labels that are
NOT yet human-validated (§8.1; α≥0.70 pending). This bounds how much label noise
could be moving the per-move Δprogression ranking:

  Perturb each block's dominant move at a single-rater DISAGREEMENT RATE (reassign
  the perturbed blocks to a uniformly-random ALTERNATIVE move), recompute the
  per-move mean-Δprogression ranking, K times, and report the Spearman rank-
  correlation distribution vs the UNPERTURBED ranking. A ranking that stays stable
  under realistic label noise is defensible pending human validation.

Disagreement rate: there is NO measured PURER IRR in data/Meta/04_validation/irr/
(that IRR is VAAMR-only; framework=='VAAMR'), so we use the masterplan default 0.30
and additionally sweep {0.15, 0.30, 0.45} for sensitivity.

Observational; the ranking itself is under-identified at n≈20 participants.
Run:  .venv/bin/python experiments/mechanism/run_e5_purer_noise.py
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import _common as C

SEED = 42
K = 200
ALL_MOVES = [0, 1, 2, 3, 4]


def measured_purer_disagreement() -> dict:
    """Try to read a PURER single-rater disagreement rate from the IRR results.
    Returns {found, rate, source}. The shipped IRR is VAAMR-only -> not found."""
    irr = os.path.join(C.ROOT, "data", "Meta", "04_validation", "irr", "irr_results.json")
    info = {"found": False, "rate": None, "source": None}
    try:
        d = json.load(open(irr))
        if str(d.get("framework", "")).upper() == "PURER":
            # pairwise agreement across human raters -> disagreement = 1 - agreement
            hh = d.get("human_human", {})
            agrs = [v.get("primary", {}).get("percent_agreement_pairwise")
                    for v in hh.values() if isinstance(v, dict)]
            agrs = [a for a in agrs if isinstance(a, (int, float))]
            if agrs:
                info.update(found=True, rate=round(1.0 - float(np.mean(agrs)), 4),
                            source="PURER human_human pairwise agreement")
    except Exception as e:
        info["source"] = f"unreadable ({type(e).__name__})"
    return info


def move_ranking(moves: np.ndarray, deltas: np.ndarray) -> dict:
    """move -> mean Δprogression (only moves present)."""
    out = {}
    for mv in ALL_MOVES:
        sel = moves == mv
        if sel.sum() > 0:
            out[mv] = float(np.nanmean(deltas[sel]))
    return out


def _spearman_on_common(base: dict, pert: dict) -> float:
    from scipy.stats import spearmanr
    common = [mv for mv in ALL_MOVES if mv in base and mv in pert]
    if len(common) < 3:
        return float("nan")
    a = [base[mv] for mv in common]
    b = [pert[mv] for mv in common]
    rho, _ = spearmanr(a, b)
    return float(rho)


def perturb_distribution(moves: np.ndarray, deltas: np.ndarray, rate: float,
                         k: int = K, seed: int = SEED) -> dict:
    base = move_ranking(moves, deltas)
    base_order = sorted(base, key=lambda m: -base[m])
    rng = np.random.default_rng(seed)
    rhos, kendalls = [], []
    from scipy.stats import kendalltau
    n = len(moves)
    for _ in range(k):
        flip = rng.random(n) < rate
        pm = moves.copy()
        for i in np.where(flip)[0]:
            alts = [m for m in ALL_MOVES if m != moves[i]]
            pm[i] = rng.choice(alts)
        pert = move_ranking(pm, deltas)
        rhos.append(_spearman_on_common(base, pert))
        # kendall tau on the common ordering for robustness
        common = [mv for mv in ALL_MOVES if mv in base and mv in pert]
        if len(common) >= 3:
            tau, _ = kendalltau([base[mv] for mv in common], [pert[mv] for mv in common])
            kendalls.append(float(tau))
    rhos = np.asarray([r for r in rhos if r == r], dtype=float)
    kendalls = np.asarray([t for t in kendalls if t == t], dtype=float)

    def q(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")

    return dict(
        rate=rate, k=int(k),
        base_ranking={C.PURER[m]: round(base[m], 4) for m in base_order},
        base_order=[C.PURER[m] for m in base_order],
        spearman=dict(mean=round(float(np.mean(rhos)), 4) if len(rhos) else None,
                      median=round(q(rhos, 50), 4), p05=round(q(rhos, 5), 4),
                      p95=round(q(rhos, 95), 4),
                      frac_ge_0_8=round(float(np.mean(rhos >= 0.8)), 4) if len(rhos) else None,
                      frac_ge_0_5=round(float(np.mean(rhos >= 0.5)), 4) if len(rhos) else None),
        kendall_tau_median=round(q(kendalls, 50), 4) if len(kendalls) else None,
    )


def main() -> int:
    print("=" * 78)
    print("E5 — PURER label-noise robustness: per-move Δprog ranking stability")
    print("=" * 78)
    D = C.load_design()
    Dm = D.dropna(subset=["move", "delta_prog"]).copy()
    Dm["move"] = Dm["move"].astype(int)
    moves = Dm["move"].to_numpy()
    deltas = Dm["delta_prog"].to_numpy(dtype=float)
    print(f"\nblocks with move+Δprog: {len(Dm)}  "
          f"participants: {Dm['participant_id'].nunique()}")

    meas = measured_purer_disagreement()
    default_rate = meas["rate"] if meas["found"] else 0.30
    print(f"PURER disagreement rate: "
          + (f"{default_rate:.3f} (measured: {meas['source']})" if meas["found"]
             else f"{default_rate:.2f} (default — no PURER IRR; shipped IRR is VAAMR-only)"))

    rates = sorted({round(default_rate, 4), 0.15, 0.30, 0.45})
    results = {}
    base_printed = False
    for rate in rates:
        try:
            r = perturb_distribution(moves, deltas, rate)
            results[f"rate_{rate:.2f}"] = r
            if not base_printed:
                print(f"\nunperturbed per-move Δprog ranking (best→worst): {r['base_order']}")
                for nm, v in r["base_ranking"].items():
                    print(f"    {nm:13} mean_delta={v:+.3f}")
                base_printed = True
            sp = r["spearman"]
            print(f"  rate={rate:.2f}  Spearman ρ vs base: median={sp['median']} "
                  f"[p05={sp['p05']}, p95={sp['p95']}]  "
                  f"frac(ρ≥0.8)={sp['frac_ge_0_8']}  frac(ρ≥0.5)={sp['frac_ge_0_5']}")
        except Exception as e:
            results[f"rate_{rate:.2f}"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  rate={rate:.2f}  ERROR: {results[f'rate_{rate:.2f}']['error']}")

    primary = results.get(f"rate_{default_rate:.2f}") or results.get("rate_0.30")
    headline = {}
    if primary and "spearman" in primary:
        headline = dict(primary_rate=default_rate,
                        spearman_median=primary["spearman"]["median"],
                        spearman_p05=primary["spearman"]["p05"],
                        frac_stable_ge_0_8=primary["spearman"]["frac_ge_0_8"])
        verdict = ("ranking is FRAGILE under label noise — gate every therapist-effect "
                   "claim on PURER validation"
                   if (primary["spearman"]["median"] is None or primary["spearman"]["median"] < 0.6)
                   else "ranking is moderately stable under label noise")
        print(f"\n=> at the {default_rate:.2f} disagreement rate: {verdict}")

    out = dict(
        design=dict(n_blocks=int(len(Dm)),
                    n_participants=int(Dm["participant_id"].nunique()), K=K, seed=SEED),
        purer_disagreement=meas, default_rate=default_rate,
        by_rate=results, headline=headline,
        note="Perturbation reassigns flipped blocks to a uniformly-random alternative "
             "move; ranking = per-move mean Δprogression; Spearman ρ vs the unperturbed "
             "ranking over moves present in both. Only 6 blocks are Utilization, so its "
             "rank is the most noise-sensitive.")
    p = C.write_json(out, os.path.join(os.path.dirname(__file__), "_e5_results.json"))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
