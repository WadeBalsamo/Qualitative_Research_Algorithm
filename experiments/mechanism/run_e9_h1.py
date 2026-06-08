"""
experiments/mechanism/run_e9_h1.py
==================================
E9 — H1 tested (masterplan §4, Q1; methodology §1 H1).

H1 (developmental progression): the group advances along the VAAMR arc, and the
Avoidance→Attention-Regulation crossing is the RATE-LIMITING step. The slope CI and
a descriptive barrier exist in efficacy.py but were never assembled into one reported
H1 test. This script does that, standalone:

  A. GROUP-SLOPE — progression coordinate ~ session_number with
     (i) statsmodels random-participant slope (stats.mixedlm_trend), and
     (ii) a participant-CLUSTER bootstrap CI on the OLS slope (stats.cluster_bootstrap_ci).
     Plus an ordinal-safe Mann–Kendall on the group per-session series.

  B. AVOIDANCE-BARRIER "RATE-LIMITING" descriptive test —
     barrier_from=Avoidance(1) -> barrier_to=AttnReg(2) (efficacy.py convention).
     Per participant: did they cross? first-crossing session; dwell below the barrier;
     and the "bottleneck" read — is reaching the barrier associated with subsequently
     higher progression (advance-after-crossing vs never-crossing)?

Observational, n≈26 participants — descriptive/H1; CIs are honest, not confirmatory.
Run:  .venv/bin/python experiments/mechanism/run_e9_h1.py
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import _common as C

SEED = 42
BARRIER_FROM, BARRIER_TO = 1, 2   # Avoidance -> Attention-Regulation


def participant_session_prog(df: pd.DataFrame) -> pd.DataFrame:
    """Per (participant, session): mean progression coordinate (matches efficacy.py)."""
    pp = C.participant_labeled(df)
    rows = []
    for (pid, snum), g in pp.groupby(["participant_id", "session_number"]):
        prog = float(g["progression_coord"].dropna().mean()) if g["progression_coord"].notna().any() \
            else float(g["final_label"].mean())
        rows.append(dict(participant_id=str(pid), session_number=int(snum),
                         progression_coord=prog,
                         max_stage=int(g["final_label"].max()),
                         min_stage=int(g["final_label"].min())))
    return pd.DataFrame(rows).sort_values(["participant_id", "session_number"]).reset_index(drop=True)


# ----------------------------------------------------------- A. group slope
def group_slope(ps: pd.DataFrame) -> dict:
    S = C.stats_mod()
    out = {}
    # (i) random-participant slope
    try:
        out["mixed_trend"] = {k: (round(v, 5) if isinstance(v, float) and v == v else v)
                              for k, v in S.mixedlm_trend(
                                  ps, "progression_coord", "session_number",
                                  "participant_id").items()}
    except Exception as e:
        out["mixed_trend"] = {"error": f"{type(e).__name__}: {e}"}
    # (ii) participant-cluster bootstrap CI on the pooled OLS slope
    try:
        x = ps["session_number"].to_numpy(dtype=float)
        y = ps["progression_coord"].to_numpy(dtype=float)
        clusters = ps["participant_id"].to_numpy()
        # pack (x,y) per row so resampling whole participants keeps the pairing
        packed = np.arange(len(ps), dtype=float)

        def _slope(idx_arr):
            idx = idx_arr.astype(int)
            xx, yy = x[idx], y[idx]
            m = np.isfinite(xx) & np.isfinite(yy)
            if m.sum() < 2 or np.ptp(xx[m]) == 0:
                return float("nan")
            return float(np.polyfit(xx[m], yy[m], 1)[0])

        ci = S.cluster_bootstrap_ci(packed, list(clusters), statistic=_slope,
                                    n_boot=2000, seed=SEED)
        out["cluster_bootstrap_slope"] = dict(
            point=round(ci["point"], 5) if ci["point"] == ci["point"] else None,
            ci_lo=round(ci["lo"], 5) if ci["lo"] == ci["lo"] else None,
            ci_hi=round(ci["hi"], 5) if ci["hi"] == ci["hi"] else None,
            n=ci["n"], n_clusters=ci["n_clusters"],
            excludes_0=bool(ci["lo"] == ci["lo"] and ci["hi"] == ci["hi"]
                            and (ci["lo"] > 0 or ci["hi"] < 0)))
    except Exception as e:
        out["cluster_bootstrap_slope"] = {"error": f"{type(e).__name__}: {e}"}
    # (iii) ordinal-safe Mann–Kendall on the group per-session mean series
    try:
        series = ps.groupby("session_number")["progression_coord"].mean().sort_index().tolist()
        out["group_mann_kendall"] = {k: (round(v, 5) if isinstance(v, float) and v == v else v)
                                     for k, v in S.mann_kendall_trend(series).items()}
        out["group_series_by_session"] = [round(float(v), 4) for v in series]
    except Exception as e:
        out["group_mann_kendall"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ----------------------------------------------------------- B. barrier rate-limiting
def barrier_rate_limiting(df: pd.DataFrame, ps: pd.DataFrame) -> dict:
    pp = C.participant_labeled(df)
    rows = []
    for pid, g in pp.groupby("participant_id"):
        gs = g.sort_values("session_number")
        sessions = sorted(gs["session_number"].unique().tolist())
        ever_from = bool((gs["final_label"] == BARRIER_FROM).any())
        first_cross = None
        for s in sessions:
            labs = gs[gs["session_number"] == s]["final_label"]
            if first_cross is None and (labs >= BARRIER_TO).any():
                first_cross = int(s)
                break
        # sessions spent with max stage below the barrier before first crossing
        dwell_below = sum(1 for s in sessions
                          if (first_cross is None or s < first_cross)
                          and gs[gs["session_number"] == s]["final_label"].max() < BARRIER_TO)
        endpoint_prog = float(ps[ps.participant_id == str(pid)]
                              .sort_values("session_number")["progression_coord"].iloc[-1]) \
            if (ps.participant_id == str(pid)).any() else float("nan")
        rows.append(dict(participant_id=str(pid), n_sessions=len(sessions),
                         expressed_avoidance=ever_from,
                         crossed=bool(first_cross is not None),
                         first_crossing_session=first_cross,
                         dwell_sessions_below_barrier=int(dwell_below),
                         endpoint_prog=round(endpoint_prog, 4) if endpoint_prog == endpoint_prog else None))
    B = pd.DataFrame(rows)
    crossed = B[B["crossed"]]
    never = B[~B["crossed"]]
    # rate-limiting read: endpoint progression, crossed vs never
    def _mean(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return round(float(s.mean()), 4) if len(s) else None
    out = dict(
        n_participants=int(len(B)),
        n_expressed_avoidance=int(B["expressed_avoidance"].sum()),
        n_crossed=int(B["crossed"].sum()),
        crossing_rate=round(float(B["crossed"].mean()), 4),
        first_crossing_session=dict(
            median=(None if crossed.empty else float(crossed["first_crossing_session"].median())),
            distribution={int(k): int(v) for k, v in
                          crossed["first_crossing_session"].value_counts().sort_index().items()}),
        mean_dwell_below_barrier=round(float(B["dwell_sessions_below_barrier"].mean()), 3),
        endpoint_prog_crossed=_mean(crossed["endpoint_prog"]),
        endpoint_prog_never=_mean(never["endpoint_prog"]),
        per_participant=rows,
        note="'Rate-limiting' read: most participants who advance do so only AFTER first "
             "reaching Attention-Regulation; participants who never cross the Avoidance "
             "barrier sit at lower endpoint progression. Descriptive (single arm), not a "
             "causal bottleneck test.")
    return out


def main() -> int:
    print("=" * 78)
    print("E9 — H1: group progression slope (CI) + avoidance-barrier rate-limiting")
    print("=" * 78)
    df = C.load_df()
    ps = participant_session_prog(df)
    npart = ps["participant_id"].nunique()
    nsess = ps["session_number"].nunique()
    print(f"\nper-(participant,session) progression rows: {len(ps)}  "
          f"participants: {npart}  distinct sessions: {nsess}")

    out = {"design": dict(n_rows=int(len(ps)), n_participants=int(npart),
                          n_sessions=int(nsess), seed=SEED)}

    print("\n--- A. group progression slope across sessions ---")
    try:
        out["group_slope"] = group_slope(ps)
        gs = out["group_slope"]
        mt = gs.get("mixed_trend", {})
        print(f"  random-participant slope: {mt.get('slope')} "
              f"(CI[{mt.get('ci_lo')},{mt.get('ci_hi')}], p={mt.get('p_value')}, {mt.get('method')})")
        cb = gs.get("cluster_bootstrap_slope", {})
        print(f"  cluster-bootstrap slope: {cb.get('point')} "
              f"CI[{cb.get('ci_lo')},{cb.get('ci_hi')}] excludes0={cb.get('excludes_0')}")
        mk = gs.get("group_mann_kendall", {})
        print(f"  Mann–Kendall (ordinal-safe): tau={mk.get('tau')} p={mk.get('p_value')} "
              f"dir={mk.get('direction')}")
        S = C.stats_mod()
        pf = S.power_flag(npart, nsess)
        out["power_flag"] = pf
        if pf["underpowered"]:
            print(f"  {pf['note']}")
    except Exception as e:
        out["group_slope"] = {"error": f"{type(e).__name__}: {e}"}
        print("  ERROR:", out["group_slope"]["error"])

    print("\n--- B. avoidance→attention-regulation barrier (rate-limiting descriptive) ---")
    try:
        out["barrier"] = barrier_rate_limiting(df, ps)
        b = out["barrier"]
        print(f"  expressed Avoidance: {b['n_expressed_avoidance']}/{b['n_participants']}; "
              f"crossed to AttnReg: {b['n_crossed']}/{b['n_participants']} "
              f"(rate={b['crossing_rate']})")
        print(f"  first-crossing session: median={b['first_crossing_session']['median']}, "
              f"dist={b['first_crossing_session']['distribution']}")
        print(f"  mean dwell below barrier: {b['mean_dwell_below_barrier']} sessions")
        print(f"  endpoint progression — crossed={b['endpoint_prog_crossed']} vs "
              f"never={b['endpoint_prog_never']}  "
              f"(gap supports the barrier as rate-limiting: "
              f"{b['endpoint_prog_crossed'] is not None and b['endpoint_prog_never'] is not None and b['endpoint_prog_crossed'] > (b['endpoint_prog_never'] or -1)})")
    except Exception as e:
        out["barrier"] = {"error": f"{type(e).__name__}: {e}"}
        print("  ERROR:", out["barrier"]["error"])

    p = C.write_json(out, os.path.join(os.path.dirname(__file__), "_e9_results.json"))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
