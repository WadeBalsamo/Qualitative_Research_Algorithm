"""
experiments/mechanism/run_e4_trajectory.py
==========================================
E4 — trajectory + within/between split + dyadic routines (masterplan §4; Q18/Q19/Q24).

The adjacent FROM→CUE→TO triple captures a *momentary* nudge; learning is a
*between-session consolidation*. This script:

  A. TRAJECTORY MODEL — per-(participant,session) modal VAAMR stage ~ lagged cue
     exposure + session_number + (1|participant). Mixed (Gaussian, ordinal-caveated)
     primary; statsmodels OrderedModel robustness (no random effect).
  B. WITHIN vs BETWEEN move effects —
       within  : mean Δprogression per dominant move on the adjacent triples
                 (momentary, from the cue blocks).
       between : Δ(modal stage) from session s→s+1 vs the dominant cue move the
                 participant received in session s (consolidation).
  C. DYADIC ROUTINES vs CONTENT-CO-OCCURRENCE NULL — observed therapist move→move
     bigram counts within sessions vs a within-session move-sequence shuffle null
     (preserves each session's move multiset; Q24 "routines beyond co-occurrence").

Observational, n≈20 participants — hypothesis-generating; under-identification expected.
Run:  .venv/bin/python experiments/mechanism/run_e4_trajectory.py
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import _common as C

SEED = 42
FORWARD_MOVES = {1, 2}   # Utilization + Reframing = prompting forward application / reframing


# ----------------------------------------------------------- per-session modal stage
def session_modal(df: pd.DataFrame) -> pd.DataFrame:
    """Per (participant, session): modal VAAMR stage + mean progression coord."""
    pp = C.participant_labeled(df)
    rows = []
    for (pid, snum), g in pp.groupby(["participant_id", "session_number"]):
        modal = int(g["final_label"].value_counts().idxmax())
        prog = float(g["progression_coord"].dropna().mean()) if g["progression_coord"].notna().any() \
            else float(g["final_label"].mean())
        rows.append(dict(participant_id=str(pid), session_number=int(snum),
                         modal_stage=modal, prog=prog, n=len(g)))
    return pd.DataFrame(rows).sort_values(["participant_id", "session_number"]).reset_index(drop=True)


def session_cue_exposure(D: pd.DataFrame) -> pd.DataFrame:
    """Per (participant, session): cue exposure from that session's blocks —
    fraction of forward-application moves (U/R) + dominant move + n_blocks."""
    rows = []
    Dm = D.dropna(subset=["move"]).copy()
    Dm["move"] = Dm["move"].astype(int)
    for (pid, snum), g in Dm.groupby(["participant_id", "session_number"]):
        moves = g["move"].tolist()
        frac_fwd = float(np.mean([m in FORWARD_MOVES for m in moves])) if moves else np.nan
        dom = int(pd.Series(moves).value_counts().idxmax()) if moves else None
        rows.append(dict(participant_id=str(pid), session_number=int(snum),
                         cue_frac_forward=frac_fwd, cue_dom_move=dom, n_blocks=len(moves)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------- A. trajectory model
def trajectory_model(modal: pd.DataFrame, expo: pd.DataFrame) -> dict:
    out = {}
    # lag cue exposure by one session: predictor = exposure in session s-1
    expo = expo.copy()
    expo["session_next"] = expo["session_number"] + 1
    lag = expo.rename(columns={"cue_frac_forward": "lag_cue_frac_forward",
                               "n_blocks": "lag_n_blocks"})[
        ["participant_id", "session_next", "lag_cue_frac_forward", "lag_n_blocks"]]
    M = modal.merge(lag, left_on=["participant_id", "session_number"],
                    right_on=["participant_id", "session_next"], how="left")
    M = M.dropna(subset=["lag_cue_frac_forward"]).copy()
    out["n_rows"] = int(len(M))
    out["n_participants"] = int(M["participant_id"].nunique())
    if len(M) < 6 or M["participant_id"].nunique() < 2:
        out["error"] = "insufficient lagged session pairs"
        return out
    # (i) Gaussian mixed model: modal_stage ~ lag_cue + session_number + (1|participant)
    try:
        import statsmodels.formula.api as smf
        r = smf.mixedlm("modal_stage ~ lag_cue_frac_forward + session_number",
                        M, groups=M["participant_id"]).fit(disp=False)
        ci = r.conf_int()
        out["mixed"] = {}
        for term in ("lag_cue_frac_forward", "session_number"):
            if term in r.params.index:
                out["mixed"][term] = dict(
                    coef=round(float(r.params[term]), 4),
                    ci_lo=round(float(ci.loc[term, 0]), 4),
                    ci_hi=round(float(ci.loc[term, 1]), 4),
                    p=round(float(r.pvalues[term]), 4),
                    excludes_0=bool((ci.loc[term, 0] > 0) or (ci.loc[term, 1] < 0)))
    except Exception as e:
        out["mixed"] = {"error": f"{type(e).__name__}: {e}"}
    # (ii) ordinal robustness (no random effect): OrderedModel on modal_stage
    try:
        import patsy
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        X = patsy.dmatrix("lag_cue_frac_forward + session_number", M,
                          return_type="dataframe").drop(columns=["Intercept"])
        rr = OrderedModel(M["modal_stage"].astype(int).to_numpy(), X,
                          distr="logit").fit(method="bfgs", disp=False, maxiter=200)
        out["ordinal"] = {t: dict(coef=round(float(rr.params[t]), 4),
                                  p=round(float(rr.pvalues[t]), 4))
                          for t in ("lag_cue_frac_forward", "session_number") if t in rr.params.index}
    except Exception as e:
        out["ordinal"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ----------------------------------------------------------- B. within vs between
def within_between(D: pd.DataFrame, modal: pd.DataFrame, expo: pd.DataFrame) -> dict:
    S = C.stats_mod()
    out = {}
    # within: mean Δprogression per dominant move on adjacent triples
    Dm = D.dropna(subset=["move", "delta_prog"]).copy()
    Dm["move"] = Dm["move"].astype(int)
    within = []
    for mv, g in Dm.groupby("move"):
        ci = S.cluster_bootstrap_ci(g["delta_prog"].to_numpy(), g["participant_id"].to_numpy(),
                                    statistic=np.nanmean, n_boot=1000, seed=SEED)
        within.append(dict(move=int(mv), move_name=C.PURER[mv], n=int(len(g)),
                           mean_delta=round(ci["point"], 4),
                           ci_lo=None if ci["lo"] != ci["lo"] else round(ci["lo"], 4),
                           ci_hi=None if ci["hi"] != ci["hi"] else round(ci["hi"], 4)))
    within.sort(key=lambda r: -r["mean_delta"])
    out["within_session_move_effects"] = within

    # between: Δ(modal stage) s->s+1 vs dominant cue move in session s
    m = modal.merge(expo, on=["participant_id", "session_number"], how="left").copy()
    m = m.sort_values(["participant_id", "session_number"]).reset_index(drop=True)
    rows = []
    for pid, g in m.groupby("participant_id"):
        g = g.sort_values("session_number")
        prev = None
        for _, r in g.iterrows():
            if prev is not None and r["session_number"] == prev["session_number"] + 1:
                if pd.notna(prev["cue_dom_move"]):
                    rows.append(dict(participant_id=pid,
                                     dom_move=int(prev["cue_dom_move"]),
                                     d_modal=int(r["modal_stage"]) - int(prev["modal_stage"])))
            prev = r
    bt = pd.DataFrame(rows)
    between = []
    if len(bt):
        for mv, g in bt.groupby("dom_move"):
            ci = S.cluster_bootstrap_ci(g["d_modal"].to_numpy(dtype=float),
                                        g["participant_id"].to_numpy(),
                                        statistic=np.nanmean, n_boot=1000, seed=SEED)
            between.append(dict(move=int(mv), move_name=C.PURER[mv], n=int(len(g)),
                               mean_d_modal=round(ci["point"], 4),
                               ci_lo=None if ci["lo"] != ci["lo"] else round(ci["lo"], 4),
                               ci_hi=None if ci["hi"] != ci["hi"] else round(ci["hi"], 4)))
        between.sort(key=lambda r: -r["mean_d_modal"])
    out["between_session_move_effects"] = between
    out["between_n_pairs"] = int(len(bt))
    return out


# ----------------------------------------------------------- C. routines vs null
def routines_vs_null(D: pd.DataFrame, n_perm: int = 2000) -> dict:
    """Observed therapist move→move bigram counts within sessions vs a
    within-session move-sequence shuffle null (preserves the session's move multiset)."""
    # ordered move sequence per session (defined moves only, in chronological block order)
    Dm = D.copy()
    seqs = []
    for sid, g in Dm.groupby("session_id", sort=False):
        moves = [int(m) for m in g["move"].tolist() if pd.notna(m)]
        if len(moves) >= 2:
            seqs.append(moves)
    if not seqs:
        return {"error": "no session has >=2 defined cue moves"}

    def bigrams(seqs):
        c = {}
        for s in seqs:
            for a, b in zip(s[:-1], s[1:]):
                c[(a, b)] = c.get((a, b), 0) + 1
        return c

    obs = bigrams(seqs)
    rng = np.random.default_rng(SEED)
    keys = list(obs.keys())
    ge = {k: 0 for k in keys}
    for _ in range(n_perm):
        perm_seqs = []
        for s in seqs:
            ss = s.copy()
            rng.shuffle(ss)
            perm_seqs.append(ss)
        pc = bigrams(perm_seqs)
        for k in keys:
            if pc.get(k, 0) >= obs[k]:
                ge[k] += 1
    rows = []
    for k in keys:
        p = (ge[k] + 1) / (n_perm + 1)
        rows.append(dict(transition=f"{C.PURER[k[0]]}->{C.PURER[k[1]]}",
                         from_move=int(k[0]), to_move=int(k[1]),
                         observed=int(obs[k]), perm_p=round(float(p), 4)))
    rows.sort(key=lambda r: (r["perm_p"], -r["observed"]))
    enriched = [r for r in rows if r["perm_p"] < 0.05]
    return dict(n_sessions_with_routine=len(seqs), n_bigram_types=len(keys),
                n_perm=n_perm, transitions=rows,
                n_exceed_null_p05=len(enriched),
                exceed_null=enriched,
                note="self-transitions are common; null preserves each session's move "
                     "multiset so only ORDER structure is tested.")


def main() -> int:
    print("=" * 78)
    print("E4 — trajectory + within/between split + dyadic routines vs content null")
    print("=" * 78)
    df = C.load_df()
    D = C.build_design(df, with_text=False)
    modal = session_modal(df)
    expo = session_cue_exposure(D)
    print(f"\nper-(participant,session) modal-stage rows: {len(modal)}  "
          f"participants: {modal['participant_id'].nunique()}")

    out = {"design": dict(n_modal_rows=int(len(modal)),
                          n_participants=int(modal["participant_id"].nunique()), seed=SEED)}

    print("\n--- A. trajectory: modal_stage ~ lagged cue exposure + session + (1|participant) ---")
    try:
        out["trajectory"] = trajectory_model(modal, expo)
        tr = out["trajectory"]
        print(f"  lagged session pairs n={tr.get('n_rows')} ({tr.get('n_participants')} participants)")
        if isinstance(tr.get("mixed"), dict) and "lag_cue_frac_forward" in tr["mixed"]:
            lc = tr["mixed"]["lag_cue_frac_forward"]
            print(f"  mixed: lag_cue_forward coef={lc['coef']} CI[{lc['ci_lo']},{lc['ci_hi']}] "
                  f"p={lc['p']} excludes0={lc['excludes_0']}")
        if isinstance(tr.get("ordinal"), dict) and "lag_cue_frac_forward" in tr["ordinal"]:
            print(f"  ordinal: lag_cue_forward coef={tr['ordinal']['lag_cue_frac_forward']['coef']} "
                  f"p={tr['ordinal']['lag_cue_frac_forward']['p']}")
    except Exception as e:
        out["trajectory"] = {"error": f"{type(e).__name__}: {e}"}
        print("  ERROR:", out["trajectory"]["error"])

    print("\n--- B. within-session (momentary) vs between-session (consolidation) move effects ---")
    try:
        out["within_between"] = within_between(D, modal, expo)
        wb = out["within_between"]
        print("  within (mean Δprog per move, top):")
        for r in wb["within_session_move_effects"][:5]:
            print(f"    {r['move_name']:13} n={r['n']:2} mean_delta={r['mean_delta']:+.3f} "
                  f"CI[{r['ci_lo']},{r['ci_hi']}]")
        print(f"  between (Δmodal s->s+1 by session dom move; {wb['between_n_pairs']} pairs):")
        for r in wb["between_session_move_effects"][:5]:
            print(f"    {r['move_name']:13} n={r['n']:2} mean_d_modal={r['mean_d_modal']:+.3f} "
                  f"CI[{r['ci_lo']},{r['ci_hi']}]")
    except Exception as e:
        out["within_between"] = {"error": f"{type(e).__name__}: {e}"}
        print("  ERROR:", out["within_between"]["error"])

    print("\n--- C. dyadic routines (therapist move→move) vs within-session shuffle null ---")
    try:
        out["routines"] = routines_vs_null(D)
        rt = out["routines"]
        print(f"  {rt['n_sessions_with_routine']} sessions w/ a routine; "
              f"{rt['n_bigram_types']} bigram types; "
              f"{rt['n_exceed_null_p05']} exceed the content-co-occurrence null (p<.05)")
        for r in rt["transitions"][:6]:
            print(f"    {r['transition']:28} obs={r['observed']:2} perm_p={r['perm_p']:.3f}")
    except Exception as e:
        out["routines"] = {"error": f"{type(e).__name__}: {e}"}
        print("  ERROR:", out["routines"]["error"])

    p = C.write_json(out, os.path.join(os.path.dirname(__file__), "_e4_results.json"))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
