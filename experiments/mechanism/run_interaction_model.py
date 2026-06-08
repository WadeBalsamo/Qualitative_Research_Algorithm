"""
experiments/mechanism/run_interaction_model.py
==============================================
E1 + E2 of the mechanism campaign (masterplan.md §4).

THE QUESTION (H2 / methodology §7.6): does a therapist PURER move's effect on the
NEXT participant VAAMR stage depend on the participant's FROM stage? That is an
*interaction* (FROM_stage x move). The shipped mechanism estimator
(analysis/mechanism.py:_mixed_effects_delta -> `delta_prog ~ C(dominant_purer)`)
fits a move MAIN EFFECT only, Gaussian, no interaction. This script fits the
interaction with the right tools and asks whether it earns its place.

ARMS
  E1a  earns-its-place: participant-grouped CV held-out log-loss of multinomial
       logistic TO_stage models  FROM-only  vs  +move (additive)  vs  FROM*move.
  E1b  frequentist inference: (i) ordinal OrderedModel LR test additive vs
       interaction; (ii) Gaussian mixed model delta_prog ~ C(from)*C(move) +
       (1|participant) — count interaction contrasts whose 95% CI excludes 0.
  E1c  Bayesian hierarchical ordinal: bambi cumulative-logit
       TO_stage ~ from*move + (1|participant) — partial pooling regularizes the
       sparse cells; report interaction HDIs + divergences.
  E2   confound sensitivity: per (from_stage x move) cell E-value (VanderWeele-Ding)
       on the standardized mean delta vs the other moves at the same FROM stage.

Everything is observational, n~=32 — hypothesis-generating, never causal.
Reuses the canonical cue-block builder (process/cue_blocks.py). Nothing committed.

Run:  .venv/bin/python experiments/mechanism/run_interaction_model.py
"""
from __future__ import annotations
import os, sys, json, math, warnings
import importlib.util
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _load_module(name: str, relpath: str):
    """Direct file-path import — bypasses the heavy src/{process,analysis}/__init__.py
    chains (which pull sentence_transformers/transformers and are numpy-pin-sensitive).
    cue_blocks.py and stats.py have only stdlib/numpy module-level deps, so this is clean."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "src", relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses w/ `from __future__ import annotations` need this
    spec.loader.exec_module(mod)
    return mod


_cue_blocks = _load_module("qra_cue_blocks_standalone", "process/cue_blocks.py")
try:
    _S = _load_module("qra_stats_standalone", "analysis/stats.py")
except Exception:  # additive-FDR reproduction is optional
    _S = None

CSV = os.path.join(ROOT, "data", "Meta", "02_meta", "training_data", "master_segments.csv")
LABELS = [0, 1, 2, 3, 4]
PURER = {0: "Phenomenology", 1: "Utilization", 2: "Reframing", 3: "Education", 4: "Reinforcement"}
VAAMR = {0: "Vigilance", 1: "Avoidance", 2: "AttnReg", 3: "Metacog", 4: "Reappraisal"}


# ---------------------------------------------------------------- build triples
def build_design() -> pd.DataFrame:
    cue_blocks_from_records = _cue_blocks.cue_blocks_from_records
    df = pd.read_csv(CSV)
    specs = cue_blocks_from_records(df.to_dict("records"), stage_key="final_label", require_stage=True)
    rows = []
    for sp in specs:
        fi, ti = sp.from_item, sp.to_item
        purers = [int(t["purer_primary"]) for t in sp.therapist_items
                  if pd.notna(t.get("purer_primary"))]
        move = Counter(purers).most_common(1)[0][0] if purers else None
        fc, tc = fi.get("progression_coord"), ti.get("progression_coord")
        delta = (tc - fc) if (pd.notna(fc) and pd.notna(tc)) else np.nan
        rows.append(dict(
            participant_id=str(fi.get("participant_id")),
            session_number=fi.get("session_number"),
            from_stage=int(sp.from_stage), to_stage=int(sp.to_stage),
            move=move, delta_prog=delta, transition_type=sp.transition_type,
            n_therapist=len(sp.therapist_items),
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------- E1a earns-its-place
def _padded_proba(clf, X, classes_full):
    """Align predict_proba columns to the full label set (missing class -> ~0)."""
    p = clf.predict_proba(X)
    out = np.full((X.shape[0], len(classes_full)), 1e-9)
    for j, c in enumerate(clf.classes_):
        out[:, classes_full.index(c)] = p[:, j]
    return out / out.sum(axis=1, keepdims=True)


def earns_its_place(D: pd.DataFrame, seed: int = 42) -> dict:
    import patsy
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import log_loss, accuracy_score

    Dm = D.dropna(subset=["move"]).copy()
    Dm["move"] = Dm["move"].astype(int)
    y = Dm["to_stage"].astype(int).values
    groups = Dm["participant_id"].astype(str).values

    specs = {
        "FROM_only":   "C(from_stage)",
        "additive":    "C(from_stage) + C(move)",
        "interaction": "C(from_stage) * C(move)",
    }
    Xs = {k: patsy.dmatrix(f, Dm, return_type="dataframe") for k, f in specs.items()}

    n_splits = min(5, len(np.unique(groups)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))

    res = {}
    for name, X in Xs.items():
        Xv = X.values
        lls, accs = [], []
        for tr, te in folds:
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit(Xv[tr], y[tr])
            proba = _padded_proba(clf, Xv[te], LABELS)
            lls.append(log_loss(y[te], proba, labels=LABELS))
            accs.append(accuracy_score(y[te], np.array(LABELS)[proba.argmax(1)]))
        res[name] = dict(logloss=float(np.mean(lls)), logloss_sd=float(np.std(lls)),
                         acc=float(np.mean(accs)), n_params=int(X.shape[1]))
    res["_n"] = int(len(y))
    res["_n_participants"] = int(len(np.unique(groups)))
    res["_n_folds"] = n_splits
    return res


# ------------------------------------------------------- E1b frequentist inference
def freq_inference(D: pd.DataFrame) -> dict:
    import patsy
    from scipy.stats import chi2
    out = {}

    # (i) ordinal OrderedModel LR test additive vs interaction
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        Dm = D.dropna(subset=["move"]).copy()
        Dm["move"] = Dm["move"].astype(int)
        y = Dm["to_stage"].astype(int).values

        def llf(formula):
            X = patsy.dmatrix(formula, Dm, return_type="dataframe").drop(columns=["Intercept"])
            r = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False, maxiter=300)
            return r.llf, X.shape[1]

        ll_a, k_a = llf("C(from_stage) + C(move)")
        ll_i, k_i = llf("C(from_stage) * C(move)")
        lr = 2.0 * (ll_i - ll_a); ddf = k_i - k_a
        out["ordinal_LR"] = dict(ll_additive=round(ll_a, 2), ll_interaction=round(ll_i, 2),
                                 LR=round(lr, 2), df=int(ddf),
                                 p=float(chi2.sf(max(lr, 0), max(ddf, 1))))
    except Exception as e:
        out["ordinal_LR"] = {"error": f"{type(e).__name__}: {e}"}

    # (ii) Gaussian mixed model delta ~ C(from)*C(move) + (1|participant)
    try:
        import statsmodels.formula.api as smf
        Dd = D.dropna(subset=["move", "delta_prog"]).copy()
        Dd["move"] = Dd["move"].astype(int)
        m = smf.mixedlm("delta_prog ~ C(from_stage)*C(move)", Dd, groups=Dd["participant_id"])
        r = m.fit(disp=False)
        params, ci = r.params, r.conf_int()
        inter = [ix for ix in params.index if ":" in ix]
        excl = [ix for ix in inter if (ci.loc[ix, 0] > 0) or (ci.loc[ix, 1] < 0)]
        out["gaussian_mixed_interaction"] = dict(
            n=int(len(Dd)), n_interaction_terms=len(inter),
            n_CI_excludes_0=len(excl), examples=excl[:6],
            note="Δprogression ~ FROM×move + (1|participant); the interaction the shipped model omits",
        )
    except Exception as e:
        out["gaussian_mixed_interaction"] = {"error": f"{type(e).__name__}: {e}"}

    # (iii) reproduce the additive per-cell FDR table (the shipped 'mechanism' read)
    try:
        S = _S
        if S is None:
            raise RuntimeError("stats module unavailable")
        Dm = D.dropna(subset=["move"]).copy()
        Dm["move"] = Dm["move"].astype(int)
        Dm["delta_prog"] = Dm["delta_prog"].astype(float)
        by_stage = {s: g for s, g in Dm.groupby("from_stage")}
        rows, pvals = [], []
        for (fs, mv), g in Dm.groupby(["from_stage", "move"]):
            if len(g) < 2 or g["delta_prog"].isna().all():
                continue
            stage_g = by_stage[fs]
            vals = stage_g["delta_prog"].fillna(0.0).tolist()
            mask = (stage_g["move"] == mv).tolist()
            perm = S.permutation_test(vals, mask, strata=None, n_perm=1000) if len(stage_g) > len(g) else {"p_value": float("nan")}
            rows.append(dict(from_stage=int(fs), move=int(mv), n=int(len(g)),
                             mean_delta=round(float(g["delta_prog"].mean()), 3),
                             perm_p=perm["p_value"]))
            pvals.append(perm["p_value"])
        bh = S.benjamini_hochberg(pvals, alpha=0.05)
        n_sig = int(sum(bool(x) for x in bh["reject"]))
        out["additive_percell_FDR"] = dict(n_cells=len(rows), n_FDR_significant=n_sig,
                                           note="the shipped per-cell read; 0 significant = under-powered at n≈32")
    except Exception as e:
        out["additive_percell_FDR"] = {"error": f"{type(e).__name__}: {e}"}

    return out


# ------------------------------------------------ E1c Bayesian hierarchical ordinal
def bayesian_ordinal(D: pd.DataFrame, draws=500, tune=500, chains=2, seed=42) -> dict:
    try:
        import bambi as bmb
        import arviz as az
        Db = D.dropna(subset=["move"]).copy()
        Db["move"] = Db["move"].astype(int)
        Db["to_stage"] = pd.Categorical(Db["to_stage"].astype(int), categories=LABELS, ordered=True)
        Db["from_stage"] = Db["from_stage"].astype("category")
        Db["move"] = Db["move"].astype("category")
        model = bmb.Model("to_stage ~ from_stage * move + (1|participant_id)",
                          Db, family="cumulative")
        idata = model.fit(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=0.9, random_seed=seed, progressbar=False)
        summ = az.summary(idata, hdi_prob=0.95)
        inter = summ[summ.index.str.contains(":") & summ.index.str.contains("from_stage")]
        excl = inter[(inter["hdi_2.5%"] > 0) | (inter["hdi_97.5%"] < 0)]
        ndiv = int(np.asarray(idata.sample_stats["diverging"]).sum()) if "diverging" in idata.sample_stats else -1
        return dict(ok=True, n=int(len(Db)),
                    n_interaction_terms=int(len(inter)),
                    n_HDI_excludes_0=int(len(excl)),
                    max_rhat=float(summ["r_hat"].max()),
                    divergences=ndiv,
                    examples=list(excl.index[:6]),
                    note="cumulative-logit + partial pooling; weak default priors shrink sparse cells")
    except Exception as e:
        import traceback
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}


# ----------------------------------------------------------------- E2 E-values
def e_value(rr: float) -> float:
    if rr <= 0 or not np.isfinite(rr):
        return float("nan")
    if rr < 1:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1.0))


def sensitivity(D: pd.DataFrame, min_n=4) -> dict:
    Dm = D.dropna(subset=["move", "delta_prog"]).copy()
    Dm["move"] = Dm["move"].astype(int)
    cells = []
    for fs, g in Dm.groupby("from_stage"):
        for mv, gc in g.groupby("move"):
            if len(gc) < min_n:
                continue
            rest = g[g["move"] != mv]["delta_prog"]
            if len(rest) < 2:
                continue
            m1, m0 = gc["delta_prog"].mean(), rest.mean()
            sd = math.sqrt((gc["delta_prog"].var(ddof=1) * (len(gc) - 1) +
                            rest.var(ddof=1) * (len(rest) - 1)) /
                           max(len(gc) + len(rest) - 2, 1))
            d = (m1 - m0) / sd if sd > 0 else 0.0
            rr = math.exp(0.91 * d)           # Chinn/VanderWeele SMD -> approx RR
            cells.append(dict(from_stage=int(fs), from_name=VAAMR[fs], move=int(mv),
                              move_name=PURER[mv], n=int(len(gc)),
                              smd=round(d, 3), approx_RR=round(rr, 3),
                              e_value=round(e_value(rr), 3)))
    cells.sort(key=lambda c: -abs(c["smd"]))
    return dict(n_cells=len(cells), cells=cells,
                note="E-value = min strength (on the RR scale) an unmeasured confounder would need "
                     "with both move-selection and Δprogression to explain away the cell association")


# --------------------------------------------------------------------- main
def main():
    print("=" * 78)
    print("E1+E2 — hierarchical ordinal INTERACTION mechanism model (FROM_stage × move)")
    print("=" * 78)
    D = build_design()
    D.to_csv(os.path.join(os.path.dirname(__file__), "_design.csv"), index=False)
    n_move = D["move"].notna().sum()
    print(f"\nTriples: {len(D)}  |  participants: {D['participant_id'].nunique()}  "
          f"|  with a defined CUE move: {n_move}  |  with Δprog: {D['delta_prog'].notna().sum()}")
    print("Transition mix:", dict(D["transition_type"].value_counts()))

    print("\n--- E1a  earns-its-place (participant-grouped CV held-out log-loss; lower=better) ---")
    eip = earns_its_place(D)
    print(f"  n={eip['_n']}  participants={eip['_n_participants']}  folds={eip['_n_folds']}")
    for name in ("FROM_only", "additive", "interaction"):
        r = eip[name]
        print(f"  {name:12} logloss={r['logloss']:.4f}±{r['logloss_sd']:.3f}  "
              f"acc={r['acc']:.3f}  params={r['n_params']}")
    base = eip["FROM_only"]["logloss"]
    print(f"  → does the cue EARN its place? additive Δlogloss vs FROM-only = "
          f"{eip['additive']['logloss']-base:+.4f}  "
          f"interaction Δ = {eip['interaction']['logloss']-base:+.4f}  (negative=better)")

    print("\n--- E1b  frequentist interaction inference ---")
    fr = freq_inference(D)
    print("  ordinal LR (additive vs interaction):", json.dumps(fr["ordinal_LR"]))
    print("  gaussian mixed interaction:", json.dumps(fr["gaussian_mixed_interaction"]))
    print("  additive per-cell FDR (shipped read):", json.dumps(fr["additive_percell_FDR"]))

    print("\n--- E1c  Bayesian hierarchical ordinal (bambi cumulative-logit + partial pooling) ---")
    bay = bayesian_ordinal(D)
    if bay.get("ok"):
        print(f"  n={bay['n']}  interaction_terms={bay['n_interaction_terms']}  "
              f"HDI-excludes-0={bay['n_HDI_excludes_0']}  max_rhat={bay['max_rhat']:.3f}  "
              f"divergences={bay['divergences']}")
        print(f"  examples (interaction terms w/ 95% HDI excluding 0): {bay['examples']}")
    else:
        print("  Bayesian arm error:", bay.get("error"))
        print(bay.get("tb", ""))

    print("\n--- E2  confound sensitivity (E-values per FROM×move cell) ---")
    sens = sensitivity(D)
    print(f"  {sens['n_cells']} cells (n≥4). Top by |SMD|:")
    for c in sens["cells"][:10]:
        print(f"    {c['from_name']:11}×{c['move_name']:13} n={c['n']:2}  "
              f"SMD={c['smd']:+.2f}  approxRR={c['approx_RR']:.2f}  E-value={c['e_value']:.2f}")

    # persist
    outp = os.path.join(os.path.dirname(__file__), "_e1e2_results.json")
    with open(outp, "w") as f:
        json.dump(dict(design=dict(n=len(D), n_participants=int(D["participant_id"].nunique()),
                                   n_move=int(n_move)),
                       e1a_earns_its_place=eip, e1b_frequentist=fr,
                       e1c_bayesian=bay, e2_sensitivity=sens), f, indent=2, default=str)
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
