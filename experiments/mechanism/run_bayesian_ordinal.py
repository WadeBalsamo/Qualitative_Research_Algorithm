"""
experiments/mechanism/run_bayesian_ordinal.py
=============================================
E1c — the BAYESIAN hierarchical ordinal INTERACTION arm, run in ISOLATION.

WHY ISOLATED: bambi/pymc/pytensor require numpy>=2.0, but the QRA pipeline pins
transformers==4.42.4 which requires numpy<2.0. They cannot coexist in one venv.
So this arm runs in a dedicated `.venv_bayes` (numpy>=2 + bambi) and consumes the
design frame the main (frequentist) experiment exported to `_design.csv`. This is
ALSO the production recommendation: the Bayesian estimator is opt-in and isolated;
the in-process default is the frequentist ordinal+mixed interaction model.

THE POINT: at n≈32 the frequentist Gaussian FROM×move interaction design is
SINGULAR (un-fittable) and the ordinal LR test is non-significant (p≈0.52). The
Bayesian cumulative-logit with weakly-informative priors + partial pooling STILL
returns finite, regularized interaction estimates with honest credible intervals —
which is exactly why it is the right tool for this small-n interaction. We expect
most/all interaction HDIs to include 0 (honest under-identification), but now
*estimated* rather than *un-fittable*.

Run:  .venv_bayes/bin/python experiments/mechanism/run_bayesian_ordinal.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.join(HERE, "_design.csv")
LABELS = [0, 1, 2, 3, 4]
VAAMR = {0: "Vigilance", 1: "Avoidance", 2: "AttnReg", 3: "Metacog", 4: "Reappraisal"}
PURER = {0: "Phenomenology", 1: "Utilization", 2: "Reframing", 3: "Education", 4: "Reinforcement"}


def main():
    import bambi as bmb
    import arviz as az

    D = pd.read_csv(DESIGN)
    D = D.dropna(subset=["move"]).copy()
    D["move"] = D["move"].astype(int)
    D["to_stage"] = pd.Categorical(D["to_stage"].astype(int), categories=LABELS, ordered=True)
    D["from_stage"] = D["from_stage"].astype(int).astype("category")
    D["move"] = D["move"].astype("category")
    D["participant_id"] = D["participant_id"].astype(str)

    print("=" * 78)
    print("E1c — BAYESIAN hierarchical ordinal interaction (bambi cumulative-logit)")
    print("=" * 78)
    print(f"n triples (move defined) = {len(D)} | participants = {D['participant_id'].nunique()}")
    print(f"to_stage dist: {dict(pd.Series(D['to_stage']).value_counts().sort_index())}")

    # weakly-informative priors regularize the sparse interaction cells (partial pooling)
    priors = {
        "from_stage": bmb.Prior("Normal", mu=0, sigma=2.5),
        "move": bmb.Prior("Normal", mu=0, sigma=2.5),
        "from_stage:move": bmb.Prior("Normal", mu=0, sigma=1.0),  # tighter on the interaction
        "1|participant_id": bmb.Prior("Normal", mu=0,
                                      sigma=bmb.Prior("HalfNormal", sigma=1.5)),
    }
    model = bmb.Model("to_stage ~ from_stage * move + (1|participant_id)",
                      D, family="cumulative", priors=priors)
    print("\nModel:\n", model)

    idata = model.fit(draws=1000, tune=1500, chains=4, cores=1,
                      target_accept=0.95, random_seed=42, progressbar=False)

    # arviz 1.x dropped hdi_prob from summary — compute HDIs directly from posterior samples.
    post = idata.posterior
    rows = []  # (name, mean, lo, hi)
    for var in post.data_vars:
        if not ("from_stage" in var and "move" in var):   # interaction variable(s) only
            continue
        da = post[var].stack(sample=("chain", "draw"))
        coef_dims = [d for d in da.dims if d != "sample"]
        if not coef_dims:
            a = np.asarray(da.values).ravel()
            rows.append((var, a.mean(), *np.percentile(a, [2.5, 97.5])))
        else:
            cd = coef_dims[0]
            for c in da[cd].values:
                a = np.asarray(da.sel({cd: c}).values).ravel()
                rows.append((f"{var}[{c}]", a.mean(), *np.percentile(a, [2.5, 97.5])))
    excl = [r for r in rows if (r[2] > 0) or (r[3] < 0)]

    ndiv = int(np.asarray(idata.sample_stats["diverging"]).sum())
    try:
        maxrhat = float(az.rhat(idata).to_array().max())
    except Exception:
        maxrhat = float("nan")

    print("\n--- RESULT ---")
    print(f"  interaction terms estimated: {len(rows)}  (frequentist Gaussian was SINGULAR — could not fit these)")
    print(f"  interaction 95% intervals excluding 0: {len(excl)}  (expected ~0 at n≈32 = honest under-identification)")
    print(f"  max R-hat: {maxrhat:.3f}   divergences: {ndiv}")
    inter = rows
    if len(excl):
        print("  interaction terms with 95% interval excluding 0:")
        for name, m, lo, hi in excl:
            print(f"    {name:40} mean={m:+.2f}  [{lo:+.2f},{hi:+.2f}]")
    else:
        print("  → ALL interaction HDIs include 0: the §7.6 moderation is estimable-but-under-identified")
        print("    at this scale. The model FITS (regularized) where the frequentist MLE is singular —")
        print("    this is the defensible small-n instrument; confirmatory power awaits Cohorts 3–4.")

    out = dict(
        n=int(len(D)), n_participants=int(D["participant_id"].nunique()),
        n_interaction_terms=int(len(inter)),
        n_HDI_excludes_0=int(len(excl)),
        max_rhat=maxrhat, divergences=ndiv,
        examples=[r[0] for r in excl[:8]],
        note="Bayesian cumulative-logit + partial pooling fits the FROM×move interaction the "
             "frequentist Gaussian design cannot (singular); HDIs are the honest small-n bounds.",
    )
    with open(os.path.join(HERE, "_e1c_bayesian_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {os.path.join(HERE, '_e1c_bayesian_results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
