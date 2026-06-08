"""
experiments/mechanism/run_e6_h6_robustness.py
=============================================
E6 — H6 discriminant-validity robustness (masterplan §4; Q8/Q27/Q28). HEAVY/OPTIONAL.

H6 (the program's strongest claim): VAAMR is DEVELOPMENTAL, not topical — a supervised
probe on segment embeddings beats a CONTENT-SIMILARITY model at recovering the VAAMR
stage (probe ≫ similarity; Δκ excludes 0). The shipped instrument (gnn_layer/discriminant.py)
uses Qwen3-8B embeddings via LM Studio. This script:

  - Q8 (embedding robustness): re-runs the contrast on all-MiniLM-L6-v2 (locally cached,
    a *different* encoder than Qwen) — if the H6 sign survives a weaker encoder it is more
    general. The Qwen arm is ATTEMPTED only if the LM Studio endpoint answers; else skipped.
  - Q27/Q28 (5 vs 6 class): runs both the 5-class (labeled-only) and the 6-class
    (+"No code") decompositions, to see how much "No code" carries the contrast.

Instruments, on shared participant-grouped folds + identical MiniLM embeddings:
  PROBE          multinomial logistic regression on the embedding (learns a stage direction)
  CONTENT-SIM    kNN-majority over training neighbours by cosine (labels follow content)
Headline = Δκ = κ(probe) − κ(content-sim), participant-clustered bootstrap CI.

Observational; κ magnitudes are small at n≈26, but H6 is a CONTRAST so it is N-robuster.
Run:  .venv/bin/python experiments/mechanism/run_e6_h6_robustness.py
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import _common as C

SEED = 42
KNN = 10


def _oof_predictions(X, y, groups, n_classes, knn=KNN, seed=SEED):
    """Participant-grouped out-of-fold predictions for PROBE and CONTENT-SIM."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    folds, n_splits = C.stratified_group_folds(y, groups, n_splits=5, seed=seed)
    probe = np.full(len(y), -1, dtype=int)
    csim = np.full(len(y), -1, dtype=int)
    homophily = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # PROBE
        clf = LogisticRegression(max_iter=4000, C=1.0)
        clf.fit(Xtr, y[tr])
        probe[te] = clf.predict(Xte)
        # CONTENT-SIM: kNN-majority over TRAIN neighbours by cosine
        sim = cosine_similarity(Xte, Xtr)
        k = min(knn, len(tr))
        nn = np.argsort(-sim, axis=1)[:, :k]
        for i in range(len(te)):
            labs = y[tr][nn[i]]
            csim[te[i]] = int(np.bincount(labs, minlength=n_classes).argmax())
            homophily.append(float(np.mean(labs == y[te[i]])))
    return probe, csim, float(np.mean(homophily)) if homophily else float("nan"), n_splits


def _contrast(X, y, groups, n_classes, label):
    probe, csim, homophily, n_splits = _oof_predictions(X, y, groups, n_classes)
    kp = C.kappa_cluster_ci(probe.tolist(), y.tolist(), groups, seed=SEED)
    kc = C.kappa_cluster_ci(csim.tolist(), y.tolist(), groups, seed=SEED)
    # paired Δκ bootstrap (resample participants once, recompute both κ, take diff)
    S = C.stats_mod()
    packed = np.arange(len(y), dtype=float)

    def _dkappa(idx_arr):
        idx = idx_arr.astype(int)
        a = C.cohen_kappa(probe[idx].tolist(), y[idx].tolist())
        b = C.cohen_kappa(csim[idx].tolist(), y[idx].tolist())
        if a != a or b != b:
            return float("nan")
        return float(a - b)

    dci = S.cluster_bootstrap_ci(packed, list(groups), statistic=_dkappa,
                                 n_boot=2000, seed=SEED)
    return dict(
        label=label, n=int(len(y)), n_classes=n_classes, n_folds=n_splits,
        kappa_probe=round(kp["point"], 4), probe_ci=[kp["lo"], kp["hi"]],
        kappa_content_sim=round(kc["point"], 4), content_ci=[kc["lo"], kc["hi"]],
        delta_kappa=round(kp["point"] - kc["point"], 4),
        delta_kappa_ci=[None if dci["lo"] != dci["lo"] else round(dci["lo"], 4),
                        None if dci["hi"] != dci["hi"] else round(dci["hi"], 4)],
        delta_excludes_0=bool(dci["lo"] == dci["lo"] and dci["hi"] == dci["hi"]
                              and (dci["lo"] > 0 or dci["hi"] < 0)),
        knn_homophily=round(homophily, 4),
        note="Δκ>0 with CI excluding 0 => the supervised probe recovers VAAMR stage better "
             "than content similarity => developmental, not topical (H6).")


def _build_xy(df, encoder, with_nocode):
    """Return (X embeddings, y labels, groups). 5-class=labeled only; 6-class=+No code(5)."""
    part = df[C.is_participant(df)].copy()
    part["segment_id"] = part["segment_id"].astype(str)
    if with_nocode:
        sub = part.copy()
        y = sub["final_label"].apply(lambda v: int(v) if pd.notna(v) else 5).to_numpy()
        n_classes = 6
    else:
        sub = part[part["final_label"].notna()].copy()
        y = sub["final_label"].astype(int).to_numpy()
        n_classes = 5
    groups = sub["participant_id"].astype(str).to_numpy()
    if encoder == "minilm":
        X = C.embed_minilm(sub["text"].tolist(),
                           cache_path=os.path.join(os.path.dirname(__file__),
                                                   "_emb_participant_minilm.npz"),
                           ids=sub["segment_id"].tolist())
    else:
        raise ValueError(encoder)
    return np.asarray(X, dtype=float), y, groups, n_classes


def _qwen_status() -> str:
    """Actually attempt a Qwen embedding call (not just /v1/models) and report the
    precise status — the model is often LISTED but fails to load (GPU busy / OOM)."""
    import urllib.request, urllib.error, json
    try:
        body = json.dumps({"model": "text-embedding-qwen3-embedding-8b",
                           "input": ["back pain"]}).encode()
        req = urllib.request.Request("http://10.0.0.58:1234/v1/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
            dim = len(d["data"][0]["embedding"])
            return f"REACHABLE+LOADS (dim={dim}) — run src/gnn_layer/discriminant.py for the Qwen arm"
    except urllib.error.HTTPError as e:
        msg = ""
        try:
            msg = e.read().decode()[:160]
        except Exception:
            pass
        return f"UNAVAILABLE — endpoint up but embeddings HTTP {e.code}: {msg}"
    except Exception as e:
        return f"UNAVAILABLE — {type(e).__name__}: {e}"


def main() -> int:
    print("=" * 78)
    print("E6 — H6 robustness: probe vs content-similarity κ contrast (MiniLM; 5 vs 6 class)")
    print("=" * 78)
    df = C.load_df()
    out = {"design": dict(seed=SEED, knn=KNN), "arms": {}}

    # MiniLM embeddings — load once via the 6-class (all participant) build, reuse subset
    try:
        for with_nocode, tag in [(False, "minilm_5class"), (True, "minilm_6class")]:
            X, y, groups, n_classes = _build_xy(df, "minilm", with_nocode)
            print(f"\n[{tag}] n={len(y)} segments, {len(np.unique(groups))} participants, "
                  f"{n_classes} classes")
            res = _contrast(X, y, groups, n_classes, tag)
            out["arms"][tag] = res
            print(f"  κ(probe)={res['kappa_probe']}  κ(content-sim)={res['kappa_content_sim']}  "
                  f"Δκ={res['delta_kappa']} CI[{res['delta_kappa_ci'][0]},{res['delta_kappa_ci'][1]}]  "
                  f"excludes0={res['delta_excludes_0']}  kNN-homophily={res['knn_homophily']}")
    except Exception as e:
        import traceback
        out["arms"]["minilm"] = {"error": f"{type(e).__name__}: {e}",
                                 "tb": traceback.format_exc()[-600:]}
        print(f"\nMiniLM arm ERROR: {out['arms']['minilm']['error']}")

    # Qwen arm — probe the real embeddings call; defer the full arm to discriminant.py
    qstat = _qwen_status()
    out["qwen"] = {"status": qstat,
                   "needed": "the shipped Qwen H6 result uses Qwen3-8B embeddings + the "
                             "Correct-&-Smooth content model in src/gnn_layer/discriminant.py; "
                             "this lightweight pass cannot adjudicate the encoder cleanly without it"}
    print(f"\nQwen arm: {qstat}")

    # headline: does the H6 sign survive a weaker (MiniLM) encoder, and on both class counts?
    headline = {}
    for tag in ("minilm_5class", "minilm_6class"):
        a = out["arms"].get(tag, {})
        if "delta_kappa" in a:
            headline[tag] = dict(delta_kappa=a["delta_kappa"], excludes_0=a["delta_excludes_0"])
    out["headline"] = headline
    out["interpretation"] = (
        "ROBUSTNESS CAVEAT, NOT an H6 refutation. The shipped H6 (discriminant.py) finds "
        "probe ≫ content-similarity (Δκ ~ +0.17 human / +0.21 LLM) on QWEN3-8B embeddings with "
        "a Correct-&-Smooth content model. Here, on the WEAKER all-MiniLM-L6-v2 encoder and a "
        "kNN-majority content instrument, the sign FLIPS: content-similarity matches/edges out "
        "the linear probe (Δκ -0.05 5-class CI incl. 0; -0.09 6-class CI EXCLUDES 0). Two honest, "
        "non-exclusive reasons: (a) MiniLM-384d carries less linearly-separable VAAMR signal than "
        "Qwen-4096d, so a nonparametric kNN extracts more local structure than a linear probe — a "
        "CLASSIFIER-CAPACITY effect, not proof of topicality; (b) the 6-class 'No code' class is "
        "strongly content-clustered (kNN homophily rises 0.30->0.36 and content-sim's edge grows "
        "with 'No code' added), so Q27/Q28: 'No code' inflates the CONTENT model, not the probe. "
        "Bottom line: H6's probe≫content contrast is NOT shown to be encoder-robust by this proxy; "
        "a faithful Q8 test must re-run the SAME discriminant.py probe-vs-C&S instrument on MiniLM "
        "AND Qwen. The Qwen embedding model was listed but failed to load at run time.")
    out["note"] = ("Q8: H6 robustness to encoder choice (MiniLM, weaker than Qwen). "
                   "Q27/Q28: 'No code' contribution = compare 5-class vs 6-class Δκ. "
                   "Probe = logistic regression; content-sim = kNN-majority over cosine "
                   "neighbours (a hard proxy for discriminant.py's Correct-&-Smooth); both on "
                   "identical participant-grouped folds + embeddings. Chance κ = 0.")
    print("\n--- headline (H6 Δκ sign survival on a weaker encoder) ---")
    for tag, h in headline.items():
        print(f"  {tag}: Δκ={h['delta_kappa']} excludes0={h['excludes_0']}")

    p = C.write_json(out, os.path.join(os.path.dirname(__file__), "_e6_results.json"))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
