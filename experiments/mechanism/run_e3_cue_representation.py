"""
experiments/mechanism/run_e3_cue_representation.py
==================================================
E3 — cue representation (masterplan §4, Q16).

THE QUESTION: which representation of the therapist CUE earns its place at
predicting the NEXT participant VAAMR stage — the PROCESS label (PURER move) or
the CONTENT (what the therapist actually said)? §7.6 / H6 predict the *process*
move carries the developmental signal, not the lexical content.

ARMS (participant-grouped CV held-out multinomial log-loss of TO_stage models;
lower = better; same rows = cue blocks with a DEFINED dominant PURER move):
  FROM_only      C(from_stage)                                  — baseline
  PURER_move     from + dominant-move one-hot                   — PROCESS feature
  content_embed  from + MiniLM(cue text), PCA fit IN-FOLD       — CONTENT feature
  both           from + move one-hot + MiniLM-PCA               — does content add?

Headline = does the PROCESS feature beat the CONTENT feature on held-out log-loss?
If the MiniLM model is unavailable, the content arms are skipped and the script
falls back to the FROM_only vs PURER_move contrast (E1 already shows move > FROM).

Observational, n≈20 participants / 160 blocks — hypothesis-generating.
Run:  .venv/bin/python experiments/mechanism/run_e3_cue_representation.py
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _common as C

SEED = 42
PCA_K = 8


def _onehot(vals, cats=C.LABELS) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    M = np.zeros((len(vals), len(cats)), dtype=float)
    for j, c in enumerate(cats):
        M[:, j] = (vals == c).astype(float)
    return M


def _padded_proba(clf, X) -> np.ndarray:
    p = clf.predict_proba(X)
    out = np.full((X.shape[0], len(C.LABELS)), 1e-9)
    for j, c in enumerate(clf.classes_):
        out[:, C.LABELS.index(int(c))] = p[:, j]
    return out / out.sum(axis=1, keepdims=True)


def _cv_logloss(arm: str, from_oh, move_oh, emb, y, folds) -> dict:
    """One arm's participant-grouped CV held-out log-loss + accuracy.
    Embedding (StandardScaler+PCA) is fit on TRAIN ONLY each fold (leakage-safe)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import log_loss, accuracy_score

    use_emb = arm in ("content_embed", "both")
    blocks = [from_oh]
    if arm in ("PURER_move", "both"):
        blocks.append(move_oh)
    base = np.hstack(blocks) if blocks else np.zeros((len(y), 0))

    lls, accs = [], []
    for tr, te in folds:
        Xtr, Xte = base[tr], base[te]
        if use_emb and emb is not None:
            k = int(min(PCA_K, emb.shape[1], max(1, len(tr) - 1)))
            sc = StandardScaler().fit(emb[tr])
            pca = PCA(n_components=k, random_state=SEED).fit(sc.transform(emb[tr]))
            Etr = pca.transform(sc.transform(emb[tr]))
            Ete = pca.transform(sc.transform(emb[te]))
            Xtr = np.hstack([Xtr, Etr])
            Xte = np.hstack([Xte, Ete])
        clf = LogisticRegression(max_iter=4000, C=1.0)
        clf.fit(Xtr, y[tr])
        proba = _padded_proba(clf, Xte)
        lls.append(log_loss(y[te], proba, labels=C.LABELS))
        accs.append(accuracy_score(y[te], np.array(C.LABELS)[proba.argmax(1)]))
    return dict(logloss=float(np.mean(lls)), logloss_sd=float(np.std(lls)),
                acc=float(np.mean(accs)), n_features_base=int(base.shape[1]))


def main() -> int:
    print("=" * 78)
    print("E3 — cue representation: PROCESS (PURER move) vs CONTENT (cue embedding)")
    print("=" * 78)
    D = C.build_design(with_text=True)
    Dm = D.dropna(subset=["move"]).copy()
    Dm["move"] = Dm["move"].astype(int)
    Dm = Dm.reset_index(drop=True)
    y = Dm["to_stage"].astype(int).to_numpy()
    groups = Dm["participant_id"].astype(str).to_numpy()
    from_oh = _onehot(Dm["from_stage"].to_numpy())
    move_oh = _onehot(Dm["move"].to_numpy())

    folds, n_splits = C.stratified_group_folds(y, groups, n_splits=5, seed=SEED)
    print(f"\nn_blocks(with move)={len(y)}  participants={len(np.unique(groups))}  "
          f"folds={n_splits}")

    # --- content embedding (MiniLM on mean-pooled cue text) ---
    emb, emb_status = None, "skipped"
    try:
        emb = C.embed_minilm(Dm["cue_text"].tolist(),
                             cache_path=os.path.join(os.path.dirname(__file__),
                                                     "_emb_cue_minilm.npz"),
                             ids=Dm["from_segment_id"].tolist())
        emb_status = f"all-MiniLM-L6-v2 ({emb.shape[1]}-d), PCA->{PCA_K}"
        print(f"cue embedding: {emb_status}")
    except Exception as e:
        emb_status = f"UNAVAILABLE ({type(e).__name__}: {e}) -> content arms skipped; "\
                     f"falling back to FROM vs PURER_move contrast (process feature)"
        print(f"cue embedding: {emb_status}")

    arms = ["FROM_only", "PURER_move"]
    if emb is not None:
        arms += ["content_embed", "both"]

    res = {}
    for arm in arms:
        try:
            res[arm] = _cv_logloss(arm, from_oh, move_oh, emb, y, folds)
            r = res[arm]
            print(f"  {arm:14} logloss={r['logloss']:.4f}±{r['logloss_sd']:.3f}  "
                  f"acc={r['acc']:.3f}")
        except Exception as e:
            res[arm] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {arm:14} ERROR: {res[arm]['error']}")

    base = res.get("FROM_only", {}).get("logloss")
    headline = {}
    if base is not None:
        for arm in ("PURER_move", "content_embed", "both"):
            if arm in res and "logloss" in res[arm]:
                headline[f"{arm}_minus_FROM"] = round(res[arm]["logloss"] - base, 4)
    proc = res.get("PURER_move", {}).get("logloss")
    cont = res.get("content_embed", {}).get("logloss")
    if proc is not None and cont is not None:
        headline["process_minus_content_logloss"] = round(proc - cont, 4)
        headline["process_beats_content"] = bool(proc < cont)

    print("\n--- headline (Δlog-loss vs FROM_only; negative = earns its place) ---")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    if "process_beats_content" in headline:
        verdict = ("PROCESS (PURER move) predicts the next stage better than CONTENT"
                   if headline["process_beats_content"]
                   else "CONTENT embedding edges out the PROCESS move")
        print(f"  => {verdict}  (Δ={headline.get('process_minus_content_logloss')})")
    print("  NOTE: at n≈20 participants the cue rarely earns its place over FROM "
          "alone in either representation — under-identified, as expected.")

    out = dict(
        design=dict(n_blocks=int(len(y)), n_participants=int(len(np.unique(groups))),
                    n_folds=int(n_splits), seed=SEED, pca_k=PCA_K),
        embedding_status=emb_status, arms=res, headline=headline,
        note="Participant-grouped StratifiedGroupKFold; embedding StandardScaler+PCA "
             "fit on train only. Cue = mean-pooled therapist text per block. "
             "Process feature = dominant PURER move one-hot.")
    p = C.write_json(out, os.path.join(os.path.dirname(__file__), "_e3_results.json"))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
