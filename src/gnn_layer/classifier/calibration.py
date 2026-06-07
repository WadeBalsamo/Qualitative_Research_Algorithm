"""
gnn_layer/calibration.py
------------------------
Track A3 — confidence calibration for domain shift.

The soft-VAAMR head's softmax is fit in-distribution; on genuinely new transcripts it
tends to be over-confident, which would let the A2 abstention floors pass labels they
should defer. Two complementary instruments:

  * **Temperature scaling** (Guo et al. 2017): a single scalar T learned on HELD-OUT
    logits by minimizing NLL against the LLM-consensus reference. Dividing logits by T>1
    softens probabilities so the abstention floors operate on calibrated confidence.
    Reliability is reported as Expected Calibration Error (ECE) before vs after.
  * **OOD score**: a new segment's mean cosine distance to its k nearest TRAINING
    segments. Above a threshold the graph is extrapolating beyond its support, so
    scale-mode forces ABSTAIN (defer to the LLM) regardless of softmax confidence.

Pure numpy/torch; no second model. Used by inference.py (apply T), runner.py (fit T +
OOD gate in scale mode), and validation.py (report T + ECE).
"""

from typing import Optional, Tuple


def apply_temperature(logits, T: float):
    """Divide logits by temperature ``T`` (no-op when T is None/<=0). Accepts np or torch."""
    if T is None or T <= 0:
        return logits
    return logits / float(T)


def fit_temperature(logits, labels, max_iter: int = 200, lr: float = 0.05) -> float:
    """Learn a single temperature T minimizing NLL of softmax(logits/T) vs ``labels``.

    ``logits`` is [N, C], ``labels`` is [N] int. Returns T (clamped to a sane [0.05, 100]
    range). Optimized in log-space so T stays positive. Falls back to T=1.0 when there is
    not enough data (< 2 rows) or a degenerate single-class reference.
    """
    import numpy as np
    import torch

    L = np.asarray(logits, dtype=np.float32)
    y = np.asarray(labels).astype(np.int64)
    if L.ndim != 2 or L.shape[0] < 2 or len(set(y.tolist())) < 2:
        return 1.0

    Lt = torch.tensor(L, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    log_T = torch.zeros((), requires_grad=True)  # T = exp(log_T), start at 1.0
    opt = torch.optim.Adam([log_T], lr=lr)
    nll = torch.nn.functional.cross_entropy
    for _ in range(int(max_iter)):
        opt.zero_grad()
        loss = nll(Lt / log_T.exp(), yt)
        loss.backward()
        opt.step()
    T = float(log_T.exp().detach())
    return float(min(100.0, max(0.05, T)))


def expected_calibration_error(probs, labels, n_bins: int = 15) -> Optional[float]:
    """ECE: mean |confidence − accuracy| over ``n_bins`` equal-width confidence bins.

    ``probs`` is [N, C] (softmax), ``labels`` is [N] int. Returns None if empty.
    """
    import numpy as np

    P = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    if P.ndim != 2 or P.shape[0] == 0:
        return None
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    n = len(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # include the right edge in the last bin so conf==1.0 is counted
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def ood_scores(query_X, ref_X, k: int = 8):
    """Mean cosine distance of each query row to its ``k`` nearest rows in ``ref_X``.

    Higher = further from the training support (more out-of-distribution). Returns a
    1-D np.ndarray of length len(query_X); empty/degenerate inputs yield zeros.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    Q = np.asarray(query_X, dtype=np.float32)
    R = np.asarray(ref_X, dtype=np.float32)
    if Q.ndim != 2 or Q.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if R.ndim != 2 or R.shape[0] == 0:
        return np.zeros((Q.shape[0],), dtype=np.float32)
    kk = min(int(k), R.shape[0])
    nn = NearestNeighbors(n_neighbors=kk, metric='cosine').fit(R)
    dist, _ = nn.kneighbors(Q)
    return dist.mean(axis=1).astype(np.float32)


def temperature_from_cv(cv: dict, df_all) -> dict:
    """Fit a temperature from a held-out CV dict that carries ``vaamr_logits``.

    Pairs each held-out VAAMR logit with the LLM-consensus reference (``final_label``) and
    returns {'temperature', 'ece_before', 'ece_after', 'n'}; T defaults to 1.0 when data is
    thin. Lets callers reuse the gate's CV (run with ``return_logits=True``) instead of
    retraining the folds a second time.
    """
    import numpy as np

    rows = (cv or {}).get('vaamr_logits', [])
    if not rows:
        return {'temperature': 1.0, 'ece_before': None, 'ece_after': None, 'n': 0}

    ref = {str(r.get('segment_id')): r.get('final_label') for _, r in df_all.iterrows()}
    L, y = [], []
    for sid, logit in rows:
        v = ref.get(str(sid))
        try:
            v = int(v)
        except (ValueError, TypeError):
            continue
        L.append(np.asarray(logit, dtype=np.float32))
        y.append(v)
    if len(L) < 2:
        return {'temperature': 1.0, 'ece_before': None, 'ece_after': None, 'n': len(L)}

    L = np.stack(L)
    y = np.asarray(y, dtype=np.int64)

    def _softmax(a):
        e = np.exp(a - a.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    ece_before = expected_calibration_error(_softmax(L), y)
    T = fit_temperature(L, y)
    ece_after = expected_calibration_error(_softmax(L / T), y)
    return {'temperature': T, 'ece_before': ece_before, 'ece_after': ece_after, 'n': len(y)}
