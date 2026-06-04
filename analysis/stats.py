"""
analysis/stats.py
-----------------
Reusable statistical-inference toolkit for the analysis layer.

The descriptive outputs (lift, slopes, transition rates, Δprogression) historically
shipped as bare point estimates. For a small (1–2 cohort) observational study that
is the central weakness: a "lift of 2.3" on n=4 blocks means nothing without
uncertainty, and a 25-cell PURER×VAAMR heatmap makes 25 simultaneous comparisons.

This module supplies the inference every downstream report now attaches:
  - Wilson score CIs for proportions
  - cluster (participant) bootstrap CIs for means / lifts / slopes — respects the
    repeated-measures nesting the raw counts ignore
  - within-stratum label-shuffle permutation tests — the methodology's named null
    control, holding base rates fixed
  - effect sizes (Cohen's h, Cramér's V, odds ratio, Cliff's delta)
  - Benjamini–Hochberg FDR control across a comparison family
  - statsmodels mixed-effects models (random participant effects) for trajectory
    slope tests and the Δprogression regression

All estimates are associational/directional — observational design, never causal.
numpy + scipy are required; statsmodels is imported lazily and degrades gracefully.
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                  # scipy is a hard dependency elsewhere; guard anyway
    from scipy import stats as _sps
except Exception:                     # pragma: no cover
    _sps = None


# ---------------------------------------------------------------------------
# Proportion CIs
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n. Returns (lo, hi)."""
    if n <= 0:
        return (0.0, 0.0)
    z = _z(alpha)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _z(alpha: float) -> float:
    """Two-sided normal critical value; falls back to 1.96 without scipy."""
    if _sps is not None:
        return float(_sps.norm.ppf(1 - alpha / 2))
    return 1.959963984540054


# ---------------------------------------------------------------------------
# Cluster (participant) bootstrap
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile CI for ``statistic`` via resampling whole clusters (participants).

    Resampling clusters rather than rows preserves within-participant correlation,
    giving honest uncertainty for repeated-measures cue-block / segment data.
    Returns {point, lo, hi, n, n_clusters}.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=object)
    out = {'point': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
           'n': int(len(values)), 'n_clusters': 0}
    if len(values) == 0:
        return out

    # Group row indices by cluster.
    groups: Dict[object, List[int]] = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    keys = list(groups.keys())
    out['n_clusters'] = len(keys)
    finite = values[np.isfinite(values)]
    out['point'] = float(statistic(finite)) if len(finite) else float('nan')
    if len(keys) < 2:
        return out                    # CI undefined with a single cluster

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    nfill = 0
    for b in range(n_boot):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        idx: List[int] = []
        for ci in chosen:
            idx.extend(groups[keys[ci]])
        sample = values[idx]
        sample = sample[np.isfinite(sample)]
        if len(sample) == 0:
            continue
        boots[nfill] = statistic(sample)
        nfill += 1
    if nfill == 0:
        return out
    boots = boots[:nfill]
    out['lo'] = float(np.percentile(boots, 100 * alpha / 2))
    out['hi'] = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return out


# ---------------------------------------------------------------------------
# Permutation test (within-stratum label shuffle)
# ---------------------------------------------------------------------------

def permutation_test(
    values: Sequence[float],
    group_mask: Sequence[bool],
    strata: Optional[Sequence] = None,
    statistic: str = 'mean_diff',
    n_perm: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided permutation p-value comparing the in-group vs out-of-group values.

    ``group_mask`` flags the rows belonging to the construct of interest (e.g. blocks
    whose dominant PURER move is Reframing). Labels are shuffled WITHIN ``strata``
    (e.g. from_stage) so the null holds base rates fixed — the methodology's named
    shuffle control. ``statistic='mean_diff'`` compares group mean vs rest; observed
    and null are built identically. Returns {observed, p_value, n_perm, n_group}.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(group_mask, dtype=bool)
    n = len(values)
    out = {'observed': float('nan'), 'p_value': float('nan'),
           'n_perm': 0, 'n_group': int(mask.sum())}
    if n == 0 or mask.sum() == 0 or (~mask).sum() == 0:
        return out
    if strata is None:
        strata = np.zeros(n, dtype=int)
    strata = np.asarray(strata, dtype=object)

    def _stat(m: np.ndarray) -> float:
        a = values[m]
        b = values[~m]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            return float('nan')
        return float(np.mean(a) - np.mean(b))

    observed = _stat(mask)
    out['observed'] = observed
    if not np.isfinite(observed):
        return out

    # Precompute stratum row-index pools for within-stratum shuffling.
    strata_pools: Dict[object, np.ndarray] = {}
    for s in np.unique(strata):
        strata_pools[s] = np.where(strata == s)[0]

    rng = np.random.default_rng(seed)
    count = 0
    done = 0
    for _ in range(n_perm):
        perm_mask = np.zeros(n, dtype=bool)
        for s, pool in strata_pools.items():
            k = int(mask[pool].sum())
            if k == 0:
                continue
            chosen = rng.choice(pool, size=k, replace=False)
            perm_mask[chosen] = True
        stat = _stat(perm_mask)
        if not np.isfinite(stat):
            continue
        if abs(stat) >= abs(observed) - 1e-12:
            count += 1
        done += 1
    if done == 0:
        return out
    out['n_perm'] = done
    out['p_value'] = (count + 1) / (done + 1)     # add-one smoothing
    return out


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def cramers_v(contingency: np.ndarray) -> Dict[str, float]:
    """Cramér's V for an r×c contingency table (+ chi-square p when scipy present)."""
    table = np.asarray(contingency, dtype=float)
    n = table.sum()
    out = {'cramers_v': float('nan'), 'chi2': float('nan'), 'p_value': float('nan')}
    if n <= 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return out
    if _sps is not None:
        chi2, p, _, _ = _sps.chi2_contingency(table, correction=False)
        out['chi2'] = float(chi2)
        out['p_value'] = float(p)
    else:                                          # manual chi-square
        row = table.sum(1, keepdims=True)
        col = table.sum(0, keepdims=True)
        expected = row @ col / n
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = np.nansum((table - expected) ** 2 / np.where(expected > 0, expected, np.nan))
        out['chi2'] = float(chi2)
    k = min(table.shape) - 1
    out['cramers_v'] = float(math.sqrt(out['chi2'] / (n * k))) if k > 0 and out['chi2'] == out['chi2'] else float('nan')
    return out


def odds_ratio_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Dict[str, float]:
    """Odds ratio with Haldane–Anscombe 0.5 correction + log-normal CI. Cells a,b / c,d."""
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (a2 * d2) / (b2 * c2)
    se = math.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
    z = _z(alpha)
    return {'odds_ratio': float(orr),
            'lo': float(math.exp(math.log(orr) - z * se)),
            'hi': float(math.exp(math.log(orr) + z * se))}


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta — nonparametric effect size in [-1, 1] (P(x>y) − P(x<y))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    gt = lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / (len(x) * len(y)))


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, list]:
    """Benjamini–Hochberg FDR. Returns {reject: [bool], qvalues: [float]} aligned to input."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = {'reject': [False] * n, 'qvalues': [float('nan')] * n}
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity of q-values
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    thresh = ranked <= (np.arange(1, m + 1) / m) * alpha
    kmax = np.where(thresh)[0].max() + 1 if thresh.any() else 0
    reject = np.zeros(m, dtype=bool)
    if kmax > 0:
        reject[:kmax] = True
    # unwind ordering back to original positions
    for local_rank, local_idx in enumerate(order):
        gi = valid[local_idx]
        out['qvalues'][gi] = float(q[local_rank])
        out['reject'][gi] = bool(reject[local_rank])
    return out


# ---------------------------------------------------------------------------
# Mixed-effects models (statsmodels; lazily imported, graceful)
# ---------------------------------------------------------------------------

def _import_statsmodels():
    try:
        import statsmodels.formula.api as smf      # noqa: F401
        import statsmodels.api as sm               # noqa: F401
        return smf, sm
    except Exception:                              # pragma: no cover
        return None, None


def mixedlm_trend(df, outcome: str, time: str, group: str,
                  random_slope: bool = True) -> Dict[str, float]:
    """Random-effects trend test: outcome ~ time + (time | group).

    Used for the program-progression slope (≠0?) with a random participant effect, so
    the test respects that sessions are nested within participants. Returns
    {slope, se, p_value, ci_lo, ci_hi, n, n_groups, method}. Falls back to an OLS
    slope when statsmodels or the fit is unavailable.
    """
    import pandas as pd
    smf, _ = _import_statsmodels()
    d = df[[outcome, time, group]].dropna()
    out = {'slope': float('nan'), 'se': float('nan'), 'p_value': float('nan'),
           'ci_lo': float('nan'), 'ci_hi': float('nan'),
           'n': int(len(d)), 'n_groups': int(d[group].nunique()), 'method': 'none'}
    if len(d) < 3 or d[group].nunique() < 2:
        return _ols_trend(d, outcome, time, out)
    if smf is None:
        return _ols_trend(d, outcome, time, out)
    try:
        re_formula = f"~{time}" if random_slope else None
        model = smf.mixedlm(f"{outcome} ~ {time}", d, groups=d[group], re_formula=re_formula)
        res = model.fit(reml=True, method='lbfgs', disp=False)
        out['slope'] = float(res.fe_params[time])
        out['se'] = float(res.bse[time])
        out['p_value'] = float(res.pvalues[time])
        ci = res.conf_int().loc[time]
        out['ci_lo'], out['ci_hi'] = float(ci[0]), float(ci[1])
        out['method'] = 'mixedlm'
        return out
    except Exception:
        return _ols_trend(d, outcome, time, out)


def _ols_trend(d, outcome, time, out) -> Dict[str, float]:
    """OLS fallback slope + CI/p via scipy linregress."""
    if _sps is None or len(d) < 2:
        return out
    try:
        x = d[time].astype(float).to_numpy()
        y = d[outcome].astype(float).to_numpy()
        lr = _sps.linregress(x, y)
        out['slope'] = float(lr.slope)
        out['se'] = float(lr.stderr)
        out['p_value'] = float(lr.pvalue)
        z = _z(0.05)
        out['ci_lo'] = float(lr.slope - z * lr.stderr)
        out['ci_hi'] = float(lr.slope + z * lr.stderr)
        out['method'] = 'ols'
    except Exception:
        pass
    return out


def mixedlm_delta(df, outcome: str = 'delta_prog', fixed: str = 'C(behavior)',
                  group: str = 'participant_id') -> Optional["object"]:
    """Fit outcome ~ fixed + (1 | group) and return the fitted result (or None).

    Used for the Δprogression mechanism model (per-move marginal effects with a random
    participant intercept). Caller extracts params/conf_int. Returns None when
    statsmodels is unavailable or the fit fails.
    """
    smf, _ = _import_statsmodels()
    if smf is None:
        return None
    cols = set()
    for token in fixed.replace('C(', '').replace(')', '').replace('*', '+').split('+'):
        cols.add(token.strip())
    needed = [c for c in cols if c and c in df.columns] + [outcome, group]
    d = df[[c for c in dict.fromkeys(needed) if c in df.columns]].dropna()
    if len(d) < 4 or d[group].nunique() < 2:
        return None
    try:
        return smf.mixedlm(f"{outcome} ~ {fixed}", d, groups=d[group]).fit(disp=False)
    except Exception:
        return None


def sign_test(n_positive: int, n_total: int) -> Dict[str, float]:
    """Two-sided sign test that the positive fraction differs from 0.5."""
    out = {'n_positive': int(n_positive), 'n_total': int(n_total), 'p_value': float('nan')}
    if n_total <= 0:
        return out
    if _sps is not None:
        out['p_value'] = float(_sps.binomtest(n_positive, n_total, 0.5).pvalue)
    return out
