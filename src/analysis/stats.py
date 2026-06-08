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
# Association
# ---------------------------------------------------------------------------

def lift_ratio(p_conditional: float, p_marginal: float) -> float:
    """Association lift: P(b|a) / P(b).

    The factor by which observing ``a`` changes the probability of ``b`` over
    its base rate (1.0 = independence, >1 = positive association). Returns 0.0
    when the marginal is non-positive — the metric is undefined there and
    callers treat it as "no association". Rounding and thresholding are left to
    the caller (sites use different precision/cutoffs).
    """
    return p_conditional / p_marginal if p_marginal > 0 else 0.0


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
    lo = center - half
    hi = center + half
    return (0.0 if lo < 1e-12 else lo, 1.0 if hi > 1 - 1e-12 else min(1.0, hi))


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


def _import_ordered_model():
    """Lazy import of statsmodels' OrderedModel + patsy. Returns (OrderedModel, patsy) or (None, None)."""
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel  # noqa: F401
        import patsy  # noqa: F401
        return OrderedModel, patsy
    except Exception:                              # pragma: no cover
        return None, None


def ordered_logit(df, outcome: str, fixed: str, distr: str = 'logit') -> Optional[Dict]:
    """Fit an ordinal cumulative-link model ``outcome ~ fixed`` (ordered logit).

    Wraps statsmodels' ``OrderedModel`` on a patsy design matrix (the intercept is
    dropped — ``OrderedModel`` carries its own thresholds). The ordinal model respects
    that the VAAMR arc (0..4) is *ordered*, unlike the Gaussian Δprogression model.

    Returns ``{llf, n_params, aic, n, converged, result}`` (``result`` is the fitted
    statsmodels object so callers can extract params), or ``None`` when statsmodels/
    patsy are unavailable or the fit fails (small / singular design). The ``llf`` +
    ``n_params`` pair powers the additive-vs-interaction likelihood-ratio test.

    CAVEAT (repeated measures): ``OrderedModel`` does not expose cluster-robust
    covariance, so any in-sample LR test built on this ``llf`` IGNORES within-participant
    correlation and is anti-conservative for nested data. It must be reported as
    in-sample/descriptive; leakage-free inference is carried by participant-grouped
    cross-validation (see ``mechanism_model._earns_its_place``), not by this p-value.
    """
    OrderedModel, patsy = _import_ordered_model()
    if OrderedModel is None:
        return None
    try:
        import numpy as _np
        y = df[outcome].to_numpy()
        X = patsy.dmatrix(fixed, df, return_type='dataframe')
        if 'Intercept' in X.columns:
            X = X.drop(columns=['Intercept'])
        if X.shape[1] == 0:
            return None
        res = OrderedModel(y, X, distr=distr).fit(method='bfgs', disp=False, maxiter=400)
        llf = float(res.llf)
        if not _np.isfinite(llf):
            return None
        return {
            'llf': llf,
            'n_params': int(X.shape[1]),
            'aic': float(getattr(res, 'aic', float('nan'))),
            'n': int(len(y)),
            'converged': bool(getattr(getattr(res, 'mle_retvals', {}), 'get', lambda *_: True)('converged', True)),
            'result': res,
        }
    except Exception:
        return None


def likelihood_ratio_test(ll_full: float, ll_reduced: float, df_diff: int) -> Dict[str, float]:
    """χ² likelihood-ratio test: LR = 2(ll_full − ll_reduced) ~ χ²(df_diff).

    Used for additive-vs-interaction model comparison on the ordinal outcome. Returns
    {LR, df, p_value}; p is NaN without scipy or for a non-positive df.
    """
    out = {'LR': float('nan'), 'df': int(df_diff), 'p_value': float('nan')}
    try:
        lr = 2.0 * (float(ll_full) - float(ll_reduced))
        out['LR'] = float(lr)
        if _sps is not None and df_diff > 0:
            out['p_value'] = float(_sps.chi2.sf(max(lr, 0.0), df_diff))
    except Exception:
        pass
    return out


def mixedlm_interaction(df, outcome: str, fixed: str, group: str) -> Dict:
    """Fit ``outcome ~ fixed + (1|group)`` and summarise the INTERACTION terms.

    The shipped mechanism mixed model fits only the move main effect (``C(dominant_purer)``);
    this variant accepts an interaction ``fixed`` (e.g. ``C(from_stage)*C(move)``) — the
    FROM×move moderation that H2/§7.6 actually claims — and counts interaction contrasts
    (terms containing ``:``) whose 95% CI excludes 0. Gaussian Δprogression outcome.

    Handles the singular / under-identified design (common at n≈32, where the full
    interaction is rank-deficient) GRACEFULLY: a failed fit or non-finite CIs are
    reported as ``singular=True`` rather than raising. Returns a dict with
    ``{n, n_interaction_terms, n_ci_excludes_0, examples, singular, method}`` (or an
    ``error`` key when statsmodels is unavailable).
    """
    import numpy as _np
    out = {'n': 0, 'n_interaction_terms': 0, 'n_ci_excludes_0': 0,
           'examples': [], 'singular': False, 'method': 'none'}
    res = mixedlm_delta(df, outcome=outcome, fixed=fixed, group=group)
    if res is None:
        out['singular'] = True
        out['method'] = 'unavailable_or_unfit'
        return out
    try:
        params = res.params
        ci = res.conf_int()
        inter = [ix for ix in params.index if ':' in str(ix)]
        out['n_interaction_terms'] = len(inter)
        excl = []
        singular = False
        for ix in inter:
            lo, hi = float(ci.loc[ix][0]), float(ci.loc[ix][1])
            if not (_np.isfinite(lo) and _np.isfinite(hi)):
                singular = True
                continue
            if lo > 0 or hi < 0:
                excl.append(str(ix))
        out['n_ci_excludes_0'] = len(excl)
        out['examples'] = excl[:6]
        out['singular'] = singular
        out['n'] = int(res.nobs) if hasattr(res, 'nobs') else 0
        out['method'] = 'mixedlm'
    except Exception:
        out['singular'] = True
        out['method'] = 'parse_failed'
    return out


# ---------------------------------------------------------------------------
# Confound sensitivity (E-value / Rosenbaum Γ)
# ---------------------------------------------------------------------------

def e_value(rr: float) -> float:
    """VanderWeele–Ding E-value for a risk-ratio-scale association.

    The E-value is the minimum strength of association (on the risk-ratio scale) that
    an unmeasured confounder would need with BOTH the exposure and the outcome to fully
    explain away an observed association of ``rr``. Symmetric in the protective
    direction (rr<1 is inverted). Monotone in |log rr| and always ≥ 1 (E=1 at rr=1).
    Returns NaN for non-positive / non-finite input.
    """
    try:
        rr = float(rr)
    except (TypeError, ValueError):
        return float('nan')
    if not math.isfinite(rr) or rr <= 0:
        return float('nan')
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1.0))


def smd_to_risk_ratio(smd: float) -> float:
    """Approximate risk ratio from a standardized mean difference (Chinn 2000; VanderWeele 2017).

    RR ≈ exp(0.91 · SMD). Used to put a continuous Δprogression cell contrast on the
    risk-ratio scale so an E-value can be computed for it.
    """
    try:
        return float(math.exp(0.91 * float(smd)))
    except (TypeError, ValueError, OverflowError):
        return float('nan')


def e_value_ci_limit(smd_lo: float, smd_hi: float) -> float:
    """E-value for the confidence limit closest to the null (SMD = 0).

    VanderWeele & Ding (2017) recommend reporting the E-value for BOTH the point estimate
    AND the CI limit nearer the null — the point-estimate E-value alone overstates
    robustness, especially under post-hoc cell selection. This returns the E-value of the
    SMD-CI bound closer to 0 (converted via ``smd_to_risk_ratio`` then ``e_value``):

      - 1.0 when the 95% CI spans 0 (a trivial confounder already overlaps the null →
        the association is not robust);
      - otherwise the E-value of the smaller-magnitude (null-side) limit — the honest
        "how robust is the *interval*" floor.

    Returns NaN when either limit is non-finite/absent.
    """
    try:
        lo = float(smd_lo)
        hi = float(smd_hi)
    except (TypeError, ValueError):
        return float('nan')
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return float('nan')
    if lo <= 0.0 <= hi:                         # CI crosses the null → no robustness
        return 1.0
    near = lo if abs(lo) < abs(hi) else hi      # both same sign: the limit nearer 0
    return e_value(smd_to_risk_ratio(near))


def rosenbaum_bounds(deltas: Sequence[float], alpha: float = 0.05,
                     max_gamma: float = 6.0, step: float = 0.05) -> Dict[str, float]:
    """Rosenbaum Γ sensitivity bound for a one-sample sign test on signed differences.

    Reports the bias factor Γ — the odds by which an unmeasured confounder could distort
    treatment (here: move) assignment within a matched stratum — at which the association
    (median Δ ≠ 0) first becomes non-significant at ``alpha``. Larger Γ ⇒ a more robust
    association (a stronger hidden bias would be required to overturn it).

    Uses the SIGN test (distribution-free, robust at tiny n) under the worst-case
    assignment probability p₊ = Γ/(1+Γ). Returns
    ``{gamma_critical, n, n_positive, direction, method}``. ``gamma_critical`` is 1.0
    when the association is already non-significant unconfounded, and capped at
    ``max_gamma`` when it survives the whole grid. NaN when degenerate / scipy absent.
    """
    out = {'gamma_critical': float('nan'), 'n': 0, 'n_positive': 0,
           'direction': 'flat', 'method': 'rosenbaum_sign'}
    vals = [float(x) for x in deltas if x is not None and x == x and x != 0.0]
    n = len(vals)
    out['n'] = n
    if n < 3 or _sps is None:
        return out
    n_pos = int(sum(1 for v in vals if v > 0))
    out['n_positive'] = n_pos
    # Majority direction sets the one-sided test; count "successes" in that direction.
    if n_pos == n - n_pos:
        out['direction'] = 'flat'
        out['gamma_critical'] = 1.0
        return out
    successes = max(n_pos, n - n_pos)
    out['direction'] = 'increasing' if n_pos > n - n_pos else 'decreasing'

    gamma = 1.0
    gamma_critical = max_gamma
    while gamma <= max_gamma + 1e-9:
        p_plus = gamma / (1.0 + gamma)
        # Upper-bound one-sided p-value: P(Binom(n, p_plus) >= successes).
        upper_p = float(_sps.binom.sf(successes - 1, n, p_plus))
        if upper_p >= alpha:
            gamma_critical = round(gamma, 4)
            break
        gamma += step
    out['gamma_critical'] = float(gamma_critical)
    return out


# ---------------------------------------------------------------------------
# Within / between split (Mundlak hybrid decomposition)
# ---------------------------------------------------------------------------

def within_between_split(df, value_col: str, group_col: str,
                         out_within: Optional[str] = None,
                         out_between: Optional[str] = None):
    """Mundlak within/between decomposition of ``value_col`` by ``group_col``.

    Splits a (possibly time-varying) predictor into its group MEAN (the *between*
    component — e.g. a participant's overall cue exposure) and the deviation from that
    mean (the *within* component — momentary fluctuation). Entered together in one
    model, the two coefficients separate a between-participant association from a
    within-participant one — the trajectory model's "consolidation vs momentary nudge"
    contrast. Returns a COPY of ``df`` with the two new columns added. Pure pandas.
    """
    import pandas as _pd
    out_within = out_within or f'{value_col}_within'
    out_between = out_between or f'{value_col}_between'
    d = df.copy()
    grp_mean = d.groupby(group_col)[value_col].transform('mean')
    d[out_between] = grp_mean
    d[out_within] = d[value_col] - grp_mean
    return d


def sign_test(n_positive: int, n_total: int) -> Dict[str, float]:
    """Two-sided sign test that the positive fraction differs from 0.5."""
    out = {'n_positive': int(n_positive), 'n_total': int(n_total), 'p_value': float('nan')}
    if n_total <= 0:
        return out
    if _sps is not None:
        out['p_value'] = float(_sps.binomtest(n_positive, n_total, 0.5).pvalue)
    return out


def mann_kendall_trend(values: Sequence[float]) -> Dict[str, float]:
    """Non-parametric Mann–Kendall test for a MONOTONIC trend in an ordered series.

    Ordinal-safe: it uses only the SIGN of pairwise differences, so it does not
    assume the VAAMR stages are equally spaced (unlike a linear OLS/mixed slope on
    E[stage]). Intended for a per-session group series (e.g. mean adaptive-stage
    occupancy by session). Reports Kendall's tau-b as the effect size and Sen's
    slope as a robust trend magnitude in the series' own units.

    Returns {n, S, tau, p_value, sen_slope, direction, method}. Degrades to NaNs
    for n < 3. This is a small-sample test; with very few points the p-value is
    indicative only — pair it with the underpowered flag in reporting.
    """
    out = {'n': 0, 'S': 0.0, 'tau': float('nan'), 'p_value': float('nan'),
           'sen_slope': float('nan'), 'direction': 'flat', 'method': 'mann_kendall'}
    v = [float(x) for x in values if x is not None and x == x]
    n = len(v)
    out['n'] = n
    if n < 3:
        return out

    # Mann–Kendall S statistic = Σ sign(v_j - v_i) for j > i.
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(v[j] - v[i])
    out['S'] = float(S)

    # Variance with tie correction; normal approximation with continuity correction.
    from collections import Counter
    ties = Counter(v)
    var = (n * (n - 1) * (2 * n + 5)
           - sum(t * (t - 1) * (2 * t + 5) for t in ties.values())) / 18.0
    if var > 0:
        z = (S - np.sign(S)) / math.sqrt(var)
        if _sps is not None:
            out['p_value'] = float(2 * (1 - _sps.norm.cdf(abs(z))))
    # Kendall's tau-b (effect size).
    if _sps is not None:
        try:
            tau, p = _sps.kendalltau(list(range(n)), v)
            out['tau'] = float(tau)
            if out['p_value'] != out['p_value']:
                out['p_value'] = float(p)
        except Exception:
            pass
    # Sen's slope: median of pairwise slopes (robust trend magnitude).
    slopes = [(v[j] - v[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    if slopes:
        out['sen_slope'] = float(np.median(slopes))
    out['direction'] = 'increasing' if S > 0 else ('decreasing' if S < 0 else 'flat')
    return out


def power_flag(n_participants: int, n_sessions: int,
               min_participants: int = 10, min_sessions: int = 4) -> Dict[str, object]:
    """Flag (do NOT suppress) when a sample is too small for trustworthy inference.

    Returns {underpowered: bool, note: str}. Reports keep showing point estimates,
    CIs and p-values, but annotate them so a reader (and a reviewer) is not invited
    to over-read inference from a handful of participants/sessions.
    """
    under = (n_participants < min_participants) or (n_sessions < min_sessions)
    note = ''
    if under:
        note = (f"UNDERPOWERED (n={n_participants} participants, {n_sessions} sessions; "
                f"targets ≥{min_participants}/≥{min_sessions}) — p-values/CIs are indicative "
                "only, not confirmatory.")
    return {'underpowered': bool(under), 'note': note}
