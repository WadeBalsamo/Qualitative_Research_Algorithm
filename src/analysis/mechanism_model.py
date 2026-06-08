"""
analysis/mechanism_model.py
---------------------------
The re-centered mechanism estimator (masterplan.md §3; methodology_assessment.md verdict).

The program's central question — *does a therapist PURER move's effect on the next
participant VAAMR stage depend on the participant's FROM stage?* — is a **FROM_stage ×
move interaction** on an ordered outcome. The shipped estimator
(`mechanism.py:_mixed_effects_delta` → ``delta_prog ~ C(dominant_purer)``) fits the move
MAIN effect only, Gaussian, no interaction. This module fits the interaction with the
right tools and asks whether it *earns its place* out of sample:

  - (a) frequentist ordinal ``OrderedModel`` for ``to_stage ~ C(from)*C(move)`` with an
        additive-vs-interaction likelihood-ratio test (ordinal-correct);
  - (b) Gaussian mixed ``delta_prog ~ C(from)*C(move) + (1|participant)`` — the
        interaction the shipped model omits — counting interaction CIs that exclude 0,
        with graceful singular/under-identified handling;
  - (c) earns-its-place participant-grouped CV (multinomial log-loss): FROM-only vs
        additive vs interaction;
  - (d) Bayesian hierarchical ordinal (opt-in; lazy bambi; degrades gracefully — bambi
        needs numpy≥2 which conflicts with the pinned transformers, so it is unavailable
        in the main venv: run it in an isolated env, see experiments/mechanism).

Plus a confound-sensitivity table (E-value / Rosenbaum Γ per cell), a PURER-label-noise
robustness check (the cue labels are not yet human-validated), and a trajectory model
that separates within- from between-session cue effects (consolidation vs momentary nudge).

Everything is observational, n≈32 — hypothesis-generating, never causal. The honest
expected outcome is "right instrument, under-identified at n≈32, bounded by sensitivity
analysis." Heavy libs are lazy-imported with graceful degradation (mirrors
``stats._import_statsmodels``); reuses ``process.cue_blocks`` upstream and the
``analysis.stats`` helpers (e_value, rosenbaum_bounds, ordered_logit, mixedlm_interaction,
within_between_split).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import stats as S

# VAAMR ordered arc (0=Vigilance .. 4=Reappraisal).
_VAAMR_LABELS = [0, 1, 2, 3, 4]


@dataclass
class MechanismModelConfig:
    """Settings for the re-centered mechanism estimator (serialized into PipelineConfig.mechanism).

    NOTE ON ``estimator``: the masterplan lists 'bayesian' as the primary arm, but the
    in-process DEFAULT here is 'frequentist' because the Bayesian stack (bambi/pymc)
    requires numpy≥2.0 while the pipeline's pinned transformers requires numpy<2.0 — the
    two cannot coexist in the main venv. 'bayesian' / 'both' are opt-in and lazy-import
    bambi, degrading gracefully when it is absent (it WILL be absent in the main venv).
    """
    enabled: bool = True                     # default-on; degrades to the additive table if libs absent
    # ---- adjacency interaction model (H2 / §7.6) ----
    estimator: str = 'frequentist'           # 'frequentist' (safe in-process default) | 'bayesian' | 'both'
    ordinal: bool = True                     # cumulative-logit on the ordered VAAMR arc (0..4)
    interaction: bool = True                 # FROM_stage × move — the §7.6 moderation term
    random_effect: str = 'participant_id'    # partial-pooling unit
    # ---- priors (weakly-informative; only used by the opt-in Bayesian arm) ----
    prior_scale: float = 2.5                 # Normal(0, prior_scale) on logit coefficients
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    seed: int = 42
    # ---- trajectory model (Q18/Q19) ----
    trajectory: bool = True                  # participant-level model w/ lagged cue exposure
    split_within_between: bool = True        # report within-session vs between-session effects separately
    # ---- sensitivity (Q20) ----
    sensitivity: bool = True                 # E-value + Rosenbaum Γ on per-cell associations
    # ---- robustness ----
    purer_noise_check: bool = True           # perturb PURER at the single-rater disagreement rate, re-rank (E5)
    purer_disagreement_rate: Optional[float] = None   # read from IRR if available, else default 0.30

    @property
    def wants_bayesian(self) -> bool:
        return str(self.estimator).lower() in ('bayesian', 'both')

    @property
    def wants_frequentist(self) -> bool:
        return str(self.estimator).lower() in ('frequentist', 'both')


# ---------------------------------------------------------------------------
# Design-frame construction
# ---------------------------------------------------------------------------

def build_design_frame(enriched_blocks: List[dict]) -> pd.DataFrame:
    """Build the FROM→CUE→TO design frame from mechanism.py's enriched blocks.

    Columns: participant_id, session_id, from_stage (int), to_stage (int),
    move (dominant PURER move of the CUE — categorical; None when the block has no
    PURER-labelled therapist segment), delta_prog (float). One row per triple.
    """
    rows = []
    for b in enriched_blocks:
        fs = b.get('from_stage')
        ts = b.get('to_stage')
        if fs is None or ts is None:
            continue
        rows.append({
            'participant_id': str(b.get('participant_id')) if b.get('participant_id') is not None else None,
            'session_id': b.get('session_id'),
            'from_stage': int(fs),
            'to_stage': int(ts),
            'move': b.get('dominant_purer'),         # readable label or None; used as a categorical
            'delta_prog': float(b['delta_prog']) if b.get('delta_prog') is not None else float('nan'),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (a)+(b)+(c)+(d)  adjacency interaction model
# ---------------------------------------------------------------------------

def _earns_its_place(D: pd.DataFrame, seed: int = 42, min_n: int = 20) -> Dict:
    """Participant-grouped CV held-out multiclass log-loss: FROM-only vs additive vs interaction.

    Lower log-loss = better held-out fit. The cue 'earns its place' iff the additive /
    interaction models beat the FROM-only baseline out of sample (negative Δlog-loss).
    Degrades to a status marker on small/degenerate data or missing sklearn/patsy.
    """
    Dm = D.dropna(subset=['move']).copy()
    if len(Dm) < min_n:
        return {'status': 'insufficient_n', 'n': int(len(Dm))}
    try:
        import patsy
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import log_loss, accuracy_score
    except Exception:
        return {'status': 'sklearn_unavailable'}

    y = Dm['to_stage'].astype(int).to_numpy()
    groups = Dm['participant_id'].astype(str).to_numpy()
    n_groups = len(set(groups))
    if len(set(y)) < 2:
        return {'status': 'single_outcome_class', 'n': int(len(y))}
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return {'status': 'too_few_participants', 'n_participants': int(n_groups)}

    specs = {
        'FROM_only':   'C(from_stage)',
        'additive':    'C(from_stage) + C(move)',
        'interaction': 'C(from_stage) * C(move)',
    }
    classes_full = list(_VAAMR_LABELS)

    def _padded_proba(clf, X):
        p = clf.predict_proba(X)
        out = np.full((X.shape[0], len(classes_full)), 1e-9)
        for j, c in enumerate(clf.classes_):
            if int(c) in classes_full:
                out[:, classes_full.index(int(c))] = p[:, j]
        return out / out.sum(axis=1, keepdims=True)

    try:
        Xs = {k: patsy.dmatrix(f, Dm, return_type='dataframe').values for k, f in specs.items()}
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    except Exception as e:
        return {'status': f'cv_setup_failed:{type(e).__name__}'}

    res: Dict[str, Dict] = {}
    for name, X in Xs.items():
        lls, accs = [], []
        for tr, te in folds:
            if len(set(y[tr])) < 2:
                continue
            try:
                clf = LogisticRegression(max_iter=3000, C=1.0)
                clf.fit(X[tr], y[tr])
                proba = _padded_proba(clf, X[te])
                lls.append(log_loss(y[te], proba, labels=classes_full))
                accs.append(accuracy_score(y[te], np.array(classes_full)[proba.argmax(1)]))
            except Exception:
                continue
        if not lls:
            return {'status': 'cv_no_valid_folds'}
        res[name] = {
            'logloss': float(np.mean(lls)),
            'logloss_sd': float(np.std(lls)),
            'acc': float(np.mean(accs)),
            'n_params': int(Xs[name].shape[1]),
        }
    base = res['FROM_only']['logloss']
    res['additive_delta_logloss'] = round(res['additive']['logloss'] - base, 4)
    res['interaction_delta_logloss'] = round(res['interaction']['logloss'] - base, 4)
    res['additive_earns_place'] = bool(res['additive']['logloss'] < base)
    res['interaction_earns_place'] = bool(res['interaction']['logloss'] < base)
    res['n'] = int(len(y))
    res['n_participants'] = int(n_groups)
    res['n_folds'] = int(n_splits)
    res['status'] = 'ok'
    return res


def _bayesian_ordinal(D: pd.DataFrame, config: MechanismModelConfig) -> Dict:
    """Opt-in Bayesian hierarchical ordinal arm (lazy bambi). Degrades gracefully.

    bambi/pymc require numpy≥2.0, which conflicts with the pinned transformers
    (numpy<2.0) in the main venv — so the ImportError path is the EXPECTED one here.
    """
    try:
        import bambi as bmb          # noqa: F401
        import arviz as az
    except Exception:
        msg = ("bambi unavailable; run the Bayesian estimator in an isolated env, "
               "see experiments/mechanism")
        print(f"  [mechanism_model] {msg}")
        return {'ok': False, 'status': 'bambi_unavailable', 'note': msg}
    try:
        Db = D.dropna(subset=['move']).copy()
        Db['to_stage'] = pd.Categorical(Db['to_stage'].astype(int), categories=_VAAMR_LABELS, ordered=True)
        Db['from_stage'] = Db['from_stage'].astype('category')
        Db['move'] = Db['move'].astype('category')
        formula = "to_stage ~ from_stage * move + (1|participant_id)"
        model = bmb.Model(formula, Db, family='cumulative')
        idata = model.fit(draws=config.draws, tune=config.tune, chains=config.chains,
                          cores=1, target_accept=config.target_accept,
                          random_seed=config.seed, progressbar=False)
        summ = az.summary(idata, hdi_prob=0.95)
        inter = summ[summ.index.str.contains(':') & summ.index.str.contains('from_stage')]
        excl = inter[(inter['hdi_2.5%'] > 0) | (inter['hdi_97.5%'] < 0)]
        ndiv = int(np.asarray(idata.sample_stats['diverging']).sum()) if 'diverging' in idata.sample_stats else -1
        return {
            'ok': True, 'status': 'ok', 'n': int(len(Db)),
            'n_interaction_terms': int(len(inter)),
            'n_hdi_excludes_0': int(len(excl)),
            'max_rhat': float(summ['r_hat'].max()),
            'divergences': ndiv,
            'examples': list(excl.index[:6]),
            'note': 'cumulative-logit + partial pooling; weakly-informative priors shrink sparse cells',
        }
    except Exception as e:                         # pragma: no cover (bambi absent in CI)
        return {'ok': False, 'status': f'bayesian_failed:{type(e).__name__}', 'note': str(e)}


def fit_adjacency_interaction(blocks_df: pd.DataFrame, config: MechanismModelConfig) -> Dict:
    """The FROM×move adjacency interaction estimate (§3.2a–d). Returns a structured dict.

    Sub-arms each degrade independently: a failed/unavailable arm reports a status string
    rather than raising, so the overall result is always returned.
    """
    out: Dict = {
        'n': int(len(blocks_df)),
        'n_with_move': int(blocks_df['move'].notna().sum()) if 'move' in blocks_df.columns else 0,
        'n_participants': int(blocks_df['participant_id'].nunique()) if 'participant_id' in blocks_df.columns else 0,
    }

    # (a) ordinal additive-vs-interaction LR test (ordinal-correct).
    ordinal: Dict = {'status': 'skipped'}
    if config.ordinal:
        Dm = blocks_df.dropna(subset=['move']).copy()
        if len(Dm) >= 8 and Dm['move'].nunique() >= 2 and Dm['from_stage'].nunique() >= 2:
            add = S.ordered_logit(Dm, 'to_stage', 'C(from_stage) + C(move)')
            inter = S.ordered_logit(Dm, 'to_stage', 'C(from_stage) * C(move)')
            if add and inter:
                lr = S.likelihood_ratio_test(inter['llf'], add['llf'], inter['n_params'] - add['n_params'])
                ordinal = {
                    'status': 'ok',
                    'll_additive': round(add['llf'], 3),
                    'll_interaction': round(inter['llf'], 3),
                    'k_additive': add['n_params'],
                    'k_interaction': inter['n_params'],
                    'LR': round(lr['LR'], 3),
                    'df': lr['df'],
                    'p_value': (round(lr['p_value'], 4) if lr['p_value'] == lr['p_value'] else None),
                }
            else:
                ordinal = {'status': 'unavailable_or_unfit'}
        else:
            ordinal = {'status': 'insufficient_n', 'n': int(len(Dm))}
    out['ordinal_lr'] = ordinal

    # (b) Gaussian mixed interaction — the term the shipped model omits.
    gaussian: Dict = {'status': 'skipped'}
    if config.interaction:
        Dd = blocks_df.dropna(subset=['move', 'delta_prog']).copy()
        if len(Dd) >= 6 and Dd['participant_id'].nunique() >= 2:
            gaussian = S.mixedlm_interaction(Dd, 'delta_prog', 'C(from_stage)*C(move)', 'participant_id')
        else:
            gaussian = {'status': 'insufficient_n', 'n': int(len(Dd))}
    out['gaussian_interaction'] = gaussian

    # (c) earns-its-place participant-grouped CV.
    out['earns_its_place'] = _earns_its_place(blocks_df, seed=config.seed)

    # (d) Bayesian arm (opt-in; graceful degrade).
    if config.wants_bayesian:
        out['bayesian'] = _bayesian_ordinal(blocks_df, config)
    else:
        out['bayesian'] = {'ok': False, 'status': 'not_requested',
                           'note': "estimator='frequentist' (in-process default); set estimator='bayesian'|'both' to enable"}
    return out


# ---------------------------------------------------------------------------
# Confound sensitivity (E-value / Rosenbaum Γ)
# ---------------------------------------------------------------------------

def _pooled_smd(cell: np.ndarray, rest: np.ndarray) -> float:
    """Pooled-SD standardized mean difference (Cohen's d) of ``cell`` vs ``rest``.

    Returns NaN when either arm is empty or the pooled denominator is non-positive;
    0.0 when the pooled SD is 0 (no spread → no standardized effect).
    """
    a = np.asarray(cell, dtype=float)
    b = np.asarray(rest, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 1 or nb < 1:
        return float('nan')
    denom = na + nb - 2
    if denom <= 0:
        return float('nan')
    va = a.var(ddof=1) if na > 1 else 0.0
    vb = b.var(ddof=1) if nb > 1 else 0.0
    sd = math.sqrt((va * (na - 1) + vb * (nb - 1)) / denom)
    if not (sd and sd == sd and sd > 0):
        return 0.0
    return float((a.mean() - b.mean()) / sd)


def _smd_cluster_ci(g: pd.DataFrame, mv, rng, n_boot: int = 1000,
                    alpha: float = 0.05) -> tuple:
    """Participant-cluster bootstrap percentile CI for the (move ``mv`` vs same-stage other
    moves) SMD within one from_stage subgroup ``g``.

    Resamples WHOLE participants (clusters) with replacement so the CI respects the
    repeated-measures nesting the raw cell counts ignore — this is what makes the
    CI-limit E-value honest. Returns ``(lo, hi)``, or ``(nan, nan)`` with < 2 participant
    clusters or too few valid resamples (a degenerate CI must not masquerade as robust).
    """
    if 'participant_id' not in g.columns:
        return (float('nan'), float('nan'))
    moves = g['move'].to_numpy(dtype=object)
    deltas = g['delta_prog'].to_numpy(dtype=float)
    pids = g['participant_id'].to_numpy(dtype=object)
    by_pid: Dict[object, List[int]] = {}
    for i in range(len(g)):
        by_pid.setdefault(pids[i], []).append(i)
    keys = list(by_pid.keys())
    if len(keys) < 2:
        return (float('nan'), float('nan'))
    boots: List[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        idx: List[int] = []
        for ci in chosen:
            idx.extend(by_pid[keys[ci]])
        idx_arr = np.asarray(idx)
        m = moves[idx_arr]
        d = deltas[idx_arr]
        s = _pooled_smd(d[m == mv], d[m != mv])
        if s == s and math.isfinite(s):
            boots.append(s)
    if len(boots) < 20:                          # too few valid draws → CI undefined
        return (float('nan'), float('nan'))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def sensitivity_bounds(blocks_df: pd.DataFrame, min_n: int = 4,
                       n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Dict:
    """Per (from_stage × move) cell: standardized Δprogression vs other moves at the same
    FROM stage → approx RR → E-value (VanderWeele–Ding), reported as BOTH a point E-value
    AND the E-value of the SMD-CI limit nearest the null (the honest robustness floor), plus
    a Rosenbaum Γ on the cell-vs-same-stage-other-moves CONTRAST. Returns
    {n_cells, cells (sorted by CI-limit E-value, then |SMD|), note}.

    Two corrections over the first cut (review P0-1, P0-3):
      - P0-1: the SMD carries a participant-cluster bootstrap 95% CI, and ``e_value_ci_limit``
        is the E-value of the CI bound nearer 0 — 1.0 when the CI spans 0. The point E-value
        alone overstates robustness under the post-hoc |SMD| cell selection; the CI-limit
        E-value is the number a methods reviewer should read (VanderWeele & Ding 2017).
      - P0-3: Γ is computed on the cell deltas CENTERED by the same-stage other-move mean
        (the same contrast the E-value bounds), not on the cell's raw signed deltas — so Γ
        and the E-value now bound the SAME association ("this move vs alternatives at this
        stage"), not "does this stage move at all".
    """
    Dm = blocks_df.dropna(subset=['move', 'delta_prog']).copy()
    rng = np.random.default_rng(seed)
    cells: List[dict] = []
    for fs, g in Dm.groupby('from_stage'):
        for mv, gc in g.groupby('move'):
            if len(gc) < min_n:
                continue
            rest = g[g['move'] != mv]
            if len(rest) < 2:
                continue
            cell_d = gc['delta_prog'].to_numpy(dtype=float)
            rest_d = rest['delta_prog'].to_numpy(dtype=float)
            smd = _pooled_smd(cell_d, rest_d)
            smd_lo, smd_hi = _smd_cluster_ci(g, mv, rng, n_boot=n_boot, alpha=alpha)
            rr = S.smd_to_risk_ratio(smd) if smd == smd else float('nan')
            e_point = S.e_value(rr) if rr == rr else float('nan')
            e_ci = S.e_value_ci_limit(smd_lo, smd_hi)
            # P0-3: sign test on the contrast (cell deltas centered by the other-move mean),
            # so Γ bounds the same "move vs alternatives at this stage" association as the E-value.
            rest_mean = float(np.nanmean(rest_d)) if len(rest_d) else 0.0
            gamma = S.rosenbaum_bounds((cell_d - rest_mean).tolist())
            cells.append({
                'from_stage': int(fs),
                'move': mv,
                'n': int(len(gc)),
                'mean_delta_prog': round(float(np.mean(cell_d)), 4),
                'smd': round(smd, 4) if smd == smd else None,
                'smd_ci_lo': round(smd_lo, 4) if smd_lo == smd_lo else None,
                'smd_ci_hi': round(smd_hi, 4) if smd_hi == smd_hi else None,
                'approx_rr': round(rr, 4) if rr == rr else None,
                'e_value': round(e_point, 4) if e_point == e_point else None,
                'e_value_ci_limit': round(e_ci, 4) if e_ci == e_ci else None,
                'rosenbaum_gamma': gamma.get('gamma_critical'),
            })
    # Surface the genuinely-robust cells first: CI-limit E-value desc, then |SMD| desc as a
    # tiebreak. When every CI spans 0 (typical at n≈32) all CI-limit E-values tie at 1.0 and
    # the order degrades to |SMD| — the honest "nothing survives the interval" presentation.
    cells.sort(key=lambda c: (-(c['e_value_ci_limit'] or 0.0), -abs(c['smd'] or 0.0)))
    return {
        'n_cells': len(cells),
        'cells': cells,
        'note': ("E-value = minimum strength (RR scale) an unmeasured confounder would need with "
                 "BOTH move-selection and Δprogression to explain away the cell association. The "
                 "CI-LIMIT E-value (E-value of the SMD 95%-CI bound nearer the null; =1.00 when the "
                 "CI spans 0) is the honest robustness floor — the point E-value alone overstates "
                 "robustness under post-hoc |SMD| cell selection (VanderWeele & Ding 2017). "
                 "Γ = matched-bias factor at which the cell-vs-same-stage-other-moves sign test "
                 "loses significance."),
    }


# ---------------------------------------------------------------------------
# PURER-label-noise robustness (E5)
# ---------------------------------------------------------------------------

def purer_noise_robustness(blocks_df: pd.DataFrame, config: MechanismModelConfig,
                           k: int = 200) -> Dict:
    """Perturb the CUE PURER move at the single-rater disagreement rate, re-rank per-move
    influence K times, report Spearman rank-stability vs the unperturbed ranking.

    The cue (PURER) labels are not yet human-validated, so this bounds how much the
    per-move Δprogression ranking could be moving under plausible label noise. Returns
    {rate, k, mean_spearman, sd_spearman, min_spearman, n_moves, status}.
    """
    rate = config.purer_disagreement_rate
    if rate is None:
        rate = 0.30
    Dm = blocks_df.dropna(subset=['move', 'delta_prog']).copy()
    moves = sorted(Dm['move'].unique().tolist(), key=lambda m: str(m))
    if Dm.empty or len(moves) < 2:
        return {'rate': rate, 'status': 'too_few_moves', 'n_moves': len(moves)}

    def _rank(df):
        means = df.groupby('move')['delta_prog'].mean()
        return means.reindex(moves)

    base = _rank(Dm).to_numpy()
    rng = np.random.default_rng(config.seed)
    move_arr = Dm['move'].to_numpy(dtype=object)
    n = len(move_arr)
    try:
        from scipy.stats import spearmanr
    except Exception:
        spearmanr = None

    rhos: List[float] = []
    for _ in range(k):
        flip = rng.random(n) < rate
        perturbed = move_arr.copy()
        for i in np.where(flip)[0]:
            alts = [m for m in moves if m != move_arr[i]]
            perturbed[i] = alts[rng.integers(0, len(alts))]
        pdf = Dm.assign(move=perturbed)
        pr = _rank(pdf).to_numpy()
        mask = np.isfinite(base) & np.isfinite(pr)
        if mask.sum() < 2:
            continue
        if np.std(base[mask]) == 0 or np.std(pr[mask]) == 0:
            continue                                 # rank correlation undefined for a constant arm
        if spearmanr is not None:
            rho, _ = spearmanr(base[mask], pr[mask])
        else:                                       # rank-correlation fallback
            a = pd.Series(base[mask]).rank().to_numpy()
            b = pd.Series(pr[mask]).rank().to_numpy()
            rho = float(np.corrcoef(a, b)[0, 1])
        if rho == rho:
            rhos.append(float(rho))
    if not rhos:
        return {'rate': rate, 'status': 'no_valid_resamples', 'n_moves': len(moves)}
    return {
        'rate': float(rate),
        'k': len(rhos),
        'mean_spearman': round(float(np.mean(rhos)), 4),
        'sd_spearman': round(float(np.std(rhos)), 4),
        'min_spearman': round(float(np.min(rhos)), 4),
        'n_moves': len(moves),
        'status': 'ok',
    }


# ---------------------------------------------------------------------------
# Trajectory / consolidation model (within- vs between-session)
# ---------------------------------------------------------------------------

def fit_trajectory(participant_df: pd.DataFrame, blocks_df: pd.DataFrame,
                   config: MechanismModelConfig) -> Dict:
    """Participant-level model of per-session stage with LAGGED cue exposure, separating
    WITHIN-session (momentary) from BETWEEN-session (consolidation) cue effects.

    Operationalization (robust at small n): per (participant, session) compute the modal
    progression coordinate and the cue-exposure intensity (number of move-labelled cue
    blocks that session); lag exposure by one session within participant; Mundlak-split
    the lagged exposure into between- (participant mean) and within- (deviation)
    components; fit ``progression ~ session_number + exposure_within + exposure_between +
    (1|participant)``. The within coefficient is the consolidation-relevant 'does a
    session's extra cue exposure precede the next session's advancement' signal.

    Returns {status, within, between, session, method, n, n_participants}. Degrades to a
    status marker on small/degenerate data or missing statsmodels.
    """
    if not config.trajectory:
        return {'status': 'disabled'}
    if participant_df is None or participant_df.empty:
        return {'status': 'no_participant_data'}
    need = {'participant_id', 'session_id', 'session_number', 'progression_coord'}
    if not need.issubset(set(participant_df.columns)):
        return {'status': 'missing_columns'}

    # Per (participant, session): modal/mean progression coordinate.
    ps = (participant_df.dropna(subset=['progression_coord'])
          .groupby(['participant_id', 'session_id', 'session_number'], as_index=False)['progression_coord']
          .mean())
    ps['participant_id'] = ps['participant_id'].astype(str)
    if len(ps) < 6 or ps['participant_id'].nunique() < 2:
        return {'status': 'insufficient_n', 'n': int(len(ps))}

    # Per (participant, session) cue exposure = # move-labelled blocks that session.
    exposure = pd.DataFrame(columns=['participant_id', 'session_id', 'cue_exposure'])
    if blocks_df is not None and not blocks_df.empty and {'participant_id', 'session_id', 'move'}.issubset(blocks_df.columns):
        bm = blocks_df.dropna(subset=['move']).copy()
        bm['participant_id'] = bm['participant_id'].astype(str)
        if not bm.empty:
            exposure = (bm.groupby(['participant_id', 'session_id'], as_index=False)
                        .size().rename(columns={'size': 'cue_exposure'}))
    d = ps.merge(exposure, on=['participant_id', 'session_id'], how='left')
    d['cue_exposure'] = d['cue_exposure'].fillna(0.0).astype(float)

    # Lag cue exposure by one session within participant (consolidation: prior cue → next stage).
    d = d.sort_values(['participant_id', 'session_number'])
    d['cue_exposure_lag'] = d.groupby('participant_id')['cue_exposure'].shift(1)
    d = d.dropna(subset=['cue_exposure_lag'])
    if len(d) < 6 or d['participant_id'].nunique() < 2:
        return {'status': 'insufficient_after_lag', 'n': int(len(d))}

    # Mundlak within/between split of the lagged exposure.
    d = S.within_between_split(d, 'cue_exposure_lag', 'participant_id')
    within_col, between_col = 'cue_exposure_lag_within', 'cue_exposure_lag_between'

    fixed = f'session_number + {within_col} + {between_col}' if config.split_within_between else f'session_number + cue_exposure_lag'
    res = S.mixedlm_delta(d, outcome='progression_coord', fixed=fixed, group='participant_id')
    if res is None:
        return {'status': 'unavailable_or_unfit', 'n': int(len(d))}
    try:
        ci = res.conf_int()

        def _term(name):
            if name not in res.params.index:
                return None
            lo, hi = float(ci.loc[name][0]), float(ci.loc[name][1])
            return {
                'estimate': round(float(res.params[name]), 4),
                'ci_lo': round(lo, 4) if lo == lo else None,
                'ci_hi': round(hi, 4) if hi == hi else None,
                'p_value': round(float(res.pvalues[name]), 4) if name in res.pvalues else None,
                'ci_excludes_0': bool((lo > 0 or hi < 0)) if (lo == lo and hi == hi) else False,
            }
        return {
            'status': 'ok',
            'method': 'mixedlm',
            'n': int(len(d)),
            'n_participants': int(d['participant_id'].nunique()),
            'within': _term(within_col),
            'between': _term(between_col),
            'session': _term('session_number'),
            'note': ('within = a session\'s extra cue exposure vs the participant\'s own mean (momentary); '
                     'between = the participant\'s overall cue exposure (consolidation). Lagged one session.'),
        }
    except Exception:
        return {'status': 'parse_failed', 'n': int(len(d))}


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_csv(rows: List[dict], path: str) -> Optional[str]:
    if not rows:
        return None
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_mechanism_models(enriched_blocks: List[dict], participant_df: pd.DataFrame,
                         output_dir: str, config: Optional[MechanismModelConfig] = None) -> Dict:
    """Run the re-centered mechanism estimator stack and write CSVs to
    ``03_analysis_data/mechanism/``. Returns a dict consumed by ``mechanism.py``'s report.

    ``available`` is True only when the modeling libs ran and produced a result; when the
    estimator is disabled or statsmodels/sklearn are unavailable it returns
    ``available=False`` so the caller preserves the legacy (estimator-off) output.
    """
    if config is None:
        config = MechanismModelConfig()
    result: Dict = {'available': False, 'files_written': []}
    if not config.enabled:
        result['status'] = 'disabled'
        return result

    D = build_design_frame(enriched_blocks)
    if D.empty:
        result['status'] = 'no_blocks'
        return result

    smf, _ = S._import_statsmodels()
    if smf is None:
        result['status'] = 'statsmodels_unavailable'
        return result

    from process import output_paths as _paths
    mech_dir = _paths.mechanism_dir(output_dir)
    os.makedirs(mech_dir, exist_ok=True)

    result['design'] = {
        'n': int(len(D)),
        'n_participants': int(D['participant_id'].nunique()),
        'n_with_move': int(D['move'].notna().sum()),
    }

    adjacency = fit_adjacency_interaction(D, config)
    result['adjacency'] = adjacency

    if config.sensitivity:
        sens = sensitivity_bounds(D)
        result['sensitivity'] = sens
        p = _write_csv(sens['cells'], os.path.join(mech_dir, 'mechanism_sensitivity_evalues.csv'))
        if p:
            result['files_written'].append(p)

    if config.purer_noise_check:
        result['noise_robustness'] = purer_noise_robustness(D, config)

    if config.trajectory:
        result['trajectory'] = fit_trajectory(participant_df, D, config)

    # Earns-its-place CV summary CSV.
    eip = adjacency.get('earns_its_place', {})
    if eip.get('status') == 'ok':
        cv_rows = []
        for name in ('FROM_only', 'additive', 'interaction'):
            r = eip.get(name, {})
            cv_rows.append({
                'model': name,
                'cv_logloss': round(r.get('logloss', float('nan')), 4),
                'cv_logloss_sd': round(r.get('logloss_sd', float('nan')), 4),
                'cv_accuracy': round(r.get('acc', float('nan')), 4),
                'n_params': r.get('n_params'),
                'delta_logloss_vs_from_only': (0.0 if name == 'FROM_only'
                                               else eip.get(f'{name}_delta_logloss')),
            })
        p = _write_csv(cv_rows, os.path.join(mech_dir, 'mechanism_interaction_cv.csv'))
        if p:
            result['files_written'].append(p)

    # Trajectory within/between CSV.
    traj = result.get('trajectory', {})
    if traj.get('status') == 'ok':
        traj_rows = []
        for comp in ('within', 'between', 'session'):
            t = traj.get(comp)
            if t:
                traj_rows.append({'component': comp, **t})
        p = _write_csv(traj_rows, os.path.join(mech_dir, 'mechanism_trajectory_within_between.csv'))
        if p:
            result['files_written'].append(p)

    result['available'] = True
    result['status'] = 'ok'
    return result
