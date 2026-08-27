"""Multiple imputation by chained equations with predictive mean matching."""

import numpy as np


def _norm_draw(y, Z, obs, rng, ridge=1e-5):
    """Draw regression parameters from their posterior — the step that makes this"""
    A, yo = Z[obs], y[obs]
    p = Z.shape[1]
    XtX = A.T @ A
    pen = np.diag(np.diag(XtX) * ridge)
    V = np.linalg.pinv(XtX + pen)
    coef = V @ (A.T @ yo)
    resid = yo - A @ coef
    df = max(len(yo) - p, 1)
    ssr = float(resid @ resid)
    chi = rng.chisquare(df)
    sigma_star = np.sqrt(ssr / chi) if chi > 0 else np.sqrt(ssr / df)
    Vs = 0.5 * (V + V.T)
    try:
        L = np.linalg.cholesky(Vs + 1e-12 * np.eye(p))
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(Vs)
        L = Q @ np.diag(np.sqrt(np.clip(w, 0, None)))
    beta_star = coef + (L @ rng.standard_normal(p)) * sigma_star
    return coef, beta_star


def _pmm_column(y, Z, obs, k, rng):
    """One predictive-mean-matching draw for a single column, matchtype = 1."""
    coef, beta_star = _norm_draw(y, Z, obs, rng)
    yhat_obs = Z[obs] @ coef
    yhat_mis = Z[~obs] @ beta_star
    dv = y[obs]
    order = np.argsort(yhat_obs)
    dp, dv = yhat_obs[order], dv[order]
    miss = np.where(~obs)[0]
    pos = np.searchsorted(dp, yhat_mis)
    out = np.empty(len(miss))
    for t, pp in enumerate(pos):
        lo, hi = max(0, pp - k), min(len(dv), pp + k)
        d = np.abs(dp[lo:hi] - yhat_mis[t])
        d = d + rng.uniform(0, 1e-12, size=d.shape)
        cand = np.argsort(d)[:k] + lo
        out[t] = dv[rng.choice(cand)]
    return miss, out


def mice_pmm(df, cols, m=10, n_iter=5, k=5, seed=42, outcome=None,
             include_outcome=False, verbose=True):
    """Generate `m` completed copies of `df` by chained equations with PMM."""
    cols = [c for c in cols if c in df.columns]
    miss_mask = {c: df[c].isna().values.copy() for c in cols}
    n_inc = int(np.any(np.column_stack([miss_mask[c] for c in cols]), axis=1).sum()) \
        if cols else 0
    if verbose:
        pct = 100 * n_inc / max(len(df), 1)

    out = []
    for d in range(m):
        rng = np.random.RandomState(seed + 1000 * d)
        work = df.copy()
        for c in cols:
            mm = miss_mask[c]
            if mm.any():
                pool = df[c].dropna().values
                work.loc[work.index[mm], c] = rng.choice(pool, size=mm.sum())
        for _ in range(n_iter):
            for c in cols:
                mm = miss_mask[c]
                if not mm.any():
                    continue
                preds = [p for p in df.columns if p != c]
                Zc = work[preds].astype(float)
                Zc = Zc.fillna(Zc.median()).values
                if include_outcome and outcome is not None:
                    Zc = np.column_stack([Zc, np.asarray(outcome, float)])
                sd = Zc.std(0)
                Zc = (Zc - Zc.mean(0)) / np.where(sd < 1e-9, 1.0, sd)
                Zc = np.column_stack([np.ones(len(Zc)), Zc])
                y = work[c].astype(float).values
                idx, vals = _pmm_column(y, Zc, ~mm, k, rng)
                work.loc[work.index[idx], c] = vals
        out.append(work)
    if verbose:
        print("MICE imputation complete.")
    return out


def rubin_pool(estimates, variances, name="", verbose=True):
    """Combine M estimates and their within-imputation variances."""
    Q = np.asarray(estimates, dtype=float)
    U = np.asarray(variances, dtype=float)
    M = len(Q)
    if M < 2:
        raise ValueError("Rubin pooling needs at least 2 imputations")
    Qbar = float(Q.mean())
    Ubar = float(U.mean())
    B = float(Q.var(ddof=1))
    T = Ubar + (1.0 + 1.0 / M) * B
    se = float(np.sqrt(T))
    fmi = float((1.0 + 1.0 / M) * B / T) if T > 0 else 0.0
    if B > 0:
        dfree = (M - 1) * (1.0 + Ubar / ((1.0 + 1.0 / M) * B)) ** 2
    else:
        dfree = float("inf")
    z = 1.959963985
    tcrit = z if dfree > 200 else z * (1.0 + (z * z + 1.0) / (4.0 * max(dfree, 1.0)))
    lo, hi = Qbar - tcrit * se, Qbar + tcrit * se
    if verbose:
        tag = f"[{name}] " if name else ""
    return {"estimate": Qbar, "se": se, "lo": lo, "hi": hi, "within": Ubar,
            "between": B, "total": T, "fmi": fmi, "df": float(dfree), "m": M}
