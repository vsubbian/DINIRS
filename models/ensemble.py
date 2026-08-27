"""Cross-validated ensemble producing the out-of-fold DINIRS estimate."""

import os
import numpy as np

from utils.metrics import c_for_benefit


def fold_assignment(n, k_folds, seed):
    """Reproduce utils.generalization.cross_fitted_ite's fold assignment."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    for i, idx in enumerate(order):
        fold_id[idx] = i % k_folds
    return fold_id


def _z(v):
    """Z-score, robust to a constant vector."""
    v = np.asarray(v, dtype=float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else v - v.mean()


def _c(tau, y, w, n_pairs=10000, seed=42):
    """c-for-benefit, returning 0.5 (chance) if it cannot be computed."""
    try:
        return c_for_benefit(tau, y, w, n_pairs=n_pairs,
                             random_state=seed)['c_for_benefit']
    except Exception:
        return 0.5


def greedy_ensemble_selection(library, y, w, mask, n_iter=25, seed=42,
                              verbose=False, names=None):
    """Forward stepwise ensemble selection WITH REPLACEMENT."""
    names = list(library) if names is None else list(names)
    zs = {m: _z(library[m]) for m in names}
    counts = {m: 0 for m in names}
    running = np.zeros(int(mask.sum()), dtype=float)
    chosen = 0
    best_hist = []

    for _ in range(n_iter):
        best_m, best_c = None, -np.inf
        for m in names:
            cand = (running * chosen + zs[m][mask]) / (chosen + 1)
            cm = _c(cand, y[mask], w[mask], seed=seed)
            if cm > best_c:
                best_m, best_c = m, cm
        running = (running * chosen + zs[best_m][mask]) / (chosen + 1)
        chosen += 1
        counts[best_m] += 1
        best_hist.append(best_c)

    out = {m: 0.0 for m in library}
    out.update({m: counts[m] / n_iter for m in names})
    return out, best_hist[-1]


def bagged_ensemble_selection(library, y, w, mask, n_iter=20, n_bags=20,
                              bag_frac=0.5, seed=42, verbose=False):
    """Bagged ensemble selection — the fix for selection overfitting."""
    all_names = list(library)
    if len(all_names) <= 2:
        return greedy_ensemble_selection(library, y, w, mask, n_iter=n_iter,
                                         seed=seed, verbose=verbose)
    rng = np.random.RandomState(seed)
    k = max(2, int(round(bag_frac * len(all_names))))
    acc = {m: 0.0 for m in all_names}
    cs = []
    for b in range(n_bags):
        sub = list(rng.choice(all_names, size=min(k, len(all_names)),
                              replace=False))
        wts, c_b = greedy_ensemble_selection(library, y, w, mask,
                                             n_iter=n_iter, seed=seed + b,
                                             names=sub)
        for m, v in wts.items():
            acc[m] += v
        cs.append(c_b)
    tot = sum(acc.values()) or 1.0
    return {m: acc[m] / tot for m in all_names}, float(np.mean(cs))


def apply_weights(library, weights):
    """Weighted mean of z-scored library members."""
    out = None
    for m, wt in weights.items():
        if wt <= 0:
            continue
        z = _z(library[m])
        out = z * wt if out is None else out + z * wt
    return out


def lofo_super_learner(library, y, w, k_folds, seed, n_iter=25, verbose=True,
                       bagged=True, n_bags=20, bag_frac=0.5,
                       guard_to_best=True):
    """Out-of-fold stacked prediction for every patient."""
    n = len(y)
    fold_id = fold_assignment(n, k_folds, seed)
    tau_z = np.full(n, np.nan, dtype=float)
    all_w, sel_cs = [], []

    for k in range(k_folds):
        sel = fold_id != k
        if bagged:
            wts, sel_c = bagged_ensemble_selection(
                library, y, w, sel, n_iter=n_iter, n_bags=n_bags,
                bag_frac=bag_frac, seed=seed)
        else:
            wts, sel_c = greedy_ensemble_selection(library, y, w, sel,
                                                   n_iter=n_iter, seed=seed)

        if guard_to_best:
            ens_c = _c(apply_weights(library, wts)[sel], y[sel], w[sel],
                       seed=seed)
            best_m, best_c = None, ens_c
            for m in library:
                cm = _c(_z(library[m])[sel], y[sel], w[sel], seed=seed)
                if cm > best_c:
                    best_m, best_c = m, cm
            if best_m is not None:
                wts = {m: (1.0 if m == best_m else 0.0) for m in library}
                sel_c = best_c

        held = fold_id == k
        full = apply_weights(library, wts)
        tau_z[held] = full[held]
        all_w.append(wts)
        sel_cs.append(sel_c)
        if verbose:
            top = sorted(((v, m) for m, v in wts.items() if v > 0), reverse=True)
            desc = "  ".join(f"{m}={v:.2f}" for v, m in top)

    assert np.isfinite(tau_z).all(), "some patient received no stacked prediction"
    return {'tau_z': tau_z, 'weights': all_w, 'sel_c': sel_cs,
            'fold_id': fold_id}


def rescale_to_days(tau_z, library, weights_list):
    """Map the z-space stacked score back into VFD-day units."""
    names = list(library)
    mean_w = {m: float(np.mean([wl.get(m, 0.0) for wl in weights_list]))
              for m in names}
    raw = None
    for m, wt in mean_w.items():
        if wt <= 0:
            continue
        v = np.asarray(library[m], dtype=float)
        raw = v * wt if raw is None else raw + v * wt
    if raw is None:
        return tau_z.copy(), mean_w
    s = tau_z.std()
    b = raw.std() / s if s > 1e-12 else 1.0
    return raw.mean() + b * (tau_z - tau_z.mean()), mean_w


def seed_ensembled_crossfit(cross_fitted_ite, build_model_fn, train_stage1,
                            train_stage2, loss_fn, config, X, W, VFD, delta,
                            pad_masks, k_folds, fold_seed, tau_base,
                            torch_seeds=(0, 1, 2), save_dir=None, verbose=True):
    """Average out-of-fold predictions over several model initialisations while"""
    import torch

    per_seed = []
    for si, ts in enumerate(torch_seeds):
        torch.manual_seed(ts)
        np.random.seed(ts)
        sd = None if save_dir is None else os.path.join(save_dir, f'ts{ts}')
        r = cross_fitted_ite(build_model_fn, train_stage1, train_stage2,
                             loss_fn, config, X, W, VFD, delta, pad_masks,
                             K=k_folds, seed=fold_seed, save_dir=sd,
                             tau_base=tau_base, verbose=verbose)
        per_seed.append(r['ite_oof'])

    ite_oof = np.mean(per_seed, axis=0)
    c_oof = _c(ite_oof, VFD, W)
    if verbose:
        singles = [_c(p, VFD, W) for p in per_seed]
    return {'ite_oof': ite_oof, 'per_seed': per_seed, 'c_oof': float(c_oof)}


def incremental_value(library, twin_keys, y, w, k_folds, seed, n_iter=25,
                      verbose=True, **kw):
    """Does the virtual twin add ranking information beyond the baselines?"""
    base_lib = {m: v for m, v in library.items() if m not in twin_keys}
    assert base_lib, "no baseline candidates in library"
    r_base = lofo_super_learner(base_lib, y, w, k_folds, seed, n_iter,
                                verbose, **kw)
    r_full = lofo_super_learner(library, y, w, k_folds, seed, n_iter,
                                verbose, **kw)
    return r_base, r_full
