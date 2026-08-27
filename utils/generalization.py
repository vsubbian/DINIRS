"""Cross-fitting, seed-stability, and subgroup evaluation."""

import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.extraction import NIRSTwinDataset
from utils.metrics import c_for_benefit


def _predict_ite(model, X, W, VFD, delta, pad_masks, device, batch_size=256,
                 tau_base=None):
    """Return (N,) ITE predictions for all rows of X (hybrid base aware)."""
    model.eval()
    dl = DataLoader(
        NIRSTwinDataset(X, W, VFD, delta, pad_masks, tau_base=tau_base),
        batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for b in dl:
            tb = b.get('tau_base')
            tb = tb.to(device) if tb is not None else None
            o, _ = model.forward_predictor(
                b['x'].to(device), b['pad_mask'].to(device), tau_base=tb)
            out.append(o['ite'].cpu().numpy().ravel())
    return np.concatenate(out)


def cross_fitted_ite(build_model_fn, train_stage1, train_stage2, loss_fn, config,
                     X, W, VFD, delta, pad_masks, K=5, seed=42, save_dir=None,
                     verbose=True, tau_base=None):
    """K-fold cross-fitted ITE: every patient receives an OUT-OF-FOLD estimate."""
    device = config['device']
    N = len(X)
    rng = np.random.RandomState(seed)
    order = rng.permutation(N)
    fold_id = np.empty(N, dtype=int)
    for i, idx in enumerate(order):
        fold_id[idx] = i % K

    ite_oof = np.full(N, np.nan, dtype=float)
    fold_c = []
    base_dir = save_dir or os.path.join(os.getcwd(), 'output', '_crossfit')
    os.makedirs(base_dir, exist_ok=True)

    for k in range(K):
        te = np.where(fold_id == k)[0]
        tr = np.where(fold_id != k)[0]
        n_val = max(1, int(0.1 * len(tr)))
        val = tr[:n_val]
        trn = tr[n_val:]
        if verbose:
            print(f"Cross-fit fold {k+1}/{K}")

        def mk(ix, shuffle):
            return DataLoader(
                NIRSTwinDataset(X[ix], W[ix], VFD[ix], delta[ix],
                                pad_masks[ix] if pad_masks is not None else None,
                                tau_base=tau_base[ix] if tau_base is not None else None),
                batch_size=config.get('batch_size', 128), shuffle=shuffle)

        fold_dir = os.path.join(base_dir, f'fold{k}')
        os.makedirs(fold_dir, exist_ok=True)

        torch.manual_seed(seed * 1000 + k)
        np.random.seed(seed * 1000 + k)
        m = build_model_fn()
        m, _ = train_stage1(m, mk(trn, True), mk(val, False), loss_fn, config, fold_dir)
        m, _ = train_stage2(m, mk(trn, True), mk(val, False), loss_fn, config, fold_dir)

        tau_te = _predict_ite(
            m, X[te], W[te], VFD[te], delta[te],
            pad_masks[te] if pad_masks is not None else None, device,
            tau_base=tau_base[te] if tau_base is not None else None)
        ite_oof[te] = tau_te
        try:
            ck = c_for_benefit(tau_te, VFD[te], W[te])['c_for_benefit']
        except Exception:
            ck = float('nan')
        fold_c.append(ck)

    c_oof = c_for_benefit(ite_oof, VFD, W)['c_for_benefit']
    if verbose:
        fc = np.array(fold_c, dtype=float)
    return {'ite_oof': ite_oof, 'fold_id': fold_id,
            'fold_c': fold_c, 'c_oof': float(c_oof)}


def seed_stability(build_model_fn, train_stage1, train_stage2, loss_fn, config,
                   X, W, VFD, delta, pad_masks, train_idx, test_idx,
                   seeds=(0, 1, 2, 3, 4), save_dir=None, verbose=True):
    """Re-train the whole pipeline under several seeds on a FIXED split and report"""
    device = config['device']
    base_dir = save_dir or os.path.join(os.getcwd(), 'output', '_seedstab')
    os.makedirs(base_dir, exist_ok=True)

    def mk(ix, shuffle):
        return DataLoader(
            NIRSTwinDataset(X[ix], W[ix], VFD[ix], delta[ix],
                            pad_masks[ix] if pad_masks is not None else None),
            batch_size=config.get('batch_size', 128), shuffle=shuffle)

    n_val = max(1, int(0.1 * len(train_idx)))
    val, trn = train_idx[:n_val], train_idx[n_val:]
    taus, cs = [], []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        d = os.path.join(base_dir, f'seed{s}')
        os.makedirs(d, exist_ok=True)
        m = build_model_fn()
        m, _ = train_stage1(m, mk(trn, True), mk(val, False), loss_fn, config, d)
        m, _ = train_stage2(m, mk(trn, True), mk(val, False), loss_fn, config, d)
        tau = _predict_ite(
            m, X[test_idx], W[test_idx], VFD[test_idx], delta[test_idx],
            pad_masks[test_idx] if pad_masks is not None else None, device)
        c = c_for_benefit(tau, VFD[test_idx], W[test_idx])['c_for_benefit']
        taus.append(tau); cs.append(c)

    cs = np.array(cs, dtype=float)
    tau_ens = np.mean(taus, axis=0)
    c_ens = c_for_benefit(tau_ens, VFD[test_idx], W[test_idx])['c_for_benefit']
    return {'seed_c': cs.tolist(), 'mean': float(cs.mean()), 'sd': float(cs.std()),
            'min': float(cs.min()), 'max': float(cs.max()),
            'ensemble_c': float(c_ens), 'ite_ensemble': tau_ens}


def subgroup_robustness(ite, vfd, treatment, strata, stratum_name='stratum',
                        min_n=100, verbose=True):
    """c-for-benefit within clinically meaningful strata (severity band, age band,"""
    ite = np.asarray(ite); vfd = np.asarray(vfd)
    treatment = np.asarray(treatment); strata = np.asarray(strata)
    out = {}
    for g in np.unique(strata):
        m = strata == g
        if m.sum() < min_n:
            continue
        try:
            c = c_for_benefit(ite[m], vfd[m], treatment[m])['c_for_benefit']
        except Exception:
            c = float('nan')
        out[str(g)] = {'n': int(m.sum()), 'c': float(c)}
    return out
