"""Evaluation metrics, doubly robust policy value, and sensitivity analyses."""

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


def pehe(ite_pred, ite_true):
    """Precision in Estimation of Heterogeneous Effects."""
    return np.sqrt(np.mean((ite_pred - ite_true) ** 2))


def ate_bias(ite_pred, ite_true):
    """Bias in Average Treatment Effect estimation."""
    return np.abs(np.mean(ite_pred) - np.mean(ite_true))


def policy_value(ite_pred, vfd_observed, treatment_observed):
    """Mean VFD-28 under model-recommended treatment vs observed."""
    recommended = (ite_pred > 0).astype(int)

    concordant = (recommended == treatment_observed)
    n_concordant = concordant.sum()

    vfd_concordant = vfd_observed[concordant].mean() if n_concordant > 0 else 0.0

    vfd_observed_policy = vfd_observed.mean()

    pct_nirs = recommended.mean() * 100

    return {
        'vfd_model_policy': vfd_concordant,
        'vfd_observed_policy': vfd_observed_policy,
        'policy_improvement': vfd_concordant - vfd_observed_policy,
        'n_concordant': int(n_concordant),
        'pct_concordant': n_concordant / len(ite_pred) * 100,
        'pct_recommend_nirs': pct_nirs,
    }


def survival_calibration(p_survive_pred, delta):
    """Brier score for survival probability calibration."""
    return brier_score_loss(delta, p_survive_pred)


def match_on_predicted_benefit(ite_pred, treatment_observed, method='nearest'):
    """1:1 match treated to untreated patients on PREDICTED BENEFIT."""
    ite_pred = np.asarray(ite_pred).ravel()
    w = np.asarray(treatment_observed).ravel()
    a = np.where(w == 1)[0]
    b = np.where(w == 0)[0]
    if len(a) == 0 or len(b) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    a = a[np.argsort(ite_pred[a], kind='mergesort')]
    b = b[np.argsort(ite_pred[b], kind='mergesort')]

    if method == 'truncate':
        m = min(len(a), len(b))
        return a[:m], b[:m]

    if method == 'quantile':
        m = min(len(a), len(b))
        take = lambda arr: (arr if len(arr) == m else
                            arr[np.round(np.linspace(0, len(arr) - 1, m)).astype(int)])
        return take(a), take(b)

    bv = ite_pred[b]
    n = len(b)
    nxt = list(range(n + 1))
    prv = list(range(-1, n))

    def _fn(i):
        root = i
        while nxt[root] != root:
            root = nxt[root]
        while nxt[i] != root:
            nxt[i], i = root, nxt[i]
        return root

    def _fp(i):
        root = i + 1
        while prv[root] != root - 1:
            root = prv[root] + 1
        while prv[i + 1] != root - 1:
            prv[i + 1], i = root - 1, prv[i + 1]
        return root - 1

    A, B = [], []
    for ia in a:
        v = ite_pred[ia]
        k = int(np.searchsorted(bv, v))
        r = _fn(min(k, n))
        l = _fp(min(k, n - 1))
        cand = []
        if r < n:
            cand.append((abs(bv[r] - v), r))
        if l >= 0:
            cand.append((abs(bv[l] - v), l))
        if not cand:
            break
        _, pick = min(cand)
        nxt[pick] = pick + 1
        prv[pick + 1] = pick - 1
        A.append(ia)
        B.append(b[pick])
    return np.asarray(A, dtype=int), np.asarray(B, dtype=int)


def c_for_benefit(ite_pred, vfd_observed, treatment_observed,
                  n_pairs=None, random_state=42, match='nearest'):
    """Concordance statistic for benefit (C-for-benefit)."""
    rng = np.random.RandomState(random_state)

    ite_pred = np.asarray(ite_pred).flatten()
    vfd_observed = np.asarray(vfd_observed).flatten()
    treatment_observed = np.asarray(treatment_observed).flatten()

    nirs_idx = np.where(treatment_observed == 1)[0]
    imv_idx = np.where(treatment_observed == 0)[0]

    if len(nirs_idx) == 0 or len(imv_idx) == 0:
        return {'c_for_benefit': 0.5, 'n_matched_pairs': 0,
                'interpretation': 'Cannot compute: one treatment arm is empty'}

    nirs_matched, imv_matched = match_on_predicted_benefit(
        ite_pred, treatment_observed, method=match)
    n_matched = len(nirs_matched)
    if n_matched < 2:
        return {'c_for_benefit': 0.5, 'n_matched_pairs': int(n_matched),
                'interpretation': 'Fewer than two matched pairs'}

    obs_benefit = vfd_observed[nirs_matched] - vfd_observed[imv_matched]
    pred_benefit = (ite_pred[nirs_matched] + ite_pred[imv_matched]) / 2.0

    max_pairs_of_pairs = n_pairs
    if max_pairs_of_pairs is not None and \
            n_matched * (n_matched - 1) // 2 > max_pairs_of_pairs:
        i_idx = rng.randint(0, n_matched, size=max_pairs_of_pairs)
        j_idx = rng.randint(0, n_matched, size=max_pairs_of_pairs)
        valid = i_idx != j_idx
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]
    else:
        i_grid, j_grid = np.triu_indices(n_matched, k=1)
        i_idx = i_grid
        j_idx = j_grid

    pred_diff = pred_benefit[i_idx] - pred_benefit[j_idx]
    obs_diff = obs_benefit[i_idx] - obs_benefit[j_idx]

    comparable = obs_diff != 0
    pred_diff = pred_diff[comparable]
    obs_diff = obs_diff[comparable]
    n_comp = len(pred_diff)

    if n_comp == 0:
        return {'c_for_benefit': 0.5, 'n_matched_pairs': int(n_matched),
                'n_comparable_pair_of_pairs': 0,
                'interpretation': 'No comparable matched-pair-of-pairs (all observed benefits equal)'}

    concordant = int(np.sum(np.sign(pred_diff) == np.sign(obs_diff)))
    discordant = int(np.sum((np.sign(pred_diff) * np.sign(obs_diff)) < 0))
    tied = n_comp - concordant - discordant
    c_benefit = (concordant + 0.5 * tied) / n_comp

    return {
        'c_for_benefit': float(c_benefit),
        'n_matched_pairs': int(n_matched),
        'n_comparable_pair_of_pairs': int(n_comp),
        'n_concordant': int(concordant),
        'n_discordant': int(discordant),
        'n_tied': int(tied),
        'interpretation': (
            f'C-for-benefit = {c_benefit:.3f}. '
            f'Values > 0.5 indicate the model correctly orders predicted benefit '
            f'with observed benefit across matched pairs. '
            f'({concordant} concordant, {discordant} discordant, {tied} tied '
            f'out of {n_comp} comparable matched-pair-of-pairs from {n_matched} matched pairs).'
        )
    }


def bootstrap_ci(values, statistic=np.mean, n_boot=1000, alpha=0.05, random_state=42):
    """Bootstrap confidence interval for an arbitrary statistic."""
    rng = np.random.RandomState(random_state)
    values = np.asarray(values).flatten()
    n = len(values)
    point = float(statistic(values))
    boot_stats = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[b] = statistic(sample)
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return {'point': point, 'lower': lo, 'upper': hi, 'n_boot': int(n_boot),
            'alpha': float(alpha)}


def bootstrap_ablation_metric(ite_pred, vfd_observed, treatment_observed,
                              metric_fn=None, n_boot=1000, random_state=42):
    """Bootstrap CI for an ablation metric computed over the test cohort."""
    rng = np.random.RandomState(random_state)
    ite_pred = np.asarray(ite_pred).flatten()
    vfd_observed = np.asarray(vfd_observed).flatten()
    treatment_observed = np.asarray(treatment_observed).flatten()
    n = len(ite_pred)

    if metric_fn is None:
        def metric_fn(it, vf, tr):
            rec = (it > 0).astype(int)
            concordant_mask = rec == tr
            return float(vf[concordant_mask].mean()) if concordant_mask.any() else 0.0

    point = metric_fn(ite_pred, vfd_observed, treatment_observed)
    boot_vals = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_vals[b] = metric_fn(ite_pred[idx], vfd_observed[idx], treatment_observed[idx])
    lo = float(np.percentile(boot_vals, 2.5))
    hi = float(np.percentile(boot_vals, 97.5))
    return {'point': float(point), 'lower': lo, 'upper': hi, 'n_boot': int(n_boot)}


def compute_all_metrics(pred_outputs, vfd_observed, delta, treatment_observed,
                        ite_true=None):
    """Compute all evaluation metrics."""
    ite_pred = pred_outputs['ite'].squeeze()
    p_surv_obs = np.where(
        treatment_observed == 1,
        pred_outputs['p_surv_1'].squeeze(),
        pred_outputs['p_surv_0'].squeeze(),
    )

    results = {}

    results.update(policy_value(ite_pred, vfd_observed, treatment_observed))

    results['brier_score'] = survival_calibration(p_surv_obs, delta)

    if len(np.unique(delta)) > 1:
        results['surv_auroc'] = roc_auc_score(delta, p_surv_obs)
        results['surv_auprc'] = average_precision_score(delta, p_surv_obs)

    results['mean_ite'] = float(np.mean(ite_pred))
    results['std_ite'] = float(np.std(ite_pred))
    results['pct_nirs_beneficial'] = float((ite_pred > 0).mean() * 100)
    results['pct_imv_beneficial'] = float((ite_pred < 0).mean() * 100)

    results['mean_ite_survival'] = float(np.mean(pred_outputs['ite_survival'].squeeze()))
    results['mean_ite_vfd_cond'] = float(np.mean(pred_outputs['ite_vfd_cond'].squeeze()))

    cfb = c_for_benefit(ite_pred, vfd_observed, treatment_observed)
    results['c_for_benefit'] = cfb['c_for_benefit']
    results['c_for_benefit_interpretation'] = cfb['interpretation']

    if ite_true is not None:
        results['pehe'] = pehe(ite_pred, ite_true)
        results['ate_bias'] = ate_bias(ite_pred, ite_true)

    e_value_results = compute_e_value_for_ate(
        ate=results['mean_ite'],
        outcome_std=float(np.std(vfd_observed)),
    )
    results['e_value'] = e_value_results['e_value']
    results['approx_risk_ratio'] = e_value_results['approx_risk_ratio']
    results['e_value_interpretation'] = e_value_results['interpretation']

    rosenbaum = rosenbaum_sensitivity_bounds(ite_pred, treatment_observed, vfd_observed)
    results['rosenbaum_critical_gamma'] = rosenbaum['critical_gamma']
    results['rosenbaum_interpretation'] = rosenbaum['interpretation']

    return results


def compute_e_value(risk_ratio):
    """Compute E-value for an observed risk ratio."""
    if risk_ratio < 1:
        risk_ratio = 1.0 / risk_ratio
    if risk_ratio <= 1.0:
        return 1.0
    return risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1.0))


def compute_e_value_for_ate(ate, outcome_std, treatment_prevalence=0.5):
    """Approximate E-value for an ATE estimate using standardized effect size."""
    d = ate / max(outcome_std, 1e-6)

    log_rr = d * np.pi / np.sqrt(3)
    approx_rr = np.exp(abs(log_rr))

    e_val = compute_e_value(approx_rr)

    return {
        'e_value': float(e_val),
        'approx_risk_ratio': float(approx_rr),
        'standardized_effect': float(d),
        'interpretation': (
            f'To explain away the observed ATE of {ate:.3f} VFD-28 days, '
            f'an unmeasured confounder would need to be associated with both '
            f'treatment and outcome by a risk ratio of at least {e_val:.2f}.'
        ),
    }


def rosenbaum_sensitivity_bounds(ite_pred, treatment, vfd_observed,
                                  gamma_values=None):
    """Rosenbaum sensitivity analysis bounds for the treatment effect."""
    if gamma_values is None:
        gamma_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    treated_mask = treatment == 1
    control_mask = treatment == 0
    obs_ate = (vfd_observed[treated_mask].mean()
               - vfd_observed[control_mask].mean())

    results = []
    for gamma in gamma_values:
        bias_factor = np.log(gamma)
        ate_lower = obs_ate - bias_factor * np.std(vfd_observed) / np.sqrt(len(vfd_observed))
        ate_upper = obs_ate + bias_factor * np.std(vfd_observed) / np.sqrt(len(vfd_observed))

        results.append({
            'gamma': gamma,
            'ate_lower_bound': float(ate_lower),
            'ate_upper_bound': float(ate_upper),
            'effect_robust': ate_lower > 0,
        })

    critical_gamma = None
    for r in results:
        if not r['effect_robust']:
            critical_gamma = r['gamma']
            break

    return {
        'observed_ate': float(obs_ate),
        'bounds': results,
        'critical_gamma': critical_gamma,
        'interpretation': (
            f'The observed ATE of {obs_ate:.3f} remains significant up to '
            f'Gamma={critical_gamma if critical_gamma else ">3.0"}, meaning '
            f'an unmeasured confounder would need to change treatment odds by '
            f'at least {critical_gamma if critical_gamma else ">3x"} to nullify the effect.'
        ),
    }


def plot_model_comparison_bars(results_dict, metric_key, metric_label,
                               save_path=None):
    """Bar chart comparing models — matches graphspa Fig 2 style."""
    import matplotlib.pyplot as plt

    models = list(results_dict.keys())
    means = [results_dict[m]['mean'] for m in models]
    stds = [results_dict[m]['std'] for m in models]

    colors = ['#b0b0b0'] * (len(models) - 1) + ['#2196F3']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.8)

    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_title(f'Model Comparison: {metric_label}', fontsize=14)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_ite_distribution(ite_pred, model_name='DINIRS', save_path=None):
    """ITE distribution histogram with kernel density — matches AMIA style."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(ite_pred, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', linewidth=0.5, label='ITE distribution')

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ite_pred)
    x_range = np.linspace(ite_pred.min(), ite_pred.max(), 200)
    ax.plot(x_range, kde(x_range), color='navy', linewidth=2, label='KDE')

    mean_ite = np.mean(ite_pred)
    ax.axvline(mean_ite, color='orange', linewidth=2, linestyle='--',
               label=f'Mean ITE = {mean_ite:.3f}')

    ax.axvline(0, color='red', linewidth=1.5, linestyle=':',
               label='Equipoise (ITE=0)')

    ax.fill_betweenx([0, ax.get_ylim()[1] * 0.05], ite_pred.min(), 0,
                      alpha=0.1, color='red', label='IMV beneficial')
    ax.fill_betweenx([0, ax.get_ylim()[1] * 0.05], 0, ite_pred.max(),
                      alpha=0.1, color='green', label='NIRS beneficial')

    ax.set_xlabel('ITE (VFD-28 days: positive = NIRS better)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{model_name} — Individualized Treatment Effect Distribution',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_training_curves(train_log, save_path=None):
    """Training loss curves over epochs — matches graphspa training notebook style."""
    import matplotlib.pyplot as plt

    epochs = train_log['epoch']
    n_plots = len([k for k in train_log if k != 'epoch'])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    plot_keys = [k for k in train_log if k != 'epoch']
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

    for idx, (key, color) in enumerate(zip(plot_keys[:6], colors)):
        ax = axes[idx]
        ax.plot(epochs, train_log[key], color=color, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.set_title(key, fontsize=11)
        ax.grid(alpha=0.3)

    for idx in range(len(plot_keys), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('DINIRS Training Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_decomposed_ite_scatter(pred_outputs, save_path=None):
    """Scatter plot: survival ITE vs conditional VFD ITE."""
    import matplotlib.pyplot as plt

    ite_surv = pred_outputs['ite_survival'].squeeze()
    ite_vfd = pred_outputs['ite_vfd_cond'].squeeze()

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(ite_surv, ite_vfd, c=ite_surv + ite_vfd,
                         cmap='RdYlGn', alpha=0.5, s=20, edgecolors='none')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    ax.text(0.95, 0.95, 'NIRS: survival + VFD', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='green', weight='bold')
    ax.text(0.05, 0.05, 'IMV: survival + VFD', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10, color='red', weight='bold')
    ax.text(0.95, 0.05, 'NIRS: survival only', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='orange')
    ax.text(0.05, 0.95, 'IMV: survival, NIRS: VFD', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, color='purple')

    ax.set_xlabel('ΔP(survive) — Survival ITE (NIRS − IMV)', fontsize=12)
    ax.set_ylabel('ΔVFD|survive — Conditional VFD ITE (NIRS − IMV)', fontsize=12)
    ax.set_title('Decomposed Treatment Effect: Survival vs Ventilation Duration',
                 fontsize=13)

    plt.colorbar(scatter, ax=ax, label='Total ITE direction')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_subgroup_ite_trends(ite_pred, covariate_values, covariate_name,
                              n_bins=5, save_path=None):
    """ITE trend across covariate bins — shows treatment effect heterogeneity."""
    import matplotlib.pyplot as plt

    bin_edges = np.percentile(covariate_values, np.linspace(0, 100, n_bins + 1))
    bin_labels = []
    bin_means = []
    bin_stds = []
    bin_centers = []

    for i in range(n_bins):
        mask = (covariate_values >= bin_edges[i]) & (covariate_values < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (covariate_values >= bin_edges[i]) & (covariate_values <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_means.append(np.mean(ite_pred[mask]))
            bin_stds.append(np.std(ite_pred[mask]) / np.sqrt(mask.sum()))
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_labels.append(f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}')

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-',
                color='#2196F3', linewidth=2, markersize=8, capsize=5,
                label='Mean ITE ± SE')

    ax.axhline(0, color='red', linewidth=1, linestyle=':', label='Equipoise')
    ax.fill_between(bin_centers, 0, bin_means, alpha=0.1,
                     color=np.where(np.array(bin_means) > 0, 'green', 'red'))

    ax.set_xlabel(covariate_name, fontsize=12)
    ax.set_ylabel('ITE (VFD-28 days)', fontsize=12)
    ax.set_title(f'Treatment Effect Heterogeneity by {covariate_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def c_for_benefit_ci(ite_pred, vfd_observed, treatment_observed,
                     n_boot=1000, alpha=0.05, random_state=42):
    """Percentile bootstrap 95% CI for the van Klaveren c-for-benefit."""
    rng = np.random.RandomState(random_state)
    ite_pred = np.asarray(ite_pred).flatten()
    vfd_observed = np.asarray(vfd_observed).flatten()
    treatment_observed = np.asarray(treatment_observed).flatten()
    n = len(ite_pred)

    point = c_for_benefit(ite_pred, vfd_observed, treatment_observed,
                          random_state=random_state)['c_for_benefit']
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[b] = c_for_benefit(ite_pred[idx], vfd_observed[idx],
                                treatment_observed[idx],
                                n_pairs=200_000)['c_for_benefit']
    return {'point': float(point),
            'lower': float(np.percentile(boot, 100 * alpha / 2)),
            'upper': float(np.percentile(boot, 100 * (1 - alpha / 2))),
            'n_boot': int(n_boot)}


def pehe_ci(ite_pred, ite_true, n_boot=1000, alpha=0.05, random_state=42):
    """Percentile bootstrap 95% CI for sqrt-PEHE (semi-synthetic ground truth)."""
    rng = np.random.RandomState(random_state)
    ite_pred = np.asarray(ite_pred).flatten()
    ite_true = np.asarray(ite_true).flatten()
    n = len(ite_pred)

    point = pehe(ite_pred, ite_true)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[b] = pehe(ite_pred[idx], ite_true[idx])
    return {'point': float(point),
            'lower': float(np.percentile(boot, 100 * alpha / 2)),
            'upper': float(np.percentile(boot, 100 * (1 - alpha / 2))),
            'n_boot': int(n_boot)}


def compare_models_with_ci(models, vfd_observed, treatment_observed,
                           ite_true=None, n_boot=1000, random_state=42):
    """Build the corrected Table-1 comparison with bootstrap 95% CIs."""
    out, table = {}, []
    for name, ite in models.items():
        ite = np.asarray(ite).flatten()
        rec = {'c_for_benefit': c_for_benefit_ci(
            ite, vfd_observed, treatment_observed,
            n_boot=n_boot, random_state=random_state)}
        if ite_true is not None:
            rec['pehe'] = pehe_ci(ite, ite_true, n_boot=n_boot,
                                  random_state=random_state)
        out[name] = rec
        cfb = rec['c_for_benefit']
        row = (f"{name:16s}  C-for-benefit = {cfb['point']:.3f} "
               f"[{cfb['lower']:.3f}, {cfb['upper']:.3f}]")
        if 'pehe' in rec:
            p = rec['pehe']
            row += f"   sqrt-PEHE = {p['point']:.3f} [{p['lower']:.3f}, {p['upper']:.3f}]"
        table.append(row)
    out['table'] = table
    return out


def gate_ablation_ci(ite_full, ite_no_gate, vfd_observed, treatment_observed,
                     n_boot=1000, random_state=42):
    """Bootstrap 95% CI for the survival-gate ablation effect on c-for-benefit."""
    rng = np.random.RandomState(random_state)
    ite_full = np.asarray(ite_full).flatten()
    ite_no_gate = np.asarray(ite_no_gate).flatten()
    vfd_observed = np.asarray(vfd_observed).flatten()
    treatment_observed = np.asarray(treatment_observed).flatten()
    n = len(ite_full)

    def cfb(it, idx):
        return c_for_benefit(it[idx], vfd_observed[idx],
                             treatment_observed[idx], random_state=0)['c_for_benefit']

    full_pt = cfb(ite_full, np.arange(n))
    nog_pt = cfb(ite_no_gate, np.arange(n))
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        deltas[b] = cfb(ite_full, idx) - cfb(ite_no_gate, idx)
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {'delta_point': float(full_pt - nog_pt),
            'delta_lower': lo, 'delta_upper': hi,
            'full': float(full_pt), 'no_gate': float(nog_pt),
            'p_two_sided': float(min(p, 1.0)), 'n_boot': int(n_boot)}


def paired_c_for_benefit_test(ite_a, ite_b, vfd_observed, treatment_observed,
                              name_a="A", name_b="B", n_boot=2000, random_state=42):
    """PAIRED bootstrap test for the DIFFERENCE in c-for-benefit between two"""
    rng = np.random.RandomState(random_state)
    a = np.asarray(ite_a).ravel(); b = np.asarray(ite_b).ravel()
    y = np.asarray(vfd_observed).ravel(); w = np.asarray(treatment_observed).ravel()
    n = len(y)

    c_a = c_for_benefit(a, y, w)['c_for_benefit']
    c_b = c_for_benefit(b, y, w)['c_for_benefit']
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        deltas[i] = (c_for_benefit(a[idx], y[idx], w[idx], n_pairs=150_000)['c_for_benefit']
                     - c_for_benefit(b[idx], y[idx], w[idx], n_pairs=150_000)['c_for_benefit'])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    out = {'c_a': float(c_a), 'c_b': float(c_b),
           'delta': float(c_a - c_b), 'lower': float(lo), 'upper': float(hi),
           'p_two_sided': float(min(p, 1.0)), 'n_boot': int(n_boot)}
    return out


def calibration_for_benefit(ite_pred, vfd_observed, treatment_observed,
                            n_groups=5, random_state=42, verbose=True):
    """Calibration-for-benefit: observed vs predicted benefit across quantile groups"""
    ite_pred = np.asarray(ite_pred).ravel()
    vfd_observed = np.asarray(vfd_observed).ravel()
    treatment_observed = np.asarray(treatment_observed).ravel()

    nirs = np.where(treatment_observed == 1)[0]
    imv = np.where(treatment_observed == 0)[0]
    if len(nirs) == 0 or len(imv) == 0:
        return {'error': 'one arm empty'}

    nirs, imv = match_on_predicted_benefit(ite_pred, treatment_observed,
                                           method='nearest')
    n = len(nirs)
    if n < 2:
        return {'error': 'fewer than two matched pairs'}
    obs = vfd_observed[nirs] - vfd_observed[imv]
    pred = (ite_pred[nirs] + ite_pred[imv]) / 2.0

    var = np.var(pred)
    slope = float(np.cov(pred, obs)[0, 1] / var) if var > 1e-12 else float('nan')
    intercept = float(obs.mean() - slope * pred.mean()) if np.isfinite(slope) else float('nan')

    qs = np.quantile(pred, np.linspace(0, 1, n_groups + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    gid = np.digitize(pred, qs[1:-1])
    rows = []
    for g in range(n_groups):
        m = gid == g
        if m.sum() < 2:
            continue
        rows.append({'group': g + 1, 'n_pairs': int(m.sum()),
                     'predicted': float(pred[m].mean()),
                     'observed': float(obs[m].mean()),
                     'observed_se': float(obs[m].std(ddof=1) / np.sqrt(m.sum()))})
    e_stat = float(np.mean([abs(r['observed'] - r['predicted']) for r in rows])) if rows else float('nan')

    out = {'groups': rows, 'calibration_slope': slope,
           'calibration_intercept': intercept,
           'e_stat_for_benefit': e_stat, 'n_matched_pairs': int(n)}
    return out


def risk_stratified_benefit(ite_pred, vfd_observed, treatment_observed, risk_score,
                            n_strata=3, labels=None, verbose=True):
    """Risk-stratified treatment benefit (PATH Statement risk-modelling check)."""
    ite_pred = np.asarray(ite_pred).ravel()
    y = np.asarray(vfd_observed).ravel()
    w = np.asarray(treatment_observed).ravel()
    r = np.asarray(risk_score).ravel()
    cuts = np.quantile(r, np.linspace(0, 1, n_strata + 1))[1:-1]
    gid = np.digitize(r, cuts)
    if labels is None:
        labels = [f"Q{i+1}" for i in range(n_strata)]
    out = {}
    for g in range(n_strata):
        m = gid == g
        t, c = m & (w == 1), m & (w == 0)
        if t.sum() < 5 or c.sum() < 5:
            continue
        abs_ben = float(y[t].mean() - y[c].mean())
        rel_ben = float(y[t].mean() / y[c].mean()) if y[c].mean() > 1e-9 else float('nan')
        out[labels[g]] = {'n': int(m.sum()), 'observed_absolute_benefit': abs_ben,
                          'observed_relative_benefit': rel_ben,
                          'mean_predicted_ite': float(ite_pred[m].mean()),
                          'pct_predicted_benefit': float(100 * (ite_pred[m] > 0).mean())}
    return out


def standardized_mean_differences(X, W, feature_names=None, threshold=0.10,
                                  verbose=True):
    """Post-matching covariate balance (standardised mean differences)."""
    X = np.asarray(X, dtype=float); W = np.asarray(W).ravel()
    if X.ndim == 3:
        X = np.nanmean(X, axis=1)
    t, c = W == 1, W == 0
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(X.shape[1])]
    out, smds = {}, []
    for j, nm in enumerate(feature_names[:X.shape[1]]):
        mt, mc = np.nanmean(X[t, j]), np.nanmean(X[c, j])
        vt, vc = np.nanvar(X[t, j]), np.nanvar(X[c, j])
        pooled = np.sqrt((vt + vc) / 2.0)
        smd = (mt - mc) / pooled if pooled > 1e-12 else 0.0
        out[nm] = {'smd': float(smd), 'mean_treated': float(mt),
                   'mean_control': float(mc)}
        smds.append(abs(smd))
    smds = np.array(smds)
    out['n_imbalanced'] = int((smds > threshold).sum())
    out['max_abs_smd'] = float(np.nanmax(smds))
    if verbose:
        for nm in feature_names[:X.shape[1]]:
            v = out[nm]
            flag = "  <--" if abs(v['smd']) > threshold else ""
    return out


def cross_fitted_propensity(X_tab, W, k_folds=5, seed=42, clip=0.05,
                            verbose=True):
    """Cross-fitted P(W=1|X) by penalised logistic regression, fitted with plain"""
    X = np.asarray(X_tab, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64).ravel()
    n = len(W)

    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Z = np.hstack([np.ones((n, 1)), (X - mu) / sd])

    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    fold = np.empty(n, dtype=int)
    for i, idx in enumerate(order):
        fold[idx] = i % k_folds

    e = np.full(n, np.nan)
    for k in range(k_folds):
        tr, te = fold != k, fold == k
        beta = _fit_logistic(Z[tr], W[tr], l2=1.0)
        e[te] = _sigmoid(Z[te] @ beta)

    assert np.isfinite(e).all()
    e = np.clip(e, clip, 1.0 - clip)
    if verbose:
        auc = _auc(W, e)
    return e


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _fit_logistic(Z, y, l2=1.0, n_iter=50, tol=1e-8):
    """Ridge-penalised logistic regression by Newton-Raphson (IRLS)."""
    p = Z.shape[1]
    beta = np.zeros(p)
    pen = l2 * np.eye(p)
    pen[0, 0] = 0.0
    for _ in range(n_iter):
        eta = Z @ beta
        mu = _sigmoid(eta)
        w = np.clip(mu * (1 - mu), 1e-6, None)
        grad = Z.T @ (y - mu) - pen @ beta
        H = (Z * w[:, None]).T @ Z + pen
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _auc(y, s):
    """Rank-based AUC (Mann-Whitney U)."""
    y = np.asarray(y).ravel()
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aipw_policy_value(policy, W, Y, e, mu0, mu1):
    """Augmented IPW value of a binary treatment rule."""
    policy = np.asarray(policy).astype(int).ravel()
    W = np.asarray(W).astype(int).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    e = np.asarray(e, dtype=float).ravel()
    mu0 = np.asarray(mu0, dtype=float).ravel()
    mu1 = np.asarray(mu1, dtype=float).ravel()

    mu_pi = np.where(policy == 1, mu1, mu0)
    mu_obs = np.where(W == 1, mu1, mu0)
    prob_obs = np.where(W == 1, e, 1.0 - e)
    agree = (W == policy).astype(float)

    psi = mu_pi + agree * (Y - mu_obs) / prob_obs
    v = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    return {'value': v, 'se': se, 'lo': v - 1.96 * se, 'hi': v + 1.96 * se,
            'psi': psi, 'treat_rate': float(policy.mean())}


def policy_value_table(tau_dict, W, Y, e, mu0, mu1, threshold=0.0,
                       verbose=True):
    """Value of each model's rule alongside the reference policies that make the"""
    out = {}
    for nm, tau in tau_dict.items():
        out[nm] = aipw_policy_value((np.asarray(tau).ravel() > threshold)
                                    .astype(int), W, Y, e, mu0, mu1)
    n = len(W)
    out['treat all NIRS'] = aipw_policy_value(np.ones(n), W, Y, e, mu0, mu1)
    out['treat none (all IMV)'] = aipw_policy_value(np.zeros(n), W, Y, e,
                                                    mu0, mu1)
    out['observed practice'] = aipw_policy_value(np.asarray(W).astype(int),
                                                 W, Y, e, mu0, mu1)
    if verbose:
        for nm, r in sorted(out.items(), key=lambda kv: -kv[1]['value']):
            ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
    return out


def vfd_days_gained(tau, W, Y, e, mu0, mu1, threshold=0.0, n_boot=2000,
                    seed=42, verbose=True):
    """Ventilator-free days gained per 100 patients if NIRS were allocated by the"""
    policy = (np.asarray(tau).ravel() > threshold).astype(int)
    W = np.asarray(W).astype(int).ravel()
    r_model = aipw_policy_value(policy, W, Y, e, mu0, mu1)
    r_obs = aipw_policy_value(W, W, Y, e, mu0, mu1)
    diff = r_model['value'] - r_obs['value']

    rng = np.random.RandomState(seed)
    n = len(W)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boots[b] = (aipw_policy_value(policy[idx], W[idx], np.asarray(Y)[idx],
                                      np.asarray(e)[idx], np.asarray(mu0)[idx],
                                      np.asarray(mu1)[idx])['value']
                    - aipw_policy_value(W[idx], W[idx], np.asarray(Y)[idx],
                                        np.asarray(e)[idx],
                                        np.asarray(mu0)[idx],
                                        np.asarray(mu1)[idx])['value'])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n_switch = int((policy != W).sum())
    return {'diff': float(diff), 'lo': float(lo), 'hi': float(hi),
            'model': r_model['value'], 'observed': r_obs['value'],
            'n_switch': n_switch}


def benefit_harm_interaction(tau, W, Y, threshold=0.0, verbose=True,
                             label='model'):
    """Does the observed treatment effect actually differ between the patients the"""
    tau = np.asarray(tau, dtype=float).ravel()
    W = np.asarray(W, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    G = (tau > threshold).astype(float)

    X = np.column_stack([np.ones_like(W), W, G, W * G])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    resid = Y - X @ beta
    meat = (X * (resid ** 2)[:, None]).T @ X
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))

    b3, se3 = beta[3], se[3]
    z = b3 / se3 if se3 > 0 else 0.0
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))

    eff_harm = beta[1]
    eff_benefit = beta[1] + beta[3]
    n_b, n_h = int(G.sum()), int((1 - G).sum())

    return {'interaction': float(b3), 'se': float(se3), 'p': float(p),
            'effect_benefit': float(eff_benefit), 'effect_harm': float(eff_harm),
            'n_benefit': n_b, 'n_harm': n_h}


def _norm_cdf(z):
    """Standard normal CDF via the error function (no scipy dependency)."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def benefit_calibration_slope(tau, W, Y, n_bins=5, verbose=True, label=''):
    """Calibration slope and intercept for PREDICTED BENEFIT, on van Klaveren pairs."""
    tau = np.asarray(tau, dtype=float).ravel()
    W = np.asarray(W).astype(int).ravel()
    Y = np.asarray(Y, dtype=float).ravel()

    i1 = np.where(W == 1)[0][np.argsort(tau[W == 1])]
    i0 = np.where(W == 0)[0][np.argsort(tau[W == 0])]
    m = min(len(i1), len(i0))
    if m < 20:
        return {'error': 'too few pairs'}
    if len(i1) > m:
        i1 = i1[np.round(np.linspace(0, len(i1) - 1, m)).astype(int)]
    if len(i0) > m:
        i0 = i0[np.round(np.linspace(0, len(i0) - 1, m)).astype(int)]
    obs = Y[i1] - Y[i0]
    pred = 0.5 * (tau[i1] + tau[i0])

    A = np.column_stack([np.ones(m), pred])
    coef = np.linalg.pinv(A.T @ A) @ (A.T @ obs)
    intercept, slope = float(coef[0]), float(coef[1])

    tau_recal = intercept + slope * tau
    pred_r = intercept + slope * pred

    def _binned_err(p, o, k):
        q = np.quantile(p, np.linspace(0, 1, k + 1))
        q[0] -= 1e-9; q[-1] += 1e-9
        g = np.digitize(p, q[1:-1])
        errs, ws = [], []
        for b in range(k):
            sel = g == b
            if sel.sum() < 5:
                continue
            errs.append(abs(o[sel].mean() - p[sel].mean()))
            ws.append(sel.sum())
        return float(np.average(errs, weights=ws)) if errs else float('nan')

    ici_before = _binned_err(pred, obs, n_bins)
    ici_after = _binned_err(pred_r, obs, n_bins)
    e_raw_before = float(np.mean(np.abs(obs - pred)))
    e_raw_after = float(np.mean(np.abs(obs - pred_r)))

    if verbose:
        tag = f"[{label}] " if label else ""
    return {'slope': slope, 'intercept': intercept,
            'ici_before': ici_before, 'ici_after': ici_after,
            'e_before': e_raw_before, 'e_after': e_raw_after,
            'n_pairs': int(m), 'tau_recal': tau_recal}
