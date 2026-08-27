"""Tree-based baselines and the cross-fitted tree base for the doubly robust learner."""

import numpy as np


class CustomRegressionTree:
    """Regression tree built from scratch via recursive binary splitting."""

    def __init__(self, max_depth=6, min_samples_leaf=20,
                 max_features=None, honest=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.honest = honest
        self.tree = None

    def fit(self, X, y):
        """Fit regression tree to data."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self.honest:
            n = len(X)
            idx = np.random.permutation(n)
            mid = n // 2
            split_idx = idx[:mid]
            est_idx = idx[mid:]

            self.tree = self._build(X[split_idx], y[split_idx], depth=0)
            self._honest_reestimate(self.tree, X[est_idx], y[est_idx])
        else:
            self.tree = self._build(X, y, depth=0)

        return self

    def _build(self, X, y, depth):
        """Recursively build tree nodes."""
        n_samples = len(y)
        node = {'value': np.mean(y), 'n': n_samples}

        if (depth >= self.max_depth or
                n_samples < 2 * self.min_samples_leaf or
                np.var(y) < 1e-10):
            node['leaf'] = True
            return node

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        if self.max_features is not None:
            feature_indices = np.random.choice(
                n_features, size=min(self.max_features, n_features),
                replace=False)
        else:
            feature_indices = np.arange(n_features)

        total_var = np.var(y) * n_samples

        for feat in feature_indices:
            col = X[:, feat]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 20:
                thresholds = unique_vals
            else:
                thresholds = np.percentile(col, np.linspace(5, 95, 20))

            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if (n_left < self.min_samples_leaf or
                        n_right < self.min_samples_leaf):
                    continue

                var_left = np.var(y[left_mask]) * n_left
                var_right = np.var(y[right_mask]) * n_right
                gain = total_var - (var_left + var_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh

        if best_feature is None:
            node['leaf'] = True
            return node

        left_mask = X[:, best_feature] <= best_threshold
        node['leaf'] = False
        node['feature'] = best_feature
        node['threshold'] = best_threshold
        node['left'] = self._build(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self._build(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def _honest_reestimate(self, node, X, y):
        """Re-estimate leaf values using estimation sample."""
        if node.get('leaf', False) or 'feature' not in node:
            if len(y) > 0:
                node['value'] = np.mean(y)
            node['n'] = len(y)
            return

        left_mask = X[:, node['feature']] <= node['threshold']
        self._honest_reestimate(node['left'], X[left_mask], y[left_mask])
        self._honest_reestimate(node['right'], X[~left_mask], y[~left_mask])

    def predict(self, X):
        """Predict by traversing tree for each sample."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if node.get('leaf', False) or 'feature' not in node:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def get_leaf_id(self, x, node=None):
        """Return leaf ID for a single sample (for Causal Forest weighting)."""
        if node is None:
            node = self.tree
        if node.get('leaf', False) or 'feature' not in node:
            return id(node)
        if x[node['feature']] <= node['threshold']:
            return self.get_leaf_id(x, node['left'])
        else:
            return self.get_leaf_id(x, node['right'])


class CustomTLearner:
    """T-Learner for heterogeneous treatment effect estimation."""

    def __init__(self, n_trees=100, max_depth=6, min_samples_leaf=20,
                 max_features_frac=0.5, subsample_frac=0.8, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_frac = max_features_frac
        self.subsample_frac = subsample_frac
        self.random_state = random_state
        self.forest_0 = []
        self.forest_1 = []

    def fit(self, X, W, Y):
        """Fit T-Learner: separate forests for each treatment arm."""
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        X_0, Y_0 = X[W == 0], Y[W == 0]
        X_1, Y_1 = X[W == 1], Y[W == 1]

        n_features = X.shape[1]
        max_features = max(1, int(n_features * self.max_features_frac))

        self.forest_0 = self._build_forest(
            X_0, Y_0, max_features)

        self.forest_1 = self._build_forest(
            X_1, Y_1, max_features)

        return self

    def _build_forest(self, X, Y, max_features):
        """Build a random forest (list of CustomRegressionTree)."""
        forest = []
        n = len(X)
        n_subsample = max(1, int(n * self.subsample_frac))

        for t in range(self.n_trees):
            boot_idx = np.random.choice(n, size=n_subsample, replace=True)
            tree = CustomRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
            )
            tree.fit(X[boot_idx], Y[boot_idx])
            forest.append(tree)
        return forest

    def predict_ite(self, X):
        """Predict ITE = μ_1(x) - μ_0(x) for each sample."""
        X = np.asarray(X, dtype=np.float64)

        mu_0 = np.mean([tree.predict(X) for tree in self.forest_0], axis=0)
        mu_1 = np.mean([tree.predict(X) for tree in self.forest_1], axis=0)
        ite = mu_1 - mu_0

        return ite, mu_0, mu_1


class CustomCausalForest:
    """Causal Forest for heterogeneous treatment effect estimation."""

    def __init__(self, n_trees=100, max_depth=5, min_samples_leaf=30,
                 max_features_frac=0.5, subsample_frac=0.8, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_frac = max_features_frac
        self.subsample_frac = subsample_frac
        self.random_state = random_state
        self.trees = []

    def fit(self, X, W, Y):
        """Fit causal forest."""
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        n = len(X)
        n_sub = max(1, int(n * self.subsample_frac))
        n_features = X.shape[1]
        max_features = max(1, int(n_features * self.max_features_frac))

        self.trees = []
        for t in range(self.n_trees):
            sub_idx = np.random.choice(n, size=n_sub, replace=True)
            X_sub = X[sub_idx]
            W_sub = W[sub_idx]
            Y_sub = Y[sub_idx]

            perm = np.random.permutation(n_sub)
            mid = n_sub // 2
            struct_idx = perm[:mid]
            est_idx = perm[mid:]

            tree = self._build_causal_tree(
                X_sub[struct_idx], W_sub[struct_idx], Y_sub[struct_idx],
                max_features, depth=0)

            self._honest_causal_reestimate(
                tree, X_sub[est_idx], W_sub[est_idx], Y_sub[est_idx])

            self.trees.append(tree)

        return self

    def _build_causal_tree(self, X, W, Y, max_features, depth):
        """Recursively build causal tree with treatment-effect-maximizing splits."""
        n = len(Y)
        n_treated = (W == 1).sum()
        n_control = (W == 0).sum()

        tau = 0.0
        if n_treated > 0 and n_control > 0:
            tau = Y[W == 1].mean() - Y[W == 0].mean()

        node = {
            'tau': tau,
            'n': n,
            'n_treated': int(n_treated),
            'n_control': int(n_control),
        }

        if (depth >= self.max_depth or
                n < 2 * self.min_samples_leaf or
                n_treated < 2 or n_control < 2):
            node['leaf'] = True
            return node

        best_score = -np.inf
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        if max_features is not None:
            feat_idx = np.random.choice(
                n_features, size=min(max_features, n_features), replace=False)
        else:
            feat_idx = np.arange(n_features)

        for feat in feat_idx:
            col = X[:, feat]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 20:
                thresholds = unique_vals
            else:
                thresholds = np.percentile(col, np.linspace(5, 95, 20))

            for thresh in thresholds:
                left = col <= thresh
                right = ~left

                n_left = left.sum()
                n_right = right.sum()

                if (n_left < self.min_samples_leaf or
                        n_right < self.min_samples_leaf):
                    continue

                if ((W[left] == 1).sum() < 1 or (W[left] == 0).sum() < 1 or
                        (W[right] == 1).sum() < 1 or (W[right] == 0).sum() < 1):
                    continue

                tau_left = (Y[left & (W == 1)].mean() -
                            Y[left & (W == 0)].mean())
                tau_right = (Y[right & (W == 1)].mean() -
                             Y[right & (W == 0)].mean())

                score = (n_left * tau_left ** 2 +
                         n_right * tau_right ** 2)

                if score > best_score:
                    best_score = score
                    best_feature = feat
                    best_threshold = thresh

        if best_feature is None:
            node['leaf'] = True
            return node

        left_mask = X[:, best_feature] <= best_threshold
        node['leaf'] = False
        node['feature'] = best_feature
        node['threshold'] = best_threshold
        node['left'] = self._build_causal_tree(
            X[left_mask], W[left_mask], Y[left_mask],
            max_features, depth + 1)
        node['right'] = self._build_causal_tree(
            X[~left_mask], W[~left_mask], Y[~left_mask],
            max_features, depth + 1)

        return node

    def _honest_causal_reestimate(self, node, X, W, Y, parent_tau=0.0):
        """Re-estimate leaf treatment effects using estimation sample."""
        if node.get('leaf', False) or 'feature' not in node:
            n_treated = (W == 1).sum()
            n_control = (W == 0).sum()
            if n_treated > 0 and n_control > 0:
                node['tau'] = Y[W == 1].mean() - Y[W == 0].mean()
            elif n_treated > 0 or n_control > 0:
                node['tau'] = float(parent_tau)
            else:
                node['tau'] = 0.0
            node['n'] = len(Y)
            node['n_treated'] = int(n_treated)
            node['n_control'] = int(n_control)
            return

        _nt, _nc = (W == 1).sum(), (W == 0).sum()
        _here = (float(Y[W == 1].mean() - Y[W == 0].mean())
                 if (_nt > 0 and _nc > 0) else float(parent_tau))

        left_mask = X[:, node['feature']] <= node['threshold']
        self._honest_causal_reestimate(
            node['left'], X[left_mask], W[left_mask], Y[left_mask], _here)
        self._honest_causal_reestimate(
            node['right'], X[~left_mask], W[~left_mask], Y[~left_mask], _here)

    def predict_ite(self, X):
        """Predict ITE by averaging leaf τ across all causal trees."""
        X = np.asarray(X, dtype=np.float64)
        tree_preds = np.array([
            self._predict_tree(X, tree) for tree in self.trees
        ])
        ite = np.mean(tree_preds, axis=0)
        return ite

    def _predict_tree(self, X, tree):
        """Predict τ for each sample by traversing one causal tree."""
        return np.array([self._predict_one(x, tree) for x in X])

    def _predict_one(self, x, node):
        if node.get('leaf', False) or 'feature' not in node:
            return node['tau']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])


def run_baselines(X_baseline, W, Y, random_state=42, delta=None):
    """Run both baseline models and return ITE predictions."""
    results = {}

    print('Running baselines...')
    tl = CustomTLearner(
        n_trees=100, max_depth=6, min_samples_leaf=20,
        max_features_frac=0.5, subsample_frac=0.8,
        random_state=random_state,
    )
    tl.fit(X_baseline, W, Y)
    ite_tl, mu0_tl, mu1_tl = tl.predict_ite(X_baseline)
    results['T-Learner'] = {
        'ite': ite_tl, 'mu_0': mu0_tl, 'mu_1': mu1_tl,
        'model': tl,
    }

    cf = CustomCausalForest(
        n_trees=100, max_depth=5, min_samples_leaf=30,
        max_features_frac=0.5, subsample_frac=0.8,
        random_state=random_state,
    )
    cf.fit(X_baseline, W, Y)
    ite_cf = cf.predict_ite(X_baseline)
    results['Causal Forest'] = {
        'ite': ite_cf,
        'model': cf,
    }

    if delta is not None:
        csf = CustomCausalSurvivalForest(
            n_trees=100, max_depth=5, min_samples_leaf=30,
            max_features_frac=0.5, subsample_frac=0.8, horizon=28.0,
            random_state=random_state)
        csf.fit(X_baseline, W, Y, delta=delta)
        ite_csf = csf.predict_ite(X_baseline)
        results['Causal Survival Forest'] = {'ite': ite_csf, 'model': csf}

    return results


class CustomCausalSurvivalForest:
    """Causal Survival Forest for right-censored / competing-risk outcomes."""

    def __init__(self, n_trees=100, max_depth=5, min_samples_leaf=30,
                 max_features_frac=0.5, subsample_frac=0.8, horizon=28.0,
                 random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_frac = max_features_frac
        self.subsample_frac = subsample_frac
        self.horizon = horizon
        self.random_state = random_state
        self._forest = None
        self.censor_weights_ = None

    @staticmethod
    def _km_censoring_weights(time, event, clip=(1.0, 10.0)):
        """IPCW weights 1/S_C(T_i) from a KM estimate of the CENSORING distribution."""
        time = np.asarray(time, dtype=float)
        cens = 1 - np.asarray(event, dtype=float)
        order = np.argsort(time)
        t_sorted, c_sorted = time[order], cens[order]
        n = len(time)
        at_risk = n - np.arange(n)
        S, surv = 1.0, np.ones(n)
        for i in range(n):
            if c_sorted[i] == 1 and at_risk[i] > 0:
                S *= (1.0 - 1.0 / at_risk[i])
            surv[i] = S
        S_at = np.empty(n); S_at[order] = surv
        S_at = np.clip(S_at, 1.0 / clip[1], 1.0)
        w = 1.0 / S_at
        return np.clip(w, clip[0], clip[1])

    def fit(self, X, W, Y, delta=None):
        """X: (N, P) covariates"""
        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W).ravel()
        Y = np.asarray(Y, dtype=np.float64).ravel()

        if delta is None:
            w_c = np.ones(len(Y))
        else:
            w_c = self._km_censoring_weights(np.clip(Y, 0, self.horizon),
                                             np.asarray(delta).ravel())
        self.censor_weights_ = w_c

        _span = float(np.max(w_c) - np.min(w_c))
        if _span > 1e-8:
            _bad = float(np.max(Y * w_c)) > self.horizon + 1e-6
            if _bad:
                print("  [CSF] IPCW DISABLED")
                w_c = np.ones(len(Y))
                self.censor_weights_ = w_c
        Y_ipcw = Y * w_c
        assert np.max(Y_ipcw) <= self.horizon + 1e-6, (
            "reweighted outcome exceeds the horizon; the weights are not a valid "
            "censoring correction")

        self._forest = CustomCausalForest(
            n_trees=self.n_trees, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features_frac=self.max_features_frac,
            subsample_frac=self.subsample_frac,
            random_state=self.random_state)
        self._forest.fit(X, W, Y_ipcw)
        return self

    def predict_ite(self, X):
        """Predict the restricted-mean treatment effect at the horizon."""
        return self._forest.predict_ite(X)


def cross_fitted_tree_base(X_tab, W, Y, K=5, seed=42, learner='t-learner',
                           verbose=True, **learner_kwargs):
    """Cross-fitted tree CATE estimate, used as the plug-in base of the DR-Learner."""
    X_tab = np.asarray(X_tab, dtype=float)
    W = np.asarray(W).ravel()
    Y = np.asarray(Y).ravel()
    N = len(Y)

    rng = np.random.RandomState(seed)
    order = rng.permutation(N)
    fold_id = np.empty(N, dtype=int)
    for i, idx in enumerate(order):
        fold_id[idx] = i % K

    tau = np.full(N, np.nan)
    mu1 = np.full(N, np.nan)
    mu0 = np.full(N, np.nan)

    for k in range(K):
        te = np.where(fold_id == k)[0]
        tr = np.where(fold_id != k)[0]
        if learner == 'causal-forest':
            m = CustomCausalForest(random_state=seed, **learner_kwargs)
            m.fit(X_tab[tr], W[tr], Y[tr])
            tau[te] = np.asarray(m.predict_ite(X_tab[te])).ravel()
        else:
            m = CustomTLearner(random_state=seed, **learner_kwargs)
            m.fit(X_tab[tr], W[tr], Y[tr])
            out = m.predict_ite(X_tab[te])
            if isinstance(out, tuple):
                t_, m0_, m1_ = out
                tau[te] = np.asarray(t_).ravel()
                mu0[te] = np.asarray(m0_).ravel()
                mu1[te] = np.asarray(m1_).ravel()
            else:
                tau[te] = np.asarray(out).ravel()

    return {'tau_base': tau, 'mu1': mu1, 'mu0': mu0, 'fold_id': fold_id}


def standardize_base(tau_base):
    """Centre/scale the base so it enters the network on a well-conditioned scale."""
    tau_base = np.asarray(tau_base, dtype=float)
    c = float(np.nanmean(tau_base))
    s = float(np.nanstd(tau_base))
    s = s if s > 1e-8 else 1.0
    return ((tau_base - c) / s).astype(np.float32), c, s
