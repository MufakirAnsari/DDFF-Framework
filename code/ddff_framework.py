"""
DDFF Framework v2 — Distributional Discriminative Feature Filtering
=====================================================================
Core implementation of the DDFF feature scoring and ranking system.
Provides individual metric scores (L1, L2, KL, Max) and an average
ensemble score with min-max normalization.

All functions return RAW SCORES (not just rankings) so that feature
score visualizations and ensemble computations are possible.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif


# ---------------------------------------------------------------------------
# DDFF Core: Compute per-feature divergence scores
# ---------------------------------------------------------------------------

def ddff_scores(X_train, y_train, norm_type, n_bins=10, epsilon=1e-6):
    """
    Compute DDFF divergence scores for each feature.

    Parameters
    ----------
    X_train : ndarray (n_samples, n_features)
        Training data (already scaled).
    y_train : ndarray (n_samples,)
        Binary class labels {0, 1}.
    norm_type : str
        One of 'L1', 'L2', 'Max', 'KL'.
    n_bins : int
        Number of equal-width histogram bins (default: 10).
    epsilon : float
        Laplace smoothing for KL divergence (default: 1e-6).

    Returns
    -------
    scores : ndarray (n_features,)
        Divergence score for each feature (higher = more discriminative).
    """
    classes = np.unique(y_train)
    if len(classes) != 2:
        raise ValueError(f"DDFF requires exactly 2 classes, got {len(classes)}: {classes}")

    mask_0 = (y_train == classes[0])
    mask_1 = (y_train == classes[1])

    X_c0 = X_train[mask_0]
    X_c1 = X_train[mask_1]

    n_features = X_train.shape[1]
    scores = np.zeros(n_features, dtype=np.float64)

    for j in range(n_features):
        feat_all = X_train[:, j]
        feat_c0 = X_c0[:, j]
        feat_c1 = X_c1[:, j]

        # Compute bin edges from ALL training samples for this feature
        _, bin_edges = np.histogram(feat_all, bins=n_bins)

        # Class-conditional histograms
        counts0, _ = np.histogram(feat_c0, bins=bin_edges)
        counts1, _ = np.histogram(feat_c1, bins=bin_edges)

        # Normalize to PMFs
        sum0 = counts0.sum()
        sum1 = counts1.sum()
        p0 = counts0 / sum0 if sum0 > 0 else np.zeros(n_bins, dtype=np.float64)
        p1 = counts1 / sum1 if sum1 > 0 else np.zeros(n_bins, dtype=np.float64)

        # Compute divergence
        if norm_type == 'L1':
            scores[j] = np.sum(np.abs(p0 - p1))
        elif norm_type == 'L2':
            scores[j] = np.sqrt(np.sum((p0 - p1) ** 2))
        elif norm_type == 'Max':
            scores[j] = np.max(np.abs(p0 - p1))
        elif norm_type == 'KL':
            p0_safe = p0 + epsilon
            p1_safe = p1 + epsilon
            kl_01 = np.sum(p0_safe * np.log(p0_safe / p1_safe))
            kl_10 = np.sum(p1_safe * np.log(p1_safe / p0_safe))
            scores[j] = 0.5 * (kl_01 + kl_10)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}. Use 'L1', 'L2', 'Max', or 'KL'.")

    return scores


def ddff_rank(X_train, y_train, norm_type, n_bins=10, epsilon=1e-6):
    """
    Compute DDFF scores and return feature indices sorted by descending score.

    Returns
    -------
    ranked_indices : ndarray (n_features,)
        Feature indices sorted from most to least discriminative.
    scores : ndarray (n_features,)
        Raw divergence scores for each feature.
    """
    scores = ddff_scores(X_train, y_train, norm_type, n_bins, epsilon)
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores


# ---------------------------------------------------------------------------
# Ensemble: Average of min-max normalized individual metric scores
# ---------------------------------------------------------------------------

def _min_max_normalize(scores):
    """Normalize scores to [0, 1] range."""
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def ddff_ensemble_scores(X_train, y_train, n_bins=10, epsilon=1e-6):
    """
    Compute the DDFF ensemble score for each feature.

    Procedure:
        1. Compute raw scores for L1, L2, KL, Max
        2. Min-max normalize each metric to [0, 1]
        3. Average the 4 normalized scores

    Returns
    -------
    ensemble_scores : ndarray (n_features,)
        Average normalized score per feature.
    individual_scores : dict
        Raw scores for each metric: {'L1': ..., 'L2': ..., 'KL': ..., 'Max': ...}
    individual_normalized : dict
        Normalized [0,1] scores for each metric.
    """
    metrics = ['L1', 'L2', 'KL', 'Max']

    individual_scores = {}
    individual_normalized = {}

    for metric in metrics:
        raw = ddff_scores(X_train, y_train, metric, n_bins, epsilon)
        individual_scores[metric] = raw
        individual_normalized[metric] = _min_max_normalize(raw)

    # Stack and average
    stacked = np.stack([individual_normalized[m] for m in metrics], axis=0)  # (4, n_features)
    ensemble_scores = stacked.mean(axis=0)  # (n_features,)

    return ensemble_scores, individual_scores, individual_normalized


def ddff_ensemble_rank(X_train, y_train, n_bins=10, epsilon=1e-6):
    """
    Compute ensemble scores and return ranked feature indices.

    Returns
    -------
    ranked_indices : ndarray (n_features,)
    ensemble_scores : ndarray (n_features,)
    individual_scores : dict
    individual_normalized : dict
    """
    ens, raw, norm = ddff_ensemble_scores(X_train, y_train, n_bins, epsilon)
    ranked_indices = np.argsort(ens)[::-1]
    return ranked_indices, ens, raw, norm


# ---------------------------------------------------------------------------
# Baseline wrappers (for unified interface)
# ---------------------------------------------------------------------------

def mutual_information_scores(X_train, y_train):
    """Compute Mutual Information scores using sklearn."""
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    return scores


def fisher_scores(X_train, y_train):
    """
    Compute Fisher Score: F_j = sum_c n_c (mu_c_j - mu_j)^2 / sum_c n_c sigma_c_j^2
    Pure numpy implementation — no skfeature dependency.
    """
    classes = np.unique(y_train)
    n_features = X_train.shape[1]
    scores = np.zeros(n_features, dtype=np.float64)

    overall_mean = X_train.mean(axis=0)

    numerator = np.zeros(n_features)
    denominator = np.zeros(n_features)

    for c in classes:
        mask = (y_train == c)
        n_c = mask.sum()
        X_c = X_train[mask]
        mu_c = X_c.mean(axis=0)
        var_c = X_c.var(axis=0)

        numerator += n_c * (mu_c - overall_mean) ** 2
        denominator += n_c * var_c

    # Avoid division by zero
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)
    scores = numerator / denominator

    return scores


# ---------------------------------------------------------------------------
# Unified scoring interface
# ---------------------------------------------------------------------------

def compute_feature_scores(X_train, y_train, method_name, n_bins=10):
    """
    Compute feature scores for any supported method.

    Parameters
    ----------
    method_name : str
        One of: 'MI', 'Fisher', 'DDFF-L1', 'DDFF-L2', 'DDFF-KL', 'DDFF-Max', 'DDFF-Ensemble'

    Returns
    -------
    ranked_indices : ndarray — feature indices sorted by descending score
    scores : ndarray — raw (or ensemble) scores per feature
    """
    if method_name == 'MI':
        scores = mutual_information_scores(X_train, y_train)
    elif method_name == 'Fisher':
        scores = fisher_scores(X_train, y_train)
    elif method_name.startswith('DDFF-') and method_name != 'DDFF-Ensemble':
        norm_type = method_name.split('-')[1]
        scores = ddff_scores(X_train, y_train, norm_type, n_bins)
    elif method_name == 'DDFF-Ensemble':
        scores, _, _ = ddff_ensemble_scores(X_train, y_train, n_bins)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores
