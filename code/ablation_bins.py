"""
ablation_bins.py — Bin-Size Sensitivity Analysis for DDFF-L1
=============================================================
Runs the full classification pipeline for DDFF-L1 across bin sizes
B ∈ {5, 10, 15, 20, 25} on all 5 datasets, recording peak kNN accuracy
and std per bin. Generates publication-quality plot with error bars.

Usage:
    python3 code/ablation_bins.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', 'Data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from ddff_framework import ddff_scores
from ddff_pipeline   import (load_crohns, load_ebola, apply_zero_imputation)

# ── Config ────────────────────────────────────────────────────────────────────
BIN_SIZES   = [5, 10, 15, 20, 25]
N_SEEDS     = 25
K_FEATURES  = [25, 50, 75, 100, 150, 200, 300, 500]
KNN_K       = 5

DATASET_DISPLAY = {
    'madelon':    'Madelon',
    'Prostate_GE':'Prostate GE',
    'ALLAML':     'ALL/AML',
    'Crohns':     "Crohn's",
    'Ebola':      'Ebola',
}

# ── Load datasets ─────────────────────────────────────────────────────────────
def load_all_datasets():
    import scipy.io
    datasets = {}

    # Madelon
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, 'madelon.mat'))
    X = mat['X'].astype(np.float64)
    Y = mat['Y'].flatten()
    Y = np.where(Y == -1, 0, 1)
    datasets['madelon'] = (X, Y)

    # Prostate GE
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, 'Prostate_GE.mat'))
    X = mat['X'].astype(np.float64)
    Y = mat['Y'].flatten().astype(int)
    Y = np.where(Y == np.unique(Y)[0], 0, 1)
    datasets['Prostate_GE'] = (X, Y)

    # ALL/AML
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, 'ALLAML.mat'))
    X = mat['X'].astype(np.float64)
    Y = mat['Y'].flatten().astype(int)
    Y = np.where(Y == np.unique(Y)[0], 0, 1)
    datasets['ALLAML'] = (X, Y)

    # Crohn's
    X, Y = load_crohns(DATA_DIR)
    X = apply_zero_imputation(X)
    datasets['Crohns'] = (X, Y)

    # Ebola
    X, Y = load_ebola(DATA_DIR)
    X = apply_zero_imputation(X)
    datasets['Ebola'] = (X, Y)

    return datasets

# ── Single run ────────────────────────────────────────────────────────────────
def run_one(X, y, seed, n_bins):
    """Return peak kNN accuracy for one seed/bin combination."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    scores = ddff_scores(X_tr, y_tr, norm_type='L1', n_bins=n_bins)
    ranked = np.argsort(scores)[::-1]

    best_acc = 0.0
    for k in K_FEATURES:
        if k > X_tr.shape[1]:
            break
        idx = ranked[:k]
        clf = KNeighborsClassifier(n_neighbors=KNN_K)
        clf.fit(X_tr[:, idx], y_tr)
        acc = clf.score(X_te[:, idx], y_te) * 100
        if acc > best_acc:
            best_acc = acc

    return best_acc

# ── Main ablation loop ─────────────────────────────────────────────────────────
def run_ablation(datasets):
    """
    For each dataset × bin size: run 25 seeds, compute mean accuracy
    at each k across seeds, then take the peak k — exactly matching
    the main pipeline aggregation (mean(seeds) → max(k)).
    """
    records = []
    total = len(datasets) * len(BIN_SIZES)
    done  = 0

    for ds_name, (X, y) in datasets.items():
        print(f"\n  Dataset: {DATASET_DISPLAY[ds_name]}  ({X.shape[0]} samples, {X.shape[1]} features)")
        for n_bins in BIN_SIZES:
            # Accumulate per-seed, per-k accuracy
            k_accs = {k: [] for k in K_FEATURES if k <= X.shape[1]}

            for seed in range(N_SEEDS):
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                train_idx, test_idx = next(sss.split(X, y))
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

                scores = ddff_scores(X_tr, y_tr, norm_type='L1', n_bins=n_bins)
                ranked = np.argsort(scores)[::-1]

                for k in k_accs:
                    idx = ranked[:k]
                    clf = KNeighborsClassifier(n_neighbors=KNN_K)
                    clf.fit(X_tr[:, idx], y_tr)
                    acc = clf.score(X_te[:, idx], y_te) * 100
                    k_accs[k].append(acc)

            # Mean across seeds per k, then peak k
            k_means = {k: np.mean(v) for k, v in k_accs.items()}
            peak_mean = max(k_means.values())
            peak_k    = max(k_means, key=k_means.get)

            # Std: take std across seeds at the peak k
            peak_std = np.std(k_accs[peak_k])

            done += 1
            print(f"    [{done}/{total}]  B={n_bins:2d}  peak={peak_mean:.2f}% ±{peak_std:.2f}  at k={peak_k}")

            records.append({
                'dataset':  ds_name,
                'n_bins':   n_bins,
                'mean_acc': peak_mean,
                'std_acc':  peak_std,
            })

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, 'ablation_bins.csv')
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    return df

# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_ablation(df):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    datasets = list(df['dataset'].unique())
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(3.2 * n_ds, 4.2), sharey=False)
    if n_ds == 1:
        axes = [axes]

    color_line  = '#1D4ED8'  # strong blue
    color_fill  = '#93C5FD'  # light blue

    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset'] == ds].sort_values('n_bins')
        means = sub['mean_acc'].values
        stds  = sub['std_acc'].values
        bins  = sub['n_bins'].values

        ax.plot(bins, means, color=color_line, linewidth=2,
                marker='o', markersize=6, markerfacecolor='white',
                markeredgecolor=color_line, markeredgewidth=2, zorder=3)
        ax.fill_between(bins, means - stds, means + stds,
                        color=color_fill, alpha=0.5, zorder=2)

        # Annotate each point with the mean value
        for x, y_val in zip(bins, means):
            ax.annotate(f'{y_val:.1f}', (x, y_val),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8, color=color_line, fontweight='bold')

        ax.set_xticks(BIN_SIZES)
        ax.set_xlabel('Number of Bins (B)', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel('Peak kNN Accuracy (%)', fontsize=10)
        ax.set_title(DATASET_DISPLAY[ds], fontweight='bold', fontsize=11, pad=4)
        ax.grid(True, alpha=0.25, linestyle='--', axis='y')

        # Mark B=10 default with a dotted line
        ax.axvline(10, color='#6B7280', linewidth=1.0, linestyle=':', alpha=0.7, label='B=10 (default)')
        ax.legend(fontsize=7, loc='lower right', framealpha=0.8)

    fig.suptitle('DDFF-L1 Bin-Size Sensitivity (Ablation Study)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle

    out = os.path.join(FIGURES_DIR, 'ablation_bins.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Print table ────────────────────────────────────────────────────────────────
def print_table(df):
    print("\n" + "=" * 70)
    print("ABLATION: DDFF-L1 Peak kNN Accuracy vs. Bin Size (mean±std)")
    print("=" * 70)
    header = f"{'Dataset':15s}"
    for b in BIN_SIZES:
        header += f"  B={b:2d}       "
    print(header)
    print("-" * 70)
    for ds in df['dataset'].unique():
        row = f"{DATASET_DISPLAY[ds]:15s}"
        for b in BIN_SIZES:
            r = df[(df['dataset'] == ds) & (df['n_bins'] == b)].iloc[0]
            row += f"  {r.mean_acc:5.1f}±{r.std_acc:4.1f}"
        print(row)
    print("=" * 70)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("DDFF-L1 Bin-Size Ablation Study")
    print("=" * 60)

    # Check if results already exist
    csv_path = os.path.join(RESULTS_DIR, 'ablation_bins.csv')
    if os.path.exists(csv_path):
        print(f"\nLoading existing results from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("\nLoading datasets...")
        datasets = load_all_datasets()
        print(f"Loaded {len(datasets)} datasets.")
        print("\nRunning ablation (this may take a few minutes)...")
        df = run_ablation(datasets)

    print_table(df)
    print("\nGenerating plot...")
    plot_ablation(df)
    print("\nDone.")
