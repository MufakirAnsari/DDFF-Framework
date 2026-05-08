"""
DDFF Publication Figures — plotResults.py
==========================================
Generates all figures for the DDFF paper:
  1. Accuracy vs. features curves (per dataset)
  2. Peak accuracy heatmap (all datasets × all methods)
  3. Feature score profiles (Madelon, Ebola, Crohn's)
  4. Variance / stability comparison

Usage:
    python plotResults.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# =====================================================================
# Configuration
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(RESULTS_DIR, 'pipeline_results.csv')

# Professional color palette
COLORS = {
    'MI':            '#6B7280',  # Gray
    'Fisher':        '#9CA3AF',  # Light gray
    'DDFF-L1':       '#3B82F6',  # Blue
    'DDFF-L2':       '#8B5CF6',  # Purple
    'DDFF-KL':       '#EC4899',  # Pink
    'DDFF-Max':      '#F59E0B',  # Amber
    'DDFF-Ensemble': '#10B981',  # Emerald
}

MARKERS = {
    'MI':            's',
    'Fisher':        'D',
    'DDFF-L1':       'o',
    'DDFF-L2':       '^',
    'DDFF-KL':       'v',
    'DDFF-Max':      'P',
    'DDFF-Ensemble': '*',
}

METHOD_ORDER = ['MI', 'Fisher', 'DDFF-L1', 'DDFF-L2', 'DDFF-KL', 'DDFF-Max', 'DDFF-Ensemble']
DATASET_NAMES = {
    'madelon': 'Madelon',
    'Prostate_GE': 'Prostate GE',
    'ALLAML': 'ALL/AML',
    'Crohns': "Crohn's",
    'Ebola': 'Ebola',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# =====================================================================
# Figure 1: Accuracy vs. Number of Features (per dataset)
# =====================================================================

def plot_accuracy_curves(df, classifier='knn'):
    """Plot accuracy vs. k for each dataset, one subplot per dataset."""
    datasets = list(df.dataset.unique())
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(4*n_ds, 3.5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    acc_col = f'{classifier}_accuracy'

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_df = df[df.dataset == ds]

        for method in METHOD_ORDER:
            m_df = ds_df[ds_df.method == method]
            if len(m_df) == 0:
                continue

            agg = m_df.groupby('k')[acc_col].agg(['mean', 'std']).reset_index()
            agg = agg.sort_values('k')

            # Filter to reasonable k values (exclude the full-feature k if too large)
            agg = agg[agg['k'] <= 500]

            ax.plot(agg['k'], agg['mean'],
                    color=COLORS[method], marker=MARKERS[method],
                    markersize=4, linewidth=1.5, label=method, alpha=0.9)
            ax.fill_between(agg['k'], agg['mean'] - agg['std'],
                           agg['mean'] + agg['std'],
                           color=COLORS[method], alpha=0.1)

        ax.set_xlabel('Number of Features (k)')
        if idx == 0:
            ax.set_ylabel(f'{"kNN" if classifier == "knn" else "SVM"} Accuracy (%)')
        ax.set_title(DATASET_NAMES.get(ds, ds), fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

    # Single legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(METHOD_ORDER),
              bbox_to_anchor=(0.5, -0.12), frameon=False)

    fig.suptitle(f'Classification Accuracy vs. Feature Count ({"kNN" if classifier == "knn" else "SVM"})',
                fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, f'accuracy_curves_{classifier}.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# =====================================================================
# Figure 2: Peak Accuracy Heatmap
# =====================================================================

def plot_peak_heatmap(df):
    """Heatmap of peak accuracy for each dataset × method — publication quality."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'wspace': 0.35})

    datasets = list(df.dataset.unique())
    display_names = [DATASET_NAMES.get(d, d) for d in datasets]

    # Short method labels for readability
    method_labels = ['MI', 'Fisher', 'DDFF-L1', 'DDFF-L2', 'DDFF-KL', 'DDFF-Max', 'Ensemble']

    for c_idx, (classifier, title) in enumerate([('knn', 'kNN (k=5)'), ('svm', 'Linear SVM')]):
        ax = axes[c_idx]
        acc_col = f'{classifier}_accuracy'

        matrix = np.zeros((len(datasets), len(METHOD_ORDER)))

        for i, ds in enumerate(datasets):
            for j, method in enumerate(METHOD_ORDER):
                sub = df[(df.dataset == ds) & (df.method == method)]
                if len(sub) == 0:
                    matrix[i, j] = np.nan
                else:
                    matrix[i, j] = sub.groupby('k')[acc_col].mean().max()

        # Sequential blue colormap — clear contrast across the full range
        cmap = LinearSegmentedColormap.from_list('ddff_heat', [
            '#F0F4FF', '#93C5FD', '#3B82F6', '#1D4ED8', '#1E3A5F'
        ], N=256)

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=55, vmax=100)

        # Find best per row for bold annotation
        row_best = np.nanargmax(matrix, axis=1)

        # Annotate cells with numbers
        for i in range(len(datasets)):
            for j in range(len(METHOD_ORDER)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                text_color = 'white' if val > 80 else 'black'
                weight = 'bold' if j == row_best[i] else 'normal'
                fontsize = 10 if j == row_best[i] else 9
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       fontsize=fontsize, fontweight=weight, color=text_color)

        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(method_labels, rotation=35, ha='right', fontsize=9)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(display_names, fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)

        # White gridlines for cell separation
        for i in range(len(datasets) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1.5)
        for j in range(len(METHOD_ORDER) + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1.5)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Accuracy (%)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Peak Classification Accuracy', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'peak_accuracy_heatmap.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path



# =====================================================================
# Figure 3: Feature Score Profiles
# =====================================================================

def plot_feature_scores(data_dir):
    """
    For Madelon, Ebola, Crohn's: compute feature scores on full data
    and plot top-30 features by ensemble score.
    """
    import scipy.io
    sys.path.insert(0, BASE_DIR)
    from ddff_framework import ddff_ensemble_scores, mutual_information_scores, fisher_scores
    from ddff_pipeline import load_crohns, load_ebola, apply_zero_imputation
    from sklearn.preprocessing import StandardScaler

    datasets_for_scores = {}

    # Madelon
    data = scipy.io.loadmat(os.path.join(data_dir, 'madelon.mat'))
    X_m = data['X'].astype(np.float64)
    Y_m = data['Y'].flatten()
    Y_m = np.where(Y_m == -1, 0, 1)
    datasets_for_scores['Madelon'] = (X_m, Y_m)

    # Crohn's & Ebola
    datasets_for_scores["Crohn's"] = load_crohns(data_dir)
    datasets_for_scores['Ebola'] = load_ebola(data_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (ds_name, (X, Y)) in enumerate(datasets_for_scores.items()):
        ax = axes[idx]

        # Scale for scoring
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute ensemble + individual scores
        ens_scores, raw_scores, norm_scores = ddff_ensemble_scores(X_scaled, Y)

        # Top 30 by ensemble
        top30_idx = np.argsort(ens_scores)[::-1][:30]

        # Normalize ensemble to [0,1] for display
        ens_norm = ens_scores[top30_idx]
        if ens_norm.max() > 0:
            ens_norm = ens_norm / ens_norm.max()

        x_pos = np.arange(30)

        # Bar for ensemble
        bars = ax.bar(x_pos, ens_norm, color='#10B981', alpha=0.6,
                     edgecolor='#059669', linewidth=0.5, label='Ensemble', zorder=2)

        # Overlay individual metrics as lines
        for metric, color in [('L1', '#3B82F6'), ('L2', '#8B5CF6'),
                               ('KL', '#EC4899'), ('Max', '#F59E0B')]:
            vals = norm_scores[metric][top30_idx]
            if vals.max() > 0:
                vals = vals / vals.max()
            ax.plot(x_pos, vals, color=color, linewidth=1.2, alpha=0.7,
                   marker='.', markersize=3, label=metric)

        # For Madelon: annotate informative features
        if ds_name == 'Madelon':
            for i, feat_idx in enumerate(top30_idx):
                if feat_idx < 5:  # truly relevant
                    ax.annotate('★', (i, ens_norm[i] + 0.02),
                              ha='center', fontsize=8, color='#DC2626')
                elif feat_idx < 20:  # derived
                    ax.annotate('◆', (i, ens_norm[i] + 0.02),
                              ha='center', fontsize=6, color='#F59E0B')

        ax.set_xlabel('Feature Rank')
        if idx == 0:
            ax.set_ylabel('Normalized Score')
        ax.set_title(ds_name, fontweight='bold')
        ax.set_xticks([0, 9, 19, 29])
        ax.set_xticklabels(['1', '10', '20', '30'])
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
        ax.set_xlim(-0.5, 29.5)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, -0.08), frameon=False)

    fig.suptitle('Feature Score Profiles — Top 30 by Ensemble Score',
                fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'feature_score_profiles.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# =====================================================================
# Figure 4: Stability (Std Dev) Comparison
# =====================================================================

def plot_stability(df):
    """Box plot of accuracy std dev across seeds, per method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for c_idx, (classifier, title) in enumerate([('knn', 'kNN'), ('svm', 'SVM')]):
        ax = axes[c_idx]
        acc_col = f'{classifier}_accuracy'

        # For each dataset×method, compute std across seeds at peak k
        std_data = {m: [] for m in METHOD_ORDER}

        for ds in df.dataset.unique():
            for method in METHOD_ORDER:
                sub = df[(df.dataset == ds) & (df.method == method)]
                if len(sub) == 0:
                    continue
                # Find peak k
                peak_k = sub.groupby('k')[acc_col].mean().idxmax()
                # Std at peak k
                std_val = sub[sub.k == peak_k][acc_col].std()
                std_data[method].append(std_val)

        # Box plot
        positions = range(len(METHOD_ORDER))
        bp_data = [std_data[m] for m in METHOD_ORDER]
        bp = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True,
                       showmeans=True, meanprops=dict(marker='D', markerfacecolor='white',
                                                       markeredgecolor='black', markersize=4))

        for patch, method in zip(bp['boxes'], METHOD_ORDER):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(METHOD_ORDER, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Std Dev of Accuracy (%)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    fig.suptitle('Classification Stability (Lower = More Stable)',
                fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'stability_comparison.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# =====================================================================
# Summary Table (text output)
# =====================================================================

def generate_summary_table(df):
    """Print summary table of peak accuracies."""
    lines = []
    for classifier in ['knn', 'svm']:
        acc_col = f'{classifier}_accuracy'
        lines.append(f'\nPEAK {classifier.upper()} ACCURACY (mean±std over 25 seeds)')
        lines.append('=' * 130)
        hdr = f'{"Dataset":15s}'
        for m in METHOD_ORDER:
            hdr += f' | {m:13s}'
        lines.append(hdr)
        lines.append('-' * 130)

        for ds in df.dataset.unique():
            row = f'{DATASET_NAMES.get(ds, ds):15s}'
            for m in METHOD_ORDER:
                sub = df[(df.dataset == ds) & (df.method == m)]
                if len(sub) == 0:
                    row += f' |     N/A      '
                    continue
                by_k = sub.groupby('k')[acc_col].mean()
                peak = by_k.max()
                best_k = by_k.idxmax()
                std = sub[sub.k == best_k][acc_col].std()
                row += f' | {peak:5.1f}±{std:4.1f}  '
            lines.append(row)

    summary = '\n'.join(lines)

    path = os.path.join(RESULTS_DIR, 'summary_table.txt')
    with open(path, 'w') as f:
        f.write(summary + '\n')
    print(f"  Saved: {path}")
    print(summary)
    return path


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("DDFF Publication Figure Generator")
    print("=" * 60)

    # Load results
    print("\n[1/5] Loading results...")
    df = pd.read_csv(RESULTS_CSV)
    print(f"  {len(df)} rows loaded")

    # Generate summary table
    print("\n[2/5] Summary table...")
    generate_summary_table(df)

    # Figure 1: Accuracy curves
    print("\n[3/5] Accuracy curves...")
    plot_accuracy_curves(df, 'knn')
    plot_accuracy_curves(df, 'svm')

    # Figure 2: Peak heatmap
    print("\n[4/5] Peak accuracy heatmap...")
    plot_peak_heatmap(df)

    # Figure 3: Feature score profiles
    print("\n[5/5] Feature score profiles...")
    DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
    plot_feature_scores(DATA_DIR)

    # Figure 4: Stability
    print("\n[Bonus] Stability comparison...")
    plot_stability(df)

    print("\n" + "=" * 60)
    print("All figures saved to:", FIGURES_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
