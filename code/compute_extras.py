"""
compute_extras.py — Rank Correlation + Convergence Speed Analysis
=================================================================
Reads pipeline_results.csv and outputs:
  1. Spearman rank correlation matrix of accuracy profiles between methods (per dataset)
  2. Convergence speed: minimum k to reach 95% of peak accuracy (per method, per dataset)
  3. LaTeX-ready table snippets for direct insertion into main.tex

Usage:
    cd "/home/ansari/Desktop/Dr Ghosh/Feature Selection/V2"
    python3 code/compute_extras.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, '..', 'results', 'pipeline_results.csv')
OUT_DIR = os.path.join(BASE_DIR, '..', 'results')

METHOD_ORDER = ['MI', 'Fisher', 'DDFF-L1', 'DDFF-L2', 'DDFF-KL', 'DDFF-Max', 'DDFF-Ensemble']
DATASET_NAMES = {
    'madelon':    'Madelon',
    'Prostate_GE':'Prostate GE',
    'ALLAML':     'ALL/AML',
    'Crohns':     "Crohn's",
    'Ebola':      'Ebola',
}
K_VALUES = [25, 50, 75, 100, 150, 200, 300, 500]

# ─────────────────────────────────────────────
# Load and aggregate
# ─────────────────────────────────────────────

df = pd.read_csv(RESULTS_CSV)
datasets = list(df.dataset.unique())

# Mean accuracy across seeds, per (dataset, method, k)
agg = (df.groupby(['dataset', 'method', 'k'])[['knn_accuracy', 'svm_accuracy']]
         .mean().reset_index())

# ─────────────────────────────────────────────
# 1. CONVERGENCE SPEED
# ─────────────────────────────────────────────
# For each dataset × method: find the smallest k where
# mean_knn_accuracy >= 0.95 * peak_knn_accuracy

print("\n" + "="*80)
print("CONVERGENCE SPEED (k to reach 95% of peak kNN accuracy)")
print("="*80)

conv_rows = []
for ds in datasets:
    ds_agg = agg[agg.dataset == ds]
    for method in METHOD_ORDER:
        m_agg = ds_agg[ds_agg.method == method].sort_values('k')
        if len(m_agg) == 0:
            continue
        peak = m_agg['knn_accuracy'].max()
        threshold = 0.95 * peak
        reached = m_agg[m_agg['knn_accuracy'] >= threshold]
        conv_k = reached['k'].min() if len(reached) > 0 else m_agg['k'].max()
        conv_rows.append({'dataset': ds, 'method': method, 'conv_k': conv_k, 'peak': peak})

conv_df = pd.DataFrame(conv_rows)

# Print table
header = f"{'Dataset':15s} | " + " | ".join(f"{m:13s}" for m in METHOD_ORDER)
print(header)
print("-"*len(header))
for ds in datasets:
    row = f"{DATASET_NAMES.get(ds,ds):15s} | "
    vals = []
    for m in METHOD_ORDER:
        sub = conv_df[(conv_df.dataset==ds) & (conv_df.method==m)]
        vals.append(f"{int(sub['conv_k'].values[0]):>5d}" if len(sub)>0 else "  N/A")
    print(row + " | ".join(vals))

# ─────────────────────────────────────────────
# 2. SPEARMAN RANK CORRELATION
# ─────────────────────────────────────────────
# For each dataset: correlate kNN accuracy-vs-k profiles between methods
# A low ρ between DDFF and MI means they select differently informative features

print("\n" + "="*80)
print("SPEARMAN ρ: accuracy-vs-k profile correlation between methods (kNN)")
print("(Higher ρ = methods agree on feature quality across feature counts)")
print("="*80)

rho_results = {}
for ds in datasets:
    ds_agg = agg[agg.dataset == ds]
    # Build matrix: rows = k values, cols = methods
    profile_matrix = []
    for method in METHOD_ORDER:
        m_agg = ds_agg[ds_agg.method == method].sort_values('k')
        profile_matrix.append(m_agg['knn_accuracy'].values)
    profile_matrix = np.array(profile_matrix)  # (7, 8)

    # Compute pairwise Spearman ρ
    n_methods = len(METHOD_ORDER)
    rho_matrix = np.ones((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            r, _ = spearmanr(profile_matrix[i], profile_matrix[j])
            rho_matrix[i, j] = r
            rho_matrix[j, i] = r
    rho_results[ds] = rho_matrix

    print(f"\n--- {DATASET_NAMES.get(ds, ds)} ---")
    header = f"{'':12s} | " + " | ".join(f"{m[:9]:9s}" for m in METHOD_ORDER)
    print(header)
    for i, m1 in enumerate(METHOD_ORDER):
        row = f"{m1[:12]:12s} | "
        row += " | ".join(f"{rho_matrix[i,j]:9.3f}" for j in range(len(METHOD_ORDER)))
        print(row)

# ─────────────────────────────────────────────
# 3. LaTeX OUTPUT
# ─────────────────────────────────────────────

print("\n" + "="*80)
print("LaTeX: Convergence Speed Table")
print("="*80)

latex_conv = r"""
\begin{table}[!htb]
\centering
\caption{Convergence speed: minimum number of features $k$ required to reach
95\% of peak kNN accuracy. Lower values indicate earlier identification of
discriminative features. \textbf{Bold} indicates fastest convergence per dataset.}
\label{tab:convergence}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcc|ccccc@{}}
\toprule
 & \textbf{MI} & \textbf{Fisher} & \textbf{DDFF-$\ell_1$} & \textbf{DDFF-$\ell_2$} & \textbf{DDFF-KL} & \textbf{DDFF-Max} & \textbf{Ens.} \\
\midrule"""

print(latex_conv)
for ds in datasets:
    row_vals = []
    min_val = None
    min_idx = None
    for idx, m in enumerate(METHOD_ORDER):
        sub = conv_df[(conv_df.dataset==ds) & (conv_df.method==m)]
        v = int(sub['conv_k'].values[0]) if len(sub)>0 else 9999
        row_vals.append(v)
        if min_val is None or v < min_val:
            min_val = v
            min_idx = idx
    row_str = f"{DATASET_NAMES.get(ds,ds):12s}"
    for idx, v in enumerate(row_vals):
        if idx == min_idx:
            row_str += f" & \\textbf{{{v}}}"
        else:
            row_str += f" & {v}"
    row_str += " \\\\"
    print(row_str)

print(r"""\bottomrule
\end{tabular}%
}
\end{table}""")

# ─────────────────────────────────────────────
# 4. MI vs Ensemble Spearman summary for paper text
# ─────────────────────────────────────────────

print("\n" + "="*80)
print("MI vs DDFF-Ensemble Spearman ρ per dataset (for paper text):")
print("="*80)
mi_idx = METHOD_ORDER.index('MI')
ens_idx = METHOD_ORDER.index('DDFF-Ensemble')
for ds in datasets:
    rho = rho_results[ds][mi_idx, ens_idx]
    print(f"  {DATASET_NAMES.get(ds,ds):15s}: ρ(MI, Ensemble) = {rho:.3f}")

fisher_idx = METHOD_ORDER.index('Fisher')
print()
for ds in datasets:
    rho = rho_results[ds][fisher_idx, ens_idx]
    print(f"  {DATASET_NAMES.get(ds,ds):15s}: ρ(Fisher, Ensemble) = {rho:.3f}")

print("\nDone. Copy LaTeX table snippets above into main.tex.")
