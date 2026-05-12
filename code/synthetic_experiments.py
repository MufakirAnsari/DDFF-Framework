"""
synthetic_experiments.py — Synthetic Distribution Motivation Experiments
=========================================================================
Demonstrates WHY DDFF outperforms Fisher Score and Mutual Information
by constructing two cases where class distributions differ in SHAPE
rather than mean, which classical filters fail to capture.

Experiment A: Same Mean, Different Variance
  Class 0: N(0, 1)    Class 1: N(0, 5)
  → Fisher Score ≈ 0 (same mean), MI is weak, DDFF-L1 is high

Experiment B: Unimodal vs. Bimodal
  Class 0: N(0, 1)    Class 1: 0.5·N(-3,1) + 0.5·N(3,1)
  → Means are equal (both ≈0), Fisher ≈ 0, DDFF-L1 is high

Usage:
    python3 code/synthetic_experiments.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from ddff_framework import ddff_scores, fisher_scores

# ── Config ─────────────────────────────────────────────────────────────────────
N_SAMPLES  = 500   # samples per class
N_BINS     = 10    # DDFF default
SEED       = 42
RNG        = np.random.default_rng(SEED)

# Colors
C0_COLOR = '#3B82F6'   # Blue  — Class 0
C1_COLOR = '#EC4899'   # Pink  — Class 1
BG_COLOR = '#F8FAFC'

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.labelsize':   11,
    'axes.titlesize':   12,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.facecolor':   BG_COLOR,
    'figure.facecolor': 'white',
})


# ══════════════════════════════════════════════════════════════════════════════
# Data Generators
# ══════════════════════════════════════════════════════════════════════════════

def generate_expA():
    """
    Experiment A: Same mean (0), different variance.
    Class 0: N(0, sigma^2=1)  ->  scale=1
    Class 1: N(0, sigma^2=5)  ->  scale=sqrt(5) ~ 2.236
    N(mu, sigma^2) convention: second argument is VARIANCE.
    """
    x0 = RNG.normal(loc=0, scale=1.0,           size=N_SAMPLES)
    x1 = RNG.normal(loc=0, scale=np.sqrt(5.0),  size=N_SAMPLES)  # std = sqrt(5)
    return x0, x1

def generate_expB():
    """Experiment B: Unimodal N(0,1) vs Bimodal 0.5·N(-3,1)+0.5·N(3,1)."""
    x0 = RNG.normal(loc=0, scale=1, size=N_SAMPLES)
    # Bimodal: half from each component
    half = N_SAMPLES // 2
    x1 = np.concatenate([
        RNG.normal(loc=-3, scale=1, size=half),
        RNG.normal(loc= 3, scale=1, size=N_SAMPLES - half)
    ])
    return x0, x1


# ══════════════════════════════════════════════════════════════════════════════
# Score Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(x0, x1):
    """
    Compute DDFF-L1, Fisher Score, and Mutual Information for a
    single synthetic 1-D feature (class 0 vs class 1).

    NO StandardScaler applied — scores are computed on the same raw
    data shown in the histograms, so the scores are directly
    interpretable from the visual. StandardScaler is only needed in
    the real pipeline to handle heterogeneous multi-feature datasets.

    Returns dict with keys: 'DDFF-L1', 'Fisher', 'MI'
    """
    X = np.concatenate([x0, x1]).reshape(-1, 1)
    y = np.array([0] * len(x0) + [1] * len(x1))

    ddff_l1 = ddff_scores(X, y, norm_type='L1', n_bins=N_BINS)[0]
    fisher   = fisher_scores(X, y)[0]
    mi       = mutual_info_classif(X, y, random_state=SEED)[0]

    return {'DDFF-L1': ddff_l1, 'Fisher': fisher, 'MI': mi}


# ══════════════════════════════════════════════════════════════════════════════
# Single-Experiment Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_single(ax, x0, x1, title, subtitle, scores, x_range=None):
    """
    Draw overlapping density histograms for x0 and x1 on ax.
    Annotates with DDFF-L1, Fisher, MI scores.
    """
    if x_range is None:
        lo = min(x0.min(), x1.min()) - 0.5
        hi = max(x0.max(), x1.max()) + 0.5
        x_range = (lo, hi)

    bins = np.linspace(x_range[0], x_range[1], 35)

    ax.hist(x0, bins=bins, density=True, color=C0_COLOR, alpha=0.55,
            label='Class 0', edgecolor='white', linewidth=0.4, zorder=2)
    ax.hist(x1, bins=bins, density=True, color=C1_COLOR, alpha=0.55,
            label='Class 1', edgecolor='white', linewidth=0.4, zorder=2)

    # Score annotation box
    score_txt = (
        f"DDFF-L1 = {scores['DDFF-L1']:.3f}\n"
        f"Fisher    = {scores['Fisher']:.4f}\n"
        f"MI           = {scores['MI']:.4f}"
    )
    ax.text(0.97, 0.97, score_txt,
            transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CBD5E1', alpha=0.92),
            family='monospace')

    ax.set_xlim(x_range)
    ax.set_xlabel(f'Feature Value\n{subtitle}', fontsize=9, color='#475569')
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=12, pad=8)
    ax.legend(fontsize=9, framealpha=0.85, loc='upper left')
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')


# ══════════════════════════════════════════════════════════════════════════════
# Individual Figures (for supplementary / separate files)
# ══════════════════════════════════════════════════════════════════════════════

def save_individual(x0, x1, scores, title, subtitle, x_range, filename):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    plot_single(ax, x0, x1, title, subtitle, scores, x_range)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Combined 2-Panel Motivation Figure (THE key paper figure)
# ══════════════════════════════════════════════════════════════════════════════

def save_combined(x0_A, x1_A, scores_A,
                  x0_B, x1_B, scores_B,
                  x_range_A, x_range_B):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    plot_single(axes[0], x0_A, x1_A,
                title='(A) Same Mean, Different Variance',
                subtitle=r'Class 0: $\mathcal{N}(0,\sigma^2{=}1)$   |   Class 1: $\mathcal{N}(0,\sigma^2{=}5)$',
                scores=scores_A,
                x_range=x_range_A)

    plot_single(axes[1], x0_B, x1_B,
                title='(B) Unimodal vs. Bimodal',
                subtitle=r'Class 0: $\mathcal{N}(0,1)$   |   Class 1: $0.5\mathcal{N}(-3,1)+0.5\mathcal{N}(3,1)$',
                scores=scores_B,
                x_range=x_range_B)

    fig.suptitle(
        'Synthetic Motivation: DDFF Detects Distributional Differences\n'
        'Invisible to Fisher Score and Mutual Information',
        fontweight='bold', fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88], w_pad=3.5)
    path = os.path.join(FIGURES_DIR, 'synthetic_motivation.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Score Table (console + LaTeX snippet)
# ══════════════════════════════════════════════════════════════════════════════

def print_score_table(scores_A, scores_B):
    methods = ['DDFF-L1', 'Fisher', 'MI']
    print("\n" + "=" * 62)
    print("SYNTHETIC EXPERIMENT SCORE COMPARISON")
    print("=" * 62)
    print(f"{'Experiment':<35s}  {'DDFF-L1':>8s}  {'Fisher':>8s}  {'MI':>8s}")
    print("-" * 62)

    row_A = (
        "A: Same Mean, Diff Variance       "
        f"  {scores_A['DDFF-L1']:>8.4f}"
        f"  {scores_A['Fisher']:>8.4f}"
        f"  {scores_A['MI']:>8.4f}"
    )
    row_B = (
        "B: Unimodal vs. Bimodal           "
        f"  {scores_B['DDFF-L1']:>8.4f}"
        f"  {scores_B['Fisher']:>8.4f}"
        f"  {scores_B['MI']:>8.4f}"
    )
    print(row_A)
    print(row_B)
    print("=" * 62)

    # LaTeX table snippet
    print("\n── LaTeX table snippet ────────────────────────────────────")
    print(r"\begin{table}[!htb]")
    print(r"\centering")
    print(r"\caption{Feature scores for the two synthetic motivation experiments.")
    print(r"DDFF-L1 detects distributional differences invisible to Fisher Score and MI.}")
    print(r"\label{tab:synthetic}")
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Experiment} & \textbf{DDFF-L1} & \textbf{Fisher} & \textbf{MI} \\")
    print(r"\midrule")
    print(f"Same mean, diff. variance (A) & {scores_A['DDFF-L1']:.4f} & {scores_A['Fisher']:.4f} & {scores_A['MI']:.4f} \\\\")
    print(f"Unimodal vs. bimodal (B)      & {scores_B['DDFF-L1']:.4f} & {scores_B['Fisher']:.4f} & {scores_B['MI']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("───────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("Synthetic Distribution Motivation Experiments")
    print("=" * 60)

    # ── Experiment A ──────────────────────────────────────────────
    print("\n[Exp A] Same mean, different variance...")
    x0_A, x1_A = generate_expA()
    scores_A = compute_scores(x0_A, x1_A)
    print(f"  DDFF-L1 = {scores_A['DDFF-L1']:.4f}")
    print(f"  Fisher  = {scores_A['Fisher']:.4f}")
    print(f"  MI      = {scores_A['MI']:.4f}")

    # ── Experiment B ──────────────────────────────────────────────
    print("\n[Exp B] Unimodal vs. bimodal...")
    x0_B, x1_B = generate_expB()
    scores_B = compute_scores(x0_B, x1_B)
    print(f"  DDFF-L1 = {scores_B['DDFF-L1']:.4f}")
    print(f"  Fisher  = {scores_B['Fisher']:.4f}")
    print(f"  MI      = {scores_B['MI']:.4f}")

    # ── Score table ───────────────────────────────────────────────
    print_score_table(scores_A, scores_B)

    # ── Shared x-axis range for fair visual comparison ────────────
    all_vals_A = np.concatenate([x0_A, x1_A])
    all_vals_B = np.concatenate([x0_B, x1_B])
    x_range_A = (np.percentile(all_vals_A, 0.5) - 0.5,
                 np.percentile(all_vals_A, 99.5) + 0.5)
    x_range_B = (np.percentile(all_vals_B, 0.5) - 0.5,
                 np.percentile(all_vals_B, 99.5) + 0.5)

    # ── Individual figures ────────────────────────────────────────
    print("\nGenerating individual figures...")
    save_individual(x0_A, x1_A, scores_A,
                    title='Same Mean, Different Variance',
                    subtitle=r'Class 0: $\mathcal{N}(0,\sigma^2{=}1)$   |   Class 1: $\mathcal{N}(0,\sigma^2{=}5)$',
                    x_range=x_range_A,
                    filename='synthetic_same_mean.pdf')

    save_individual(x0_B, x1_B, scores_B,
                    title='Unimodal vs. Bimodal',
                    subtitle=r'Class 0: $\mathcal{N}(0,1)$   |   Class 1: $0.5\mathcal{N}(-3,1)+0.5\mathcal{N}(3,1)$',
                    x_range=x_range_B,
                    filename='synthetic_bimodal.pdf')

    # ── Combined motivation figure ────────────────────────────────
    print("Generating combined motivation figure...")
    save_combined(x0_A, x1_A, scores_A,
                  x0_B, x1_B, scores_B,
                  x_range_A, x_range_B)

    print("\nDone. All figures saved to figures/")
