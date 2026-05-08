"""
DDFF Master Pipeline v2
========================
Complete experiment pipeline for the DDFF paper.
Runs all 5 datasets × 7 methods × 2 classifiers × 8 k-values × 25 seeds.
Saves results incrementally to CSV.

Usage:
    python ddff_pipeline.py
"""

import os
import sys
import time
import gzip
import warnings
import numpy as np
import pandas as pd
import scipy.io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from ddff_framework import compute_feature_scores

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =====================================================================
# Configuration
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

METHODS = ['MI', 'Fisher', 'DDFF-L1', 'DDFF-L2', 'DDFF-KL', 'DDFF-Max', 'DDFF-Ensemble']
FEATURE_SUBSETS = [25, 50, 75, 100, 150, 200, 300, 500]
N_REPS = 25
SPLIT_RATIO = 0.8

RESULTS_CSV = os.path.join(RESULTS_DIR, 'pipeline_results.csv')


# =====================================================================
# Data Loaders
# =====================================================================

def load_mat_dataset(name, data_dir):
    """Load a .mat dataset. Returns X, Y with Y as {0, 1}."""
    mat_path = os.path.join(data_dir, f'{name}.mat')
    data = scipy.io.loadmat(mat_path)
    X = data['X'].astype(np.float64)
    Y = data['Y'].flatten()

    # Standardize labels to {0, 1}
    unique_labels = np.sort(np.unique(Y))
    if len(unique_labels) != 2:
        raise ValueError(f"Expected 2 classes in {name}, got {len(unique_labels)}")

    # Map: smallest label → 0, largest → 1
    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    Y = np.array([label_map[y] for y in Y])

    return X, Y


def apply_zero_imputation(X, threshold=0.20):
    """
    Remove features with >threshold fraction of zeros,
    then mean-impute remaining zeros.
    """
    n_samples = X.shape[0]
    zero_pcts = np.sum(X == 0, axis=0) / n_samples

    # Filter features
    valid_mask = zero_pcts <= threshold
    X_filtered = X[:, valid_mask]
    n_removed = np.sum(~valid_mask)
    print(f"    [Cleaner] Removed {n_removed} features with >{threshold*100:.0f}% zeros "
          f"({X.shape[1]} → {X_filtered.shape[1]})")

    # Mean-impute remaining zeros
    X_imputed = X_filtered.copy()
    for j in range(X_imputed.shape[1]):
        col = X_imputed[:, j]
        zero_mask = col == 0
        if zero_mask.any():
            non_zero_vals = col[~zero_mask]
            if len(non_zero_vals) > 0:
                X_imputed[zero_mask, j] = non_zero_vals.mean()

    return X_imputed


def load_crohns(data_dir):
    """
    Load Crohn's Disease dataset (GSE317503).
    Cross-sectional, single time point, colon biopsy tissue.
    90 CD + 50 NIBD = 140 samples.
    """
    print("  Loading Crohn's Disease (GSE317503)...")
    crohns_dir = os.path.join(data_dir, 'Crohn_Disease')
    meta_path = os.path.join(crohns_dir, 'GSE317503_family.soft.gz')
    counts_path = os.path.join(crohns_dir, 'GSE317503_TPMSalmonCounts_final.txt.gz')
    if not os.path.exists(counts_path):
        counts_path = counts_path.replace('.gz', '')

    # --- Step 1: Parse SOFT metadata ---
    sample_disease = {}
    gsm_to_title = {}
    current_gsm = None

    opener = gzip.open if meta_path.endswith('.gz') else open
    with opener(meta_path, 'rt') as f:
        for line in f:
            if line.startswith('^SAMPLE ='):
                current_gsm = line.strip().split('=')[1].strip()
            elif line.startswith('!Sample_title =') and current_gsm:
                title = line.strip().split('=', 1)[1].strip()
                gsm_to_title[current_gsm] = title
            elif line.startswith('!Sample_characteristics_ch1 = disease status:') and current_gsm:
                status = line.strip().split(':')[-1].strip()
                if status in ('CD', 'NIBD'):
                    sample_disease[current_gsm] = status

    # --- Step 2: Map GSM → title → counts column ---
    # Titles look like "IBD_697394, colon" → strip to "IBD_697394"
    title_to_disease = {}
    for gsm, status in sample_disease.items():
        if gsm in gsm_to_title:
            base_title = gsm_to_title[gsm].split(',')[0].strip()
            title_to_disease[base_title] = status

    # --- Step 3: Load counts and align ---
    df_counts = pd.read_csv(counts_path, sep='\t', index_col=0)

    cols_to_keep = [c for c in df_counts.columns if c in title_to_disease]
    if len(cols_to_keep) == 0:
        raise ValueError("No matching columns found between metadata and counts file!")

    df_counts = df_counts[cols_to_keep]
    X_df = df_counts.T  # rows=samples, cols=genes

    # --- Step 4: Labels ---
    Y = np.array([1 if title_to_disease[c] == 'CD' else 0 for c in X_df.index])

    # --- Step 5: Zero-imputation ---
    X_raw = X_df.values.astype(np.float64)
    X_cleaned = apply_zero_imputation(X_raw)

    print(f"    Final: {X_cleaned.shape[0]} samples × {X_cleaned.shape[1]} features "
          f"(CD={np.sum(Y==1)}, NIBD={np.sum(Y==0)})")
    return X_cleaned, Y


def load_ebola(data_dir):
    """
    Load Ebola dataset (GSE226106) with B2 filtering.
    Retains ONLY sacrifice-day organ-harvest samples:
      - Infected: D003-D008 tissue (18 monkeys, 213 samples)
      - Control: D000 tissue (3 monkeys, 33 samples)
    Excludes all pre-infection blood draws and early longitudinal samples.
    """
    print("  Loading Ebola (GSE226106) with B2 filter...")
    ebola_dir = os.path.join(data_dir, 'ebola')
    meta_path = os.path.join(ebola_dir, 'GSE226106_series_matrix.txt.gz')
    counts_path = os.path.join(ebola_dir, 'GSE226106_20230121_counts_submission.txt.gz')

    # --- Step 1: Parse series matrix metadata ---
    sample_titles = []
    treatments = []
    times = []

    opener = gzip.open if meta_path.endswith('.gz') else open
    with opener(meta_path, 'rt') as f:
        for line in f:
            if line.startswith('!Sample_title'):
                sample_titles = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif '"treatment:' in line and line.startswith('!Sample_characteristics_ch1'):
                treatments = [s.strip('"').replace('treatment: ', '') for s in line.strip().split('\t')[1:]]
            elif '"time:' in line and line.startswith('!Sample_characteristics_ch1'):
                times = [s.strip('"').replace('time: ', '') for s in line.strip().split('\t')[1:]]

    if not sample_titles or not treatments or not times:
        raise ValueError("Failed to parse Ebola metadata — check file format")

    # --- Step 2: Build mapping with B2 filter ---
    # B2: sacrifice-day organs only
    INFECTED_DAYS = {'D003', 'D004', 'D005', 'D006', 'D007', 'D008'}
    CONTROL_DAYS = {'D000'}

    title_to_label = {}
    n_discarded = 0

    for i, title in enumerate(sample_titles):
        trt = treatments[i] if i < len(treatments) else 'NA'
        t = times[i] if i < len(times) else 'NA'

        # Extract the [key] part from title for matching to counts columns
        key = title
        if '[' in title and ']' in title:
            key = title.split('[')[-1].split(']')[0]

        if trt == 'infected with Ebola virus' and t in INFECTED_DAYS:
            title_to_label[key] = 1
        elif trt == 'non_infected_control' and t in CONTROL_DAYS:
            title_to_label[key] = 0
        else:
            n_discarded += 1

    print(f"    B2 filter: kept {len(title_to_label)} samples, discarded {n_discarded}")

    # --- Step 3: Load counts and align ---
    df_counts = pd.read_csv(counts_path, sep='\t', index_col=0)

    cols_to_keep = []
    y_labels = []
    for c in df_counts.columns:
        if c in title_to_label:
            cols_to_keep.append(c)
            y_labels.append(title_to_label[c])

    if len(cols_to_keep) == 0:
        raise ValueError("No matching columns after B2 filter!")

    df_counts = df_counts[cols_to_keep]
    X_df = df_counts.T
    Y = np.array(y_labels)

    # --- Step 4: Zero-imputation ---
    X_raw = X_df.values.astype(np.float64)
    X_cleaned = apply_zero_imputation(X_raw)

    print(f"    Final: {X_cleaned.shape[0]} samples × {X_cleaned.shape[1]} features "
          f"(Infected={np.sum(Y==1)}, Control={np.sum(Y==0)})")
    return X_cleaned, Y


# =====================================================================
# Dataset Registry
# =====================================================================

def load_all_datasets(data_dir):
    """Load all 5 datasets. Returns dict of {name: (X, Y)}."""
    datasets = {}

    # .mat files
    for name in ['madelon', 'Prostate_GE', 'ALLAML']:
        print(f"  Loading {name}...")
        X, Y = load_mat_dataset(name, data_dir)
        print(f"    Final: {X.shape[0]} samples × {X.shape[1]} features "
              f"(class0={np.sum(Y==0)}, class1={np.sum(Y==1)})")
        datasets[name] = (X, Y)

    # RNA-seq
    datasets['Crohns'] = load_crohns(data_dir)
    datasets['Ebola'] = load_ebola(data_dir)

    return datasets


# =====================================================================
# Evaluation Engine
# =====================================================================

def run_single_experiment(X, Y, method, seed, feature_subsets):
    """
    Run one seed of the experiment for one method on one dataset.
    Returns list of result dicts.
    """
    results = []

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, train_size=SPLIT_RATIO, stratify=Y, random_state=seed
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Score and rank features
    ranked_indices, scores = compute_feature_scores(X_train, y_train, method)

    # Evaluate at each k
    for k in feature_subsets:
        k_actual = min(k, X_train.shape[1])
        top_k = ranked_indices[:k_actual]

        X_tr_k = X_train[:, top_k]
        X_te_k = X_test[:, top_k]

        # kNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_tr_k, y_train)
        knn_acc = accuracy_score(y_test, knn.predict(X_te_k)) * 100

        # SVM
        try:
            svm = LinearSVC(max_iter=5000, dual='auto')
            svm.fit(X_tr_k, y_train)
            svm_acc = accuracy_score(y_test, svm.predict(X_te_k)) * 100
        except Exception:
            svm_acc = np.nan

        results.append({
            'method': method,
            'seed': seed,
            'k': k,
            'knn_accuracy': knn_acc,
            'svm_accuracy': svm_acc
        })

    return results


def run_pipeline():
    """Run the complete experiment pipeline."""
    print("=" * 70)
    print("DDFF Pipeline v2")
    print("=" * 70)

    # Load all datasets
    print("\n[1/3] Loading datasets...")
    datasets = load_all_datasets(DATA_DIR)

    # Check for existing results (resume support)
    if os.path.exists(RESULTS_CSV):
        existing_df = pd.read_csv(RESULTS_CSV)
        print(f"\n  Found existing results: {len(existing_df)} rows")
        completed = set()
        for _, row in existing_df.iterrows():
            completed.add((row['dataset'], row['method'], int(row['seed'])))
    else:
        existing_df = pd.DataFrame()
        completed = set()

    # Run experiments
    print("\n[2/3] Running experiments...")
    total_combos = len(datasets) * len(METHODS) * N_REPS
    done_count = 0
    new_results = []

    for ds_name, (X, Y) in datasets.items():
        # Adjust k values for small feature counts
        k_values = [k for k in FEATURE_SUBSETS if k <= X.shape[1]]
        if X.shape[1] not in k_values:
            k_values.append(X.shape[1])
        k_values = sorted(k_values)

        for method in METHODS:
            for seed in range(N_REPS):
                done_count += 1
                key = (ds_name, method, seed)

                if key in completed:
                    continue

                t_start = time.time()
                try:
                    results = run_single_experiment(X, Y, method, seed, k_values)
                    for r in results:
                        r['dataset'] = ds_name
                    new_results.extend(results)
                    elapsed = time.time() - t_start
                    print(f"  [{done_count}/{total_combos}] {ds_name} | {method:15s} | seed={seed:2d} | {elapsed:.1f}s")
                except Exception as e:
                    print(f"  [{done_count}/{total_combos}] {ds_name} | {method:15s} | seed={seed:2d} | ERROR: {e}")

                # Save incrementally every 25 experiments
                if len(new_results) >= 25 * len(k_values):
                    _save_results(existing_df, new_results)

    # Final save
    _save_results(existing_df, new_results)

    print(f"\n[3/3] Done! Results saved to {RESULTS_CSV}")
    print(f"  Total experiments: {done_count}")


def _save_results(existing_df, new_results):
    """Append new results to existing and save to CSV."""
    if not new_results:
        return
    new_df = pd.DataFrame(new_results)
    if len(existing_df) > 0:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(RESULTS_CSV, index=False)


# =====================================================================
# Entry point
# =====================================================================

if __name__ == '__main__':
    run_pipeline()
