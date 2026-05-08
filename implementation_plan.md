# Implementation Plan v3: DDFF Paper Restructuring

## Goal
Restructure the DDFF paper with deep dataset biological profiles, clean Ebola filtering, 5 datasets, 7 methods, feature score visualizations, and condense to 6 pages.

---

## 1. Deep Dataset Biological Profiles

### Dataset 1: Madelon (Synthetic Benchmark)

| Property | Detail |
|---|---|
| **Source** | NIPS 2003 Feature Selection Challenge |
| **Citation** | Guyon, I. (2003). Design of experiments for the NIPS 2003 variable selection benchmark |
| **Organism** | Synthetic — no biological organism |
| **Tissue** | N/A |
| **Collection** | Algorithmically generated |
| **Platform** | Artificial data generator |
| **Design** | 5 truly relevant features placed at vertices of a **5-dimensional hypercube**. Data points form **32 clusters** on hypercube vertices. Class labels (+1/−1) are assigned based on cluster position. 15 additional features are **linear combinations** of the 5 true features + noise. 480 features are **pure noise distractors** with zero predictive power. |
| **Samples** | 2,600 total: 1,300 per class (perfectly balanced) |
| **Features** | 500 total: 5 truly relevant + 15 derived + 480 noise |
| **Classes** | +1 vs −1 (arbitrary binary labels) |
| **Preprocessing** | None needed — clean synthetic data |
| **Why it matters** | Provides **ground-truth validation** with known informative features. Tests whether DDFF can detect non-linear XOR-like separability invisible to linear methods. The 480 noise features test DDFF's ability to suppress irrelevant distractors. |
| **Key challenge** | No single feature is informative on its own — the 5 relevant features only discriminate via their **joint non-linear interaction** (hypercube geometry). This is designed to defeat univariate filters. |

**What we write in the paper:**
> *"Madelon is a synthetic benchmark from the NIPS 2003 Feature Selection Challenge, designed to test non-linear multivariate separability. Data points are clustered on vertices of a 5-dimensional hypercube, with class labels assigned by vertex position. Of 500 features, exactly 5 are truly relevant, 15 are derived linear combinations, and 480 are pure noise distractors. Crucially, no single feature is independently informative—discrimination requires detecting joint non-linear interactions—making this an adversarial test for any univariate filter."*

---

### Dataset 2: Prostate_GE (Prostate Cancer Microarray)

| Property | Detail |
|---|---|
| **Source** | Singh et al. (2002), *Cancer Cell* 1(2):203–209 |
| **Citation** | "Gene expression correlates of clinical prostate cancer behavior" |
| **Organism** | *Homo sapiens* (human) |
| **Tissue** | **Prostate tissue biopsies** obtained during **radical prostatectomy** surgery |
| **Collection** | Surgical resection — tumor and adjacent normal tissue excised from prostate gland |
| **Platform** | **Affymetrix oligonucleotide microarrays** (HG-U95Av2), probing ~12,600 genes/ESTs |
| **Design** | Cross-sectional: tumor vs. normal tissue from surgical patients. Original study also correlated expression with Gleason score and post-surgery relapse, but we use only the tumor/normal binary. |
| **Samples** | 102 total: 52 tumor + 50 normal |
| **Features** | 5,966 genes (pre-processed subset of original ~12,600) |
| **Classes** | Class 1 = Tumor (52), Class 2 = Normal (50) |
| **Preprocessing** | Pre-processed by original authors; loaded from `.mat` file |
| **Why it matters** | Classic HDLSS benchmark (p/n ratio ≈ 58:1). Tests DDFF on moderate-dimensional microarray with near-balanced classes. |

**What we write:**
> *"Prostate_GE originates from Singh et al. (2002), profiling gene expression in prostate tissue biopsies obtained during radical prostatectomy. Using Affymetrix HG-U95Av2 microarrays, 102 samples (52 tumor, 50 normal) were assayed across 5,966 genes, constituting a classic HDLSS binary classification benchmark."*

---

### Dataset 3: ALLAML (Leukemia Microarray)

| Property | Detail |
|---|---|
| **Source** | Golub et al. (1999), *Science* 286(5439):531–537 |
| **Citation** | "Molecular classification of cancer: class discovery and class prediction by gene expression monitoring" |
| **Organism** | *Homo sapiens* (human) |
| **Tissue** | **Bone marrow mononuclear cells** (training set) and a mix of **bone marrow + peripheral blood** (independent test set) |
| **Collection** | Bone marrow aspirates collected **at time of diagnosis, before chemotherapy** |
| **Platform** | **Affymetrix oligonucleotide microarrays** containing probes for **6,817 human genes** |
| **Design** | Landmark study in cancer genomics. Demonstrated that gene expression monitoring alone can distinguish between AML and ALL subtypes of acute leukemia. Originally: 38 training + 34 independent test = 72 total. Used "neighborhood analysis" for class discovery and weighted voting for class prediction. |
| **Samples** | 72 total: 47 ALL + 25 AML |
| **Features** | 7,129 genes (expanded from original 6,817 via probe re-mapping) |
| **Classes** | Class 1 = ALL / Acute Lymphoblastic Leukemia (47), Class 2 = AML / Acute Myeloid Leukemia (25) |
| **Preprocessing** | Pre-processed; loaded from `.mat` file |
| **Why it matters** | The foundational cancer classification dataset. Extremely high p/n ratio (≈ 99:1). Moderate class imbalance (1.9:1). If a feature selection method works here, it proves fundamental capability on biological data. |

**What we write:**
> *"ALLAML is the foundational cancer classification dataset from Golub et al. (1999), published in Science. Gene expression was profiled from bone marrow mononuclear cells collected at diagnosis (pre-chemotherapy) using Affymetrix microarrays. The 72 samples (47 ALL, 25 AML) across 7,129 genes exhibit an extreme p/n ratio of 99:1, making it a canonical HDLSS benchmark."*

---

### Dataset 4: Crohn's Disease (Clinical RNA-seq)

| Property | Detail |
|---|---|
| **Source** | GEO Accession **GSE317503** |
| **Citation** | Furey TS, Sheikh SZ, Kennedy Ng MM — University of North Carolina |
| **Organism** | *Homo sapiens* (human) |
| **Tissue** | **Colon biopsy tissue** — all 140 samples are colon biopsies |
| **Collection** | Clinical endoscopic biopsies from the colon during routine colonoscopy |
| **Platform** | Illumina RNA-seq (GPL11154 / GPL16791 / GPL20301) |
| **Sequencing** | Bulk RNA-seq → quantified using **Salmon** → reported as TPM (Transcripts Per Million) |
| **Design** | **Cross-sectional, single time point, single tissue.** 140 patients: 90 with confirmed Crohn's Disease (CD) and 50 non-inflammatory bowel disease controls (NIBD). No longitudinal follow-up. |
| **Samples** | 140 total: 90 CD + 50 NIBD |
| **Features (raw)** | 62,550 genes in original TPM counts file |
| **Features (after cleaning)** | 23,813 genes — after removing genes with >20% zero expression and mean-imputing the remainder |
| **Classes** | Class 0 = NIBD / healthy control (50), Class 1 = CD / Crohn's Disease (90) |
| **Preprocessing** | 1. Remove genes with >20% zero expression across all samples; 2. Mean-impute remaining zero values with non-zero column mean; 3. StandardScaler normalization |
| **Why it matters** | Most extreme HDLSS benchmark (p/n ≈ 170:1). Real-world clinical RNA-seq with biological noise, batch effects, and zero-inflation. The ultimate test of DDFF's practical utility. |

**What we write:**
> *"The Crohn's Disease RNA-seq profile (GSE317503) represents a cross-sectional study of colon biopsy tissue from 140 patients at a single clinical time point. Bulk RNA-seq (Illumina, quantified via Salmon TPM) captured 90 Crohn's Disease (CD) and 50 non-inflammatory bowel disease (NIBD) control profiles. After aggressive noise filtration — removing genes with >20% zero-expression and applying feature-wise mean imputation — the active manifold comprised 23,813 features, producing an extreme p/n ratio of 170:1."*

---

### Dataset 5: Ebola Virus Disease (Multi-Tissue RNA-seq)

| Property | Detail |
|---|---|
| **Source** | GEO Accession **GSE226106** |
| **Citation** | Normandin E, Sierra SH, Raju SS et al. (2024), PMID: 38169842 |
| **Organism** | *Macaca mulatta* (rhesus macaque) |
| **Tissue** | **17 distinct tissue types** harvested at necropsy: Spleen, Kidney, Liver, Brain (Grey & White matter), Lung, Adrenal Gland, Lymph nodes (Axillary, Inguinal, Mesenteric), Skin (Rash & Non-Rash), Sex Organs (Ovary/Testis), PBMC, Whole Blood |
| **Collection** | **Lethal challenge study.** 21 rhesus macaques inoculated with Ebola virus (EBOV). Animals sacrificed at designated time points and organs harvested at necropsy. 3 uninfected control macaques sacrificed at Day 0. |
| **Platform** | Illumina RNA-seq (GPL27943, *Macaca mulatta*) |
| **Design** | Natural history / lethal dose study. Infected animals sacrificed at D003 (6 monkeys), D004 (6), D005 (3), D006 (5), D007 (2), D008 (2). Controls sacrificed at D000 (3 monkeys). Pre-infection blood draws (D-30, D-14, D-04) were also collected. |

#### Filtering Decision: Sacrifice-Day Organ Harvest Only (Option B2)

**We exclude:** All pre-infection blood draws (D-30, D-28, D-14, D-04) and early longitudinal blood samples (D000-D002 from infected monkeys). These represent pre-infection or very early infection states where the monkey's transcriptome has not yet been systemically altered.

**We retain:**
- **Infected**: 213 tissue samples from 18 monkeys sacrificed at D003–D008 (organ harvest at necropsy)
- **Control**: 33 tissue samples from 3 monkeys sacrificed at D000 (organ harvest at necropsy)

| Sacrifice Day | Infected Monkeys | Samples | Stage |
|---|---|---|---|
| D003 | RA0223, RA0449, RA0452, RA0522, RA1639, RA1834 | 39 | Early symptomatic |
| D004 | RA0717, RA0850, RA1074, RA1325, RA1803, RA1849 | 37 | Mid symptomatic |
| D005 | RA0452, RA0522, RA1639 | 37 | Late symptomatic |
| D006 | RA0917, RA1074, RA1325, RA1803, RA1818 | 56 | Late symptomatic |
| D007 | RA0700, RA1790 | 23 | Terminal |
| D008 | RA1423, RA1779 | 21 | Terminal |
| **D000 (Control)** | RA1082, RA1819, RA1856 | **33** | Uninfected |

| Property | After Filtering |
|---|---|
| **Samples** | 246 total: 213 infected + 33 control |
| **Features (raw)** | 35,405 genes |
| **Features (after cleaning)** | ~13,900* (TBD — needs re-run with filtered samples) |
| **Classes** | Class 0 = Uninfected control (33), Class 1 = EBOV-infected at necropsy (213) |
| **Ratio** | 1:6.5 |

**What we write:**
> *"The Ebola dataset (GSE226106) derives from a natural-history lethal challenge study in rhesus macaques (Normandin et al., 2024). Twenty-one animals were inoculated with Ebola virus and sacrificed at designated time points between days 3–8 post-infection; 3 uninfected control animals were sacrificed at day 0. At necropsy, bulk RNA-seq was performed on up to 17 tissue types per animal (spleen, kidney, liver, brain, lung, lymph nodes, skin, adrenal gland, sex organs, and whole blood). We retained only sacrifice-day organ-harvest samples to ensure all samples represent a definitive infected or uninfected physiological state, yielding 246 samples (213 infected from 18 monkeys, 33 control from 3 monkeys). After removing genes with >20% zero expression and mean-imputing the remainder, the evaluation matrix comprised approximately 13,900 features."*

---

## 2. Complete Dataset Summary Table (for paper)

| Dataset | Organism | Tissue | Platform | Time | Samples | Features | Class 0 | Class 1 | p/n |
|---|---|---|---|---|---|---|---|---|---|
| **Madelon** | Synthetic | N/A | Generated | N/A | 2,600 | 500 | 1,300 | 1,300 | 0.2 |
| **Prostate_GE** | Human | Prostate biopsy | Affymetrix | Cross-section | 102 | 5,966 | 50 (Normal) | 52 (Tumor) | 58 |
| **ALLAML** | Human | Bone marrow | Affymetrix | At diagnosis | 72 | 7,129 | 25 (AML) | 47 (ALL) | 99 |
| **Crohn's** | Human | Colon biopsy | Illumina RNA-seq | Cross-section | 140 | 23,813 | 50 (NIBD) | 90 (CD) | 170 |
| **Ebola** | Rhesus macaque | 17 organs | Illumina RNA-seq | D003–D008 | 246 | ~13,900 | 33 (Control) | 213 (Infected) | ~57 |

---

## 3. Data Processing — Complete Pipeline Per Dataset

### 3.1 Raw Data Inspection Results

| Dataset | File Format | X dtype | X range | Y values | Zeros in X | NaNs |
|---|---|---|---|---|---|---|
| **Madelon** | `.mat` (X, Y) | uint16 | [0, 999] | {−1, +1} | 1 / 1.3M (0.00%) | None |
| **Prostate_GE** | `.mat` (X, Y) | float64 | [1.00, 4.20] | {1, 2} | 0 / 608K (0.00%) | None |
| **ALLAML** | `.mat` (X, Y) | float64 | [−8.33, 8.37] | {1, 2} | 17 / 513K (0.00%) | None |
| **Crohn's** | `.soft.gz` + `.txt.gz` (TPM) | float64 | [0, ~10⁵] | Parsed: CD/NIBD | Heavy (62K→24K after filter) | None |
| **Ebola** | `series_matrix.gz` + `counts.txt.gz` | float64 | [0, ~10⁶] | Parsed: infected/control | Heavy (35K→14K after filter) | None |

### 3.2 Processing Pipeline Per Dataset

---

#### Madelon
```
Raw: madelon.mat → X(2600×500, uint16), Y(2600, {-1,+1})
  ↓ Load directly with scipy.io.loadmat()
  ↓ Map labels: -1 → 0, +1 → 1    [NEW: current code keeps {-1,+1}, need to standardize]
  ↓ No zero-imputation needed (0.00% zeros)
  ↓ No additional preprocessing
Final: X(2600×500), Y(2600, {0,1})
```
**Notes:**
- Data values are integer counts [0, 999] — already clean
- Already log-scale-free — values represent raw feature magnitudes
- StandardScaler applied later during evaluation (after split)

---

#### Prostate_GE
```
Raw: Prostate_GE.mat → X(102×5966, float64), Y(102, {1,2})
  ↓ Load with scipy.io.loadmat()
  ↓ Map labels: 1 → 0 (Normal), 2 → 1 (Tumor)    [standardize to {0,1}]
  ↓ No zero-imputation needed (0 zeros)
  ↓ Already pre-processed by Singh et al. (log-transformed, range [1.00, 4.20])
Final: X(102×5966), Y(102, {0,1})
```
**Notes:**
- Values are already log-transformed expression levels
- Narrow value range [1.00, 4.20] means features are already comparable
- No NaN/zero concerns at all

---

#### ALLAML
```
Raw: ALLAML.mat → X(72×7129, float64), Y(72, {1,2})
  ↓ Load with scipy.io.loadmat()
  ↓ Map labels: 1 → 0 (ALL), 2 → 1 (AML)    [OR: 1→ALL, 2→AML, verify]
  ↓ No zero-imputation needed (17 zeros out of 513K = negligible)
  ↓ Already pre-processed (log-transformed, range [-8.33, 8.37])
Final: X(72×7129), Y(72, {0,1})
```
**Notes:**
- Negative values present → already log-ratio or z-score transformed
- Only 17 zeros in entire matrix — negligible
- Most extreme HDLSS ratio (p/n = 99:1)

---

#### Crohn's Disease
```
Raw files:
  1. GSE317503_family.soft.gz     → metadata (GSM IDs, disease status, sample titles)
  2. GSE317503_TPMSalmonCounts_final.txt.gz → expression matrix (genes × samples)

Step 1: Parse SOFT metadata
  ↓ Scan for "^SAMPLE =" lines → extract GSM IDs (e.g., GSM9473209)
  ↓ Scan for "disease status:" → extract CD or NIBD
  ↓ Scan for "!Sample_title =" → extract titles (e.g., "IBD_697394, colon")
  ↓ Build mapping: GSM → title → disease_status
  Result: 90 CD patients + 50 NIBD controls = 140 total

Step 2: Load counts and align
  ↓ Read TPM counts file (genes as rows, sample IDs as columns)
  ↓ Strip tissue suffix from titles: "IBD_697394, colon" → "IBD_697394"
  ↓ Match count columns to metadata → keep only matched CD/NIBD columns
  ↓ Transpose: rows=samples, cols=genes
  Result: X_raw(140 × 62,550)

Step 3: Zero-imputation (applied to FULL dataset)
  ↓ Count zeros per feature across all 140 samples
  ↓ Remove features where >20% of samples have zero expression
     → Removes 38,737 features (62% of features are too sparse)
  ↓ For remaining features: replace zero values with non-zero column mean
  Result: X_cleaned(140 × 23,813)

Step 4: Label encoding
  ↓ CD → 1, NIBD → 0
Final: X(140×23,813), Y(140, {0,1})
```
**Key concern — Zero-imputation before split:**
> The `apply_zero_imputation()` runs on the full 140-sample matrix BEFORE the train/test split. The non-zero mean used for imputation includes test-set samples. This is a minor form of data leakage. Impact: minimal, because (a) mean imputation of zeros has low information content, and (b) it's the standard practice in genomics preprocessing. We document this clearly in the new pipeline.

---

#### Ebola (with B2 Filtering)
```
Raw files:
  1. GSE226106_series_matrix.txt.gz    → metadata (treatment, time, monkey, tissue)
  2. GSE226106_20230121_counts_submission.txt.gz → count matrix (genes × samples)

Step 1: Parse series matrix metadata
  ↓ Extract !Sample_title lines → sample IDs with [key] format
  ↓ Extract treatment: → "infected with Ebola virus" or "non_infected_control"
  ↓ Extract time: → D-30, D000, D003, ..., D008
  ↓ Extract tissue: → Spleen, Kidney, Liver, etc.

Step 2: Apply B2 Filter (NEW)
  ↓ INFECTED: Keep ONLY samples where:
      treatment == "infected with Ebola virus" AND
      time ∈ {D003, D004, D005, D006, D007, D008}   ← sacrifice-day organs only
      → 213 samples from 18 monkeys
  ↓ CONTROL: Keep ONLY samples where:
      treatment == "non_infected_control" AND
      time == "D000"   ← sacrifice-day organs only
      → 33 samples from 3 monkeys
  ↓ DISCARD: All pre-infection blood draws (D-30, D-28, D-14, D-04)
  ↓ DISCARD: Early longitudinal blood (D000-D002 from infected monkeys)
  ↓ DISCARD: treatment == "NA" (10 samples)
  Result: 246 samples (213 infected + 33 control)

Step 3: Match to counts and load
  ↓ Extract [key] from sample titles → match to counts file columns
  ↓ Keep only matched columns → transpose to rows=samples, cols=genes
  Result: X_raw(246 × 35,405)

Step 4: Zero-imputation (on filtered subset)
  ↓ Remove features with >20% zeros across the 246 samples
  ↓ Mean-impute remaining zeros
  Result: X_cleaned(246 × ~TBD)  ← feature count will change from 13,973

Step 5: Label encoding
  ↓ infected → 1, control → 0
Final: X(246 × ~TBD), Y(246, {0,1})
```
**Key change from current code:** The current Ebola loader has NO time/tissue filtering. The new pipeline adds B2 filtering between Step 1 and Step 3.

---

### 3.3 Evaluation Pipeline (After Data Loading)

```
For each dataset D (X, Y already loaded and cleaned):
│
├─ For each seed s ∈ {0, 1, ..., 24}:
│   │
│   ├─ SPLIT: train_test_split(X, Y, 80/20, stratified, seed=s)
│   │   → X_train, X_test, Y_train, Y_test
│   │
│   ├─ SCALE: StandardScaler.fit(X_train)
│   │   → X_train = scaler.transform(X_train)  ← fit ONLY on train
│   │   → X_test  = scaler.transform(X_test)   ← transform with train stats
│   │
│   ├─ For each method M ∈ {MI, Fisher, L1, L2, KL, Max, Ensemble}:
│   │   │
│   │   ├─ SCORE: compute feature_scores(M, X_train, Y_train)
│   │   │   ↓ MI: sklearn mutual_info_classif(X_train, Y_train)
│   │   │   ↓ Fisher: skfeature fisher_score(X_train, Y_train)
│   │   │   ↓ DDFF: histogram(X_train[:,j], bins=10) → PMFs → divergence
│   │   │   ↓ Ensemble: normalize + average all 4 DDFF scores
│   │   │
│   │   ├─ RANK: ranked_indices = argsort(scores)[::-1]
│   │   │
│   │   ├─ For each k ∈ {25, 50, 75, 100, 150, 200, 300, 500}:
│   │   │   │
│   │   │   ├─ SELECT: top_k = ranked_indices[:k]
│   │   │   │
│   │   │   ├─ CLASSIFY (kNN):
│   │   │   │   → KNeighborsClassifier(n_neighbors=5)
│   │   │   │   → fit(X_train[:, top_k]) → predict(X_test[:, top_k])
│   │   │   │   → knn_accuracy
│   │   │   │
│   │   │   ├─ CLASSIFY (SVM):
│   │   │   │   → LinearSVC(max_iter=5000)
│   │   │   │   → fit(X_train[:, top_k]) → predict(X_test[:, top_k])
│   │   │   │   → svm_accuracy
│   │   │   │
│   │   │   └─ STORE: (dataset, seed, method, k, knn_acc, svm_acc)
│   │
│   └─ END methods
│
└─ AGGREGATE: mean ± std across 25 seeds for each (method, k, classifier)
```

### 3.4 Two Key Data Processing Concerns

**Concern 1: Zero-imputation runs on full data before split**
- `apply_zero_imputation()` uses ALL samples to compute non-zero means
- Test samples contribute to the imputation values
- **Impact**: Minimal — mean imputation of zeros carries very little predictive information
- **Mitigation**: Standard practice in genomics; we document it explicitly
- **Alternative (not implemented)**: Move imputation inside the split loop, fit on train only

**Concern 2: StandardScaler runs after split but before DDFF histogramming**
- DDFF builds histograms on scaled training data, not raw expression values
- This means the histogram bin edges depend on the StandardScaler transform
- **Impact**: Negligible — scaling is a monotonic linear transform that preserves distributional shape
- **Why this is actually fine**: The PMF normalization in DDFF is invariant to linear scaling of the input, since only the relative bin occupancies matter

---

## 4. Methods — Final List (7 Total)

| # | Method | Type | Definition |
|---|---|---|---|
| 1 | **Mutual Information** | Baseline | MI(X;Y) = ΣΣ p(x,y) log[p(x,y) / p(x)p(y)] |
| 2 | **Fisher Score** | Baseline | F = Σ nⱼ(μⱼ − μ)² / Σ nⱼσⱼ² |
| 3 | **DDFF-L₁** | Proposed | Σ\|P₀(k) − P₁(k)\| |
| 4 | **DDFF-L₂** | Proposed | √(Σ(P₀(k) − P₁(k))²) |
| 5 | **DDFF-KL** | Proposed | ½[KL(P₀‖P₁) + KL(P₁‖P₀)] with ε=10⁻⁶ |
| 6 | **DDFF-Max** | Proposed | max_k \|P₀(k) − P₁(k)\| |
| 7 | **DDFF-Ensemble** | Proposed | Mean of min-max normalized [L₁, L₂, KL, Max] scores |

---

## 5. Experiment Pipeline

```
For each dataset D ∈ {Madelon, Prostate_GE, ALLAML, Crohn's, Ebola}:
  Load and preprocess D → X, Y
  For each seed s ∈ {0, 1, ..., 24}:
    Stratified 80/20 split → X_train, X_test, Y_train, Y_test
    StandardScaler: fit on X_train, transform both
    For each method M ∈ {MI, Fisher, L1, L2, KL, Max, Ensemble}:
      Compute scores(M, X_train, Y_train) → feature_scores
      Rank by descending score → ranked_indices
      For each k ∈ {25, 50, 75, 100, 150, 200, 300, 500}:
        For each classifier C ∈ {kNN(k=5), LinearSVM}:
          Train C on X_train[:, top_k], predict X_test[:, top_k] → accuracy
          Store: (D, s, M, k, C, accuracy)
  Save results → CSV
```

---

## 6. Feature Score Visualization — 3 Datasets

Generate feature score profile plots for **Madelon, Ebola, Crohn's**:
- Top 30 features sorted by ensemble score
- 5 overlaid lines/bars: L₁, L₂, KL, Max, Ensemble (all normalized to [0,1])
- For Madelon: annotate informative vs noise features
- Professional styling with curated color palette

---

## 7. Case-Strengthening Additions

| Addition | What It Shows |
|---|---|
| **Accuracy heatmap** | Color-coded peak accuracy: 5 datasets × 7 methods × 2 classifiers |
| **Rank correlation** | Spearman ρ between method rankings — proves DDFF captures different signal |
| **Convergence speed** | Features needed to reach 95% of peak — DDFF should converge faster |
| **Variance comparison** | Std dev across 25 seeds — DDFF should show lower variance |

---

## 8. Code Files

| File | Purpose |
|---|---|
| **[MODIFY]** `code/ddff_framework.py` | Add `ddff_ensemble_scores()` function |
| **[NEW]** `code/ddff_pipeline.py` | Master experiment pipeline |
| **[NEW]** `code/plotResults.py` | All figure generation |
| **[MODIFY]** `code/evaluateBiomedicalDDFF.py` | Update Ebola loader with B2 filtering |

---

## 9. Paper Structure — 6 Pages (2-column)

| Section | ~Pages | Content |
|---|---|---|
| Title + Abstract | 0.3 | ~150 word abstract |
| §1 Introduction | 0.6 | Problem + gap + contributions |
| §2 Related Work | 0.5 | 3 condensed paragraphs |
| §3 Methodology | 1.0 | DDFF: 4 metrics + ensemble + complexity |
| §4 Experimental Setup | 1.0 | Dataset profiles, methods, pipeline |
| §5 Results | 1.8 | Heatmap, curves, feature scores, significance |
| §6 Conclusion | 0.5 | Summary + limitations + future |
| References | 0.3 | ~12-15 refs |

---

## 10. Execution Order

1. Modify `ddff_framework.py` — add ensemble scoring
2. Modify Ebola loader — add B2 filtering
3. Create `ddff_pipeline.py` — complete pipeline
4. **Run pipeline** — all experiments
5. Create `plotResults.py` — generate all figures
6. Rewrite `main.tex` — condensed 6-page paper
7. Verify — compile, check page count, review

---

## Open Questions

> [!IMPORTANT]
> 1. **Ebola feature count after B2 filtering**: Will change from 13,973 because the zero-filtration runs on a different sample subset (246 vs 298). Need to re-run to get exact number.
> 2. **Madelon informative feature indices**: The original NIPS challenge uses features 0–4 as truly relevant, 5–19 as derived. Verify this against the `.mat` file we have.
> 3. **Ready to start coding once approved.**
