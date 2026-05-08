# DDFF: Distributional Discriminative Feature Filtering

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Distributional Discriminative Feature Filtering: A Scalable Framework for High-Dimensional Low-Sample-Size Biomedical Data**  
> Mufakir Ansari, Susmit Ghosh — The Ohio State University

## Overview

DDFF is a filter-based feature selection framework that ranks features by measuring the divergence between class-conditional empirical probability mass functions (PMFs). It operates in **O(M)** time, making it scalable to high-dimensional genomics data.

Four divergence metrics are provided:
| Method | Formula |
|---|---|
| **DDFF-L₁** | Σ\|P₀(k) − P₁(k)\| |
| **DDFF-L₂** | √(Σ(P₀(k) − P₁(k))²) |
| **DDFF-KL** | ½[KL(P₀‖P₁) + KL(P₁‖P₀)] |
| **DDFF-Max** | max_k \|P₀(k) − P₁(k)\| |
| **DDFF-Ensemble** | Mean of min-max normalized [L₁, L₂, KL, Max] |

## Results

Peak kNN accuracy (mean ± std over 25 seeds):

| Dataset | MI | Fisher | DDFF-L₁ | DDFF-L₂ | DDFF-KL | DDFF-Max | **Ensemble** |
|---|---|---|---|---|---|---|---|
| Madelon | 58.6±2.2 | 72.3±2.0 | **76.3±2.6** | 76.0±3.4 | 72.5±3.2 | 71.4±3.8 | 75.8±3.0 |
| Prostate GE | 93.1±5.3 | 91.4±5.1 | 92.4±5.1 | 91.2±6.3 | **93.5±4.1** | 89.7±6.3 | 91.4±6.6 |
| ALL/AML | 97.3±4.7 | 97.6±4.7 | 96.5±5.8 | 97.1±5.1 | 96.8±5.1 | **98.4±2.9** | 97.1±5.1 |
| Crohn's | 77.9±6.4 | 80.3±6.4 | 79.0±8.3 | 80.4±7.8 | 76.1±8.0 | 79.9±7.5 | **81.0±6.3** |
| Ebola | 90.6±3.3 | 90.1±2.6 | 90.3±2.4 | 90.6±3.3 | 90.8±3.2 | 90.4±3.2 | **90.9±3.2** |

## Repository Structure

```
DDFF-Framework/
├── code/
│   ├── ddff_framework.py      # Core DDFF implementation (4 metrics + ensemble)
│   ├── ddff_pipeline.py       # Master experiment runner (875 experiments)
│   ├── plotResults.py         # Figure generation (5 publication figures)
│   └── compute_extras.py      # Spearman ρ + convergence speed analysis
├── Data/
│   ├── madelon.mat            # Madelon synthetic benchmark (NIPS 2003)
│   ├── Prostate_GE.mat        # Singh et al. (2002) Cancer Cell
│   ├── ALLAML.mat             # Golub et al. (1999) Science
│   ├── Crohn_Disease/         # GSE317503 (Furey et al.)
│   └── ebola/                 # GSE226106 (Normandin et al. 2024)
├── figures/                   # All 5 publication-ready PDF figures
├── results/
│   ├── pipeline_results.csv   # Full experiment records (7,701 rows)
│   └── summary_table.txt      # Peak accuracy summary
├── main.tex                   # 6-page Elsevier manuscript
└── papers.bib                 # Bibliography
```

## Quick Start

### Requirements
```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### Run Experiments
```bash
cd V2/
python3 code/ddff_pipeline.py          # Run all 875 experiments → results/pipeline_results.csv
python3 code/plotResults.py            # Generate all 5 figures → figures/
python3 code/compute_extras.py         # Rank correlation + convergence analysis
```

### Compile Paper
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Datasets

| Dataset | Source | GEO Accession |
|---|---|---|
| Madelon | NIPS 2003 Feature Selection Challenge | N/A |
| Prostate GE | Singh et al. (2002), *Cancer Cell* | N/A (.mat) |
| ALL/AML | Golub et al. (1999), *Science* | N/A (.mat) |
| Crohn's Disease | Furey et al., UNC | GSE317503 |
| Ebola (Rhesus macaque) | Normandin et al. (2024) | GSE226106 |

**Note on Ebola data**: We apply a sacrifice-day organ-harvest (B2) filter, retaining only necropsy-day tissue samples (D003–D008 infected, D000 controls). See `ddff_pipeline.py:load_ebola()`.

## Citation

If you use this code, please cite:

```bibtex
@article{Ansari2025DDFF,
  title={Distributional Discriminative Feature Filtering: A Scalable Framework
         for High-Dimensional Low-Sample-Size Biomedical Data},
  author={Ansari, Mufakir and Ghosh, Susmit},
  journal={Knowledge-Based Systems},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
