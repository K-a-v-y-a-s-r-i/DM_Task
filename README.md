# Task 1: Outlier Detection for Houston Weather Dataset (HW2023)
## COSC6335 Spring 2026 

---

## Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

---

## How to Run

1. Place `HW2023.csv` in the **same folder** as the notebook.
2. Open and run all cells top-to-bottom:

```bash
jupyter notebook "File name".ipynb
```

Or run as a script:
```bash
jupyter nbconvert --to notebook --execute "File name.ipynb
```

---

## What the Code Does

| Section | Description |
|---------|-------------|
| **Section 1** | Load & explore HW2023 dataset; fill missing rain values with 0; standardize features |
| **Section 2** | Define the SIMPLE 2D dataset (ground truth known) |
| **Section 3** | **Method 1 — Density-Based (GMM):** Fit Gaussian Mixture Model; compute negative log-likelihood as OLS; run 3 hyperparameter settings (k=2, 4, 6 components) |
| **Section 4** | **Method 2 — Reconstruction-Based (Autoencoder):** Train a deep autoencoder; compute per-sample MSE reconstruction error as OLS; run 3 hyperparameter settings (latent_dim=2, 3, 5) |
| **Section 5** | Compare the two methods (correlation, top-10 overlap, scatter plot) |
| **Section 6** | All visualizations (OLS over time, histograms, SIMPLE scatter plots) |
| **Section 7** | *(Optional)* Quarterly outlier drift analysis — Q1-Q4 outlier rates and top-3 anomalous days per quarter |
| **Section 8** | Save augmented CSV files; print Appendix SIMPLE results |

---

## Output Files Generated

| File | Description |
|------|-------------|
| `HW2023_GMM_Outliers.csv` | HW2023 + GMM OLS columns (k=2,4,6) |
| `HW2023_AE_Outliers.csv`  | HW2023 + Autoencoder OLS columns (latent=2,3,5) |
| `SIMPLE_Outliers.csv`     | SIMPLE dataset with OLS_GMM and OLS_AE |
| `method_comparison_scatter.png` | Scatter plot of GMM vs AE OLS |
| `ols_over_time.png`        | Both OLS scores over the full year |
| `gmm_hyperparameter_comparison.png` | GMM OLS distributions for 3 settings |
| `ae_hyperparameter_comparison.png`  | AE OLS distributions for 3 settings |
| `simple_dataset_visualization.png`  | SIMPLE dataset outlier visualization |
| `quarterly_drift.png`      | Quarterly outlier drift chart |

---

## Hyperparameters

### Method 1 — GMM
| Setting | n_components (k) | Effect |
|---------|-----------------|--------|
| GMM_k2  | 2 | Coarse density model |
| GMM_k4  | 4 | Balanced (primary) |
| GMM_k6  | 6 | Fine-grained density |

### Method 2 — Autoencoder
| Setting | latent_dim | Effect |
|---------|-----------|--------|
| AE_latent2 | 2 | Very compressed — catches global outliers |
| AE_latent3 | 3 | Balanced (primary) |
| AE_latent5 | 5 | More expressive — catches subtle outliers |

---

