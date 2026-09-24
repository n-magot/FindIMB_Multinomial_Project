# FindIMB — Multinomial (Discrete) Implementation

Python implementation of **FindIMB** for discrete data: a Bayesian method that combines a large **observational** dataset $D_o$ with a small **experimental** (randomized) dataset $D_e$ to predict post-intervention outcomes $P(Y \mid do(X), \mathbf{Z})$.

For each candidate covariate subset $\mathbf{Z}$, FindIMB compares two hypotheses:

- $H^c_{\mathbf{Z}}$: conditioning on $\mathbf{Z}$ makes the observational and experimental conditionals equal, $P_o(Y \mid X, \mathbf{Z}) = P_e(Y \mid X, \mathbf{Z})$, so $D_o$ and $D_e$ can be pooled.
- $\bar H^c_{\mathbf{Z}}$: they differ, so only $D_e$ is informative about $P(Y \mid do(X), \mathbf{Z})$.

Predictions are then obtained by Bayesian model averaging (BMA) over subsets and both hypotheses, and compared against observational-only, experimental-only and naively pooled baselines.

## Repository structure

```
.
├── run_experiment.py      # Entry point: data → search → cross-validation → metrics
├── requirements.txt
└── src/
    ├── preprocessing.py   # Synthetic example data + check that covariates are discrete
    ├── counting.py        # Contingency counts N_jk for (X, Z) configurations × Y
    ├── scoring.py         # Dirichlet–multinomial marginal likelihoods
    ├── search.py          # Forward searches: FindIMB (Do+De) and single-dataset
    └── bma.py             # BMA prediction, treatment recommendation, evaluation metrics
```

## Installation

Requires Python ≥ 3.11 (needed by the pinned `numpy 2.4` / `pandas 3.0`).

```bash
git clone https://github.com/n-magot/FindIMB_Multinomial_Project.git
cd FindIMB_Multinomial_Project
python -m venv FindIMB_venv
source FindIMB_venv/bin/activate          # Windows: FindIMB_venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
python run_experiment.py
```

This generates a synthetic example (`src/preprocessing.py`), runs 2-fold cross-validation over $D_e$, and prints per-fold metrics. On the default synthetic data (5,000 observational and 100 experimental samples, two binary covariates) it takes about two minutes.

Synthetic data-generating process:

| Variable    | Role                                                                    |
|-------------|-------------------------------------------------------------------------|
| `age`       | Confounder: affects treatment assignment in $D_o$ and the outcome       |
| `noise_var` | Affects the outcome only                                                |
| `treatment` | $X$: depends on `age` in $D_o$, randomized ($p=0.5$) in $D_e$           |
| `outcome`   | $Y$: `logit P(Y=1) = -1 + 0.7·age + 0.6·noise_var - 1.2·treatment`      |

## Using your own data

Replace `create_synthetic_data()` in `run_experiment.py` with two `pandas.DataFrame`s, `Do` and `De`, that have the same columns:

- `treatment`: binary, coded `0`/`1`
- `outcome`: binary, coded `0`/`1`, where **`0` is the desirable outcome**. Treatment recommendation maximizes $P(Y=0)$, and AUC/ECE treat $Y=0$ as the positive class.
- every other column is used as a candidate covariate and **must be discrete**.

Column names and settings are in the configuration block of `run_experiment.py`:

| Parameter   | Default       | Meaning                                                       |
|-------------|---------------|---------------------------------------------------------------|
| `TREATMENT` | `"treatment"` | Treatment column name                                         |
| `OUTCOME`   | `"outcome"`   | Outcome column name                                           |
| `N_SPLITS`  | `2`           | Number of `KFold` splits of $D_e$ (unshuffled)                |
| `THRESHOLD` | `0.1`         | Pruning threshold of the forward searches                     |
| `SEED`      | `42`          | Declared; `create_synthetic_data` uses its own `seed` argument |

## Method

For every candidate covariate set $\mathbf{Z}$ (always including $X$), `greedy_search_FindIMB_forward` in `src/search.py` scores $H^c_{\mathbf{Z}}$ and $\bar H^c_{\mathbf{Z}}$ with Dirichlet–multinomial marginal likelihoods (`src/scoring.py`). It explores subsets with a forward search that prunes them using `THRESHOLD`. `src/bma.py` then predicts $P(Y \mid do(X), \mathbf{Z})$ by averaging over the visited subsets and both hypotheses, weighted by their posterior probabilities.

### Compared models

| Key in results | Training data | Subset weights |
|---|---|---|
| `alg` | $D_o$ and $D_e$ | FindIMB: $P(H^c_{\mathbf{Z}}\mid D)$, $P(\bar H^c_{\mathbf{Z}}\mid D)$ |
| `exp` | $D_e$ only | single-dataset search on $D_e$ |
| `obs` | $D_o$ only | single-dataset search on $D_o$ (learned once, outside the CV loop) |
| `all` | $D_o \cup D_e$ pooled | single-dataset search on the pooled data |

### Metrics

All metrics are computed on the held-out fold of $D_e$:

- **Log-loss:** `sklearn.metrics.log_loss`.
- **AUC:** ROC AUC with $Y=0$ as the positive class.
- **ECE:** expected calibration error with 10 equal-width bins, $Y=0$ as the positive class.
- **DEU (direct expected utility):** each test unit is assigned $\hat x = \arg\max_x P(Y=0 \mid do(X=x), \mathbf{z})$, then
  $\mathrm{DEU} = \sum_{a\in\{0,1\}} \hat P(Y=0 \mid \hat x = a = x_{\text{received}})\,\hat P(\hat x = a)$,
  estimated only on units whose received treatment matches the recommendation, with additive smoothing ($a=0.1$, $b=0.2$).

`run_experiment` returns a DataFrame with one row per fold and the columns `DEU_*`, `AUC_*`, `logloss_*`, `ece_*` for `* ∈ {alg, exp, obs, all}`.


## Citation

If you use this code, please cite the accompanying paper:

```bibtex
% TODO: add BibTeX entry
```

## Author

Nandia Lelova
