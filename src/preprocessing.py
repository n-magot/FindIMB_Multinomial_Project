"""This is just an example dataset"""
import numpy as np
import pandas as pd

def create_synthetic_data(n_obs=5000, n_exp=100, seed=42):
    np.random.seed(seed)

    # ======================================================
    # 1. Covariates
    # ======================================================

    # Confounder: affects both treatment and outcome
    age_obs = np.random.binomial(1, 0.5, n_obs)   # 0=young, 1=old
    age_exp = np.random.binomial(1, 0.5, n_exp)

    # Non-confounder: affects outcome but NOT treatment
    noise_obs = np.random.binomial(1, 0.5, n_obs)
    noise_exp = np.random.binomial(1, 0.5, n_exp)

    # ======================================================
    # 2. Observational treatment (biased by confounder only)
    # ======================================================

    logits_obs = 0.6 * age_obs   # noise_var intentionally excluded
    p_t_obs    = 1 / (1 + np.exp(-logits_obs))
    T_obs      = np.random.binomial(1, p_t_obs)

    # ======================================================
    # 3. Outcome model (confounder + non-confounder + treatment)
    # ======================================================

    def outcome_model(age, noise_var, T):
        logit = (
            -1.0
            + 0.7 * age        # confounder affects outcome
            + 0.6 * noise_var  # non-confounder affects outcome
            - 1.2 * T          # treatment effect
        )
        p = 1 / (1 + np.exp(-logit))
        return np.random.binomial(1, p)

    Y_obs = outcome_model(age_obs, noise_obs, T_obs)

    Do = pd.DataFrame({
        "age":       age_obs,
        "noise_var": noise_obs,
        "treatment": T_obs,
        "outcome":   Y_obs
    })

    # ======================================================
    # 4. Experimental dataset (randomized treatment)
    # ======================================================

    T_exp = np.random.binomial(1, 0.5, n_exp)
    Y_exp = outcome_model(age_exp, noise_exp, T_exp)

    De = pd.DataFrame({
        "age":       age_exp,
        "noise_var": noise_exp,
        "treatment": T_exp,
        "outcome":   Y_exp
    })

    return Do, De

def check_for_continuous_variables(df, exclude=None, max_unique_ratio=0.05):
    """
    Raises error if continuous variables are detected.

    A variable is considered continuous if:
    - dtype is numeric AND
    - it has too many unique values relative to sample size
    """

    if exclude is None:
        exclude = []

    n = len(df)

    continuous_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        # only check numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):

            nunique = df[col].nunique(dropna=True)
            unique_ratio = nunique / max(n, 1)

            # heuristic: too many unique values → continuous
            if unique_ratio > max_unique_ratio:
                continuous_cols.append(col)

    if continuous_cols:
        raise ValueError(
            f"❌ Continuous variables detected: {continuous_cols}\n"
            f"Your BMA model requires DISCRETE variables.\n"
            f"Please discretize (e.g. pd.cut / pd.qcut) before running."
        )