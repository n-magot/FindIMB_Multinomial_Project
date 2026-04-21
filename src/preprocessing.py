"""This is just an example dataset"""
import numpy as np
import pandas as pd

def create_synthetic_data(n_obs=5000, n_exp=100, seed=42):
    np.random.seed(seed)

    # ======================================================
    # 1. DISCRETIZE covariates from the start
    # ======================================================

    # OBSERVATIONAL
    age_obs = np.random.choice(["young", "middle", "old"], n_obs, p=[0.5, 0.3, 0.2])
    sex_obs = np.random.choice([0, 1], n_obs, p=[0.8, 0.2])

    # EXPERIMENTAL
    age_exp = np.random.choice(["young", "middle", "old"], n_exp, p=[0.5, 0.3, 0.2])
    sex_exp = np.random.choice([0, 1], n_exp, p=[0.8, 0.2])

    # ======================================================
    # 2. Observational treatment (biased)
    # ======================================================

    age_score = np.array([0, 1, 2])  # young < middle < old
    age_map = {"young": 0, "middle": 1, "old": 2}

    logits_obs = (
        0.6 * np.vectorize(age_map.get)(age_obs)
        + 0.3 * sex_obs
    )

    p_t_obs = 1 / (1 + np.exp(-logits_obs))
    T_obs = np.random.binomial(1, p_t_obs)

    # ======================================================
    # 3. Outcome model (discrete inputs)
    # ======================================================

    def outcome_model(age_cat, sex, T):
        age_val = np.vectorize(age_map.get)(age_cat)

        logit = (
            -2
            + 0.7 * age_val
            + 0.5 * sex
            - 1.2 * T
        )

        p = 1 / (1 + np.exp(-logit))
        return np.random.binomial(1, p)

    Y_obs = outcome_model(age_obs, sex_obs, T_obs)

    Do = pd.DataFrame({
        "age": age_obs,
        "sex": sex_obs,
        "treatment": T_obs,
        "outcome": Y_obs
    })

    # ======================================================
    # 4. Experimental dataset (independent + randomized)
    # ======================================================

    T_exp = np.random.binomial(1, 0.5, n_exp)

    Y_exp = outcome_model(age_exp, sex_exp, T_exp)

    De = pd.DataFrame({
        "age": age_exp,
        "sex": sex_exp,
        "treatment": T_exp,
        "outcome": Y_exp
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