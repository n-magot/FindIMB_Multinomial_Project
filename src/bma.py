import numpy as np
import pandas as pd
import itertools

from sklearn.metrics import log_loss, roc_auc_score

from .counting import get_counts_multiZ

all_preds = []
all_preds_DEU = []

def bma_predict_and_evaluate(Do, De, De_test, df_scores, treatment, outcome, df_Score_De, df_Score_Do,
                                    df_Score_Do_De):
    """
    Perform BMA over subsets and evaluate the metrics using the new expected utility function.
    The evaluation metrics are binary cross-entropy, expected calibration error, and expected utility.
    """
    Do_De = pd.concat([Do, De], ignore_index=True)

    Ntest = len(De_test)
    K = 2
    P_Y_alg_accum = np.zeros((Ntest, K))

    # Keep track of subsets for expected outcome evaluation
    subsets = [list(t) for t in df_scores['Variables']]

    # Iterate over subsets to compute BMA
    for _, row in df_scores.iterrows():
        Z_cols = list(row['Variables'])
        w_c = float(row['P_HZ_c'])
        w_cb = float(row['P_HZ_c_bar'])

        # Build a universal Z_reference across both datasets
        Z_categories = []
        for col in Z_cols:
            cats_Do = Do[col].astype('category').cat.categories
            cats_De = De[col].astype('category').cat.categories
            all_cats = sorted(set(cats_Do) | set(cats_De))  # union of both
            Z_categories.append(all_cats)

        Z_reference = list(itertools.product(*Z_categories))

        Z_vals_Do, _, N_o_j, N_o_jk, _ = get_counts_multiZ(Do, Z_cols, outcome, Z_reference=Z_reference)
        Z_vals_De, _, N_e_j, N_e_jk, _ = get_counts_multiZ(De, Z_cols, outcome, Z_reference=Z_reference)

        alpha_jk = np.ones_like(N_o_jk)
        alpha_j = np.sum(alpha_jk, axis=1)

        probs_Y_HZc, probs_Y_HZc_bar, probs_Y_obs_tmp = compute_posterior_predictive_both_hypotheses(
            N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j
        )

        Z_config_to_index = {tuple(z): i for i, z in enumerate(Z_vals_De)}
        probs_rows_HZc, probs_rows_HZc_bar = [], []

        for _, rtest in De_test.iterrows():
            z_tuple = tuple(rtest[col] for col in Z_cols)
            idx = Z_config_to_index.get(z_tuple)
            if idx is not None:
                probs_rows_HZc.append(probs_Y_HZc[idx])
                probs_rows_HZc_bar.append(probs_Y_HZc_bar[idx])
            else:
                probs_rows_HZc.append(np.ones(K) / K)
                probs_rows_HZc_bar.append(np.ones(K) / K)

        probs_rows_HZc = np.asarray(probs_rows_HZc)
        probs_rows_HZc_bar = np.asarray(probs_rows_HZc_bar)

        P_Y_alg_accum += w_c * probs_rows_HZc + w_cb * probs_rows_HZc_bar
    # Normalize BMA
    P_Y_alg = P_Y_alg_accum / np.clip(P_Y_alg_accum.sum(axis=1, keepdims=True), 1e-12, None)

    def bma_predict_single_source(D_train, D_test, df_scores, outcome, K=2):
        Ntest = len(D_test)
        P_Y_accum = np.zeros((Ntest, K))

        for _, row in df_scores.iterrows():
            Z_cols = list(row['Variables'])
            w = float(row['P(data|Z)'])

            # Build universal Z_reference
            Z_categories = []
            for col in Z_cols:
                cats = D_train[col].astype('category').cat.categories
                Z_categories.append(cats)

            Z_reference = list(itertools.product(*Z_categories))

            Z_vals, _, N_j, N_jk, _ = get_counts_multiZ(
                D_train, Z_cols, outcome, Z_reference=Z_reference
            )

            probs_Y = compute_posterior_predictive_single_model(
                N_jk, N_j)

            Z_config_to_index = {tuple(z): i for i, z in enumerate(Z_vals)}

            probs_rows = []
            for _, rtest in D_test.iterrows():
                z_tuple = tuple(rtest[col] for col in Z_cols)
                idx = Z_config_to_index.get(z_tuple)
                if idx is not None:
                    probs_rows.append(probs_Y[idx])
                else:
                    probs_rows.append(np.ones(K) / K)

            P_Y_accum += w * np.asarray(probs_rows)

        # Normalize across Y
        P_Y = P_Y_accum / np.clip(P_Y_accum.sum(axis=1, keepdims=True), 1e-12, None)

        return P_Y

    # print('df_scores', df_scores)
    # print('df_Score_De', df_Score_De)
    # print('df_Score_Do', df_Score_Do)
    # print('df_Score_Do_De', df_Score_Do_De)


    P_Y_Do = bma_predict_single_source(Do, De_test, df_Score_Do, outcome)
    P_Y_De = bma_predict_single_source(De, De_test, df_Score_De, outcome)
    P_Y_all = bma_predict_single_source(Do_De, De_test, df_Score_Do_De, outcome)

    P_Y_exp = np.asarray(P_Y_Do)
    P_Y_obs = np.asarray(P_Y_De)
    P_Y_all = np.asarray(P_Y_all)

    # Evaluate expected outcomes using the new function
    T0_obs, T1_obs, avgY1_obs, T0_exp, T1_exp, avgY1_exp, T0_alg, T1_alg, avgY1_alg, T0_all, T1_all, avgY1_all, DEU_alg, DEU_exp, DEU_obs, DEU_all = evaluate_expected_outcome(
        De_test=De_test,
        Do=Do,
        De=De,
        df_scores=df_scores,
        df_Score_De=df_Score_De,
        df_Score_Do=df_Score_Do,
        df_Score_Do_De=df_Score_Do_De,
        subsets=subsets,
        treatment=treatment,
        outcome=outcome

    )

    # Log-loss
    y_true = De_test[outcome].to_numpy()
    alg_loss = log_loss(y_true, P_Y_alg)
    exp_loss = log_loss(y_true, P_Y_exp)
    obs_loss = log_loss(y_true, P_Y_obs)
    all_loss = log_loss(y_true, P_Y_all)

    # ECE and we assume as the possitive outcome NO PONV (PONV=0)
    y_true_no_PONV = (y_true == 0).astype(int)
    ece_alg = expected_calibration_error(y_true_no_PONV, P_Y_alg[:, 0], n_bins=10)
    ece_exp = expected_calibration_error(y_true_no_PONV, P_Y_exp[:, 0], n_bins=10)
    ece_obs = expected_calibration_error(y_true_no_PONV, P_Y_obs[:, 0], n_bins=10)
    ece_all = expected_calibration_error(y_true_no_PONV, P_Y_all[:, 0], n_bins=10)

    # AUC and we assume as the possitive outcome NO PONV (PONV=0)
    y_true_no_PONV = (y_true == 0).astype(int)
    auc_alg = roc_auc_score(y_true_no_PONV, P_Y_alg[:, 0])
    auc_exp = roc_auc_score(y_true_no_PONV, P_Y_exp[:, 0])
    auc_obs = roc_auc_score(y_true_no_PONV, P_Y_obs[:, 0])
    auc_all = roc_auc_score(y_true_no_PONV, P_Y_all[:, 0])

    # build fold dataframe
    fold_df = pd.DataFrame({
        'patient_id': De_test.index,
        'y_true': y_true,
        'P_Y_alg': P_Y_alg[:, 0],
        'P_Y_exp': P_Y_exp[:, 0],
        'P_Y_obs': P_Y_obs[:, 0],
        'P_Y_all': P_Y_all[:, 0]
    })

    # add to list
    all_preds.append(fold_df)
    predictions_df = pd.concat(all_preds, ignore_index=True)

    # Pack results
    metrics = {
        "algorithm": dict(T0=T0_alg, T1=T1_alg, avgY1=avgY1_alg, logloss=alg_loss, ece=ece_alg, auc=auc_alg,
                          DEU=DEU_alg),
        "experimental": dict(T0=T0_exp, T1=T1_exp, avgY1=avgY1_exp, logloss=exp_loss, ece=ece_exp, auc=auc_exp,
                             DEU=DEU_exp),
        "observational": dict(T0=T0_obs, T1=T1_obs, avgY1=avgY1_obs, logloss=obs_loss, ece=ece_obs, auc=auc_obs,
                              DEU=DEU_obs),
        "all": dict(T0=T0_all, T1=T1_all, avgY1=avgY1_all, logloss=all_loss, ece=ece_all, auc=auc_all, DEU=DEU_all),
        "P_Y": dict(alg=P_Y_alg, exp=P_Y_exp, obs=P_Y_obs)
    }

    return metrics

def compute_posterior_predictive_both_hypotheses(N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j):
    """
    Returns:
    - probs_HZc: P(Y|do(X), Z, HZc) predictive probs under H_Z^c
    - probs_HZc_bar: P(Y|do(X), Z, HZc_bar) predictive probs under H̄_Z^c
    """
    numerator_HZc = N_o_jk + N_e_jk + alpha_jk
    denominator_HZc = (N_o_j + N_e_j + alpha_j)[:, np.newaxis]
    probs_Y_HZc = numerator_HZc / denominator_HZc

    numerator_HZc_bar = N_e_jk + alpha_jk
    denominator_HZc_bar = (N_e_j + alpha_j)[:, np.newaxis]
    probs_Y_HZc_bar = numerator_HZc_bar / denominator_HZc_bar

    numerator_obs = N_o_jk + alpha_jk
    denominator_obs = (N_o_j + alpha_j)[:, np.newaxis]
    probs_Y_obs = numerator_obs / denominator_obs

    return probs_Y_HZc, probs_Y_HZc_bar, probs_Y_obs


def compute_posterior_predictive_single(N_o_jk, N_o_j, N_e_jk, N_e_j, N_o_e_jk, N_o_e_j):
    """
    Returns:
    - probs_HZc: P(Y|do(X), Z, HZc) predictive probs under H_Z^c
    - probs_HZc_bar: P(Y|do(X), Z, HZc_bar) predictive probs under H̄_Z^c
    """
    alpha_o_jk = np.ones_like(N_o_jk)
    alpha_o_j = np.sum(alpha_o_jk, axis=1)

    alpha_e_jk = np.ones_like(N_e_jk)
    alpha_e_j = np.sum(alpha_e_jk, axis=1)

    alpha_o_e_jk = np.ones_like(N_o_e_jk)
    alpha_o_e_j = np.sum(alpha_o_e_jk, axis=1)

    numerator_HZc_bar = N_e_jk + alpha_e_jk
    denominator_HZc_bar = (N_e_j + alpha_e_j)[:, np.newaxis]
    probs_Y_HZc_bar = numerator_HZc_bar / denominator_HZc_bar

    numerator_obs = N_o_jk + alpha_o_jk
    denominator_obs = (N_o_j + alpha_o_j)[:, np.newaxis]
    probs_Y_obs = numerator_obs / denominator_obs

    numerator_all = N_o_e_jk + alpha_o_e_jk
    denominator_all = (N_o_e_j + alpha_o_e_j)[:, np.newaxis]
    probs_Y_all = numerator_all / denominator_all

    return probs_Y_HZc_bar, probs_Y_obs, probs_Y_all


def compute_posterior_predictive_single_model(N_jk, N_j):
    """
    Compute posterior predictive probabilities P(Y | Z) for a single dataset.

    Parameters
    ----------
    N_jk : array (J x K)
        Counts for each Z configuration j and outcome k
    N_j : array (J,)
        Total counts per Z configuration

    Returns
    -------
    probs_Y : array (J x K)
        Posterior predictive probabilities
    """
    alpha_jk = np.ones_like(N_jk)
    alpha_j = np.sum(alpha_jk, axis=1)

    numerator = N_jk + alpha_jk
    denominator = (N_j + alpha_j)[:, np.newaxis]

    probs_Y = numerator / denominator
    return probs_Y


def evaluate_expected_outcome(
        De_test, Do, De,
        df_scores, df_Score_De, df_Score_Do, df_Score_Do_De,
        subsets, treatment, outcome
):
    """
    Vectorized evaluation of expected outcome under optimal treatment.

    Returns:
    avgY_obs, avgY_exp, avgY_alg, avgY_all and DEUs
    """

    Do_De = pd.concat([Do, De], ignore_index=True)
    N = len(De_test)
    K = 2

    # Counterfactual datasets
    Dtest_T0 = De_test.copy()
    Dtest_T0[treatment] = 0
    Dtest_T1 = De_test.copy()
    Dtest_T1[treatment] = 1

    # --------------------------------------------------
    # (1) BMA with competing hypotheses (UNCHANGED)
    # --------------------------------------------------
    def compute_BMA_alg(De_input):
        P_Y_accum = np.zeros((N, K))

        score_map_c = {
            tuple(r['Variables']): r['P_HZ_c']
            for _, r in df_scores.iterrows()
        }

        score_map_cb = {
            tuple(r['Variables']): r['P_HZ_c_bar']
            for _, r in df_scores.iterrows()
        }

        for Z_cols in subsets:
            row = df_scores.loc[df_scores['Variables'] == tuple(Z_cols)]

            w_c = score_map_c.get(tuple(Z_cols), 0.0)
            w_cb = score_map_cb.get(tuple(Z_cols), 0.0)

            Z_categories = []
            for col in Z_cols:
                cats = sorted(
                    set(Do[col].astype('category').cat.categories) |
                    set(De[col].astype('category').cat.categories)
                )
                Z_categories.append(cats)

            Z_reference = list(itertools.product(*Z_categories))

            Z_vals_Do, _, N_o_j, N_o_jk, _ = get_counts_multiZ(Do, Z_cols, outcome, Z_reference)
            Z_vals_De, _, N_e_j, N_e_jk, _ = get_counts_multiZ(De, Z_cols, outcome, Z_reference)

            alpha_jk = np.ones_like(N_o_jk)
            alpha_j = np.sum(alpha_jk, axis=1)

            probs_HZc, probs_HZc_bar, _ = compute_posterior_predictive_both_hypotheses(
                N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j
            )

            Z_index = {tuple(z): i for i, z in enumerate(Z_vals_De)}

            rows_c, rows_cb = [], []
            for _, r in De_input.iterrows():
                idx = Z_index.get(tuple(r[col] for col in Z_cols))
                if idx is not None:
                    rows_c.append(probs_HZc[idx])
                    rows_cb.append(probs_HZc_bar[idx])
                else:
                    rows_c.append(np.ones(K) / K)
                    rows_cb.append(np.ones(K) / K)

            P_Y_accum += w_c * np.asarray(rows_c) + w_cb * np.asarray(rows_cb)

        return P_Y_accum / np.clip(P_Y_accum.sum(axis=1, keepdims=True), 1e-12, None)

    # --------------------------------------------------
    # (2) Generic single-dataset BMA
    # --------------------------------------------------
    def compute_BMA_single(D_train, De_input, df_score):
        P_Y_accum = np.zeros((N, K))

        for _, row in df_score.iterrows():
            Z_cols = list(row['Variables'])
            w = float(row['P(data|Z)'])

            Z_categories = []
            for col in Z_cols:
                cats = sorted(
                    set(D_train[col].astype('category').cat.categories) |
                    set(De_input[col].astype('category').cat.categories)
                )
                Z_categories.append(cats)

            Z_reference = list(itertools.product(*Z_categories))

            Z_vals, _, N_j, N_jk, _ = get_counts_multiZ(
                D_train, Z_cols, outcome, Z_reference
            )

            probs_Y = compute_posterior_predictive_single_model(N_jk, N_j)

            Z_index = {tuple(z): i for i, z in enumerate(Z_vals)}

            rows = []
            for _, r in De_input.iterrows():
                idx = Z_index.get(tuple(r[col] for col in Z_cols))
                if idx is not None:
                    rows.append(probs_Y[idx])
                else:
                    rows.append(np.ones(K) / K)

            P_Y_accum += w * np.asarray(rows)

        return P_Y_accum / np.clip(P_Y_accum.sum(axis=1, keepdims=True), 1e-12, None)

    # --------------------------------------------------
    # (3) Compute probabilities
    # --------------------------------------------------
    P_Y_alg_T0 = compute_BMA_alg(Dtest_T0)
    P_Y_alg_T1 = compute_BMA_alg(Dtest_T1)

    P_Y_exp_T0 = compute_BMA_single(De, Dtest_T0, df_Score_De)
    P_Y_exp_T1 = compute_BMA_single(De, Dtest_T1, df_Score_De)

    P_Y_obs_T0 = compute_BMA_single(Do, Dtest_T0, df_Score_Do)
    P_Y_obs_T1 = compute_BMA_single(Do, Dtest_T1, df_Score_Do)

    P_Y_all_T0 = compute_BMA_single(Do_De, Dtest_T0, df_Score_Do_De)
    P_Y_all_T1 = compute_BMA_single(Do_De, Dtest_T1, df_Score_Do_De)

    real_T = De_test[treatment].tolist()
    real_Y = De_test[outcome].tolist()

    # --------------------------------------------------
    # (4) Optimal treatment + DEU (UNCHANGED)
    # --------------------------------------------------
    def best_treatment_probs(P_T0, P_T1):
        """
        Compute best treatment per row, expected outcome (avgY0), and Direct Expected Utility (DEU),
        applying Laplace smoothing to avoid undefined probabilities.

        Inputs:
            P_T0, P_T1: predicted probabilities for outcome=0, shape (N, K)
            real_T: actual treatment received
            real_Y: actual outcome
            a, b: Laplace smoothing parameters
        """

        a = 0.1
        b = 0.2

        # Determine best treatment per row (maximize probability of Y=0)
        best_T = (P_T1[:, 0] > P_T0[:, 0]).astype(int)

        # Use best treatment probabilities for expected outcome
        best_probs = np.where(best_T[:, None] == 1, P_T1, P_T0)
        avgY0 = best_probs[:, 0].mean()

        # Count number of rows assigned to each treatment
        T0Count = int(np.sum(best_T == 0))
        T1Count = int(np.sum(best_T == 1))

        # Apply Laplace smoothing for P(BX=A_i)
        P_BX_0 = (T0Count + a) / (T0Count + T1Count + b)
        P_BX_1 = (T1Count + a) / (T0Count + T1Count + b)

        # Determine outcomes corresponding to the best treatment
        common_T = [r if r == t else None for r, t in zip(real_T, best_T)]  # None if mismatch

        # Extract Y values for rows where treatment matches best_T
        Y_T0 = [real_Y[i] for i, t in enumerate(common_T) if t == 0]
        Y_T1 = [real_Y[i] for i, t in enumerate(common_T) if t == 1]

        # Apply Laplace smoothing to conditional probabilities
        P_Y0_T0 = (Y_T0.count(0) + a) / (len(Y_T0) + b) if len(Y_T0) > 0 else a / b
        P_Y0_T1 = (Y_T1.count(0) + a) / (len(Y_T1) + b) if len(Y_T1) > 0 else a / b

        # Direct Expected Utility
        DEU = P_Y0_T0 * P_BX_0 + P_Y0_T1 * P_BX_1

        return T0Count, T1Count, avgY0, DEU

    y_true = De_test[outcome]
    # build fold dataframe
    fold_df_DEU = pd.DataFrame({
        'patient_id': De_test.index,
        'y_true': y_true,
        'real_T': real_T,
        'P_Y_alg_T0': P_Y_alg_T0[:, 0],
        'P_Y_alg_T1': P_Y_alg_T1[:, 0],
        'P_Y_exp_T0': P_Y_exp_T0[:, 0],
        'P_Y_exp_T1': P_Y_exp_T1[:, 0],
        'P_Y_obs_T0': P_Y_obs_T0[:, 0],
        'P_Y_obs_T1': P_Y_obs_T1[:, 0],
        'P_Y_all_T0': P_Y_all_T0[:, 0],
        'P_Y_all_T1': P_Y_all_T1[:, 0]
    })

    all_preds_DEU.append(fold_df_DEU)
    predictions_DEU = pd.concat(all_preds_DEU, ignore_index=True)

    T0_alg, T1_alg, avgY_alg, DEU_alg = best_treatment_probs(P_Y_alg_T0, P_Y_alg_T1)
    T0_exp, T1_exp, avgY_exp, DEU_exp = best_treatment_probs(P_Y_exp_T0, P_Y_exp_T1)
    T0_obs, T1_obs, avgY_obs, DEU_obs = best_treatment_probs(P_Y_obs_T0, P_Y_obs_T1)
    T0_all, T1_all, avgY_all, DEU_all = best_treatment_probs(P_Y_all_T0, P_Y_all_T1)

    return (
        T0_obs, T1_obs, avgY_obs,
        T0_exp, T1_exp, avgY_exp,
        T0_alg, T1_alg, avgY_alg,
        T0_all, T1_all, avgY_all,
        DEU_alg, DEU_exp, DEU_obs, DEU_all
    )


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        bin_size = np.sum(mask)
        if bin_size > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (bin_size / N) * np.abs(acc - conf)

    return ece
