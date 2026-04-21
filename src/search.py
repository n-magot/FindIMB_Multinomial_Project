"""
Search algorithms for selecting covariate subsets.

Includes:
- Forward search for single dataset
- FindIMB forward search (core method)
"""

import numpy as np
import pandas as pd
import itertools
from scipy.special import logsumexp

from src.counting import get_counts_multiZ
from src.scoring import P_De_given_HZc_log, P_De_given_HZc_bar_log


def make_subset(X_col, Z_iterable):
    return (X_col,) + tuple(sorted(set(Z_iterable)))


def greedy_search_FindIMB_forward(
        Do, De, X_col, Z_cols, Y_col,
        threshold=0.1, priors_val=1
):
    Z_cols = tuple(Z_cols)
    FS_vars = make_subset(X_col, Z_cols)

    # --- ALWAYS canonical ---
    list_to_invest = [make_subset(X_col, [])]

    log_P_HZc = {}
    log_P_HZc_bar = {}

    # --- Cache categories ---
    cat_cache = {}
    for col in FS_vars:
        cats_Do = Do[col].astype('category').cat.categories
        cats_De = De[col].astype('category').cat.categories
        cat_cache[col] = sorted(set(cats_Do) | set(cats_De))

    # --- Cache Z references (canonical keys!) ---
    Zref_cache = {}

    def get_Z_reference(Z_subset):
        Z_subset = make_subset(X_col, Z_subset[1:])
        if Z_subset not in Zref_cache:
            Z_categories = [cat_cache[col] for col in Z_subset]
            Zref_cache[Z_subset] = list(itertools.product(*Z_categories))
        return Zref_cache[Z_subset]

    # --- INITIAL NODE ---
    base_subset = make_subset(X_col, [])
    Z_reference = get_Z_reference(base_subset)

    _, _, _, N_o_jk, _ = get_counts_multiZ(
        Do, list(base_subset), Y_col, Z_reference=Z_reference
    )

    _, _, _, N_e_jk, _ = get_counts_multiZ(
        De, list(base_subset), Y_col, Z_reference=Z_reference
    )

    priors = np.full_like(N_o_jk, priors_val - 1)

    log_P_HZc[base_subset] = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
    log_P_HZc_bar[base_subset] = P_De_given_HZc_bar_log(N_e_jk, priors)

    # -----------------------------------
    # Forward layered search
    # -----------------------------------
    for _ in range(len(Z_cols)):

        next_layer = set()

        for subset in list_to_invest:

            used_Zs = set(subset[1:])
            remaining = [z for z in Z_cols if z not in used_Zs]

            for z in remaining:
                new_subset = make_subset(X_col, list(used_Zs) + [z])
                next_layer.add(new_subset)

        list_to_invest = []

        for Z_subset in next_layer:

            if Z_subset in log_P_HZc:
                continue

            Z_reference = get_Z_reference(Z_subset)

            _, _, _, N_o_jk, _ = get_counts_multiZ(
                Do, list(Z_subset), Y_col,
                Z_reference=Z_reference
            )

            _, _, _, N_e_jk, _ = get_counts_multiZ(
                De, list(Z_subset), Y_col,
                Z_reference=Z_reference
            )

            priors = np.full_like(N_o_jk, priors_val - 1)

            log_P_HZc[Z_subset] = P_De_given_HZc_log(
                N_o_jk, N_e_jk, priors
            )

            log_P_HZc_bar[Z_subset] = P_De_given_HZc_bar_log(
                N_e_jk, priors
            )

            list_to_invest.append(Z_subset)

        if not log_P_HZc:
            break

        # --- Global normalization ---
        log_total = logsumexp([
            np.logaddexp(log_P_HZc[z], log_P_HZc_bar[z])
            for z in log_P_HZc
        ])

        Scores_HZc = {
            z: np.exp(log_P_HZc[z] - log_total)
            for z in log_P_HZc
        }

        Scores_HZc_bar = {
            z: np.exp(log_P_HZc_bar[z] - log_total)
            for z in log_P_HZc_bar
        }

        list_to_invest = [
            z for z in list_to_invest
            if Scores_HZc[z] > threshold
               or Scores_HZc_bar[z] > threshold
        ]

        if not list_to_invest:
            break

    # -----------------------------------
    # Final dataframe
    # -----------------------------------
    df_results = pd.DataFrame({
        'Variables': list(log_P_HZc.keys()),
        'P(De|Do,HcZ) log': list(log_P_HZc.values()),
        'P(De|Do,HcZ_bar) log': list(log_P_HZc_bar.values())
    })

    len_MB = len(FS_vars)

    log_total = logsumexp([
        np.logaddexp(
            log_P_HZc[z] if z == FS_vars else -np.inf,
            log_P_HZc_bar[z] + np.log(1 / (len_MB - len(z) + 1))
        )
        for z in log_P_HZc
    ])
    print("Full set of variables:", FS_vars)

    df_results['P_HZ_c'] = df_results['Variables'].apply(
        lambda z: float(np.exp(log_P_HZc[z] - log_total)) if z == FS_vars else 0.0
    )

    df_results['P_HZ_c_bar'] = df_results['Variables'].apply(
        lambda z: float(
            np.exp(
                log_P_HZc_bar[z]
                + np.log(1 / (len_MB - len(z) + 1))
                - log_total
            )
        )
    )

    return df_results


def greedy_search_single_dataset_forward(
        data, X_col, Z_cols, Y_col,
        threshold=0.1, priors_val=1
):
    Z_cols = tuple(Z_cols)
    FS_vars = tuple([X_col] + list(Z_cols))

    # Start with X only
    list_to_invest = [(X_col,)]
    log_P_data = {}

    # --- Cache category levels ---
    cat_cache = {
        col: data[col].astype('category').cat.categories
        for col in FS_vars
    }

    # --- Cache Z references ---
    Zref_cache = {}

    def get_Z_reference(Z_subset):
        if Z_subset not in Zref_cache:
            Z_categories = [cat_cache[col] for col in Z_subset]
            Zref_cache[Z_subset] = list(itertools.product(*Z_categories))
        return Zref_cache[Z_subset]

    # -----------------------------------
    # 🔹 NEW: Evaluate (X_col,) first
    # -----------------------------------
    Z_reference = get_Z_reference((X_col,))

    _, _, _, N_jk, _ = get_counts_multiZ(
        data, [X_col], Y_col,
        Z_reference=Z_reference
    )

    priors = np.full_like(N_jk, priors_val - 1)
    log_P_data[(X_col,)] = P_De_given_HZc_bar_log(N_jk, priors)

    # -----------------------------------
    # Forward layered search
    # -----------------------------------
    for _ in range(len(Z_cols)):

        next_layer = []

        for subset in list_to_invest:

            used_Zs = set(subset[1:])
            remaining = [z for z in Z_cols if z not in used_Zs]

            for z in remaining:
                new_subset = (X_col,) + tuple(
                    list(subset[1:]) + [z]
                )
                next_layer.append(new_subset)

        next_layer = list(set(next_layer))
        list_to_invest = []

        for Z_subset in next_layer:

            if Z_subset in log_P_data:
                continue

            Z_reference = get_Z_reference(Z_subset)

            _, _, _, N_jk, _ = get_counts_multiZ(
                data, list(Z_subset), Y_col,
                Z_reference=Z_reference
            )

            priors = np.full_like(N_jk, priors_val - 1)
            log_P_data[Z_subset] = P_De_given_HZc_bar_log(N_jk, priors)

            list_to_invest.append(Z_subset)

        if not log_P_data:
            break

        # --- Global normalization (unchanged) ---
        log_vals = np.array(list(log_P_data.values()))
        log_total = logsumexp(log_vals)

        Scores = {
            z: np.exp(v - log_total)
            for z, v in log_P_data.items()
        }

        list_to_invest = [
            z for z in list_to_invest
            if Scores[z] > threshold
        ]

        if not list_to_invest:
            break

    # -----------------------------------
    # Final results
    # -----------------------------------
    df_results = pd.DataFrame({
        'Variables': list(log_P_data.keys()),
        'logP(data|Z)': list(log_P_data.values())
    })

    log_total = logsumexp(np.array(list(log_P_data.values())))
    df_results['P(data|Z)'] = np.exp(
        df_results['logP(data|Z)'] - log_total
    )

    return df_results
