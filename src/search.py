"""
Search algorithms for selecting covariate subsets.

Includes:
- Forward search for single dataset
- FindIMB forward search (core method)
"""

import numpy as np
import pandas as pd
import itertools
from scipy.special import logsumexp, comb

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

    list_to_invest = [make_subset(X_col, [])]

    log_P_HZc     = {}
    log_P_HZc_bar = {}
    log_P_HZo     = {}  # NEW

    # --- Build tree T as parent->children dict ---
    T = {}  # NEW

    # --- Cache categories ---
    cat_cache = {}
    for col in FS_vars:
        cats_Do = Do[col].astype('category').cat.categories
        cats_De = De[col].astype('category').cat.categories
        cat_cache[col] = sorted(set(cats_Do) | set(cats_De))

    # --- Cache Z references ---
    Zref_cache = {}

    def get_Z_reference(Z_subset):
        Z_subset = make_subset(X_col, Z_subset[1:])
        if Z_subset not in Zref_cache:
            Z_categories = [cat_cache[col] for col in Z_subset]
            Zref_cache[Z_subset] = list(itertools.product(*Z_categories))
        return Zref_cache[Z_subset]

    # --- INITIAL NODE ---
    base_subset   = make_subset(X_col, [])
    Z_reference   = get_Z_reference(base_subset)

    _, _, _, N_o_jk, _ = get_counts_multiZ(
        Do, list(base_subset), Y_col, Z_reference=Z_reference
    )
    _, _, _, N_e_jk, _ = get_counts_multiZ(
        De, list(base_subset), Y_col, Z_reference=Z_reference
    )

    priors = np.full_like(N_o_jk, priors_val - 1)

    log_P_HZc[base_subset]     = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
    log_P_HZc_bar[base_subset] = P_De_given_HZc_bar_log(N_e_jk, priors)
    log_P_HZo[base_subset]     = P_De_given_HZc_bar_log(N_o_jk, priors)  # NEW
    T[base_subset]             = []  # NEW

    # -----------------------------------
    # Forward layered search
    # -----------------------------------
    for _ in range(len(Z_cols)):

        next_layer = set()

        for subset in list_to_invest:
            used_Zs   = set(subset[1:])
            remaining = [z for z in Z_cols if z not in used_Zs]
            for z in remaining:
                new_subset = make_subset(X_col, list(used_Zs) + [z])
                next_layer.add((new_subset, subset))  # CHANGED: track parent

        list_to_invest = []

        for Z_subset, parent in next_layer:  # CHANGED: unpack parent

            if Z_subset in log_P_HZc:
                continue

            Z_reference = get_Z_reference(Z_subset)

            _, _, _, N_o_jk, _ = get_counts_multiZ(
                Do, list(Z_subset), Y_col, Z_reference=Z_reference
            )
            _, _, _, N_e_jk, _ = get_counts_multiZ(
                De, list(Z_subset), Y_col, Z_reference=Z_reference
            )

            priors = np.full_like(N_o_jk, priors_val - 1)

            log_P_HZc[Z_subset]     = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
            log_P_HZc_bar[Z_subset] = P_De_given_HZc_bar_log(N_e_jk, priors)
            log_P_HZo[Z_subset]     = P_De_given_HZc_bar_log(N_o_jk, priors)  # NEW

            T[parent].append(Z_subset)  # NEW: record edge
            T[Z_subset] = []            # NEW: initialise child list

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

        # NEW: running normalization for HZo over all subsets seen so far
        log_total_HZo = logsumexp(list(log_P_HZo.values()))
        Scores_HZo = {
            z: np.exp(log_P_HZo[z] - log_total_HZo)
            for z in log_P_HZo
        }

        list_to_invest = [
            z for z in list_to_invest
            if Scores_HZc[z]     > threshold
            or Scores_HZc_bar[z] > threshold
            or Scores_HZo[z]     > threshold  # NEW
        ]

        if not list_to_invest:
            break

    # -----------------------------------
    # Final phase: compute Equation 3
    # -----------------------------------

    def get_subtree_node_sets(node):
        """Return S_Z: the node sets of all nodes in the subtree rooted at node."""
        result = [node]
        for child in T[node]:
            result.extend(get_subtree_node_sets(child))
        return result

    V_size = len(Z_cols)

    def compute_log_P_HZo_bar(Z_subset):
        """Approximate P(Do|HoZ_bar) using the subtree S_Z."""
        S_Z   = get_subtree_node_sets(Z_subset)
        Z_len = len(Z_subset) - 1  # exclude X_col
        denom = V_size - Z_len     # |V| - |Z|

        log_terms = []
        for U in S_Z:
            U_len     = len(U) - 1  # exclude X_col
            diff      = U_len - Z_len
            binom     = comb(denom, diff, exact=True)
            log_w     = -np.log(binom) - np.log(denom + 1)
            log_terms.append(log_P_HZo[U] + log_w)

        return logsumexp(log_terms)

    # Compute log numerators and log P(Do|HoZ_bar) for all nodes
    log_numerators_Hz   = {}
    log_numerators_Hz_bar   = {}
    log_P_HZo_bar    = {}

    for Z_subset in log_P_HZc:
        log_numerators_Hz[Z_subset] = log_P_HZc[Z_subset] + log_P_HZo[Z_subset]
        log_P_HZo_bar[Z_subset]  = compute_log_P_HZo_bar(Z_subset)
        log_numerators_Hz_bar[Z_subset] = log_P_HZc_bar[Z_subset] + log_P_HZo_bar[Z_subset]


    # Compute log denominator: sum over all Z of (HZc*HZo + HZc_bar*HZo_bar)
    log_denom_terms = []
    for Z_subset in log_P_HZc:
        log_term_c    = log_P_HZc[Z_subset]     + log_P_HZo[Z_subset]
        log_term_cbar = log_P_HZc_bar[Z_subset] + log_P_HZo_bar[Z_subset]
        log_denom_terms.append(np.logaddexp(log_term_c, log_term_cbar))

    log_denominator = logsumexp(log_denom_terms)

    # Equation 3 scores P(Hzc)
    P_HZ_c = {
        z: np.exp(log_numerators_Hz[z] - log_denominator)
        for z in log_P_HZc
    }

    # Equation 3 scores P(Hzc_bar)
    P_HZ_c_bar = {
        z: np.exp(log_numerators_Hz_bar[z] - log_denominator)
        for z in log_P_HZc_bar
    }

    # -----------------------------------
    # Final dataframe
    # -----------------------------------
    df_results = pd.DataFrame({
        'Variables':                list(log_P_HZc.keys()),
        'P(De|Do,HcZ) log':         list(log_P_HZc.values()),
        'P(De|Do,HcZ_bar) log':     list(log_P_HZc_bar.values()),
        'P(Do|HoZ) log':            list(log_P_HZo.values()),
        'P(Do|HoZ_bar) log':        [log_P_HZo_bar[z] for z in log_P_HZc],
        'P_HZ_c':                   [P_HZ_c[z]     for z in log_P_HZc],
        'P_HZ_c_bar':               [P_HZ_c_bar[z] for z in log_P_HZc_bar]
    })

    return df_results, T

# def greedy_search_FindIMB_forward(
#         Do, De, X_col, Z_cols, Y_col,
#         threshold=0.1, priors_val=1
# ):
#     Z_cols = tuple(Z_cols)
#     FS_vars = make_subset(X_col, Z_cols)
#
#     # --- ALWAYS canonical ---
#     list_to_invest = [make_subset(X_col, [])]
#
#     log_P_HZc = {}
#     log_P_HZc_bar = {}
#     log_P_HZo = {}  # NEW: P(D_o | H_Z^o)
#
#     # --- Cache categories ---
#     cat_cache = {}
#     for col in FS_vars:
#         cats_Do = Do[col].astype('category').cat.categories
#         cats_De = De[col].astype('category').cat.categories
#         cat_cache[col] = sorted(set(cats_Do) | set(cats_De))
#
#     # --- Cache Z references (canonical keys!) ---
#     Zref_cache = {}
#
#     def get_Z_reference(Z_subset):
#         Z_subset = make_subset(X_col, Z_subset[1:])
#         if Z_subset not in Zref_cache:
#             Z_categories = [cat_cache[col] for col in Z_subset]
#             Zref_cache[Z_subset] = list(itertools.product(*Z_categories))
#         return Zref_cache[Z_subset]
#
#     # --- INITIAL NODE ---
#     base_subset = make_subset(X_col, [])
#     Z_reference = get_Z_reference(base_subset)
#
#     _, _, _, N_o_jk, _ = get_counts_multiZ(
#         Do, list(base_subset), Y_col, Z_reference=Z_reference
#     )
#
#     _, _, _, N_e_jk, _ = get_counts_multiZ(
#         De, list(base_subset), Y_col, Z_reference=Z_reference
#     )
#
#     priors = np.full_like(N_o_jk, priors_val - 1)
#
#     log_P_HZc[base_subset] = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
#     log_P_HZc_bar[base_subset] = P_De_given_HZc_bar_log(N_e_jk, priors)
#     log_P_HZo[base_subset] = P_De_given_HZc_bar_log(N_o_jk, priors)  # NEW
#
#
#     # -----------------------------------
#     # Forward layered search
#     # -----------------------------------
#     for _ in range(len(Z_cols)):
#
#         next_layer = set()
#
#         for subset in list_to_invest:
#
#             used_Zs = set(subset[1:])
#             remaining = [z for z in Z_cols if z not in used_Zs]
#
#             for z in remaining:
#                 new_subset = make_subset(X_col, list(used_Zs) + [z])
#                 next_layer.add(new_subset)
#
#         list_to_invest = []
#
#         for Z_subset in next_layer:
#
#             if Z_subset in log_P_HZc:
#                 continue
#
#             Z_reference = get_Z_reference(Z_subset)
#
#             _, _, _, N_o_jk, _ = get_counts_multiZ(
#                 Do, list(Z_subset), Y_col,
#                 Z_reference=Z_reference
#             )
#
#             _, _, _, N_e_jk, _ = get_counts_multiZ(
#                 De, list(Z_subset), Y_col,
#                 Z_reference=Z_reference
#             )
#
#             priors = np.full_like(N_o_jk, priors_val - 1)
#
#             log_P_HZc[Z_subset] = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
#             log_P_HZc_bar[Z_subset] = P_De_given_HZc_bar_log(N_e_jk, priors)
#             log_P_HZo[Z_subset] = P_De_given_HZc_bar_log(N_o_jk, priors)  # NEW
#
#             list_to_invest.append(Z_subset)
#
#         if not log_P_HZc:
#             break
#
#         # --- Global normalization ---
#         log_total = logsumexp([
#             np.logaddexp(log_P_HZc[z], log_P_HZc_bar[z])
#             for z in log_P_HZc
#         ])
#
#         Scores_HZc = {
#             z: np.exp(log_P_HZc[z] - log_total)
#             for z in log_P_HZc
#         }
#
#         Scores_HZc_bar = {
#             z: np.exp(log_P_HZc_bar[z] - log_total)
#             for z in log_P_HZc_bar
#         }
#
#         # NEW: running normalization for HZo over all subsets seen so far
#         log_total_HZo = logsumexp(list(log_P_HZo.values()))
#         Scores_HZo = {
#             z: np.exp(log_P_HZo[z] - log_total_HZo)
#             for z in log_P_HZo
#         }
#
#         list_to_invest = [
#             z for z in list_to_invest
#             if Scores_HZc[z] > threshold
#                or Scores_HZc_bar[z] > threshold
#                or Scores_HZo[z] > threshold  # NEW
#         ]
#
#         if not list_to_invest:
#             break
#
#     # -----------------------------------
#     # Final dataframe
#     # -----------------------------------
#     df_results = pd.DataFrame({
#         'Variables': list(log_P_HZc.keys()),
#         'P(De|Do,HcZ) log': list(log_P_HZc.values()),
#         'P(De|Do,HcZ_bar) log': list(log_P_HZc_bar.values()),
#         'P(Do|HoZ) log': list(log_P_HZo.values())  # NEW
#     })
#
#     """Building T as a parent→children dict during the search, recording edges when a subset is first computed
#         The final phase where for each node Z in T:
#             - Traverse T downward from Z to collect S_Z
#             - Compute P(Do|HoZ_bar) using the approximation formula with U ∈ S_Z
#             - Compute the Equation 3 score using P(De|Do,HcZ)*P(Do|HoZ) in the numerator and summing both terms over
#              all Z in T for the denominator
#
#         Return a dataframe with one row per node, with columns for numerator, denominator, and final score"""
#
#     len_MB = len(FS_vars)
#
#     log_total = logsumexp([
#         np.logaddexp(
#             log_P_HZc[z] if z == FS_vars else -np.inf,
#             log_P_HZc_bar[z] + np.log(1 / (len_MB - len(z) + 1))
#         )
#         for z in log_P_HZc
#     ])
#     print("Full set of variables:", FS_vars)
#
#     df_results['P_HZ_c'] = df_results['Variables'].apply(
#         lambda z: float(np.exp(log_P_HZc[z] - log_total)) if z == FS_vars else 0.0
#     )
#
#     df_results['P_HZ_c_bar'] = df_results['Variables'].apply(
#         lambda z: float(
#             np.exp(
#                 log_P_HZc_bar[z]
#                 + np.log(1 / (len_MB - len(z) + 1))
#                 - log_total
#             )
#         )
#     )
#
#     return df_results


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
