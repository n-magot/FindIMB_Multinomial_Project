"""
Main script to reproduce experiments.

Steps:
1. Load and preprocess data
2. Learn subset scores (Do, De, Do+De)
3. Run FindIMB forward search
4. Perform cross-validation
5. Compute metrics (DEU, AUC, ECE, LogLoss)
6. Save results

In this document Eq.7b new updated and FindIMB implemented:
    For 𝐔* = MB(Y), 𝐕 ′ = 𝐕 ⋃ 𝑋, 𝐙 ⊆ 𝐔* and hen the full set is assumed as MB(Y), then 𝐕 ′ = 𝐔*
    and the 2nd term of Eq.7b_new is equal to 1. So we have:

        (a) 𝑃(𝐷𝑜|𝐻𝐙𝑐) = 𝑃(𝐷𝑜|𝐻_𝐔*) for 𝐙 = 𝐔*; otherwise 0.
        (b) 𝑃(𝐷𝑜|𝐻𝐙c̅) = 𝑃(𝐷𝑜|𝐻_𝐔*) *1* (1/ ( |𝐕 ′|-|𝐙| + 1)) for 𝐙 ⊆ 𝐔*; otherwise 0.

@author: LELOVA
"""

import pandas as pd
from sklearn.model_selection import KFold
from datetime import datetime
import time

# --- Your modules ---
from src.preprocessing import create_synthetic_data, check_for_continuous_variables
from src.search import (
    greedy_search_single_dataset_forward,
    greedy_search_FindIMB_forward
)
from src.bma import bma_predict_and_evaluate


# ======================================================
# CONFIGURATION
# ======================================================
TREATMENT = "treatment"
OUTCOME = "outcome"
N_SPLITS = 2
THRESHOLD = 0.1
SEED = 42


# ======================================================
# MAIN PIPELINE
# ======================================================
def run_experiment(Do, De_all):

    start_time = time.time()

    covariates = [c for c in Do.columns if c not in [OUTCOME, TREATMENT]]

    # -----------------------------------
    # 2. Learn observational model ONCE
    # -----------------------------------
    print("Learning observational model...")
    df_Score_Do = greedy_search_single_dataset_forward(
        Do, TREATMENT, covariates, OUTCOME, threshold=THRESHOLD
    )

    # -----------------------------------
    # 3. Cross-validation
    # -----------------------------------
    kf = KFold(n_splits=N_SPLITS, shuffle=False)

    results = []
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(De_all)):

        print(f"\nFold {fold+1}/{N_SPLITS}")

        De = De_all.iloc[train_idx]
        De_test = De_all.iloc[test_idx]

        ATEforRx0 = (De_test.loc[De_test[TREATMENT] == 0, OUTCOME] == 0).mean()
        ATEforRx1 = (De_test.loc[De_test[TREATMENT] == 1, OUTCOME] == 0).mean()

        Do_De = pd.concat([Do, De], ignore_index=True)

        # -----------------------------------
        # 3a. Learn models
        # -----------------------------------
        df_Score_De = greedy_search_single_dataset_forward(
            De, TREATMENT, covariates, OUTCOME, threshold=THRESHOLD
        )

        df_Score_Do_De = greedy_search_single_dataset_forward(
            Do_De, TREATMENT, covariates, OUTCOME, threshold=THRESHOLD
        )

        # -----------------------------------
        # 3b. FindIMB
        # -----------------------------------
        df_scores, T = greedy_search_FindIMB_forward(
            Do, De, TREATMENT, covariates, OUTCOME, threshold=THRESHOLD
        )
        print(df_scores)

        # -----------------------------------
        # 3c. BMA prediction + evaluation
        # -----------------------------------
        metrics = bma_predict_and_evaluate(
            Do, De, De_test,
            df_scores,
            TREATMENT, OUTCOME,
            df_Score_De,
            df_Score_Do,
            df_Score_Do_De
        )

        # -----------------------------------
        # 3d. Store results
        # -----------------------------------
        results.append({
            "fold": fold + 1,
            "DEU_alg": metrics["algorithm"]["DEU"],
            "DEU_exp": metrics['experimental']['DEU'],
            "DEU_obs": metrics['observational']['DEU'],
            "DEU_all": metrics['all']['DEU'],
            "AUC_alg": metrics["algorithm"]["auc"],
            "AUC_exp": metrics['experimental']['auc'],
            "AUC_obs": metrics['observational']['auc'],
            "AUC_all": metrics['all']['auc'],
            "logloss_alg": metrics["algorithm"]["logloss"],
            "logloss_exp": metrics['experimental']['logloss'],
            "logloss_obs": metrics['observational']['logloss'],
            "logloss_all": metrics['all']['logloss'],
            "ece_alg": metrics["algorithm"]["ece"],
            "ece_exp": metrics['experimental']['ece'],
            "ece_obs": metrics['observational']['ece'],
            "ece_all": metrics['all']['ece']
        })

    # -----------------------------------
    # 4. Save results
    # -----------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)

    print(results_df)

    # results_df.to_csv(f"results_{timestamp}.csv", index=False)

    print("\nDone!")
    print(f"Total time: {time.time() - start_time:.2f} sec")

    return results_df

if __name__ == "__main__":

    Do, De_all = create_synthetic_data()

    check_for_continuous_variables(Do, exclude=[OUTCOME, TREATMENT])
    check_for_continuous_variables(De_all, exclude=[OUTCOME, TREATMENT])

    results = run_experiment(Do, De_all)

    print("\nFINAL RESULTS:")
    print(results[["fold", "logloss_alg", "logloss_exp", "logloss_obs"]])
