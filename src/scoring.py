import numpy as np
from scipy.special import gammaln

def dirichlet_bayesian_score(counts, priors=None):
    counts = np.asarray(counts)

    if priors is None:
        priors = np.zeros_like(counts)

    N = np.sum(counts, axis=-1)
    sum_priors = np.sum(priors + 1, axis=-1)

    score = (
        gammaln(sum_priors)
        - np.sum(gammaln(priors + 1), axis=-1)
        + np.sum(gammaln(counts + priors + 1), axis=-1)
        - gammaln(N + sum_priors)
    )

    return np.sum(score)

def P_De_given_HZc_log(N_o_jk, N_e_jk, priors):
    return dirichlet_bayesian_score(N_o_jk + N_e_jk, priors) - \
           dirichlet_bayesian_score(N_o_jk, priors)

def P_De_given_HZc_bar_log(N_e_jk, priors):
    return dirichlet_bayesian_score(N_e_jk, priors)