import numpy as np
import pandas as pd
import itertools

def get_counts_multiZ(df, Z_cols, Y_col, Z_reference=None):
    df = df.copy()

    for col in Z_cols:
        df[col] = df[col].astype('category')

    df[Y_col] = df[Y_col].astype('category')
    Y_values = df[Y_col].cat.categories.tolist()

    # If no external reference provided, use all combos from this df
    if Z_reference is None:
        Z_categories = [sorted(df[col].cat.categories) for col in Z_cols]
        Z_values = list(itertools.product(*Z_categories))
    else:
        # Use the pre-defined reference (same across datasets)
        Z_values = Z_reference

    index_map = {z: i for i, z in enumerate(Z_values)}
    y_map = {y: i for i, y in enumerate(Y_values)}

    N_j = np.zeros(len(Z_values), dtype=int)
    N_jk = np.zeros((len(Z_values), len(Y_values)), dtype=int)

    for _, row in df.iterrows():
        z_tuple = tuple(row[Z_cols])
        y_val = row[Y_col]
        if any(pd.isnull(v) for v in z_tuple) or pd.isnull(y_val):
            continue
        if z_tuple in index_map:
            i = index_map[z_tuple]
            k = y_map[y_val]
            N_j[i] += 1
            N_jk[i, k] += 1

    df_counts = pd.DataFrame(N_jk, index=Z_values, columns=Y_values)
    df_counts.index.name = "Z_configuration"
    df_counts["Total"] = N_j

    return Z_values, Y_values, N_j, N_jk, df_counts