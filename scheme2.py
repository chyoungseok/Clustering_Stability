import numpy as np
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist

def scheme2(df, nk, B=20, r=5):
    results = {}
    n = df.shape[0]
    # %%
    # define a matrix of cluster memberships
    clst_mat = np.empty((B + 1, n))

    km = kmeans(df, nk, iter=r)
    center = km[0]
    clst = km[1]

    clst_mat[0, :] = clst

    stab_matrix = np.empty((n, B))

    for b in range(B):
        resample = np.random.choice(n, size=n, replace=True)

        df_star = df[resample, :]
        km_star = kmeans(df_star, nk, iter=r)
        center_star = km_star[0]

        class_star = cdist(center_star, df, metric='euclidean').argmin(axis=0)
        clst_mat[b + 1, :] = class_star

    B1 = B + 1
    agree_mat = np.empty((B1, B1))
    np.fill_diagonal(agree_mat, 1)

    for i in range(B1 - 1):
        for j in range(i + 1, B1):
            agree_mat[i, j] = np.mean(clst_mat[i, :] == clst_mat[j, :])
            agree_mat[j, i] = agree_mat[i, j]

    mean_agr = np.mean(agree_mat, axis=1)
    ref = np.argmax(mean_agr)
    clst_ref = clst_mat[ref, :]

    stab_mat = np.empty((B1, n))

    for i in range(B1):
        stab_mat[i, :] = (clst_ref == clst_mat[i, :]).mean()

    results['membership'] = clst_ref
    results['obs_wise'] = np.mean(stab_mat, axis=0)
    results['cluster_matrix'] = clst_mat
    results['agree_matrix'] = agree_mat
    results['ref_cluster'] = ref
    return results