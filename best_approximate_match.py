import numpy as np
from scipy.spatial import cKDTree
from globals import *

def best_approximate_match(A_features, A_pyramid, B_pyramid, B_features, l, i, j):
    '''
    cDKTree 需要的是N*d， N个点 d维
    (i, j, k) 有k页，每一页有 i*j 个像素，所以应该是 k * (i * j)维 25 * 896维
    但是 有i*j 个点，每个点的特征数为k, 所以应该是 (i*j) * k 896*25 维
    '''

    ## cDKTree 需要二维
    A_l = A_features[l].reshape((A_features[l].shape[0] * A_features[l].shape[1], A_features[l].shape[2]))
    tree = cKDTree(A_l)

    # Get the query point from B_features
    query_pnt = B_features[l][i, j, :] #(1,25) 应该是

    # print("shape of query_pnt: ", query_pnt.shape)

    # Find the index of the best_match in A_features
    idx = tree.query(query_pnt, k = 1)[1]

    # Get the dimensions of A_pyramid
    A_h, A_w, _ = A_pyramid[l].shape

    best_app_i, best_app_j = np.unravel_index(idx, (A_h, A_w))

    return best_app_i, best_app_j

# brute force just to sanity-check ANN (implement with cKDTree)
def brute_force_check(A_features, B_features, l, i, j):
    query_pnt = B_features[l][i, j, :]

    # initalize the minimum distance
    min_dist = float('inf')
    min_idx = -1

    num_features = A_features[l].shape[0]

    for ii in range(num_features):
        dist = np.sum((A_features[l][ii, :] - query_pnt) ** 2)

        if dist < min_dist:
            min_dist = dist
            min_idx = ii

    return min_idx, min_dist