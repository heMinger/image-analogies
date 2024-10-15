import numpy as np
from concat_feature import concat_feature
from globals import *

def best_coherence_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s_pyramid, l, L, i, j):
    # BEST_COHERENCE_MATCH
    # (i,j) are q in the paper
    A_h, A_w, _ = A_pyramid[l].shape
    border_big = int(np.floor(N_BIG / 2))

    F_q = concat_feature(B_pyramid, B_prime_pyramid, l, i, j, L)

    min_dist = np.inf
    r_star_i = -1
    r_star_j = -1
    done = False

    # Loop over neighborhood
    for ii in range(i-border_big, i+border_big+1):
        for jj in range(j-border_big, j+border_big+1):
            # Skip the pixel itself
            if ii == i and jj == j:
                done = True
                break

            s_i = s_pyramid[l][ii, jj, 0]
            s_j = s_pyramid[l][ii, jj, 1]

            F_sr_i = s_i + (i - ii)
            F_sr_j = s_j + (j - jj)

            # Check for out-of-bounds
            if (F_sr_i >= A_h or F_sr_i < 0 or
                F_sr_j >= A_w or F_sr_j < 0):
                continue

            F_sr = concat_feature(A_pyramid, A_prime_pyramid, l, F_sr_i, F_sr_j, L)

            dist = np.sum((F_sr - F_q) ** 2)

            if dist < min_dist:
                min_dist = dist
                r_star_i = ii
                r_star_j = jj
                best_coh_i = F_sr_i
                best_coh_j = F_sr_j

        if done:
            break

    if r_star_i == -1 or r_star_j == -1:
        best_coh_i = -1
        best_coh_j = -1
        return best_coh_i, best_coh_j

    return best_coh_i, best_coh_j