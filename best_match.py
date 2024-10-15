from best_coherence_match import best_coherence_match
from best_approximate_match import best_approximate_match
from concat_feature import concat_feature
import numpy as np
from globals import *

def best_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s_pyramid,
               A_features, B_features, l, L, i, j):
    # BEST_MATCH
    best_app_i, best_app_j = best_approximate_match(A_features, A_pyramid, B_pyramid, B_features, l, i, j)

    # return int(best_app_i), int(best_app_j)

    h, w, _ = B_pyramid[l].shape
    if i < 4 or j < 4 or i >= h - 4 - 1 or j >= w - 4 - 1:
        return int(best_app_i), int(best_app_j)

    best_coh_i, best_coh_j = best_coherence_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s_pyramid, l,
                                                  L, i, j)

    if best_coh_i == -1 or best_coh_j == -1:
        return int(best_app_i), int(best_app_j)

    F_p_app = concat_feature(A_pyramid, A_prime_pyramid, l, best_app_i, best_app_j, L)
    F_p_coh = concat_feature(A_pyramid, A_prime_pyramid, l, best_coh_i, best_coh_j, L)
    F_q = concat_feature(B_pyramid, B_prime_pyramid, l, i, j, L)

    d_app = np.sum((F_p_app - F_q) ** 2)
    d_coh = np.sum((F_p_coh - F_q) ** 2)

    if d_coh <= d_app * (1 + 2 ** (l - L) * kappa):
        return int(best_coh_i), int(best_coh_j)
    else:
        return int(best_app_i), int(best_app_j)
