'''
该文件中，所有的slice,  matlab中的[i, j]对应了[i - 1, j]
'''
import numpy as np
from globals import *
from get_indices import get_indices
from scipy.ndimage import gaussian_filter

# 定义gaussian_filter的参数
sigma = 0.5

def concat_feature(X_pyramid, X_prime_pyramid, l, i, j, L):
    # CONCAT_FEATURE Concatenate the neighborhood around (i,j)
    # on levels l and l-1 for X and X'

    X_fine = X_pyramid[l]
    X_prime_fine = X_prime_pyramid[l]
    x_fine_height, x_fine_width, _ = X_fine.shape

    # if l + 1 <= L:  # In range, so make a coarse guy
    if l + 1 < L:  # In range, so make a coarse guy
        X_coarse = X_pyramid[l + 1]
        X_prime_coarse = X_prime_pyramid[l + 1]
        x_coarse_height, x_coarse_width, _ = X_coarse.shape

    X_fine_nhood = np.zeros((N_BIG, N_BIG, NUM_FEATURES))
    X_prime_fine_nhood = np.zeros((N_BIG, N_BIG, NUM_FEATURES))
    X_coarse_nhood = np.zeros((N_SMALL, N_SMALL, NUM_FEATURES))
    X_prime_coarse_nhood = np.zeros((N_SMALL, N_SMALL, NUM_FEATURES))

    x_fine_start_i, x_fine_end_i, x_fine_start_j, x_fine_end_j, pad_top, pad_bot, pad_left, pad_right, _ = get_indices(i, j, N_BIG, x_fine_height, x_fine_width)
    # result = get_indices(i, j, N_BIG, x_fine_height, x_fine_width)

    # A or B
    # 第三维取0， 我觉得也可以直接在前面建一个二维的数组
    #               1: 6              1: 6                               3: 8 3:8
    # print(pad_top, pad_bot + 1, pad_left, pad_right + 1)
    # print(N_BIG)
    '''
    1. matlab 从0开始 Python 从1开始
    2. MATLAB的下标取得到j Python取到j-1
    所以 MATLAB [i, j] 对应Python [i - 1, j]
    '''
    # X_fine_nhood[pad_top:pad_bot + 1, pad_left:pad_right + 1, 0] = X_fine[x_fine_start_i:x_fine_end_i+1,
    # print("2: shape of X_fine_nhood: \n", X_fine_nhood.shape)
    # print(pad_top - 1, pad_bot, pad_left - 1, pad_right, x_fine_start_i - 1, x_fine_end_i, x_fine_start_j - 1, x_fine_end_j)
    # X_fine_nhood[pad_top - 1:pad_bot, pad_left - 1:pad_right, 0] = X_fine[x_fine_start_i:x_fine_end_i + 1, x_fine_start_j:x_fine_end_j + 1, NUM_FEATURES]
    X_fine_nhood[pad_top: pad_bot + 1, pad_left: pad_right + 1, 0] = X_fine[x_fine_start_i: x_fine_end_i + 1, x_fine_start_j: x_fine_end_j + 1, NUM_FEATURES - 1]
    # print("3: shape of X_fine_nhood: \n", X_fine_nhood.shape)
    # G_big: ndarry(5, 5)
    # X_fine_nhood: ndarry(5, 5, 1)

    ######## 1.
    X_fine_nhood = X_fine_nhood[:,:,0] * G_big
    # X_fine_nhood = gaussian_filter(X_fine_nhood[:, :, 0], sigma=sigma)
    # print("4: shape of X_fine_nhood: \n", X_fine_nhood.shape)

    # A' or B'
    # A' or B'
    # X_prime_fine_nhood[pad_top-1:pad_bot, pad_left-1:pad_right, 0] = X_prime_fine[x_fine_start_i:x_fine_end_i + 1,
    #                                                                      x_fine_start_j:x_fine_end_j + 1,
    #                                                                      NUM_FEATURES]
    # X_prime_fine_nhood[pad_top-1:pad_bot, pad_left-1:pad_right, 0] = X_prime_fine[x_fine_start_i - 1:x_fine_end_i,
    #                                                                      x_fine_start_j - 1:x_fine_end_j,
    #                                                                      NUM_FEATURES]
    X_prime_fine_nhood[pad_top: pad_bot + 1, pad_left: pad_right + 1, 0] = X_prime_fine[x_fine_start_i: x_fine_end_i + 1,
                                                                         x_fine_start_j:x_fine_end_j + 1,
                                                                         NUM_FEATURES - 1]
    ########## 2.
    X_prime_fine_nhood = X_prime_fine_nhood[:,:,0] * G_big
    # X_prime_fine_nhood = gaussian_filter(X_prime_fine_nhood[:,:,0], sigma=sigma)

    # if l + 1 <= L:
    if l + 1 < L:
        x_coarse_start_i, x_coarse_end_i, x_coarse_start_j, x_coarse_end_j, \
        pad_top, pad_bot, pad_left, pad_right, flag = get_indices(i // 2, j // 2, N_SMALL, x_coarse_height,
                                                                  x_coarse_width)

        if flag == False:
            # X_coarse_nhood[pad_top - 1:pad_bot, pad_left - 1:pad_right, 0] = X_coarse[
            #                                                                  x_coarse_start_i: x_coarse_end_i + 1,
            #                                                                  x_coarse_start_j:x_coarse_end_j + 1,
            #                                                                  NUM_FEATURES]
            # X_coarse_nhood[pad_top - 1:pad_bot, pad_left - 1:pad_right, 0] = X_coarse[
            #                                                                  x_coarse_start_i - 1: x_coarse_end_i,
            #                                                                  x_coarse_start_j - 1:x_coarse_end_j,
            #                                                                  NUM_FEATURES]
            X_coarse_nhood[pad_top: pad_bot + 1, pad_left: pad_right + 1, 0] = X_coarse[
                                                                             x_coarse_start_i: x_coarse_end_i + 1,
                                                                             x_coarse_start_j: x_coarse_end_j + 1,
                                                                             NUM_FEATURES - 1]
            ############## 3.
            X_coarse_nhood = X_coarse_nhood[:,:,0] * G_small
            # X_coarse_nhood = gaussian_filter(X_coarse_nhood[:, :, 0], sigma=sigma)

            # X_prime_coarse_nhood[pad_top - 1:pad_bot, pad_left - 1:pad_right, 0] = X_prime_coarse[
            #                                                                        x_coarse_start_i:x_coarse_end_i + 1,
            #                                                                        x_coarse_start_j:x_coarse_end_j + 1,
            #                                                                        NUM_FEATURES]
            # X_prime_coarse_nhood[pad_top - 1:pad_bot, pad_left - 1:pad_right, 0] = X_prime_coarse[
            #                                                                        x_coarse_start_i - 1:x_coarse_end_i,
            #                                                                        x_coarse_start_j - 1:x_coarse_end_j,
            #                                                                        NUM_FEATURES]
            X_prime_coarse_nhood[pad_top: pad_bot + 1, pad_left: pad_right + 1, 0] = X_prime_coarse[
                                                                                   x_coarse_start_i: x_coarse_end_i + 1,
                                                                                   x_coarse_start_j: x_coarse_end_j + 1,
                                                                                   NUM_FEATURES - 1]
            ############### 4.
            X_prime_coarse_nhood = X_prime_coarse_nhood[:,:,0] * G_small
            # X_prime_coarse_nhood = gaussian_filter(X_prime_coarse_nhood[:, :, 0], sigma = sigma)

    F = np.zeros((1, NNF*2+nnf*2))
    # print("shape of F: ", F.shape)
    # F = np.zeros(1 + NNF + NNF + (nnf if l + 1 <= L else 0) + (nnf if l + 1 <= L else 0))

    # This is neighborhood of A{l} or B{l}
    # ValueError: could not broadcast input array from shape (125,) into shape (1,68)
    # print("shape of X_fine_nhood: ", X_fine_nhood.shape) #(5, 5, 5)
    # print("shape of X_fine_nhood.reshape(-1): ", X_fine_nhood.reshape(-1).shape) #(125,)
    # print("NNF: ", NNF) # 25
    # print("shape of F[:NNF]: ", F[:NNF].shape) # (1:68)
    # ValueError: cannot reshape array of size 125 into shape (1,25)
    # tmp = F[0, :NNF]
    # print("NNF = ", NNF)
    # print("shape of F: ", F.shape)
    # print("shape of tmp: ", tmp.shape)

    # F[0, :NNF] = X_fine_nhood.reshape((1, NNF))
    F[0, :NNF] = X_fine_nhood.T.reshape((1, NNF))

    # Only copies over the parts of B' (and A') that are already made
    # temp = X_prime_fine_nhood.T
    temp = X_prime_fine_nhood
    temp = temp.reshape((1, NNF))
    # temp[2*N_BIG + 3*N_BIG - 1:] = 0
    temp[0, 2+ (3 - 1)*N_BIG:] = 0
    F[0, NNF:2 * NNF] = temp

    # Edge case: make sure the l-1 level exists
    if l + 1 < L:
        # F[0, 2 * NNF:2 * NNF + nnf] = X_coarse_nhood.reshape((1, nnf))
        # F[0, 2 * NNF + nnf:] = X_prime_coarse_nhood.reshape((1, nnf))
        F[0, 2 * NNF:2 * NNF + nnf] = X_coarse_nhood.T.reshape((1, nnf))
        F[0, 2 * NNF + nnf:] = X_prime_coarse_nhood.T.reshape((1, nnf))

    #### F 就是 Contenation of features
    return F