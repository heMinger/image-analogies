import numpy as np
import pandas as pd
from numpy import reshape
import cv2
from skimage import color, transform, io
from skimage.color import rgb2yiq, yiq2rgb
from skimage import io
from skimage.util import view_as_windows
from best_match import best_match
from extend_image import extend_image
from globals import *

def my_rgb2yiq(A):
    A_float = A.astype(np.float32) / 255.0

    # 定义RGB到YIQ的转换矩阵
    rgb2yiq_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])

    # 执行矩阵乘法
    yiq_image = np.dot(A_float, rgb2yiq_matrix.T)
    # yiq_image = np.dot(A, rgb2yiq_matrix.T)

    # # 将结果乘以255，转换回0-255范围的整数
    # yiq_image = np.clip(yiq_image * 255, 0, 255).astype(np.uint8)

    return yiq_image

def my_yiq2rgb(A):
    # A_float = A.astype(np.float32) / 255.0

    # 定义RGB到YIQ的转换矩阵
    yiq2rgb_matrix = np.array([
        [1.000, 1.000, 1.000],
        [0.956, -0.272, -1.106],
        [0.621, -0.647, 1.703]
    ])

    # 执行矩阵乘法
    rgb_image = np.dot(A, yiq2rgb_matrix)

    # 将结果乘以255，转换回0-255范围的整数
    # rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

    return rgb_image

# Assuming the necessary functions (best_match, extend_image, rgb2ntsc, ntsc2rgb) are defined
def create_image_analogy(A, A_prime, B):
    B_height, B_width, _ = B.shape
    A_height, A_width, _ = A.shape

    # Create luminance maps
    A = my_rgb2yiq(A)
    # 归一化
    # A = A.astype(np.float32) / 255
    # x = pd.DataFrame(A_prime[:,:,0])
    # x.to_excel('A_prime_python.xlsx')
    A_prime = my_rgb2yiq(A_prime)
    # x = pd.DataFrame(A_prime)
    # x.to_excel('A_prime_after_transform_python.xlsx')
    '''
    # 检查一下rgb2yiq结果
    A = A_prime[:, :, 0]
    return A
    my_rgb2yiq 结果的亮度通道没问题
    rgb2yiq看起来不太好
    '''
    # A = my_yiq2rgb(A)
    # return A
    B = my_rgb2yiq(B)

    # Construct Gaussian pyramids for A, A' and B
    A_pyramid = [A]
    A_prime_pyramid = [A_prime]
    B_pyramid = [B]
    B_prime_pyramid = [np.zeros(B.shape)]
    s_pyramid = [np.zeros((B_height, B_width, 2))]

    # 构建金字塔
    while A_height >= 50 and A_width >= 50:
        A_pyramid.append(cv2.pyrDown(A_pyramid[-1]))
        # A_prime = cv2.pyrDown(A_prime)
        # A_prime_pyramid.append(A_prime)
        A_prime_pyramid.append(cv2.pyrDown(A_prime_pyramid[-1]))
        # B = cv2.pyrDown(B)
        # B_pyramid.append(B)
        B_pyramid.append(cv2.pyrDown(B_pyramid[-1]))
        # B_prime = cv2.pyrDown(B_prime_pyramid)
        # B_prime_pyramid.append(B_prime)
        B_prime_pyramid.append(cv2.pyrDown(B_prime_pyramid[-1]))
        # s = cv2.pyrDown(s_pyramid[-1])
        # s_pyramid.append(s)
        s_pyramid.append(cv2.pyrDown(s_pyramid[-1]))


        A_height, A_width, _ = A_pyramid[-1].shape

    L = len(A_pyramid)

    A_pyramid_extend = [extend_image(A_pyramid[l], N_BIG // 2, NUM_FEATURES) for l in range(L)]
    B_pyramid_extend = [extend_image(B_pyramid[l], N_BIG // 2, NUM_FEATURES) for l in range(L)]
    s_pyramid = [np.zeros((B_height + 4, B_width + 4, 2)) for _ in range(L)]

#### 这里还没有验证对不对 先去看extend_images.py了
    # A_features = [np.zeros((A_pyramid[l].shape[0] * A_pyramid[l].shape[1], NNF)) for l in range(L)]
    # B_features = [np.zeros((B_pyramid[l].shape[0] * B_pyramid[l].shape[1], NNF)) for l in range(L)]
    A_features = [None] * L
    B_features = [None] * L

    for l in range(L):
        A_l = A_pyramid_extend[l]
        B_l = B_pyramid_extend[l]

        A_height, A_width, _ = A_pyramid[l].shape
        B_height, B_width, _ = B_pyramid[l].shape

        A_features[l] = np.zeros((A_height, A_width, NNF))
        B_features[l] = np.zeros((B_height, B_width, NNF))

        for i in range(A_height):
            for j in range(A_width):
                A_features[l][i, j, :] = A_l[i:i+N_BIG, j:j+N_BIG, 0].reshape((1, NNF))

        for i in range(B_height):
            for j in range(B_width):
                B_features[l][i, j, :] = B_l[i:i + N_BIG, j:j + N_BIG, 0].reshape((1, NNF))

    print('Finding best match...\n\n')

    for l in range(L-1, -1, -1):
        print(f'\nl: {l}/{L}\n===========')
        B_prime_l = B_prime_pyramid[l]
        height, width, _ = B_prime_l.shape
        for i in range(height):
            print(f'i: {i}/{height}\n')
            for j in range(width):
                best_i, best_j = best_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s_pyramid, A_features, B_features, l, L, i, j)

                s_pyramid[l][i, j, 0] = best_i
                s_pyramid[l][i, j, 1] = best_j

                B_prime_pyramid[l][i, j, 0] = A_prime_pyramid[l][best_i, best_j, 0]
                B_prime_pyramid[l][i, j, 1:] = B_pyramid[l][i, j, 1:]

    B_prime = my_yiq2rgb(B_prime_pyramid[0])
    # B_prime = B_prime_pyramid[0]
    return B_prime

# Assuming the best_match function is defined elsewhere
# Make sure to define the global variables N_BIG, NUM_FEATURES, and NNF