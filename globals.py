from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
from skimage import io

'''
定义整个项目中的全局变量
'''
N_BIG = 5
N_SMALL = 3
NUM_FEATURES = 1
NNF = N_BIG * N_BIG * NUM_FEATURES
nnf = N_SMALL * N_SMALL * NUM_FEATURES
kappa = 2
# kappa = 0

# end_idx = sub2ind([N_BIG N_BIG], 2, 3)
end_idx = 2+ (3 - 1)*N_BIG # 用的时候不需要再+1
# Python不需要下标转换 但是 下标是2， 3


sigma = 0.5
# G_big = gaussian_filter(np.zeros((N_BIG, N_BIG)), sigma=1)
# G_small = gaussian_filter(np.zeros((N_SMALL, N_SMALL)), sigma=1)
G_big = np.multiply(cv2.getGaussianKernel(N_BIG, sigma), (cv2.getGaussianKernel(N_BIG, sigma)).T)
G_small = np.multiply(cv2.getGaussianKernel(N_SMALL, sigma), (cv2.getGaussianKernel(N_SMALL, sigma)).T)

# 这两个是读入的图像 在run_this.py里面读入的
# Read images
# A = io.imread('images/swan.jpg')
# A_prime = io.imread('images/swan-pastel.jpg')
# B = io.imread('images/chicago.jpg')