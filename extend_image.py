import numpy as np
from globals import *

def extend_image(image, border, layers):
    # EXTEND_IMAGE Extend the borders of the image by border size

    border = int(np.floor(border))
    h, w, _ = image.shape
    result = np.zeros((h + 2 * border, w + 2 * border, layers))
    # result = np.zeros((h + 2 * border, w + 2 * border))

    '''
    matlab image(:, :, 1:1) 得到的是 二维数组但Python得到的是第三维维度为0 的三维数组，所以result要建立一个三维数组
    额其实可以直接 image(:,:,0) Python得到的也是二维
    '''
    # Put original pic in center
    result[border:-border, border:-border] = image[:, :, :layers]

    # Fill in borders
    # tmp1 = np.tile(image[0, :, :layers], (border, 1, 1))
    # tt1 = result[:border, border:-border]
    result[:border, border:-border] = np.tile(image[0, :, :layers], (border, 1, 1))
    # tt2 = result[-border:, border:-border]
    # tmp2 = np.tile(image[-1, :, :layers], (border, 1, 1))
    result[-border:, border:-border] = np.tile(image[-1, :, :layers], (border, 1, 1))
    # tt3 = result[border:-border, :border]
    # tmp = image[:, 0, :layers]
    # tmp3 = np.tile(image[:, 0, :layers], (1, border, 1))

    x = np.tile(image[:, 0, :layers], (1, border))
    result[border:-border, :border] = x.reshape((x.shape[0], x.shape[1], 1)) #.transpose((1,2,0))
    #
    # tt4 =  result[border:-border, -border:]
    # tmp4 = np.tile(image[:, -1, :layers], (1, border, 1))
    x = np.tile(image[:, -1, :layers], (1, border))
    result[border:-border, -border:] = x.reshape((x.shape[0],x.shape[1], 1))

    # Fill in corners
    x = np.tile(image[0, 0, :layers], (border, border))
    result[:border, :border, :] = x.reshape((x.shape[0], x.shape[1], 1))
    x = np.tile(image[0, -1, :layers], (border, border))
    result[:border, -border:, :] = x.reshape((x.shape[0], x.shape[1], 1))
    x = np.tile(image[-1, 0, :layers], (border, border))
    result[-border:, :border, :] = x.reshape((x.shape[0], x.shape[1], 1))
    x = np.tile(image[-1, -1, :layers], (border, border))
    result[-border:, -border:, :] = x.reshape((x.shape[0], x.shape[1], 1))

    return result