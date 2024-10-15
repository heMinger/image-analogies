import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, img_as_float
from skimage.transform import resize, rescale
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab, lab2rgb
from create_image_analogy import create_image_analogy
from globals import *
from skimage.color import *

# # Calculate Gaussian filter
# G_big = gaussian_filter(np.zeros((N_BIG, N_BIG)), sigma=1)
# G_small = gaussian_filter(np.zeros((N_SMALL, N_SMALL)), sigma=1)

# # Indices
# end_idx = int(np.ravel_multi_index((2, 3), (N_BIG, N_BIG)))

# # Read images # 9.28 10:57第一次测试 效果不好
# A = io.imread('images/swan.jpg')
# # A_prime = io.imread('images/swan.jpg')
# A_prime = io.imread('images/swan-pastel.jpg')
# B = io.imread('images/chicago.jpg')

# # Texture transfer
# '''
# test: 10.5
# '''
# A = io.imread('images/transfer2_A1.jpg')
# A_prime = io.imread('images/transfer2_A2.jpg')
# B = io.imread('images/transfer2_B1.jpg')

# # Painting
# '''
# test: 10.5
# '''
# A = io.imread('images/rhone-src.jpg')
# A_prime = io.imread('images/rhone.jpg')
# B = io.imread('images/jakarta.jpg')

# # Blur
# '''
# test: 10.5
# '''
# A = io.imread('images/newflower-src.jpg')
# A_prime = io.imread('images/newflower-blur.jpg')
# B = io.imread('images/toy-newshore-src.jpg')


# # Blur2
# '''
# test: 10.5
# '''
# A = io.imread('images/blurA1.jpg')
# A_prime = io.imread('images/blurA2.jpg')
# B = io.imread('images/blurB1.jpg')

# # Identity Test
# '''
# test: 10.5
# '''
# A = io.imread('images/IdentityA.jpg')
# A_prime = io.imread('images/IdentityA.jpg')
# B = io.imread('images/IdentityB.jpg')

# # Emboss Test
# '''
# test: 10.7
# '''
# A = io.imread('images/rose-src.jpg')
# A_prime = io.imread('images/rose-emboss.jpg')
# B = io.imread('images/dandilion-src.jpg')

# # Brightness Test
# '''
# test: 10.8
# '''
# A = io.imread('images/brightnessA1.png')
# A = cv2.cvtColor(A, cv2.COLOR_RGBA2RGB)
# # plt.imshow(A)
# # plt.show()
# A_prime = io.imread('images/brightnessA2.png')
# A_prime = cv2.cvtColor(A_prime, cv2.COLOR_RGBA2RGB)
# B = io.imread('images/brightnessB1.png')
# B = cv2.cvtColor(B, cv2.COLOR_RGBA2RGB)

# # Contrast Test
# '''
# test: 10.8
# '''
# A = io.imread('images/contrastA1.png')
# A = cv2.cvtColor(A, cv2.COLOR_RGBA2RGB)
# # plt.imshow(A)
# # plt.show()
# A_prime = io.imread('images/contrastA2.png')
# A_prime = cv2.cvtColor(A_prime, cv2.COLOR_RGBA2RGB)
# B = io.imread('images/contrastB1.png')
# B = cv2.cvtColor(B, cv2.COLOR_RGBA2RGB)

# Recolorization Test
'''
test: 10.8
'''
A = io.imread('images/recolorizationA1.png')
A = cv2.cvtColor(A, cv2.COLOR_RGBA2RGB)
plt.imshow(A)
plt.show()
A_prime = io.imread('images/recolorizationA2.png')
A_prime = cv2.cvtColor(A_prime, cv2.COLOR_RGBA2RGB)
plt.imshow(A_prime)
plt.show()
B = io.imread('images/recolorizationB1.png')
B = cv2.cvtColor(B, cv2.COLOR_RGBA2RGB)
plt.imshow(B)
plt.show()

# # Oil Painting Test
# '''
# test: 10.8
# '''
# A = io.imread('images/oilA1.png')
# A = cv2.cvtColor(A, cv2.COLOR_RGBA2RGB)
# # plt.imshow(A)
# # plt.show()
# A_prime = io.imread('images/oilA2.png')
# A_prime = cv2.cvtColor(A_prime, cv2.COLOR_RGBA2RGB)
# B = io.imread('images/oilB1.png')
# B = cv2.cvtColor(B, cv2.COLOR_RGBA2RGB)

# Oil Painting Test
'''
test: 10.8
'''
A = io.imread('images/oil2A1.png')
A = cv2.cvtColor(A, cv2.COLOR_RGBA2RGB)
# plt.imshow(A)
# plt.show()
A_prime = io.imread('images/oil2A2.png')
A_prime = cv2.cvtColor(A_prime, cv2.COLOR_RGBA2RGB)
B = io.imread('images/oil2B1.png')
B = cv2.cvtColor(B, cv2.COLOR_RGBA2RGB)


# # Resize images for testing
A_scale = 0.5
B_scale = 0.5
# A = resize(A, (int(A.shape[0] * A_scale), int(A.shape[1] * A_scale)), anti_aliasing=True)
# A_prime = resize(A_prime, (int(A_prime.shape[0] * A_scale), int(A_prime.shape[1] * A_scale)), anti_aliasing=True)
# B = resize(B, (int(B.shape[0] * B_scale), int(B.shape[1] * B_scale)), anti_aliasing=True)
#
A = rescale(A, (A_scale, A_scale, 1), preserve_range = True, order = 2)
# A = np.round(A)
A_prime = rescale(A_prime, (A_scale, A_scale, 1), preserve_range = True, order = 2)
# A_prime = np.round(A_prime)
B = rescale(B, (B_scale, B_scale, 1), preserve_range =True, order = 2)
# B = np.round(B)

print(A.shape)
print(B.shape)

# # Convert images to float for processing
# A = img_as_float(A)
# A_prime = img_as_float(A_prime)
# B = img_as_float(B)

# Assuming the create_image_analogy function is defined
B_prime = create_image_analogy(A, A_prime, B)

# Save or display the result
# io.imshow(yiq2rgb(B_prime))
io.imshow(B_prime)
io.show()
