import numpy as np
import scipy
import matplotlib.pyplot as plt

from sample_images import sample_images_raw
from display_network import display_network

def get_optimal_k(threshold, s):
    """
    Return the opticmal k.
    k is the minimum value satisfying (sum_j=1^k s_j) / (sum_j=1^n s_j).
    where n is the total number of eigenvalues. 
    
    threshold : threshold value.
    s : array of eigenvalues.
    """
    k = 0
    total_sum = np.sum(s)
    sum_ev = 0.0 # Sum of eigenvalues
    for i in range(s.size):
        sum_ev += s[i]     
        ratio = sum_ev / total_sum
        if ratio > threshold: break
        k += 1
        
    return k   

"""
Step 0a: Load data
  Here we provide the code to load natural image data into x.
  x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
  the raw image data from the kth 12x12 image patch sampled.
  You do not need to change the code below.
"""

# Return patches from sample images
x = sample_images_raw('data/IMAGES_RAW.mat')

# n is the number of dimensions and m is the number of patches
n, m = x.shape
random_sel = np.random.randint(0, m, 200)
image_x = display_network(x[:, random_sel])

fig = plt.figure()
plt.imshow(image_x, cmap=plt.cm.gray)
plt.title('Raw patch images')

"""
Step 0b: Zero-mean the data (by row)
"""

x -= np.mean(x, axis=1).reshape(-1, 1)

"""
Step 1a: Implement PCA to obtain x_rot
  Implement PCA to obtain x_rot, the matrix in which the data is expressed
  with respect to the eigenbasis of sigma, which is the matrix U.
"""

# n is the number of dimensions and m is the number of patches
sigma = x.dot(x.T) / m 

# Sigular value decomposition
U, s, V = np.linalg.svd(sigma)

# Compute x_rot
x_rot = U.T.dot(x)

"""
Step 1b: Check your implementation of PCA
  The covariance matrix for the data expressed with respect to the basis U
  should be a diagonal matrix with non-zero entries only along the main
  diagonal. We will verify this here.
  Write code to compute the covariance matrix, covar. 
  When visualised as an image, you should see a straight line across the
  diagonal (non-zero entries) against a blue background (zero entries).
"""

covar = np.cov(x_rot)

# Because of the range of the diagonal entries, the diagonal line may not be apparent
fig = plt.figure()
plt.imshow(covar)
plt.title('Covariance matrix of x_rot')

"""
Step 2: Find k, the number of components to retain
  Write code to determine k, the number of components to retain in order
  to retain at least 99% of the variance.
"""
    
opt_k_99 = get_optimal_k(0.99, s) # Optimal k to retain at least 99% variance

"""
Step 3: Implement PCA with dimension reduction
  Now that you have found k, you can reduce the dimension of the data by
  discarding the remaining dimensions. In this way, you can represent the
  data in k dimensions instead of the original 144, which will save you
  computational time when running learning algorithms on the reduced
  representation.
 
  Following the dimension reduction, invert the PCA transformation to produce 
  the matrix x_hat, the dimension-reduced data with respect to the original basis.
  Visualise the data and compare it to the raw data. You will observe that
  there is little loss due to throwing away the principal components that
  correspond to dimensions with low variation.
"""

x_tilde = x_rot[0:opt_k_99, :]
x_hat = U[:, 0:opt_k_99].dot(x_tilde)

image_x_hat_99 = display_network(x_hat[:, random_sel])

# Visualise the data, and compare it to the raw data
# You should observe that the raw and processed data are of comparable quality.
# For comparison, you may wish to generate a PCA reduced image which
# retains only 90% of the variance.

opt_k_90 = get_optimal_k(0.90, s) # Optimal k to retain at least 90% variance
x_tilde = x_rot[0:opt_k_90, :]
x_hat = U[:, 0:opt_k_90].dot(x_tilde)

image_x_hat_90 = display_network(x_hat[:, random_sel])

f, ax = plt.subplots(1, 3)
ax[0].imshow(image_x, cmap=plt.cm.gray)
ax[0].set_title('Raw data')
ax[1].imshow(image_x_hat_99, cmap=plt.cm.gray)
ax[1].set_title('99% variance')
ax[2].imshow(image_x_hat_90, cmap=plt.cm.gray)
ax[2].set_title('90% variance')

"""
Step 4a: Implement PCA with whitening and regularisation
  Implement PCA with whitening and regularisation to produce the matrix
  x_PCAWhite. 
""" 

# PCA white with regulation
epsilon = 0.1 # Regulation
x_PCA_white = np.diag(1.0/np.sqrt(s + epsilon)).dot(x_rot)

"""
Step 4b: Check your implementation of PCA whitening 
  Check your implementation of PCA whitening with and without regularisation. 
  PCA whitening without regularisation results a covariance matrix 
  that is equal to the identity matrix. PCA whitening with regularisation
  results in a covariance matrix with diagonal entries starting close to 
  1 and gradually becoming smaller. We will verify these properties here.
  Write code to compute the covariance matrix, covar. 

  Without regularisation (set epsilon to 0 or close to 0), 
  when visualised as an image, you should see a red line across the
  diagonal (one entries) against a blue background (zero entries).
  With regularisation, you should see a red line that slowly turns
  blue across the diagonal, corresponding to the one entries slowly
  becoming smaller.
"""

# PCA white without regulation
x_PCA_white_without_regulation = np.diag(1.0/np.sqrt(s)).dot(x_rot)

covar = np.cov(x_PCA_white)
covar_without_regulation = np.cov(x_PCA_white_without_regulation)

# Visualise the covariance matrix. You should see a red line across the
# diagonal against a blue background.
f, ax = plt.subplots(1, 2)
ax[0].imshow(covar)
ax[0].set_title('PCA white With Regulation')
ax[1].imshow(covar_without_regulation)
ax[1].set_title('PCA white Without Regulation')

"""
Step 5: Implement ZCA whitening
  Now implement ZCA whitening to produce the matrix xZCAWhite.
  Visualise the data and compare it to the raw data. You should observe
  that whitening results in, among other things, enhanced edges.
"""
epsilon = 0.1 # Regulation
x_PCA_white = np.diag(1.0/np.sqrt(s + epsilon)).dot(x_rot)
x_ZCA_white = U.dot(x_PCA_white)

# Visualise the data, and compare it to the raw data.
# You should observe that the whitened images have enhanced edges.
image_raw = display_network(x[:, random_sel])
image_ZCA_white = display_network(x_ZCA_white[:, random_sel])

f, ax = plt.subplots(1, 2)
ax[0].imshow(image_ZCA_white, cmap=plt.cm.gray)
ax[0].set_title('ZCA whitened images')
ax[1].imshow(image_raw, cmap=plt.cm.gray)
ax[1].set_title('Raw images')
plt.show()

plt.imsave('pca_raw.png', image_raw, cmap=plt.cm.gray)
plt.imsave('pca_zca_white.png', image_ZCA_white, cmap=plt.cm.gray)
