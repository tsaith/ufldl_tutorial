# CS294A/CS294W Linear Decoder Exercise

import numpy as np
import scipy.optimize
import scipy.io
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from sparse_autoencoder import initialize_parameters, sparse_autoencoder_linear_cost
from check_numerical_gradient import compute_numerical_gradient
from display_network import display_color_network


"""
  Instructions
  ------------

  This file contains code that helps you get started on the
  linear decoder exericse. For this exercise, you will only need to modify
  the code in sparseAutoencoderLinearCost.m. You will not need to modify
  any code in this file.
"""

"""
  STEP 0: Initialization
  Here we initialize some parameters used for the exercise.
"""

image_channels = 3   # number of channels (rgb, so 3)

patch_dim = 8        # patch dimension
n_patches = 100000   # number of patches

visible_size = patch_dim * patch_dim * image_channels  # number of input units
output_size  = visible_size  # number of output units
hidden_size  = 400           # number of hidden units

sparsity_param = 0.035 # desired average activation of the hidden units.
lambda_ = 3e-3         # weight decay parameter
beta = 5               # weight of sparsity penalty term

epsilon = 0.1          # epsilon for ZCA whitening

"""
  STEP 1: Create and modify sparse_autoencoder_linear_cost to use a linear decoder,
          and check gradients
"""

# To speed up gradient checking, we will use a reduced network and some
# dummy patches
debug = False
if debug:
    debug_hidden_size = 5
    debug_visible_size = 8
    patches = np.random.rand(8, 10)
    theta = initialize_parameters(debug_hidden_size, debug_visible_size)

    cost, grad = sparse_autoencoder_linear_cost(theta,
                     debug_visible_size, debug_hidden_size, lambda_, sparsity_param, beta, patches)

    # Check that the numerical and analytic gradients are the same
    J = lambda theta : sparse_autoencoder_linear_cost(theta,
                            debug_visible_size, debug_hidden_size, lambda_, sparsity_param, beta, patches)[0]

    nume_grad = compute_numerical_gradient(J, theta)

    # Use this to visually compare the gradients side by side
    for i in range(grad.size):
        print("{0:20.12f} {1:20.12f}".format(nume_grad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Compare numerically computed gradients with the ones obtained from backpropagation
    # The difference should be small. In our implementation, these values are usually less than 1e-9.
    # When you got this working, Congratulations!!!
    diff = np.linalg.norm(nume_grad - grad) / np.linalg.norm(nume_grad + grad)
    print("Norm of difference = ", diff)
    print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n')

    assert diff < 1e-9, 'Difference too large. Check your gradient computation again'

    # NOTE: Once your gradients check out, you should run step 0 again to
    #       reinitialize the parameters

"""
  STEP 2: Learn features on small patches
  In this step, you will use your sparse autoencoder (which now uses a
  linear decoder) to learn features on small patches sampled from related
  images.
"""

"""
  STEP 2a: Load patches
  In this step, we load 100k patches sampled from the STL10 dataset and
  visualize them. Note that these patches have been scaled to [0,1]
"""

patches = scipy.io.loadmat('data/stlSampledPatches.mat')['patches']

image = display_color_network(patches[:, :100])
plt.imsave('linear_decoder_raw_patches.png', image)
#plt.imshow(image)

"""
  STEP 2b: Apply preprocessing
  In this sub-step, we preprocess the sampled patches, in particular,
  ZCA whitening them.

  In a later exercise on convolution and pooling, you will need to replicate
  exactly the preprocessing steps you apply to these patches before
  using the autoencoder to learn features on them. Hence, we will save the
  ZCA whitening and mean image matrices together with the learned features
  later on.
"""

# Subtract mean patch (hence zeroing the mean of the patches)
mean_patch = np.mean(patches, axis=1);
patches -= mean_patch.reshape((-1, 1))

# Apply ZCA whitening
sigma = patches.dot(patches.T) / n_patches
u, s, v = np.linalg.svd(sigma) # Sigular value decomposition
D = np.diag(1.0/np.sqrt(s + epsilon))

zca_white = u.dot(D).dot(u.T)
patches = zca_white.dot(patches)

image = display_color_network(patches[:, :100])
plt.imsave('linear_decoder_zca_patches.png', image)
#plt.imshow(image)

"""
  STEP 2c: Learn features
  You will now use your sparse autoencoder (with linear decoder) to learn
  features on the preprocessed patches. This should take around 45 minutes.
"""
theta = initialize_parameters(hidden_size, visible_size)

# Train the model
J = lambda theta : sparse_autoencoder_linear_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, patches)

options = {'maxiter': 400, 'disp': True}
results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)

# Save the learned features and the preprocessing matrices for use in
# the later exercise on convolution and pooling
print('Saving learned features and preprocessing matrices...\n')

params = {}
params['opt_theta'] = opt_theta
params['zca_white'] = zca_white
params['mean_patch'] = mean_patch
joblib.dump(params, "data/STL10_features.pkl", compress=3)

# STEP 2d: Visualize learned features
W = opt_theta[0:visible_size * hidden_size].reshape((hidden_size, visible_size))
b = opt_theta[2*hidden_size*visible_size:2*hidden_size*visible_size + hidden_size]

image = display_color_network( (W.dot(zca_white)).T )
plt.imsave('linear_decoder_features.png', image)
#plt.imshow(image)
