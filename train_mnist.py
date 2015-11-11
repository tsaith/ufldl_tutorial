import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from sparse_autoencoder import sparse_autoencoder_cost, initialize_parameters
from display_network import display_network
from load_MNIST import load_MNIST_images

# Loading 10K images from MNIST database
images = load_MNIST_images('data/mnist/train-images-idx3-ubyte')
patches = images[:, :10000]

n_patches = patches.shape[1]    # Number of patches

# Randomly sample 200 patches and save as an image file
image = display_network(patches[:, [np.random.randint(n_patches) for i in range(200)]])

plt.figure()
plt.imsave('sparse_autoencoder_minist_patches.png', image, cmap=plt.cm.gray)
plt.imshow(image, cmap=plt.cm.gray)

visible_size = patches.shape[0] # Number of input units
hidden_size = 196               # Number of hidden units

weight_decay_param = 3e-3       # Weight decay parameter, which is the lambda in lecture notes
beta = 3                        # Weight of sparsity penalty term
sparsity_param = 0.1            # Desired average activation of the hidden units.

#  Randomly initialize the fitting parameters
theta = initialize_parameters(hidden_size, visible_size)

J = lambda theta : sparse_autoencoder_cost(theta, visible_size, hidden_size, weight_decay_param, sparsity_param, beta, patches)

# The number of maximun iterations is set as 400,
# which is good enough to get reasonable results.
options = {'maxiter': 400, 'disp': True, 'gtol': 1e-5, 'ftol': 2e-9}
results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)

# Visualization
W1 = opt_theta[0:hidden_size*visible_size].reshape((hidden_size, visible_size))

image = display_network(W1.T)
plt.figure()
plt.imsave('sparse_autoencoder_minist_weights.png', image, cmap=plt.cm.gray)
plt.imshow(image, cmap=plt.cm.gray)

plt.show()


