import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from sample_images import sample_images
from sparse_autoencoder import sparse_autoencoder_cost, initialize_parameters
from check_numerical_gradient import check_numerical_gradient, compute_numerical_gradient
from display_network import display_network

"""
STEP 0: Implement sample_images
        After implementing sample_images, the display_network command should
        display a random sample of 200 patches from the dataset
"""
patches = sample_images('data/IMAGES.mat') # Read sample form the Matlab file

n_patches = patches.shape[1] # Number of patches

# Randomly sample 200 patches and save as an image file
image = display_network(patches[: , [np.random.randint(n_patches) for i in range(200)]])

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)

plt.imsave('sparse_autoencoder_patches.png', image, cmap=plt.cm.gray)


"""
STEP 1: Here we provide the relevant parameters values that will
        allow your sparse autoencoder to get good filters; you do not need to
        change the parameters below.
"""
visible_size = patches.shape[0] # number of input units
hidden_size = 25                # number of hidden units

weight_decay_param = 0.0001 # weight decay parameter, which is the lambda in lecture notes
beta = 3                    # weight of sparsity penalty term
sparsity_param = 0.01       # desired average activation of the hidden units.

#  Obtain random parameters theta
theta = initialize_parameters(hidden_size, visible_size)

"""
STEP 2: Implement sparse_autoencoder_cost
  You can implement all of the components (squared error cost, weight decay term,
  sparsity penalty) in the cost function at once, but it may be easier to do
  it step-by-step and run gradient checking (see STEP 3) after each step.  We
  suggest implementing the sparse_autoencoder_cost function using the following steps:

  (a) Implement forward propagation in your neural network, and implement the
      squared error term of the cost function.  Implement backpropagation to
      compute the derivatives. Then (using lambda=beta=0), run Gradient Checking
      to verify that the calculations corresponding to the squared error cost
      term are correct.

  (b) Add in the weight decay term (in both the cost function and the derivative
      calculations), then re-run Gradient Checking to verify correctness.

  (c) Add in the sparsity penalty term, then re-run Gradient Checking to
      verify correctness.

  Feel free to change the training settings when debugging your
  code.  (For example, reducing the training set size or
  number of hidden units may make your code run faster; and setting beta
  and/or weight_decay_param to zero may be helpful for debugging.)  However, in your
  final submission of the visualized weights, please use parameters we
  gave in Step 0 above.
"""

cost, grad = sparse_autoencoder_cost(theta, visible_size, hidden_size, weight_decay_param, sparsity_param, beta, patches)

"""
STEP 3: Gradient Checking
  Hint: If you are debugging your code, performing gradient checking on smaller models
        and smaller training sets (e.g., using only 10 training examples and 1-2 hidden
        units) may speed things up.
"""
debug = False # Please switch this to True when you need to check gradient

if debug:

    # First, lets make sure your numerical gradient computation is correct for a
    # simple function.  After you have implemented compute_numerical_gradient.py,
    # run the following:
    check_numerical_gradient()

    # Now we can use it to check your cost function and derivative calculations
    # for the sparse autoencoder.
    J = lambda theta : sparse_autoencoder_cost(theta, visible_size, hidden_size,
        weight_decay_param, sparsity_param, beta, patches)[0]
    numgrad = compute_numerical_gradient(J, theta)

    # Use this to visually compare the gradients side by side
    n = min(grad.size, 20) # Number of gradients to display
    for i in range(n):
        print("{0:20.12f} {1:20.12f}".format(numgrad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Compare numerically computed gradients with the ones obtained from backpropagation
    # This should be small. In our implementation, these values are usually less than 1e-9.
    # When you got this working, Congratulations!!!
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("Norm of difference = ", diff)


"""
STEP 4: After verifying that your implementation of
  sparse_autoencoder_cost is correct, You can start training your sparse
  autoencoder with minFunc (L-BFGS).
"""
#  Randomly initialize the parameters
theta = initialize_parameters(hidden_size, visible_size)

J = lambda theta : sparse_autoencoder_cost(theta, visible_size, hidden_size,
    weight_decay_param, sparsity_param, beta, patches)

# In case you want to see the details of optimization,
# Please set 'disp' as True
options = {'maxiter': 400, 'disp': True, 'gtol': 1e-5, 'ftol': 2e-9}
results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)


"""
STEP 5: Visualization
"""
W1 = opt_theta[0:hidden_size*visible_size].reshape((hidden_size, visible_size))

print("Save and show the W1")
image = display_network(W1.T)

plt.figure()
plt.imsave('sparse_autoencoder_weights.png', image, cmap=plt.cm.gray)
plt.imshow(image, cmap=plt.cm.gray)

plt.show()


