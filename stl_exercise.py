# CS294A/CS294W Self-taught Learning Exercise

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from load_MNIST import load_MNIST_images, load_MNIST_labels
from sparse_autoencoder import initialize_parameters, sparse_autoencoder_cost, feedforward_autoencoder
from display_network import display_network
from softmax import softmax_train, softmax_predict


"""
Instructions
------------

This file contains code that helps you get started on the self-taught learning.
You will need to complete code in feed_forward_autoencoder.py
You will also need to have implemented sparse_autoencoder_cost and
softmax_cost from previous exercises.

======================================================================
STEP 0: Here we provide the relevant parameters values that will
  allow your sparse autoencoder to get good filters; you do not need to
  change the parameters below.

"""

input_size  = 28 * 28
n_labels  = 5
hidden_size = 200
sparsity_param = 0.1 # desired average activation of the hidden units.
                     # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                     #  in the lecture notes).
lambda_ = 3e-3       # weight decay parameter
beta = 3             # weight of sparsity penalty term
maxiter = 400

"""
STEP 1: Load data from the MNIST database

  This loads our training and test data from the MNIST database files.
  We have sorted the data for you in this so that you will not have to
  change it.
"""

# Load MNIST database files
mnist_data   = load_MNIST_images('data/mnist/train-images-idx3-ubyte')
mnist_labels = load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')

# Set Unlabeled Set (All Images)

# Simulate a Labeled and Unlabeled set
labeled_set   = np.argwhere(mnist_labels < 5).flatten()
unlabeled_set = np.argwhere(mnist_labels >= 5).flatten()

n_train = round(labeled_set.size / 2) # Number of training data
train_set = labeled_set[:n_train]
test_set  = labeled_set[n_train:]

train_data   = mnist_data[:, train_set]
train_labels = mnist_labels[train_set]

test_data   = mnist_data[:, test_set]
test_labels = mnist_labels[test_set]

unlabeled_data = mnist_data[:, unlabeled_set]

# Output Some Statistics
print('# examples in unlabeled set: {}'.format(unlabeled_data.shape[1]))
print('# examples in supervised training set: {}'.format(train_data.shape[1]))
print('# examples in supervised testing set: {}\n'.format(test_data.shape[1]))


"""
STEP 2: Train the sparse autoencoder

  This trains the sparse autoencoder on the unlabeled training images.
"""

#  Randomly initialize the parameters
theta = initialize_parameters(hidden_size, input_size)

#  Find optimal theta by running the sparse autoencoder on
#  unlabeled training images
J = lambda theta : sparse_autoencoder_cost(theta, input_size, hidden_size,
    lambda_, sparsity_param, beta, unlabeled_data)

options = {'maxiter': maxiter, 'disp': True}
results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)

# Visualize weights
W1 = opt_theta[0:hidden_size*input_size].reshape((hidden_size, input_size))
image = display_network(W1.T)
plt.figure()
plt.imsave('stl_weights.png', image, cmap=plt.cm.gray)
#plt.imshow(image, cmap=plt.cm.gray)


"""
STEP 3: Extract Features from the Supervised Dataset

  You need to complete the code in feed_forward_autoencoder so that the
  following command will extract features from the data.

"""

train_features = feedforward_autoencoder(opt_theta, hidden_size, input_size, train_data)
test_features  = feedforward_autoencoder(opt_theta, hidden_size, input_size, test_data)


"""
STEP 4: Train the softmax classifier

  Use softmax_train from the previous exercise to train a multi-class classifier.

  Use lambda = 1e-4 for the weight regularization for softmax

  You need to compute softmax_model using softmax_train on train_features and
  train_labels
"""

lambda_ = 1e-4 # weight decay parameter
options = {'maxiter': maxiter, 'disp': True}
softmax_model = softmax_train(hidden_size, n_labels, lambda_, train_features, train_labels, options)
"""
STEP 5: Testing

  Compute Predictions on the test set (test_features) using softmax_predict
  and softmax_model
"""

# Make predictions
pred = softmax_predict(softmax_model, test_features)

acc = np.mean(test_labels == pred)
print("The Accuracy (with learned features): {:5.2f}% \n".format(acc*100))

"""
 Accuracy is the proportion of correctly classified images
 The results for our implementation was:

 Accuracy: 98.3%
"""

# As a comparison, when raw pixels are used (instead of the learned features),
# we obtained a test accuracy of only around 96% (for the same train and test sets).

softmax_model = softmax_train(input_size, n_labels, lambda_, train_data, train_labels, options)
pred = softmax_predict(softmax_model, test_data)

acc = np.mean(test_labels == pred)
print("The Accuracy (with raw pixels as features): {:5.2f}% \n".format(acc*100))
