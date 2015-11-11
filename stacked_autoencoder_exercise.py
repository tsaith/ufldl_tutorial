# CS294A/CS294W Stacked Autoencoder Exercise

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from load_MNIST import load_MNIST_images, load_MNIST_labels
from sparse_autoencoder import initialize_parameters, sparse_autoencoder_cost, feedforward_autoencoder
from display_network import display_network
from softmax import softmax_train
from stacked_autoencoder import stack2params, params2stack, stacked_ae_cost, check_stacked_ae_cost, stacked_ae_predict


"""
Instructions
------------

  This file contains code that helps you get started on the
  sstacked autoencoder exercise. You will need to complete code in
  stacked_ae_cost
  You will also need to have implemented sparse_autoencoder_cost and
  softmax_cost from previous exercises. You will need the initialize_parameters
  load_mnist_images, and load_mnist_labels from previous exercises.


STEP 0: Here we provide the relevant parameters values that will
  allow your sparse autoencoder to get good filters; you do not need to
  change the parameters below.

"""

input_size = 28 * 28
n_classes = 10         # Number of classes
hidden_size_L1 = 200   # Layer 1 Hidden Size
hidden_size_L2 = 200   # Layer 2 Hidden Size
sparsity_param = 0.1   # desired average activation of the hidden units.
                       # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                       #  in the lecture notes).
lambda_ = 3e-3         # weight decay parameter
beta = 3               # weight of sparsity penalty term

maxiter = 400          # Maximum iterations for training

"""
STEP 1: Load data from the MNIST database

  This loads our training data from the MNIST database files.
"""

# Load MNIST database files
# Load MNIST database files
train_data   = load_MNIST_images('data/mnist/train-images-idx3-ubyte')
train_labels = load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')


"""
STEP 2: Train the first sparse autoencoder

  This trains the first sparse autoencoder on the unlabelled STL training images.
  If you've correctly implemented sparse_autoencoder_cost, you don't need
  to change anything here.
"""

# Randomly initialize the parameters
sae1_theta = initialize_parameters(hidden_size_L1, input_size)

#  Instructions: Train the first layer sparse autoencoder, this layer has
#                an hidden size of "hidden_size_L1"
#                You should store the optimal parameters in sae1_opt_theta

J = lambda theta : sparse_autoencoder_cost(theta, input_size, hidden_size_L1, lambda_, sparsity_param, beta, train_data)

options = {'maxiter': maxiter, 'disp': True}

results = scipy.optimize.minimize(J, sae1_theta, method='L-BFGS-B', jac=True, options=options)
sae1_opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)

# Visualize weights
visualize_weights = False
if visualize_weights:
    W1 = sae1_opt_theta[0:hidden_size_L1*input_size].reshape((hidden_size_L1, input_size))
    image = display_network(W1.T)
    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


"""
STEP 2: Train the second sparse autoencoder

  This trains the second sparse autoencoder on the first autoencoder featurse.
  If you've correctly implemented sparse_autoencoder_cost, you don't need
  to change anything here.
"""

sae1_features = feedforward_autoencoder(sae1_opt_theta, hidden_size_L1, input_size, train_data)

#  Randomly initialize the parameters
sae2_theta = initialize_parameters(hidden_size_L2, hidden_size_L1)

#  Instructions: Train the second layer sparse autoencoder, this layer has
#                an hidden size of "hidden_size_L2" and an input size of "hidden_size_L1"
#                You should store the optimal parameters in sae2_opt_theta
J = lambda theta : sparse_autoencoder_cost(theta, hidden_size_L1, hidden_size_L2,
    lambda_, sparsity_param, beta, sae1_features)

options = {'maxiter': maxiter, 'disp': True}

results = scipy.optimize.minimize(J, sae2_theta, method='L-BFGS-B', jac=True, options=options)
sae2_opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)


"""
STEP 3: Train the softmax classifier

  This trains the sparse autoencoder on the second autoencoder features.
  If you've correctly implemented softmax_cost, you don't need
  to change anything here.
"""

sae2_features = feedforward_autoencoder(sae2_opt_theta, hidden_size_L2, hidden_size_L1, sae1_features)

#  Instructions: Train the softmax classifier, the classifier takes in
#                input of dimension "hidden_sizeL2" corresponding to the
#                hidden layer size of the 2nd layer.
#
#                You should store the optimal parameters in sae_softmax_opt_theta


options = {'maxiter': maxiter, 'disp': True}
softmax_model = softmax_train(hidden_size_L2, n_classes, lambda_, sae2_features, train_labels, options)
softmax_opt_theta = softmax_model['opt_theta']


"""
STEP 5: Finetune softmax model
"""

# Implement the stacked_ae_cost to give the combined cost of the whole model then run this cell.

# Initialize the stack using the parameters learned

n_stack = 2 # Two layers
stack = [{} for i in range(n_stack)]

stack[0]['w'] = sae1_opt_theta[0:hidden_size_L1*input_size].reshape((hidden_size_L1, input_size))
stack[0]['b'] = sae1_opt_theta[2*hidden_size_L1*input_size: 2*hidden_size_L1*input_size + hidden_size_L1]

stack[1]['w'] = sae2_opt_theta[0:hidden_size_L2*hidden_size_L1].reshape((hidden_size_L2, hidden_size_L1))
stack[1]['b'] = sae2_opt_theta[2*hidden_size_L2*hidden_size_L1: 2*hidden_size_L2*hidden_size_L1 + hidden_size_L2]

# Initialize the parameters for the deep model
stack_params, net_config = stack2params(stack)
stacked_ae_theta = np.concatenate((softmax_opt_theta, stack_params))

# Instructions: Train the deep network, hidden size here refers to the
#               dimension of the input to the classifier, which corresponds
#               to "hidden_size_L2".

J = lambda theta : stacked_ae_cost(theta, input_size, hidden_size_L2, n_classes, net_config, lambda_, train_data, train_labels)

#check_stacked_ae_cost() # Verify the correctness

# Find out the optimal theta
options = {'maxiter': maxiter, 'disp': True}
results = scipy.optimize.minimize(J, stacked_ae_theta, method='L-BFGS-B', jac=True, options=options)
stacked_ae_opt_theta = results['x']

print(results)

"""
STEP 6: Test
  Instructions: You will need to complete the code in stacked_ae_predict
                before running this part of the code
"""

# Get labelled test images
# Note that we apply the same kind of preprocessing as the training set
test_data   = load_MNIST_images('/Users/tsaith/projects/ufldl_tutorial/data/mnist/t10k-images-idx3-ubyte')
test_labels = load_MNIST_labels('/Users/tsaith/projects/ufldl_tutorial/data/mnist/t10k-labels-idx1-ubyte')

pred = stacked_ae_predict(stacked_ae_theta, input_size, hidden_size_L2, n_classes, net_config, test_data)

acc = np.mean(test_labels == pred)
print("Before Finetuning Test Accuracy: {:5.2f}% \n".format(acc*100))

pred = stacked_ae_predict(stacked_ae_opt_theta, input_size, hidden_size_L2, n_classes, net_config, test_data)

acc = np.mean(test_labels == pred)
print("After Finetuning Test Accuracy: {:5.2f}% \n".format(acc*100))

"""
Accuracy is the proportion of correctly classified images
The results for our implementation were:

Before Finetuning Test Accuracy: 87.7%
After Finetuning Test Accuracy:  97.6%

If your values are too low (accuracy less than 95%), you should check
your code for errors, and make sure you are training on the
entire data set of 60000 28x28 training images
(unless you modified the loading code, this should be the case)
"""
