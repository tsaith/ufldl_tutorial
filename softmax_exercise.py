import numpy as np

from load_MNIST import load_MNIST_images, load_MNIST_labels
from check_numerical_gradient import compute_numerical_gradient
from softmax import softmax_cost, softmax_train, softmax_predict


"""
Instructions
------------
  This file contains code that helps you get started on the
  softmax exercise. You will need to write the softmax cost function
  in softmaxCost.m and the softmax prediction function in softmaxPred.m.
  For this exercise, you will not need to change any code in this file,
  or any other files other than those mentioned above.
  (However, you may be required to do so in later exercises)

======================================================================
STEP 0: Initialise constants and parameters

  Here we define and initialise some constants which allow your code
  to be used more generally on any arbitrary input.
  We also initialise some parameters used for tuning the model.
"""

input_size = 28 * 28 # Size of input vector (MNIST images are 28x28)
n_classes = 10       # Number of classes (MNIST images fall into 10 classes)

lambda_ = 1e-4       # Weight decay parameter

"""
STEP 1: Load data

  In this section, we load the input and output data.
  For softmax regression on MNIST pixels,
  the input data is the images, and
  the output data is the labels.

  Change the filenames if you've saved the files under different names
  On some platforms, the files might be saved as
  train-images.idx3-ubyte / train-labels.idx1-ubyte
"""

images = load_MNIST_images('data/mnist/train-images-idx3-ubyte')
labels = load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')
input_data = images

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking.
# Here, we create synthetic dataset using random data for testing
debug = False
if debug:
    input_size = 8 * 8
    input_data = np.random.randn(input_size, 100)
    labels = np.random.randint(n_classes, size=100)

# Randomly initialise theta
theta = 0.005 * np.random.randn(n_classes * input_size)

"""
STEP 2: Implement softmax_cost

  Implement softmax_cost in softmax_cost.py.
"""

cost, grad = softmax_cost(theta, n_classes, input_size, lambda_, input_data, labels)

"""
STEP 3: Gradient checking

  As with any learning algorithm, you should always check that your
  gradients are correct before learning the parameters.
"""

if debug:

    # Now we can use it to check your cost function and derivative calculations
    # for the sparse autoencoder.
    J = lambda theta : softmax_cost(theta, n_classes, input_size, lambda_, input_data, labels)[0]
    numgrad = compute_numerical_gradient(J, theta)

    # Use this to visually compare the gradients side by side
    n = min(grad.size, 20) # Number of gradients to display
    for i in range(n):
        print("{0:20.12f} {1:20.12f}".format(numgrad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Compare numerically computed gradients with those computed analytically
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("Norm of difference = ", diff)

    # The difference should be small.
    # In our implementation, these values are usually less than 1e-7.

    # When your gradients are correct, congratulations!

"""
STEP 4: Learning parameters

  Once you have verified that your gradients are correct,
  you can start training your softmax regression code using softmax_train.
"""

options = {'maxiter': 100, 'disp': True}
model = softmax_train(input_size, n_classes, lambda_, input_data, labels, options)

# Although we only use 100 iterations here to train a classifier for the
# MNIST data set, in practice, training for more iterations is usually
# beneficial.

"""
STEP 5: Testing

  You should now test your model against the test images.
  To do this, you will first need to write softmax_predict,
  which should return predictions given a softmax model and the input data.
"""
images = load_MNIST_images('data/mnist/t10k-images-idx3-ubyte')
labels = load_MNIST_labels('data/mnist/t10k-labels-idx1-ubyte')
input_data = images

# Make predictions
pred = softmax_predict(model, input_data)

acc = np.mean(labels == pred)
print("Accuracy: {:5.2f}% \n".format(acc*100))

"""
Accuracy is the proportion of correctly classified images
After 100 iterations, the results for our implementation were:

Accuracy: 92.200%

If your values are too low (accuracy less than 0.91), you should check
your code for errors, and make sure you are training on the
entire data set of 60000 28x28 training images
(unless you modified the loading code, this should be the case)
"""
