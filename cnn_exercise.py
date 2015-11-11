# CS294A/CS294W Convolutional Neural Networks Exercise

import numpy as np
import scipy.optimize
import scipy.io
import matplotlib.pyplot as plt
import time
import datetime

from cnn import cnn_convolve, cnn_pool
from sparse_autoencoder import initialize_parameters, sparse_autoencoder_linear_cost, feedforward_autoencoder
from softmax import softmax_train, softmax_predict
from display_network import display_color_network
from sklearn.externals import joblib


"""
  Instructions
  ------------

  This file contains code that helps you get started on the
  convolutional neural networks exercise. In this exercise, you will only
  need to modify cnnConvolve.m and cnnPool.m. You will not need to modify
  this file.

  ======================================================================
  STEP 0: Initialization
  Here we initialize some parameters used for the exercise.

"""

image_dim = 64         # image dimension
image_channels = 3     # number of channels (rgb, so 3)

patch_dim = 8          # patch dimension
n_patches = 50000      # number of patches

visible_size = patch_dim * patch_dim * image_channels # number of input units
output_size = visible_size  # number of output units
hidden_size = 400           # number of hidden units

epsilon = 0.1          # epsilon for ZCA whitening

pool_dim = 19          # dimension of pooling region

"""
  STEP 1: Train a sparse autoencoder (with a linear decoder) to learn
   features from color patches. If you have completed the linear decoder
   execise, use the features that you have obtained from that exercise,
   loading them into opt_theta. Recall that we have to keep around the
   parameters used in whitening (i.e., the ZCA whitening matrix and the
   mean_patch)
"""

# Loading the learned features and preprocessing matrices
load_data = joblib.load("data/STL10_features.pkl")

opt_theta  = load_data['opt_theta']
zca_white  = load_data['zca_white']
mean_patch = load_data['mean_patch']

W = opt_theta[0:visible_size * hidden_size].reshape((hidden_size, visible_size))
b = opt_theta[2*hidden_size*visible_size:2*hidden_size*visible_size + hidden_size]

# Display and check to see that the features look good
image = display_color_network( (W.dot(zca_white)).T )
plt.imsave('cnn_learned_features.png', image)
plt.imshow(image)
"""
  STEP 2: Implement and test convolution and pooling
  In this step, you will implement convolution and pooling, and test them
  on a small part of the data set to ensure that you have implemented
  these two functions correctly. In the next step, you will actually
  convolve and pool the features with the STL10 images.
"""

"""
  STEP 2a: Implement convolution
  Implement convolution in the function cnnConvolve in cnnConvolve.m
"""
# Note that we have to preprocess the images in the exact same way
# we preprocessed the patches before we can obtain the feature activations.

# Load training images and labels
train_subset = scipy.io.loadmat('data/stlTrainSubset.mat')
n_train_images = train_subset['numTrainImages'][0, 0]
train_images   = train_subset['trainImages'] # shape (rows, cols, channels, n_train_images)
train_labels   = train_subset['trainLabels'][:, 0]

# Use only the first 8 images for testing
conv_images = train_images[:, :, :, :8]

# NOTE: Implement cnn_convolve first!
convolved_features = cnn_convolve(patch_dim, hidden_size, conv_images, W, b, zca_white, mean_patch)


"""
  STEP 2b: Checking your convolution
  To ensure that you have convolved the features correctly, we have
  provided some code to compare the results of your convolution with
  activations from the sparse autoencoder
"""

# For 1000 random points
for i in range(1000):
    feature_num = np.random.randint(hidden_size)
    image_num = np.random.randint(8)
    image_row = np.random.randint(image_dim - patch_dim + 1)
    image_col = np.random.randint(image_dim - patch_dim + 1)

    patch = conv_images[image_row:image_row + patch_dim, image_col:image_col + patch_dim, :, image_num]
    patch = np.concatenate((patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
    patch = (patch-mean_patch).reshape((-1, 1))
    patch = zca_white.dot(patch) # ZCA whitening

    features = feedforward_autoencoder(opt_theta, hidden_size, visible_size, patch)

    if abs(features[feature_num, 0] - convolved_features[feature_num, image_num, image_row, image_col]) > 1e-9:
        print('Convolved feature does not match activation from autoencoder\n')
        print('Feature Number    : {}\n'.format(feature_num))
        print('Image Number      : {}\n'.format(image_num))
        print('Image Row         : {}\n'.format(image_row))
        print('Image Column      : {}\n'.format(image_col))
        print('Convolved feature : {:f}\n'.format(convolved_features[feature_num, image_num, image_row, image_col]))
        print('Sparse AE feature : {:f}\n'.format(features[feature_num, 0]))
        print('Error! Convolved feature does not match activation from autoencoder')

print('Congratulations! Your convolution code passed the test.')


"""
  STEP 2c: Implement pooling
  Implement pooling in the function cnnPool in cnnPool.m
"""

# NOTE: Implement cnn_pool first!
pooled_features = cnn_pool(pool_dim, convolved_features)

"""
  STEP 2d: Checking your pooling
  To ensure that you have implemented pooling, we will use your pooling
  function to pool over a test matrix and check the results.
"""

test_matrix = np.arange(64, dtype=np.float64).reshape((8, 8))
expected_matrix = np.array([[test_matrix[0:4, 0:4].mean(), test_matrix[0:4, 4:8].mean()],
                            [test_matrix[4:8, 0:4].mean(), test_matrix[4:8, 4:8].mean()]])

test_matrix = test_matrix.reshape((1, 1, 8, 8))

pooled_features = cnn_pool(4, test_matrix)

if not (pooled_features == expected_matrix).all():
    print('Pooling incorrect')
    print('Expected');
    print(expected_matrix)
    print('Got');
    print(pooled_features)
else:
    print('Congratulations! Your pooling code passed the test.')


"""
  STEP 3: Convolve and pool with the dataset
  In this step, you will convolve each of the features you learned with
  the full large images to obtain the convolved features. You will then
  pool the convolved features to obtain the pooled features for
  classification.

  Because the convolved features matrix is very large, we will do the
  convolution and pooling 50 features at a time to avoid running out of
  memory. Reduce this number if necessary
"""

step_size = 50
assert hidden_size % step_size == 0, 'step_size should divide hidden_size'

train_subset = scipy.io.loadmat('data/stlTrainSubset.mat')
n_train_images = train_subset['numTrainImages'][0, 0]
train_images   = train_subset['trainImages'] # shape (rows, cols, channels, n_train_images)
train_labels   = train_subset['trainLabels'][:, 0]
train_labels = train_labels - 1 # Start from 0 instead of 1

test_subset = scipy.io.loadmat('data/stlTestSubset.mat')
n_test_images = test_subset['numTestImages'][0, 0]
test_images   = test_subset['testImages'] # shape (rows, cols, channels, n_test_images)
test_labels   = test_subset['testLabels'][:, 0]
test_labels = test_labels - 1 # Start from 0 instead of 1

region_dim = int(np.floor((image_dim - patch_dim + 1) / pool_dim))
pooled_features_train = np.zeros((hidden_size, n_train_images, region_dim, region_dim))
pooled_features_test  = np.zeros((hidden_size, n_test_images, region_dim, region_dim))


start_time = time.time()
for conv_part in range(int(hidden_size / step_size)):

    feature_start = conv_part * step_size
    feature_end = (conv_part + 1) * step_size

    print('Step {:d}: features {:d} to {:d}\n'.format(conv_part, feature_start, feature_end))
    Wt = W[feature_start:feature_end, :]
    bt = b[feature_start:feature_end]

    print('Convolving and pooling train images\n')
    convolved_features_this = cnn_convolve(patch_dim, step_size,
        train_images, Wt, bt, zca_white, mean_patch)
    pooled_features_this = cnn_pool(pool_dim, convolved_features_this)
    pooled_features_train[feature_start:feature_end, :, :, :] = pooled_features_this

    print("Time elapsed: {} \n".format(datetime.timedelta(seconds=time.time() - start_time)))

    print('Convolving and pooling test images\n')
    convolved_features_this = cnn_convolve(patch_dim, step_size,
        test_images, Wt, bt, zca_white, mean_patch)
    pooled_features_this = cnn_pool(pool_dim, convolved_features_this)
    pooled_features_test[feature_start:feature_end, :, :, :] = pooled_features_this

    print("Time elapsed: {} \n".format(datetime.timedelta(seconds=time.time() - start_time)))

# You might want to save the pooled features since convolution and pooling takes a long time
saved_data = {}
saved_data['pooled_features_train'] = pooled_features_train
saved_data['pooled_features_test'] = pooled_features_test
joblib.dump(saved_data, "data/cnn_pooled_features.pkl", compress=3)

"""
  STEP 4: Use pooled features for classification
  Now, you will use your pooled features to train a softmax classifier,
  using softmax_train from the softmax exercise.
  Training the softmax classifer for 1000 iterations should take less than
  10 minutes.
"""

# Load pooled features
pooled_features_data = joblib.load("data/cnn_pooled_features.pkl")
pooled_features_train = pooled_features_data['pooled_features_train']
pooled_features_test  = pooled_features_data['pooled_features_test']

# Setup parameters for softmax
softmax_lambda = 1e-4
n_classes = 4

softmax_input_size = int(pooled_features_train.size / n_train_images)

# Reshape the pooled_features to form an input vector for softmax
softmax_X = np.transpose(pooled_features_train, axes=[0, 2, 3, 1])
softmax_X = softmax_X.reshape((softmax_input_size, n_train_images))
softmax_Y = train_labels

options = {'maxiter': 200, 'disp': True}
softmax_model = softmax_train(softmax_input_size,
                              n_classes, softmax_lambda, softmax_X, softmax_Y, options)


"""
  STEP 5: Test classifer
  Now you will test your trained classifer against the test images

"""
softmax_input_size = int(pooled_features_test.size / n_test_images)

softmax_X = np.transpose(pooled_features_test, axes=[0, 2, 3, 1])
softmax_X = softmax_X.reshape((softmax_input_size, n_test_images))
softmax_Y = test_labels

# Make predictions
pred = softmax_predict(softmax_model, softmax_X)

acc = np.mean(softmax_Y == pred)
print("Accuracy: {:5.2f}% \n".format(acc*100))

# You should expect to get an accuracy of around 80% on the test images.
