import numpy as np
import scipy.signal


def cnn_convolve(patch_dim, n_features, images, W, b, zca_white, mean_patch):
    """
    Returns the convolution of the features given by W and b with the given images.

    Parameters
    ----------
    patch_dim: patch (feature) dimension
    n_features: number of features
    images: large images to convolve with,
            matrix with shape (rows, cols, channels, n_images)
    W, b: weights and bias for features from the sparse autoencoder
    zca_white: ZCA whitening matrix
    mean_patch: mean patch

    Return
    ------
%   convolved_features: convolved features with shape (n_features, n_images, rows, cols)
    """

    image_dim, _, n_channels, n_images = images.shape
    patch_size = patch_dim * patch_dim

    """
    Instructions:
%     Convolve every feature with every large image here to produce the
%     n_features x n_images x (image_dim - patch_dim + 1) x (image_dim - patch_dim + 1)
%     matrix convolved_features, such that
%     convolved_features[feature_num, image_num, image_row, image_col] is the
%   value of the convolved feature_num feature for the image_num image over
%     the region (image_cow, image_col) to (image_row + patch_dim - 1, image_col + patch_dim - 1)
%
%   Expected running times:
%     Convolving with 100 images should take less than 3 minutes
%     Convolving with 5000 images should take around an hour
%     (So to save time when testing, you should convolve with less images, as
%     described earlier)
    """

    # Precompute the matrices that will be used during the convolution. Recall
    #that you need to take into account the whitening and mean subtraction steps

    convolved_features = np.zeros((n_features, n_images, image_dim - patch_dim + 1, image_dim - patch_dim + 1), dtype=np.float64)

    WT = W.dot(zca_white) # With shape (n_features, n_channels*patch_size)
    b_eff = b - WT.dot(mean_patch) # Effective b

    for image_num in range(n_images):
        for feature_num in range(n_features):

            # convolution of image with feature matrix for each channel
            convolved_image = np.zeros((image_dim - patch_dim + 1, image_dim - patch_dim + 1), dtype=np.float64)

            for channel in range(n_channels):

                # Obtain the feature with shape (patch_dim, patch_dim) needed during the convolution
                ia = patch_size * channel
                iz = patch_size * (channel + 1)
                feature = WT[feature_num, ia:iz].reshape((patch_dim, patch_dim))

                # Flip the feature matrix because of the definition of convolution, as explained later
                feature = np.flipud(np.fliplr(feature))

                # Obtain the image
                im = images[:, :, channel, image_num]

                # Convolve "feature" with "im", adding the result to convolved_image
                # be sure to do a 'valid' convolution
                convolved_image += scipy.signal.convolve2d(im, feature, mode='valid')

            # Add the bias unit (correcting for the mean subtraction as well)
            # Then, apply the sigmoid function to get the hidden activation
            # The convolved feature is the sum of the convolved values for all channels
            z = convolved_image + b_eff[feature_num]
            convolved_features[feature_num, image_num, :, :] = sigmoid(z)

    return convolved_features

def cnn_pool(pool_dim, convolved_features):
    """
    Pools the given convolved features.

    Parameters
    ----------
    pool_dim: integer
       Dimension of pooling region.

    convolved_features: array, shape (n_features, n_images, rows, cols)
        Convolved features to pool.

    Returns
    -------
    pooled_features: array, shape (n_features, n_images, rows, cols)
        Matrix of pooled features.
    """

    n_features, n_images, convolved_dim, _ = convolved_features.shape

    assert convolved_dim % pool_dim == 0, "Convolved dimension should be a multiple of pooling dimension"
    region_dim = int(convolved_dim / pool_dim)

    pooled_features = np.zeros((n_features, n_images, region_dim, region_dim))

    """
    Instructions:

    Now pool the convolved features in regions of pool_dim x pool_dim,
    to obtain the
    n_features x n_images x region_dim x region_dim
    matrix pooled_features, such that
    pooled_features[feature_num, image_num, pool_row, pool_col] is the
    value of the feature_num feature for the image_num image pooled over the
    corresponding (pool_row, pool_col) pooling region
    (see http://ufldl/wiki/index.php/Pooling )

    Use mean pooling here.
    """

    for i in range(region_dim): # Row
        for j in range(region_dim): # Column
            pool = convolved_features[:, :, i*pool_dim: (i+1)*pool_dim, j*pool_dim: (j+1)*pool_dim]
            pooled_features[:, :, i, j] = np.mean(pool, axis=(2, 3))

    return pooled_features


def sigmoid(x):
    # Return value of the Sigmoid function
    return 1.0 / (1.0 + np.exp(-x))
