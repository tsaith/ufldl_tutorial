import numpy as np
from scipy.io import loadmat


def sample_images(fname):
    """
    Sample images form a Matlab file.
    """

    image_data = loadmat(fname)['IMAGES']
    image_rows = image_data.shape[0]
    image_cols = image_data.shape[1]
    n_images = image_data.shape[2]

    n_patches = 10000
    patch_rows = 8
    patch_cols = 8
    patches = np.zeros((patch_rows*patch_cols, n_patches), dtype=np.float64)

    rows_diff = image_rows - patch_rows
    cols_diff = image_cols - patch_cols
    for i in range(n_patches):
        image_id = np.random.randint(0, n_images)
        x = np.random.randint(0, rows_diff)
        y = np.random.randint(0, cols_diff)
        patch = image_data[y:y+patch_rows, x:x+patch_cols, image_id].ravel()
        patches[:, i] = patch

    # Normalize data
    patches = normalize_data(patches)

    return patches

def normalize_data(patches):
    """
    Squash data to [0.1, 0.9] since we use sigmoid as the activation
    function in the output layer
    """

    # Remove the DC (mean of images)
    patches -= patches.mean(axis=0)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3.0 * np.std(patches)
    patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd

    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches + 1) * 0.4 + 0.1

    return patches

def sample_images_raw(fname):
    image_data = loadmat(fname)['IMAGESr']

    patch_size = 12
    n_patches = 10000
    image_size = image_data.shape[0]
    n_images = image_data.shape[2]

    patches = np.zeros(shape=(patch_size * patch_size, n_patches))

    for i in range(n_patches):
        image_id = np.random.randint(0, n_images)
        image_x = np.random.randint(0, image_size - patch_size)
        image_y = np.random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(-1)
        patches[:, i] = patch

    return patches
