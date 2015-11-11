import numpy as np


def load_MNIST_images(filename):
    """
    Return a 2d array of images from MNIST dataset.
    images : array, shape (n_images, n_pixels)
             n_pixels = 28*28 = 784
    filename: input data file
    """
    with open(filename, "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        n_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((n_images, rows * cols))
        images = images.T
        images = images.astype(np.float64) / 255

        f.close()

        return images


def load_MNIST_labels(filename):
    """
    Return an array of image labels from MNIST dataset.
    labels : array, shape (n_labels)
    filename: input file for labels
    """
    with open(filename, 'r') as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        n_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        labels = np.fromfile(f, dtype=np.uint8)

        f.close()

        return labels
