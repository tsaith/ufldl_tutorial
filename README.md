## Unsupervised Feature Learning and Deep Learning Tutorial 

Implement the exercises of UFLDL Tutorial with python 3

Tutorial Website: http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

### Packages required

* [Numpy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)
* [scikit-learn](http://scikit-learn.org/stable/)

### Sparse Autoencoder

* load_MNIST.py: Load MNIST dataset
* sample_images.py: Load sample images for testing sparse autoencoder
* sparse_autoencoder.py: Functions used in sparse autoencoder
* train.py: Train sparse autoencoder with MNIST data and visualize learnt features
* check_numerical_gradient.py: Check numerical gradients
* display_network.py: Display visualized features

### Preprocessing: PCA & Whitening

* pca_2d.py: PCA, PCA whitening and ZCA whitening in 2D
* pca_gen.py: PCA and Whitening on natural images

### Softmax Regression

* softmax.py: Functions used in softmax regression
* softmax_exercise.py: Classify MNIST digits

### Self-Taught Learning and Unsupervised Feature Learning

* stl_exercise.py: Classify MNIST digits with self-taught learning and softmax regression

### Building Deep Networks for Classification

* stacked_autoencoder.py: Functions used in stacked autoencoder
* stacked_autoencoder_exercise.py: Use a stacked autoencoder for digit classification

### Linear Decoders with Autoencoders

* linear_decoder_exercise.py: Implement a linear decoder and apply it to learn features on color images

### Working with Large Images (Convolutional Neural Networks)

* cnn.py: Functions used in convolution neural networks
* cnn_exercise.py: Classify STL-10 images
