import numpy as np
import scipy


def sparse_autoencoder_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    """
    Vectorized version to compute the cost and derivative of sparse autoencoder.

    visible_size: the number of input units (probably 64)
    hidden_size: the number of hidden units (probably 25)
    lambda_: weight decay parameter
    sparsity_param: The desired average activation for the hidden units (denoted in the lecture
                    notes by the greek alphabet rho, which looks like a lower-case "p").
    beta: weight of sparsity penalty term
    data: Our 10000x64 matrix containing the training data.  So, data(i, :) is the i-th training example.
    """

    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    W2 = theta[hidden_size*visible_size : 2*hidden_size*visible_size].reshape((visible_size, hidden_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

    # Number of instances
    m = data.shape[1]

    # Forward pass
    a1 = data              # Input activation
    z2 = W1.dot(a1) + b1.reshape((-1, 1))
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2.reshape((-1, 1))
    h  = sigmoid(z3)       # Output activation
    y  = a1

    # Compute rho_hat used in sparsity penalty
    rho = sparsity_param
    rho_hat = np.mean(a2, axis=1)
    sparsity_delta = (-rho/rho_hat + (1.0-rho)/(1-rho_hat)).reshape((-1, 1))

    # Backpropagation
    delta3 = (h-y)*sigmoid_prime(z3)
    delta2 = (W2.T.dot(delta3) + beta*sparsity_delta)*sigmoid_prime(z2)

    # Compute the cost
    squared_error_term = np.sum((h-y)**2) / (2.0*m)
    weight_decay = 0.5*lambda_*(np.sum(W1*W1) + np.sum(W2*W2))
    sparsity_term = beta*np.sum(KL_divergence(rho, rho_hat))
    cost = squared_error_term + weight_decay + sparsity_term

    # Compute the gradients
    W1grad = delta2.dot(a1.T)/m + lambda_*W1
    W2grad = delta3.dot(a2.T)/m + lambda_*W2
    b1grad = np.mean(delta2, axis=1)
    b2grad = np.mean(delta3, axis=1)
    grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

    return cost, grad

def sparse_autoencoder_cost_original(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    """
    Partially vectorized version to compute the cost and derivative of sparse autoencoder.

    visible_size: the number of input units (probably 64)
    hidden_size: the number of hidden units (probably 25)
    lambda_: weight decay parameter
    sparsity_param: The desired average activation for the hidden units (denoted in the lecture
                    notes by the greek alphabet rho, which looks like a lower-case "p").
    beta: weight of sparsity penalty term
    data: Our 10000x64 matrix containing the training data.  So, data(i, :) is the i-th training example.
    """

    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    W2 = theta[hidden_size*visible_size : 2*hidden_size*visible_size].reshape((visible_size, hidden_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

    # Cost and gradient variables (your code needs to compute these values).
    # Here, we initialize them to zeros.
    cost = 0
    W1grad = np.zeros(W1.shape)
    W2grad = np.zeros(W2.shape)
    b1grad = np.zeros(b1.shape)
    b2grad = np.zeros(b2.shape)

    rho = np.tile(sparsity_param, hidden_size)
    rho_hat = np.zeros(hidden_size)
    squared_error_term = 0.0

    """
    Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
                  and the corresponding gradients W1grad, W2grad, b1grad, b2grad.

    W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
    Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
    as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
    respect to W1.  I.e., W1grad[i,j] should be the partial derivative of J_sparse(W,b)
    with respect to the input parameter W1[i,j].  Thus, W1grad should be equal to the term
    [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2
    of the lecture notes (and similarly for W2grad, b1grad, b2grad).

    Stated differently, if we were using batch gradient descent to optimize the parameters,
    the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
    """

    m = data.shape[1]

    # For rho_hat
    for i in range(m):

        # Forward pass to a2
        a1 = data[:, i]
        z2 = W1.dot(a1) + b1
        a2 = sigmoid(z2)

        rho_hat += a2

    rho_hat /= m

    # For cost and derivative
    for i in range(m):

        # Forward pass
        a1 = data[:, i]
        z2 = W1.dot(a1) + b1
        a2 = sigmoid(z2)
        z3 = W2.dot(a2) + b2
        h = sigmoid(z3)
        y = a1

        # Accumulate the squared error
        diff = h - y
        squared_error_term += 0.5*np.sum(diff*diff)

        # Backpropagation
        delta3 = (h-y)*sigmoid_prime(z3)
        sparsity_delta = -rho/rho_hat + (1.0-rho)/(1-rho_hat)
        delta2 = (W2.T.dot(delta3) + beta*sparsity_delta)*sigmoid_prime(z2)

        W1grad += delta2.reshape((-1, 1))*a1
        W2grad += delta3.reshape((-1, 1))*a2
        b1grad += delta2
        b2grad += delta3

    # Compute the cost
    squared_error_term /= m
    weight_decay = 0.5*lambda_*(np.sum(W1*W1) + np.sum(W2*W2))
    sparsity_term = beta*np.sum(KL_divergence(rho, rho_hat))

    cost = squared_error_term + weight_decay + sparsity_term

    # Compute the gradients
    W1grad = W1grad/m + lambda_*W1
    W2grad = W2grad/m + lambda_*W2
    b1grad /= m
    b2grad /= m
    grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

    return cost, grad

def predict(data, theta, visible_size, hidden_size):
    """
    x: input data.
    visible_size: the number of input units (probably 64)
    hidden_size: the number of hidden units (probably 25)
    """

    # The input theta is a vector (because minFunc expects the parameters to be a vector). 
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
    # follows the notation convention of the lecture notes. 

    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    W2 = theta[hidden_size*visible_size : 2*hidden_size*visible_size].reshape((visible_size, hidden_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size].reshape((-1, 1))
    b2 = theta[2*hidden_size*visible_size+hidden_size:].reshape((-1, 1))

    # Forward pass
    a1 = data
    z2 = W1.dot(a1) + b1
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2
    h = sigmoid(z3)

    return h

def sparse_autoencoder_linear_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    """
    Compute the cost and derivative of sparse autoencoder with linear decoder.

    visible_size: the number of input units
    hidden_size: the number of hidden units
    lambda_: weight decay parameter
    sparsity_param: The desired average activation for the hidden units (denoted in the lecture
                    notes by the greek alphabet rho, which looks like a lower-case "p").
    beta: weight of sparsity penalty term
    data: Our 10000x64 matrix containing the training data.  So, data(i, :) is the i-th training example.
    """

    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    W2 = theta[hidden_size*visible_size : 2*hidden_size*visible_size].reshape((visible_size, hidden_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

    # Number of instances
    m = data.shape[1]

    # Forward pass
    a1 = data              # Input activation
    z2 = W1.dot(a1) + b1.reshape((-1, 1))
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2.reshape((-1, 1))
    h  = z3                # Output activation
    y  = a1

    # Compute rho_hat used in sparsity penalty
    rho = sparsity_param
    rho_hat = np.mean(a2, axis=1)
    sparsity_delta = (-rho/rho_hat + (1.0-rho)/(1-rho_hat)).reshape((-1, 1))

    # Backpropagation
    delta3 = -(y-h)
    delta2 = (W2.T.dot(delta3) + beta*sparsity_delta)*sigmoid_prime(z2)

    # Compute the cost
    squared_error_term = np.sum((h-y)**2) / (2.0*m)
    weight_decay = 0.5*lambda_*(np.sum(W1*W1) + np.sum(W2*W2))
    sparsity_term = beta*np.sum(KL_divergence(rho, rho_hat))
    cost = squared_error_term + weight_decay + sparsity_term

    # Compute the gradients
    W1grad = delta2.dot(a1.T)/m + lambda_*W1
    W2grad = delta3.dot(a2.T)/m + lambda_*W2
    b1grad = np.mean(delta2, axis=1)
    b2grad = np.mean(delta3, axis=1)
    grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

    return cost, grad


def feedforward_autoencoder(theta, hidden_size, visible_size, data):
    """
    Feedforward autoencoder.

    theta: trained weights from the autoencoder
    visible_size: the number of input units
    hidden_size: the number of hidden units
    data: Our matrix containing the training data as columns.  So, data[:, i] is the i-th training example. 
    """

    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
    # follows the notation convention of the lecture notes. 

    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size].reshape((-1, 1))
    
    # Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.
    a1 = data
    z2 = W1.dot(a1) + b1
    a2 = sigmoid(z2)

    return a2


def sigmoid(x):
    """
    Return the sigmoid (aka logistic) function, 1 / (1 + exp(-x)). 
    """
    return scipy.special.expit(x)


def sigmoid_prime(x):
    """
    Return the first derivative of the sigmoid function. 
    """
    f = sigmoid(x)
    df = f*(1.0-f)
    return df

def initialize_parameters(hidden_size, visible_size):
    """
    Return the initial theta.
    """

    # Initialize parameters randomly based on layer sizes.
    r  = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    # we'll choose weights uniformly from the interval [-r, r)
    W1 = np.random.random((hidden_size, visible_size)) * 2.0 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2.0 * r - r

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)

    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all
    # your parameters into a vector.
    theta = np.hstack((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))

    return theta


def KL_divergence(p, q):
    """
    Kullback-Leiber divergence.
    """
    
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
