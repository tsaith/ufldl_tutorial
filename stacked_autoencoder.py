import numpy as np
import scipy.optimize
from softmax import softmax_predict
from check_numerical_gradient import compute_numerical_gradient


def stacked_ae_cost(theta, input_size, hidden_size,
                    n_classes, net_config, lambda_, data, labels):
    """
    Takes a trained softmax_theta and a training data set with labels,
    and returns cost and gradient using a stacked autoencoder model. Used for finetuning.

    theta: trained weights from the autoencoder
    input_size:  the number of input units
    hidden_size: the number of hidden units *at the 2nd layer*
    n_classes:   the number of categories
    net_config:  the network configuration of the stack
    lambda_:     the weight regularization penalty
    data: our matrix containing the training data as columns.  So, data[:,i] is the i-th training example.
    labels: a vector containing labels, where labels[i] is the label for the i-th training example
    """

    # We first extract the part which compute the softmax gradient
    softmax_theta = theta[0:hidden_size*n_classes].reshape((n_classes, hidden_size))

    # Extract out the "stack"
    stack = params2stack(theta[hidden_size*n_classes:], net_config)

    # Number of examples
    m = data.shape[1]

    # Forword pass
    z = [np.zeros(1)] # Note that z[0] is dummy
    a = [data]
    for s in stack:
        z.append(s['w'].dot(a[-1]) + s['b'].reshape((-1, 1)) )
        a.append(sigmoid(z[-1]))

    learned_features = a[-1]

    # Probability with shape (n_classes, m)
    theta_features = softmax_theta.dot(learned_features)
    alpha = np.max(theta_features, axis=0)
    theta_features -= alpha # Avoid numerical problem due to large values of exp(theta_features)
    proba = np.exp(theta_features) / np.sum(np.exp(theta_features), axis=0)

    # Matrix of indicator fuction with shape (n_classes, m)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.arange(m))))
    indicator = np.array(indicator.todense())

    # Compute softmax cost and gradient
    cost = -1.0/m * np.sum(indicator * np.log(proba)) + 0.5*lambda_*np.sum(softmax_theta*softmax_theta)
    softmax_grad = -1.0/m * (indicator - proba).dot(learned_features.T) + lambda_*softmax_theta

    # Backpropagation
    delta = [- softmax_theta.T.dot(indicator - proba) * sigmoid_prime(z[-1])]
    n_stack = len(stack)
    for i in reversed(range(n_stack)): # Note that delta[0] will not be used
        d = stack[i]['w'].T.dot(delta[0])*sigmoid_prime(z[i])
        delta.insert(0, d) # Insert element at beginning

    stack_grad = [{} for i in range(n_stack)]
    for i in range(n_stack):
        stack_grad[i]['w'] = delta[i+1].dot(a[i].T) / m
        stack_grad[i]['b'] = np.mean(delta[i+1], axis=1)

    stack_grad_params = stack2params(stack_grad)[0]

    grad = np.concatenate((softmax_grad.flatten(), stack_grad_params))

    return cost, grad


def stacked_ae_predict(theta, input_size, hidden_size,
                       n_classes, net_config, data):
    """
    theta: optimal theta
    input_size:  the number of input units
    hidden_size: the number of hidden units *at the 2nd layer*
    n_classes:   the number of categories
    net_config:  the network configuration of the stack
    data: our matrix containing the testing data as columns.  So, data[:,i] is the i-th training example.

    pred: the prediction array.
    """

    # We first extract the part which compute the softmax gradient
    softmax_theta = theta[0:hidden_size*n_classes].reshape((n_classes, hidden_size))

    # Extract out the "stack"
    stack = params2stack(theta[hidden_size*n_classes:], net_config)

    # Number of examples
    m = data.shape[1]

    # Forword pass
    z = [np.zeros(1)]
    a = [data]
    for s in stack:
        z.append(s['w'].dot(a[-1]) + s['b'].reshape((-1, 1)) )
        a.append(sigmoid(z[-1]))

    learned_features = a[-1]

    # Softmax model
    model = {}
    model['opt_theta']  = softmax_theta
    model['n_classes']  = n_classes
    model['input_size'] = hidden_size

    # Make predictions
    pred = softmax_predict(model, learned_features)

    return pred


def check_stacked_ae_cost():
    """
    Check the gradients for the stacked autoencoder.

    In general, we recommend that the creation of such files for checking
    gradients when you write new cost functions.
    """

    # Setup random data / small model
    input_size = 4;
    hidden_size = 5;
    lambda_ = 0.01;
    data   = np.random.randn(input_size, 5)
    labels = np.array([ 0, 1, 0, 1, 0], dtype=np.uint8)
    n_classes = 2
    n_stack = 2

    stack = [{} for i in range(n_stack)]
    stack[0]['w'] = 0.1 * np.random.randn(3, input_size)
    stack[0]['b'] = np.zeros(3)
    stack[1]['w'] = 0.1 * np.random.randn(hidden_size, 3)
    stack[1]['b'] = np.zeros(hidden_size)

    softmax_theta = 0.005 * np.random.randn(hidden_size * n_classes)

    stack_params, net_config = stack2params(stack)
    stacked_ae_theta = np.concatenate((softmax_theta, stack_params))

    cost, grad = stacked_ae_cost(stacked_ae_theta, input_size, hidden_size,
                                 n_classes, net_config, lambda_, data, labels)

    # Check that the numerical and analytic gradients are the same
    J = lambda theta : stacked_ae_cost(theta, input_size, hidden_size,
                                       n_classes, net_config, lambda_, data, labels)[0]
    nume_grad = compute_numerical_gradient(J, stacked_ae_theta)

    # Use this to visually compare the gradients side by side
    for i in range(grad.size):
        print("{0:20.12f} {1:20.12f}".format(nume_grad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Compare numerically computed gradients with the ones obtained from backpropagation
    # The difference should be small. In our implementation, these values are usually less than 1e-9.
    # When you got this working, Congratulations!!!
    diff = np.linalg.norm(nume_grad - grad) / np.linalg.norm(nume_grad + grad)
    print("Norm of difference = ", diff)
    print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')


def stack2params(stack):
    """
    Converts a "stack" structure into a flattened parameter vector and also
    stores the network configuration.

    stack: the stack structure, where
           stack[0]['w'] = weights of first layer
           stack[0]['b'] = weights of first layer
           stack[1]['w'] = weights of second layer
           stack[1]['b'] = weights of second layer
                                           ... etc.
    params: parameter vector.
    net_config: configuration of network.
    """

    # Setup the compressed param vector
    params = []
    for i in range(len(stack)):
        w = stack[i]['w']
        b = stack[i]['b']
        params.append(w.flatten())
        params.append(b.flatten())

        # Check that stack is of the correct form
        assert w.shape[0] == b.size, \
            'The size of bias should equals to the column size of W for layer {}'.format(i)
        if i < len(stack)-1:
            assert stack[i]['w'].shape[0] == stack[i+1]['w'].shape[1], \
                'The adjacent layers L {} and L {} should have matching sizes.'.format(i, i+1)

    params = np.concatenate(params)

    # Setup network configuration
    net_config = {}
    if len(stack) == 0:
        net_config['input_size'] = 0
        net_config['layer_sizes'] = []
    else:
        net_config['input_size'] = stack[0]['w'].shape[1]
        net_config['layer_sizes'] = []
        for s in stack:
            net_config['layer_sizes'].append(s['w'].shape[0])

    return params, net_config


def params2stack(params, net_config):
    """
    Converts a flattened parameter vector into a nice "stack" structure
    for us to work with. This is useful when you're building multilayer
    networks.

    params: flattened parameter vector
    net_config: auxiliary variable containing the configuration of the network
    """

    # Map the params (a vector into a stack of weights)
    layer_sizes = net_config['layer_sizes']
    prev_layer_size = net_config['input_size'] # the size of the previous layer
    depth = len(layer_sizes)
    stack = [{} for i in range(depth)]
    current_pos = 0                           # mark current position in parameter vector

    for i in range(depth):
        # Extract weights
        wlen = layer_sizes[i] * prev_layer_size
        stack[i]['w'] = params[current_pos:current_pos+wlen].reshape((layer_sizes[i], prev_layer_size))
        current_pos += wlen

        # Extract bias
        blen = layer_sizes[i]
        stack[i]['b'] = params[current_pos:current_pos+blen]
        current_pos += blen

        # Set previous layer size
        prev_layer_size = layer_sizes[i]

    return stack


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

