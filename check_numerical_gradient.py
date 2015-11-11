import numpy as np

def check_numerical_gradient():
    """
    This code can be used to check your numerical gradient implementation 
    in computeNumericalGradient.m
    It analytically evaluates the gradient of a very simple function called
    simpleQuadraticFunction (see below) and compares the result with your numerical
    solution. Your numerical gradient implementation is incorrect if
    your numerical solution deviates too much from the analytical solution.  
    """

    # Evaluate the function and gradient at x = [4, 10]
    x = np.array([4, 10], dtype=np.float64)
    value, grad = simple_quadratic_function(x)

    # Use your code to numerically compute the gradient of simple_quadratic_function at x.
    func = lambda x : simple_quadratic_function(x)[0] 
    numgrad = compute_numerical_gradient(func, x)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    n_grad = grad.size
    for i in range(n_grad):
        print("{0:20.12f} {1:20.12f}".format(numgrad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be 2.1452e-12 
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("Norm of difference = ", diff) 
    print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')


def simple_quadratic_function(x):
    """
    This function accepts a vector as input. 
    Its outputs are:
    value: h(x0, x1) = x0^2 + 3*x0*x1
    grad: A vector that gives the partial derivatives of h with respect to x0 and x1       
    """

    value = x[0]*x[0] + 3*x[0]*x[1]

    grad = np.zeros(2)
    grad[0]  = 2*x[0] + 3*x[1]
    grad[1]  = 3*x[0]

    return value, grad


def compute_numerical_gradient(J, theta):
    """
    J: a function that outputs a real-number. Calling y = J(theta) will return the
       function value at theta. 
    theta: a vector of parameters
    """
    n = theta.size
    grad = np.zeros(n)
    eps = 1.0e-4
    eps2 = 2*eps
    
    """
    Instructions: 
    Implement numerical gradient checking, and return the result in grad.  
    (See Section 2.3 of the lecture notes.)
    You should write code so that grad[i] is (the numerical approximation to) the 
    partial derivative of J with respect to the i-th input argument, evaluated at theta.  
    I.e., grad(i) should be the (approximately) the partial derivative of J with 
    respect to theta[i].
               
    Hint: You will probably want to compute the elements of grad one at a time. 
    """    
    for i in range(n):
        theta_p = theta.copy()
        theta_n = theta.copy()
        theta_p[i] = theta[i] + eps
        theta_n[i] = theta[i] - eps
        
        grad[i] = (J(theta_p) - J(theta_n)) / eps2
    
    return grad
