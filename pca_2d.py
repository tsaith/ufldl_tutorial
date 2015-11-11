import numpy as np
import matplotlib.pyplot as plt


"""
Step 0: Load data
  We have provided the code to load data from pcaData.txt into x.
  x is a 45 * 2 matrix, where the k-th column x[:, k] corresponds to
  the kth data point. Here we provide the code to load natural image data into x.
  You do not need to change the code below.
"""

x = np.loadtxt('data/pcaData.txt', dtype=np.float64)

fig = plt.figure()
plt.scatter(x[0, :], x[1, :], s=40, facecolors='none', edgecolors='b')
plt.title('Raw data')


"""
Step 1a: Implement PCA to obtain U
 Implement PCA to obtain the rotation matrix U, which is the eigenbasis sigma.
"""

n, m = x.shape # m is the number of points
sigma = x.dot(x.T) / m # Assuming x is zero-mean

# Sigular value decomposition
U, s, V = np.linalg.svd(sigma)

# Make a plot
fig = plt.figure()
plt.scatter(x[0, :], x[1, :], s=40, facecolors='none', edgecolors='green')
plt.title('Plot u1 and u2')
plt.plot([0, U[0, 0]], [0, U[1, 0]], color='blue')
plt.plot([0, U[0, 1]], [0, U[1, 1]], color='blue')
plt.xlim([-0.8, 0.8])
plt.ylim([-0.8, 0.8])


"""
Step 1b: Compute x_rot, the projection on to the eigenbasis
  Now, compute x_rot by projecting the data on to the basis defined
  by U. Visualize the points by performing a scatter plot.
"""

x_rot = U.T.dot(x)

# Visualise the covariance matrix. You should see a line across the
# diagonal against a blue background.
fig = plt.figure()
plt.scatter(x_rot[0, :], x_rot[1, :], s=40, facecolors='none', edgecolors='blue')
plt.title('x_rot')


"""
Step 2: Reduce the number of dimensions from 2 to 1.
  Compute x_rot again (this time projecting to 1 dimension).
  Then, compute x_hat by projecting the x_rot back onto the original axes
  to see the effect of dimension reduction
"""

k = 1 # Use k = 1 and project the data onto the first eigenbasis
x_tilde = x_rot[0:k, :] # Reduce dimensions
x_hat = U[:, 0:k].dot(x_tilde)

# Make x_hat
fig = plt.figure()
plt.scatter(x_hat[0, :], x_hat[1, :], s=40, facecolors='none', edgecolors='blue')
plt.title('x_hat')
"""
Step 3: PCA Whitening
 Complute x_PCAWhite and plot the results.
"""
epsilon = 1e-5;

x_PCA_white = np.diag(1.0/np.sqrt(s + epsilon)).dot(x_rot)

# Make x_PCAwhite
fig = plt.figure()
plt.scatter(x_PCA_white[0, :], x_PCA_white[1, :], s=40, facecolors='none', edgecolors='blue')
plt.title('x_PCA_white')

"""
Step 3: ZCA Whitening
 Complute x_ZCAWhite and plot the results.

"""

x_ZCA_white = U.dot(x_PCA_white)

# Make x_PCAwhite
fig = plt.figure()
plt.scatter(x_ZCA_white[0, :], x_ZCA_white[1, :], s=40, facecolors='none', edgecolors='blue')
plt.title('x_ZCA_white')

plt.show() # Show all figures

# Congratulations! When you have reached this point, you are done!
# You can now move onto the next PCA exercise. :)
