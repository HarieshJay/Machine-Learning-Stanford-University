# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat


class RegularizedLineareRegression:

    def linear_reg_cost(self, X, y, theta, lambda_=0.0):

        # number of training observations
        m = y.size

        # gradient vector of theta values
        grad = np.zeros(theta.shape)

        # predicted values for each observation
        hypothesis = X.dot(np.transpose(theta))

        # vectorized implementation of cost function

        # difference between output values and actual response
        error = hypothesis - y

        # mean squared difference
        cost = error.dot(error) / (2 * m)

        # regularization term
        reg = (lambda_ / (2 * m)) * theta[1:, ].dot(theta[1:, ])

        # cost including the regularization term
        cost += reg

        # gradient values
        grad = (1/m) * error.dot(X)

        # regularization term for gradients
        reg_grad = (lambda_ / (2 * m)) * theta

        # gradient values including regularization
        # bias term is ommited
        grad[1:, ] = grad[1:, ] + reg_grad[1:, ]

        return cost, grad

    def learning_curve(self, X, y, Xval, yval, lambda_=0):

        # Number of training examples
        m = y.size

        # error term on training data
        error_train = np.zeros(m)

        # error term on cross validation error
        error_val = np.zeros(m)

        for i in range(1, m+1):

            # compute theta by optimizing the cost functions
            theta = self.train_linear_reg(
                    self.linear_reg_cost, X[:i, :], y[:i], lambda_)

            # cost function applied to training data
            error_train[i-1] = self.linear_reg_cost(
                    X[:i, :], y[:i], theta, lambda_)[0]

            # cost function applied to training data
            error_val[i-1] = self.linear_reg_cost(
                Xval, yval, theta, lambda_)[0]

        return error_train, error_val

    def train_linear_reg(
            self, linear_reg_cost, X, y, lambda_=0.0, maxiter=200):

        # initialize theta
        initial_theta = np.zeros(X.shape[1])

        # Create "short hand" for the cost function to be minimized
        cost_function = lambda t: self.linear_reg_cost(X, y, t, lambda_)

        # cost_function is a function that takes in only one argument
        options = {'maxiter': maxiter}

        # minimize using scipy
        res = optimize.minimize(
                cost_function,
                initial_theta,
                jac=True,
                method='TNC',
                options=options)

        return res.x

    def polyFeatures(self, X, p):

        # expand the training predictors by including polynomial terms of X
        X_poly = np.zeros((X.shape[0], p))

        for i in range(0, p):

            # add X to the power of 1..p to training predictors
            X_poly[:, i] = np.power(X, i+1).ravel()

        return X_poly

    def validation_curve(self, X, y, Xval, yval):

        # sequence of lambda values to be tested
        lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

        # error values in the training data with corresponding lambda values
        error_train = np.zeros(len(lambda_vec))

        # error values in the cross validation data
        # with corresponding lambda values
        error_val = np.zeros(len(lambda_vec))

        for i in range(len(lambda_vec)):

            lambda_ = lambda_vec[i]

            # optimized theta with given lambda value
            theta = self.train_linear_reg(self.linear_reg_cost, X, y, lambda_)

            # error from training data with given lambda
            error_train[i] = self.linear_reg_cost(X, y, theta, lambda_)[0]

            # error from cross validation with given lambda
            error_val[i] = self.linear_reg_cost(Xval, yval, theta, lambda_)[0]

        return lambda_vec, error_train, error_val
