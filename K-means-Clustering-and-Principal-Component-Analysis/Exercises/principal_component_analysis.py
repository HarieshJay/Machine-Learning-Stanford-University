# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Import regular expressions to process emails
import re

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat


class principal_component_analysis:

    @staticmethod
    def pca(X):

        # number of training examples
        m = X.shape[0]

        # compute the covariance matrix
        covariance = 1/m * (np.transpose(X)).dot(X)

        # U is the eigenvectors that represent the principal components of X
        # S contains the singular values for each principal component
        U, S, V = np.linalg.svd(covariance)

        return U, S

    @staticmethod
    def project_data(X, U, K):

        # eigenvector projection matrix to convert the current feature space
        # into a feature space of K dimensions
        Ureduce = U[:, :K]

        # create new features of K dimensions with the projection matrix
        Z = X.dot(Ureduce)

        return Z

    @staticmethod
    def recoverData(Z, U, K):

        # approximation of the data by projecting back
        # onto the original space using the top K eigenvectors in U
        X_rec = Z.dot(np.transpose(U[:, :K]))

        return X_rec
