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


class k_means_clustering:

    @staticmethod
    def findClosestCentroids(X, centroids):

        # number of clusters
        K = centroids.shape[0]

        # assign each observation to closest centroids
        idx = np.zeros(X.shape[0], dtype=int)

        for ex in range(X.shape[0]):

            # intializae closest distance to infinity
            min_dist = np.inf

            for cent in range(centroids.shape[0]):

                # difference between cluster and observation
                diff = X[ex] - centroids[cent]

                dist = 0

                # iterate through all features and compute the magnitude
                for feature in range(X.shape[1]):

                    dist += diff[feature]**2

                dist = np.sqrt(dist)

                # assign the observation to the closest centroid
                if (min_dist > dist):

                    idx[ex] = cent

                    min_dist = dist

        return idx

    @staticmethod
    def computeCentroids(X, idx, K):

        # observations and feature size
        m, n = X.shape

        # You need to return the following variables correctly.
        centroids = np.zeros((K, n))

        # iterate through all the centeroids
        for cent in range(K):

            points = 0

            for obs in range(X.shape[0]):

                if (idx[obs] == cent):

                    # count of observation in cluster
                    points += 1

                    # compute the sum of the features in that cluster
                    centroids[cent] += X[obs]

            # the average of the features in the cluster
            centroids[cent] = centroids[cent] / points

        return centroids

    @staticmethod
    def init_centroids(self, X, K):

        # Randomly reorder the indices of examples
        randidx = np.random.permutation(X.shape[0])

        # Take the first K examples as centroids
        centroids = X[randidx[:K], :]

        return centroids
