import os

import numpy as np

from matplotlib import pyplot

from mpl_toolkits.mplot3d import axes3d


class LinearRegression:

    def __init__(self):

        # load housing data
        housing_data = np.loadtxt(os.path.join(
            'Data',
            'housing_data.txt'), delimiter=',')

        # save all rows of columns 1 and 2 as predictors
        self.predictors = housing_data[:, :2]

        # save all rows of column 2 as the response
        self.response = housing_data[:, 2]

        # number of observations
        self.m = self.response.size

        # normalize the predictors
        self.predictors_norm = self.feature_normalize(self.predictors)

    @staticmethod
    def feature_normalize(predictors):

        # copy the predictors since np passes by reference
        predictors_norm = predictors.copy()

        # init array for the mean of each predictor
        mu = np.zeros(predictors.shape[1])

        # init array for the standard deviations of predictors
        sigma = np.zeros(predictors.shape[1])

        for i in range(predictors.shape[1]):

            # the mean of the observations in each predictor
            mu[i] = np.mean(predictors[:, i])

            # standard deviation of the observations in each predictor
            sigma[i] = np.std(predictors[:, i])

        for row in range(predictors.shape[0]):

            for col in range(predictors.shape[1]):

                # normalize every observation
                predictors_norm[row][col] = (
                    predictors[row][col] - mu[col]) / sigma[col]

        return predictors_norm

    @staticmethod
    def linear_model(theta, predictors):

        predicted_response = 0

        for factor in range(len(theta)):

            # apply the multivariate linear regression model
            predicted_response += theta[factor] * predictors[factor]

        return predicted_response

    def compute_cost(self, predictor_norm, response, theta):

        # add a column of ones to treat the intercept as a feature
        predictor_adj = np.concatenate(
            [np.ones((self.m, 1)), predictor_norm], axis=1)

        rss = 0.0

        for observation in range(0, len(self.response)):

            # computer the residual sum of squares
            rss += (
                self.linear_model(
                    theta,
                    predictor_adj[observation])
                - self.response[observation])**2

        # compute the mean squared error, or the cost function
        mse = rss/self.m

        # value of the cost function, the mean squared error divided by 2
        return mse/2

    def gradient_descent(self, theta, alpha, num_iters):

        predictors_adj = np.concatenate(
            [np.ones((self.m, 1)), self.predictors_norm], axis=1)

        # make a copy of theta since numpy arrays are pass by reference
        theta = theta.copy()

        # save history of cost function at each iteration
        J_history = []

        for iteration in range(num_iters):

            pd_sum = np.zeros(predictors_adj.shape[1])

            for observation in range(self.m):

                for pred in range(predictors_adj.shape[1]):

                    # sum in the partial derivative
                    pd_sum[pred] += ((
                                self.linear_model(
                                    theta, predictors_adj[observation])
                                - self.response[observation])
                                * predictors_adj[observation][pred])

                for pred in range(predictors_adj.shape[1]):

                    # apply gradient descent to all predictors
                    theta[pred] -= alpha * (1/self.m) * pd_sum[pred]

            J_history.append(
                self.compute_cost(predictors_adj, self.response, theta))

        # plot the cost function as a function of the iterations
        fig = pyplot.figure()

        pyplot.ylabel('Cost Function')

        pyplot.xlabel('Iterations')

        pyplot.plot(list(range(0, num_iters)), J_history, '-')

        pyplot.show()

        return theta

    def normal_eqn(self):

        # add a column of ones to the predictor data
        predictor_adj = np.concatenate(
            [np.ones((self.m, 1)), self.predictors], axis=1)

        # apple the normal equations function to find the coefficients
        inv = np.linalg.pinv(np.transpose(predictor_adj).dot(predictor_adj))

        result = (inv.dot(np.transpose(predictor_adj))).dot(self.response)

        return result
