# used for miscellaneous operating system interfaces
import os

# multidimensional array-processing package
import numpy as np

# plotting library
from matplotlib import pyplot

# plotting 3-D services
from mpl_toolkits.mplot3d import axes3d


class SimpleLinearRegression:

    def __init__(self):

        # loading data about the population and profit into a numpy variable
        food_truck_population_profit = np.loadtxt(os.path.join(
            'Data',
            'food_truck_data.txt'), delimiter=',')

        # the population of each city will act as the predictor variables
        self.population_predictor = food_truck_population_profit[:, 0]

        # the profit at each city will act as the response variable
        self.profit_reponse = food_truck_population_profit[:, 1]

        # number of training observations
        self.m = self.profit_reponse.size

        # add a column of ones to treat the intercept as a feature
        self.predictor_with_int = np.stack(
            [np.ones(self.m), self.population_predictor], axis=1)

    def plot_data(self):

        # opens a figure
        fig = pyplot.figure()

        # plots the response and predictors with
        # red(ro), markersize of 10, and color black(k)
        pyplot.plot(self.population_predictor, self.profit_reponse,
                    'ro', ms=10, mec='k')

        # label the x and y axis
        pyplot.ylabel('Profit in $10,000')

        pyplot.xlabel('Population of City in 10,000s')

        # visualizes the data
        pyplot.show()

    @staticmethod
    def linear_model(theta, predictors):

        # compute the response value with given coefficients
        predicted_response = (theta[0] * predictors[0]
                              + theta[1] * predictors[1])

        return predicted_response

    def compute_cost(self, theta):

        rss = 0.0

        for observation in range(0, len(self.profit_reponse)):

            # computer the residual sum of squares
            rss += (
                self.linear_model(
                    theta,
                    self.predictor_with_int[observation])
                - self.profit_reponse[observation])**2

        # compute the mean squared error, or the cost function
        mse = rss/self.m

        # value of the cost function, the mean squared error divided by 2
        return mse/2

    def gradient_descent(self, theta, alpha, num_iters):

        # make a copy of theta since numpy arrays are pass by reference
        theta = theta.copy()

        for iteration in range(num_iters):

            # the sum in the partial derivative
            pd_sum_0 = 0

            pd_sum_1 = 0

            for observation in range(self.m):

                # compute the sum in the partial derivative for the intercept
                pd_sum_0 += (self.linear_model(
                    theta,
                    self.predictor_with_int[observation])
                    - self.profit_reponse[observation])

                # sum in the partial derivative for the population parameter
                pd_sum_1 += ((
                            self.linear_model(
                                theta, self.predictor_with_int[observation])
                            - self.profit_reponse[observation])
                            * self.predictor_with_int[observation][1])

            # subtract theta by the partial derivative
            theta[0] -= alpha * (1/self.m) * pd_sum_0

            theta[1] -= alpha * (1/self.m) * pd_sum_1

        return theta

    def plot_regression_line(self):

        fig = pyplot.figure()

        pyplot.plot(self.predictor_with_int[:, 1], self.profit_reponse,
                    'ro', ms=10, mec='k')

        pyplot.ylabel('Profit in $10,000')

        pyplot.xlabel('Population of City in 10,000s')

        theta = np.zeros(2)

        # find the coefficients after 1500 iterations of gradient descent
        theta = self.gradient_descent(theta, 0.01, 1500)

        # np.dot calculates the dot product applying linear regression
        pyplot.plot(self.population_predictor,
                    np.dot(self.predictor_with_int, theta), '-')

        # create a legend for the training data, and regression line
        pyplot.legend(['Training data', 'Linear regression'])

        pyplot.show()

    def graph_cost_func(self):

        # area the values of the cost function will be graph
        theta0_vals = np.linspace(-10, 10, 100)

        theta1_vals = np.linspace(-1, 4, 100)

        # create a matrix of 0's to intialize
        j_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

        # fill out the cost function values
        for i, theta0 in enumerate(theta0_vals):
            for j, theta1 in enumerate(theta1_vals):

                j_vals[i, j] = self.compute_cost([theta0, theta1])

        # transpose the values otherwise the axis will be flipped
        j_vals = j_vals.T

        # create surface plot
        fig = pyplot.figure(figsize=(12, 5))

        ax = fig.add_subplot(121, projection='3d')

        ax.plot_surface(theta0_vals, theta1_vals, j_vals, cmap='viridis')

        pyplot.xlabel('theta0')

        pyplot.ylabel('theta1')

        pyplot.title('Surface')

        pyplot.show()

        # find the coefficients for the contour plot

        theta = np.zeros(2)

        theta = self.gradient_descent(theta, 0.01, 1500)

        # create contour plot
        ax = pyplot.subplot(122)

        pyplot.contour(theta0_vals, theta1_vals, j_vals, linewidths=2,
                       cmap='viridis', levels=np.logspace(-2, 3, 20))

        pyplot.xlabel('theta0')

        pyplot.ylabel('theta1')

        pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)

        pyplot.title('Contour, showing minimum')

        pyplot.show()
