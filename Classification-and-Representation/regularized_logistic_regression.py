# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize


class RegularizedLogisticRegression:

    def __init__(self):

        microchip_data = np.loadtxt(os.path.join(
            'Data',
            'microchip_data.txt'), delimiter=',')

        self.predictors = microchip_data[:, :2]

        self.response = microchip_data[:, 2]

        self.m = self.response.size

        # predictors adjusted to include column of 1 for intercept
        self.predictors_adj = np.concatenate(
            [np.ones((self.m, 1)), self.predictors], axis=1)

        # predictors with polynomial combinations
        self.feature_mapping = self.map_features()

    def plot_data(self):

        # boolean array of positive and negative cases
        positive = self.response == 1

        negative = self.response == 0

        fig = pyplot.figure()

        # lw = line width
        # go -> g = green, o = circle
        # mec = marker edge color
        # ms = marker size
        pyplot.plot(
            self.predictors[positive, 0],
            self.predictors[positive, 1],
            'go', lw=2, mec='k', ms=8)

        # p = pluses
        # mfc = marker face color
        # mew = marker edge width
        pyplot.plot(
            self.predictors[negative, 0],
            self.predictors[negative, 1],
            'p', mfc='c', ms=8, mec='k', mew=1)

        pyplot.xlabel('Microchip Test 1')

        pyplot.ylabel('Microchip Test 2')

        pyplot.legend(
            ['Accepted Quality', 'Rejected Quality'],
            loc='upper right')

        pyplot.show()

    def map_features(self, degree=6):

        # first predictor
        x_1 = self.predictors[:, 0]

        # second predictor
        x_2 = self.predictors[:, 1]

        if x_1.ndim > 0:

            out = [np.ones(x_1.shape[0])]

        else:
            out = [np.ones(1)]

        for i in range(1, degree + 1):

            for j in range(i + 1):

                # create polynomial combinations of the 2 predictors
                out.append((x_1 ** (i - j)) * (x_2 ** j))

        if x_1.ndim > 0:

            # join sequence of arrays along a new axis
            return np.stack(out, axis=1)

        else:

            return np.array(out)

    @staticmethod
    @staticmethod
    def sigmoid_f(z):

        # sigmoid function for a scalar z
        return 1 / (1 + np.exp(-z))

    def sigmoid(self, z):

        # convert input into numpy array
        z = np.array(z)

        # different controls for dimensions of z
        dim = z.ndim

        g = np.zeros(z.shape)

        # z is a scalar
        if (dim == 0):

            return self.sigmoid_f(z)

        # z is an array
        if (dim == 1):

            for i in range(len(z)):

                g[i] = self.sigmoid_f(z[i])

            return g

        # z is a matrix
        if (dim == 2):

            for row in range(z.shape[0]):

                for col in range(z.shape[1]):

                    g[col][row] = self.sigmoid_f(z[col][row])

            return g

    def regularized_cost_function(self, theta, learn):

        sum_cost = 0

        theta = np.array(theta)

        grad = np.zeros(theta.shape)

        for i in range(self.m):

            hypothesis = self.sigmoid(
                (np.transpose(theta)).dot(self.feature_mapping[i]))

            sum_cost += (-self.response[i] * np.log(hypothesis)
                         - (1 - self.response[i]) * np.log(1 - hypothesis))

        reg = sum(theta, 1)

        # the regularization term is added at the end of the cost
        cost = sum_cost / self.m + learn / self.m * reg

        for pred in range(len(self.feature_mapping[1])):

            sum_grad = 0

            for i in range(self.m):

                hypothesis = self.sigmoid(
                    (np.transpose(theta)).dot(self.feature_mapping[i]))

                sum_grad += ((hypothesis - self.response[i])
                             * self.feature_mapping[i][pred])

            # gradient function including the regularization term
            grad[pred] = sum_grad / self.m + learn / self.m * theta[pred]

        return cost, grad

    def optimize_cost_func(self):

        options = {'maxiter': 400}

        res = optimize.minimize(self.regularized_cost_function,
                                np.zeros(self.feature_mapping.shape[1]),
                                (1),
                                jac=True,
                                method='TNC',
                                options=options)

        # the fun property of `OptimizeResult` object returns
        # the value of costFunction at optimized theta
        cost = res.fun

        # the optimized theta is in the x property
        theta = res.x

        return theta
