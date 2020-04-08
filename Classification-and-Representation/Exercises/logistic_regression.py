# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize


class LogisticRegression:

    def __init__(self):

        university_data = np.loadtxt(os.path.join(
            'Data',
            'university_data.txt'), delimiter=',')

        self.predictors = university_data[:, 0:2]

        self.response = university_data[:, 2]

        self.m = self.response.size

        self.predictors_adj = np.concatenate(
            [np.ones((self.m, 1)), self.predictors], axis=1)

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

        pyplot.xlabel('Exam 1 score')

        pyplot.ylabel('Exam 2 score')

        pyplot.legend(['Admitted', 'Not admitted'])

        pyplot.show()

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

    def cost_function(self, theta):

        # riemann sum of cost function
        sum_cost = 0

        # stores gradients for each predictor
        grad = np.zeros(theta.shape)

        for i in range(self.m):

            # predicted value from the hypothesis function
            hypothesis = self.sigmoid(
                ((np.transpose(theta))).dot(self.predictors_adj[i]))

            sum_cost += (- self.response[i] * np.log(hypothesis)
                         - (1 - self.response[i]) * np.log(1 - hypothesis))

        cost = sum_cost / self.m

        for pred in range(self.predictors_adj.shape[1]):

            # riemann sum of gradient function
            sum_grad = 0

            for i in range(self.m):

                hypothesis = self.sigmoid(
                    ((np.transpose(theta))).dot(self.predictors_adj[i]))

                sum_grad += ((hypothesis - self.response[i])
                             * self.predictors_adj[i][pred])

            grad[pred] = sum_grad / self.m

        return cost, grad

    def optimize_cost_func(self):

        # set maximum number of iterations to 400
        options = {'maxiter': 400}

        # function to be optimzed is the cost function
        # jac = true ->  fun is assumed to return the gradient
        #    along with the objective function
        # method='TNC' -> uses a truncated Newton (TNC) algorithm
        # [0,0,0] inital parameter
        res = optimize.minimize(self.cost_function,
                                [0, 0, 0],
                                (),
                                jac=True,
                                method='TNC',
                                options=options)

        # the fun property of `OptimizeResult` object returns
        # the value of costFunction at optimized theta
        cost = res.fun

        # the optimized theta is in the x property
        theta = res.x

        return theta

    def predict(self, theta):

        predictions = np.zeros(self.m)

        for i in range(self.m):

            # probability of student being accepted using the sigmoid function
            hypothesis = self.sigmoid(
                (np.transpose(theta)).dot(self.predictors_adj[i]))

            # assign to class with 0.5 threshold
            if (hypothesis > 0.5):

                predictions[i] = 1

            else:

                predictions[i] = 0

        return predictions
