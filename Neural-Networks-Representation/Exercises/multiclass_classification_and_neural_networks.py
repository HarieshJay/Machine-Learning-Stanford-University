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


class MultiClassClassification:

    def __init__(self):

        # 20x20 Input Images of Digits
        self.input_layer_size = 400

        # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
        self.num_labels = 10

        #  training data stored in arrays X, y
        self.number_data = loadmat(os.path.join('Data', 'ex3data1.mat'))

        self.predictors = self.number_data['X']

        self.response = self.number_data['y'].ravel()

        # set the zero digit to 0, rather than its mapped 10 in this dataset
        # This is an artifact due to the fact that this dataset was used in
        # MATLAB where there is no index 0
        self.response[self.response == 10] = 0

        # observation count
        self.m = self.response.size

    def display_data(self, example_width=None, figsize=(10, 10)):

        # Randomly select 100 data points to display
        rand_indices = np.random.choice(self.m, 100, replace=False)

        # store randomyl chosed data points
        data_points = self.predictors[rand_indices, :]

        # Compute rows, cols
        if data_points.ndim == 2:

            m, n = data_points.shape

        elif data_points.ndim == 1:

            n = data_points.size

            m = 1

            data_points = data_points[None]  # Promote to a 2 dimensional array

        else:

            raise IndexError('Input X should be 1 or 2 dimensional.')

        example_width = int(np.round(np.sqrt(n)))

        example_height = n / example_width

        # Compute number of items to display
        display_rows = int(np.floor(np.sqrt(m)))

        display_cols = int(np.ceil(m / display_rows))

        fig, ax_array = pyplot.subplots(
            display_rows, display_cols, figsize=figsize)

        fig.subplots_adjust(wspace=0.025, hspace=0.025)

        ax_array = [ax_array] if m == 1 else ax_array.ravel()

        for i, ax in enumerate(ax_array):

            ax.imshow(
                self.predictors[i].reshape(
                    example_width, example_width, order='F'),
                cmap='Greys', extent=[0, 1, 0, 1])

            ax.axis('off')

        pyplot.show()

    @staticmethod
    def sigmoid(z):

        # sigmoid function for a scalar z
        return 1 / (1 + np.exp(-z))

    def cost_function(self, theta, X, y, learn):

        # convert labels to ints if their type is bool
        if self.response.dtype == bool:
            self.response = self.response.astype(int)

        # observation count
        m = y.size

        # You need to return the following variables correctly
        cost = 0

        # inital gradient array
        grad = np.zeros(theta.shape)

        # predicted values for observations
        hypothesis = self.sigmoid(X.dot(theta))

        # logistic regression cost function
        cost = ((- y.dot(np.log(hypothesis))
                - (1-y).dot(np.log(1-hypothesis)))/m
                + (learn/(2*m))*(theta[1:].dot(theta[1:])))

        # gradient value for bias
        grad[0] = (1/m) * (hypothesis - y).dot(X[:, 0])

        # set the gradient values for j = 1 ... n
        for pred in range(1, len(theta)):
            grad[pred] = ((1/m) * (hypothesis - y).dot(X[:, pred])
                          + learn/m*theta[pred])

        return cost, grad

    def one_vs_all(self, num_labels=10, learn=3):

        # count of input nodes
        inputs = self.predictors.shape[1]

        # you need to return the following variables correctly
        all_theta = np.zeros((num_labels, inputs + 1))

        # add ones to the X data matrix
        X = np.concatenate([np.ones((m, 1)), self.predictors], axis=1)

        # set Initial theta
        initial_theta = np.zeros(inputs + 1)

        # set options for minimize
        options = {'maxiter': 50}

        for classifier in range(num_labels):

            # custom classifier for each class
            res = optimize.minimize(
                self.cost_function,
                initial_theta,
                (X, (self.response == classifier), learn),
                jac=True,
                method='TNC',
                options=options)

            all_theta[classifier] = res.x

        return all_theta

    def one_vs_all_prediction(self, all_theta):

        m = self.predictors.shape[0]
        num_labels = all_theta.shape[0]

        # You need to return the following variables correctly
        p = np.zeros(m)

        # Add column of 1 for bias term
        predictors_adj = np.concatenate(
            [np.ones((m, 1)), self.predictors], axis=1)

        pred = predictors_adj.dot(np.transpose(all_theta))

        result = np.argmax(pred, axis=1)

        return result

    def neural_network_prediction(self):

        # Setup the parameters you will use for this exercise
        input_layer_size = 400  # 20x20 Input Images of Digits
        hidden_layer_size = 25   # 25 hidden units
        num_labels = 10          # 10 labels, from 0 to 9

        # Load the .mat file, which returns a dictionary
        weights = loadmat(os.path.join('Data', 'ex3weights.mat'))

        # get the model weights from the dictionary
        # Theta1 has size 25 x 401
        # Theta2 has size 10 x 26
        Theta1, Theta2 = weights['Theta1'], weights['Theta2']

        # swap first and last columns of Theta2,
        # due to legacy from MATLAB indexing,
        # since the weight file ex3weights.mat
        # was saved based on MATLAB indexing
        Theta2 = np.roll(Theta2, 1, axis=0)

        # Make sure the input has two dimensions
        if self.predictors.ndim == 1:

            self.predictors = self.predictors[None]  # promote to 2-dimensions

        # possible classes
        num_labels = Theta2.shape[0]

        # adjust the predictor for the bias
        predictors_adj = np.concatenate(
            [np.ones((self.m, 1)), self.predictors], axis=1)

        # second layer of the neural network
        # using theta 1 as weights for mapping
        layer2 = self.sigmoid(predictors_adj.dot(np.transpose(Theta1)))

        # adjusting the second layer for the bias
        layer2_adj = np.concatenate([np.ones((self.m, 1)), layer2], axis=1)

        # output layer using theta 2 as the weights for mapping
        layer3 = self.sigmoid(layer2_adj.dot(np.transpose(Theta2)))

        # predicted outcomes assiging to the class with the highest prediction
        result = np.argmax(layer3, axis=1)

        return result
