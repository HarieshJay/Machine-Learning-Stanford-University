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


class NeuralNetworkLearning:

    def __init__(self):

        # 20x20 Input Images of Digits
        self.input_layer_size = 400

        # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
        self.num_labels = 10

        #  training data stored in arrays X, y
        self.number_data = loadmat(os.path.join('Data', 'ex4data1.mat'))

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

            # count of training observations
            observations = data_points.shape[0]

            # count of predictors
            predictors = data_points.shape[1]

        elif data_points.ndim == 1:

            predictors = data_points.size

            observations = 1

            # Promote to a 2 dimensional array
            data_points = data_points[None]

        else:

            raise IndexError('Input X should be 1 or 2 dimensional.')

        example_width = int(np.round(np.sqrt(n)))

        example_height = predictors / example_width

        # Compute number of items to display
        display_rows = int(np.floor(np.sqrt(m)))

        display_cols = int(np.ceil(observations / display_rows))

        fig, ax_array = pyplot.subplots(
            display_rows, display_cols, figsize=figsize)

        fig.subplots_adjust(wspace=0.025, hspace=0.025)

        ax_array = [ax_array] if observations == 1 else ax_array.ravel()

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

    @staticmethod
    def sigmoid_gradient(z):

        # sigmoid function applied to z
        z_sigmoid = NeuralNetworkLearning.sigmoid(z)

        return np.multiply((z_sigmoid), (1 - z_sigmoid))

    def random_intialize_weights(self, l_in, l_out, epsilon_init=0.12):

        # intialize the starting weights randomly to avoid symmetrical updates
        weights = (np.random.rand(l_out, 1 + l_in)
                   * 2 * epsilon_init - epsilon_init)

        return weights

    def cost_function(self,
                      nn_params,
                      input_layer_size,
                      hidden_layer_size,
                      num_labels,
                      lambda_=0.0):

        Theta1 = (np.reshape(
                    nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1))))

        Theta2 = (np.reshape(
                    nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1))))

        # forward propagation

        # layer 1 of neural network
        a_1 = self.predictors

        # adjust the first layer for the bias
        a_1_adj = np.concatenate(
            [np.ones((self.m, 1)), self.predictors], axis=1)

        # second layer of the activation units in the
        # neural network using theta 1 as weights for mapping
        a_2 = self.sigmoid(a_1_adj.dot(np.transpose(Theta1)))

        # adjusting the second layer for the bias
        a_2_adj = np.concatenate([np.ones((self.m, 1)), a_2], axis=1)

        # output layer using theta 2 as the weights for mapping
        a_3 = self.sigmoid(a_2_adj.dot(np.transpose(Theta2)))

        # calculating cost

        cost = 0

        # response values of the observations
        y = np.zeros((self.m, num_labels))

        for obs in range(self.m):

            # assigns 1 to correct output class
            # 0 to incorrect output class
            y[obs][self.response[obs]] = 1

        for i in range(self.m):

            for k in range(num_labels):

                # compute the cost function
                cost += (- y[i][k] * np.log(a_3[i][k])
                         - (1 - y[i][k]) * np.log(1-(a_3[i][k])))

        # sum of theta values for regularization
        reg_sum = ((lambda_ / (2 * self.m))
                   * (np.power(Theta1[:, 1:], 2).sum()
                   + np.power(Theta2[:, 1:], 2).sum()))

        # add the regularization term to the cost function
        cost = cost / self.m + reg_sum

        # back propagation

        # delta values of layer 3 is the difference between the
        # actual responses and activation units
        d_3 = np.subtract(a_3, y)

        # activation units of layer 2 prior to the activation function
        z_2 = a_1_adj.dot(np.transpose(Theta1))

        # sigmoid gradient of z_2
        u = NeuralNetworkLearning.sigmoid_gradient(z_2)

        # delta values of layer 2
        d_2 = np.multiply(d_3.dot(Theta2[:, 1:]), u)

        # capital delta values of layer 1
        Delta_1 = np.transpose(d_2).dot(a_1_adj)

        # capital delta values of layer 2
        Delta_2 = np.transpose(d_3).dot(a_2_adj)

        # gradients without regularization
        Theta1_grad = Delta_1 / self.m

        Theta2_grad = Delta_2 / self.m

        # apply regularization to all terms by the bias
        Theta1_grad[:, 1:] += (Theta1[:, 1:] * lambda_) / self.m

        Theta2_grad[:, 1:] += (Theta2[:, 1:] * lambda_) / self.m

        # concatenate into one arrays
        grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

        return cost, grad

    def neural_network_prediction(self, nn_params):

        # Setup the parameters you will use for this exercise
        input_layer_size = 400  # 20x20 Input Images of Digits
        hidden_layer_size = 25   # 25 hidden units
        num_labels = 10          # 10 labels, from 0 to 9

        Theta1 = (np.reshape(
                    nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1))))

        Theta2 = np.reshape(
                    nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

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
