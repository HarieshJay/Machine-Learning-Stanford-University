# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Import regular expressions to process emails
import re

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions
# for assignment submission, and others
import utils


class support_vector_machines:

    @staticmethod
    def gaussianKernel(x1, x2, sigma):

        sim = 0

        # difference between input vectors
        diff = x1 - x2

        # magnitiude of the difference
        dist = np.sum(np.power(diff, 2))

        # result of the similarity function
        sim = np.exp(- dist / (2 * sigma**2))

        return sim

    @staticmethod
    def processEmail(email_contents, verbose=True):

        # load vocabulary
        vocabList = utils.getVocabList()

        # init return value
        word_indices = set()

        # Lower case
        email_contents = email_contents.lower()

        # Strip all HTML
        # Looks for any expression that starts with < and ends with >
        # and does not have any < or > in the tag it with a space
        email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)

        # Handle Numbers
        # Look for one or more characters between 0-9
        email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

        # Handle URLS
        # Look for strings starting with http:// or https://
        email_contents = re.compile(
            '(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

        # Handle Email Addresses
        # Look for strings with @ in the middle
        email_contents = re.compile(
            '[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)

        # Handle $ sign
        email_contents = re.compile('[$]+').sub(' dollar ', email_contents)

        # get rid of any punctuation
        email_contents = re.split(
            '[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

        # remove any empty word string
        email_contents = [word for word in email_contents if len(word) > 0]

        # Stem the email contents word by word
        stemmer = utils.PorterStemmer()

        processed_email = []

        for word in email_contents:

            # Remove any remaining non alphanumeric characters in word
            word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()

            # reduce words to their core
            word = stemmer.stem(word)

            processed_email.append(word)

            if len(word) < 1:
                continue

            try:

                index = vocabList.index(word)

                word_indices.add(index)

            except ValueError:

                pass

        if verbose:

            print('----------------')

            print('Processed email:')

            print('----------------')

            print(' '.join(processed_email))

        return list(word_indices)

    @staticmethod
    def dataset3Params(X, y, Xval, yval):

        # You need to return the following variables correctly.
        c_final = 1

        sigma_final = 0.3

        test_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

        lowest_error = np.inf

        for c in test_vals:

            for s in test_vals:

                model = utils.svmTrain(X, y, c, gaussianKernel, (s))

                predictions = utils.svmPredict(model, Xval)

                error = np.mean(predictions != yval)

                if (error < lowest_error):

                    lowest_error = error

                    c_final = c

                    sigma_final = s

        return c_final, sigma_final

    @staticmethod
    def emailFeatures(word_indices):

        # Total number of words in the dictionary
        n = 1899

        # You need to return the following variables correctly.
        x = np.zeros(n)

        for i in range(n):

            if i in word_indices:

                x[i] = 1

        return x
