import numpy as np
from collections import Counter

class KNN(object):
    '''
    KNN is an implementation of k-nearest neighbors for classification
    '''
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        '''
        Fits the training data to the model
        '''
        self.X1 = X
        self.y = y

    def predict(self, X):
        '''
        Returns the predictions of a dataset
        '''
        predictions = []
        for x2 in X:
            # distances is a list of tuples containing the distance from each point in the training set (and its corresponding label)
            #   to a point in the test set
            distances = [(self.distance(x1, x2), self.y[index]) for index, x1 in enumerate(self.X1)]

            # for the k nearest points, append the most common occuring class to the predictions list
            predictions.append(Counter([i[1] for i in sorted(distances)[:self.k]]).most_common(1)[0][0])

        return predictions

    @staticmethod
    def distance(x1, x2):
        '''
        Returns the Euclidean distance between two points
        '''
        return np.sqrt(np.sum((x2 - x1)**2))