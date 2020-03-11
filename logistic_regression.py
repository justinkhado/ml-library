import numpy as np 

class LogisticRegression(object):
    '''
    LogisticRegression is an implementation of the logistic regression model with gradient descent
    '''
    def __init__(self, iterations=1000, learning_rate=0.00001):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        '''
        Fits the data and target values to the model
        '''
        pass

    def predict(self, X):
        '''
        Returns the predictions for a dataset
        '''
        pass

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        '''
        Returns the cost based on the binary cross entropy loss function
        '''