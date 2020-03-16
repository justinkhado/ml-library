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
        # Transform X so that each column is a sample
        X = X.T
        num_features, num_samples = X.shape
        
        self.weights = np.zeros(num_features).reshape(num_features, 1)
        self.bias = 0.0

        for i in range(num_samples):
            # affine function
            Z = np.dot(self.weights.T, X) + self.bias
            # sigmoid function
            sigmoid = 1 / (1 + np.exp(-Z))

            # partial derivative of log loss function with respect to weights
            dL_dw = (1/num_samples) * np.sum(np.multiply((sigmoid - y), X), axis=1).reshape(num_features, 1)
            # partial derivative of log loss function with respect to bias
            dL_db = (1/num_samples) * np.sum(sigmoid - y)

            # update weights and bias
            self.weights -= self.learning_rate * dL_dw
            self.bias -= self.learning_rate * dL_db

    def predict(self, X):
        '''
        Returns the predictions for a dataset
        '''
        X = X.T
        Z = np.dot(self.weights.T, X) + self.bias
        sigmoid = 1 / (1 + np.exp(-Z))
        return (sigmoid >= 0.5).astype(int).reshape(-1)