import numpy as np

class LinearRegression(object):
    '''
    LinearRegression is an implementation of linear regression with gradient descent
    '''
    def __init__(self, learning_rate=0.00001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        '''
        Takes in a matrix of features and trains the model
        '''
        # Transform X so that each column is a sample
        X = X.T
        num_features, num_samples = X.shape

        self.weights = np.zeros(num_features).reshape(num_features,1)
        self.bias = 0.0

        for i in range(self.iterations):
            # affine function
            Z = np.dot(self.weights.T, X) + self.bias

            # partial derivate of cost function with respect to the weights
            dL_dw = (-2/num_samples) * np.sum(np.multiply((y-Z), X), axis=1).reshape(num_features, 1)
            # partial derivate of cost function with respect to the bias
            dL_db = (-2/num_samples) * np.sum(Z - y)

            # update the weights and bias
            self.weights -= (self.learning_rate * dL_dw)     
            self.bias -= (self.learning_rate * dL_db)
                        
    def predict(self, X):
        '''
        Returns a vector of predictions based on the input, X
        '''
        # Transform X so that each column is a sample
        X = X.T
        return np.dot(self.weights.T, X) + self.bias

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        '''
        Returns the mean squared error
        '''
        return (1/(len(y_true))) * np.sum((y_true - y_pred) ** 2)
