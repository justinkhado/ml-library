import pandas as pd
import numpy as np

class NaiveBayesClassifier(object):
    '''
    Implementation of the Naive Bayes classifier
    '''
    def __init__(self, distribution='gaussian'):
        self.distribution = distribution

    def fit(self, X, y):
        '''
        Fits the dataset to the model
        '''
        self.X = X
        self.y = y

        # combine the data and the labels
        labeled_data = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
        # get the priors for the classes
        self.classes = labeled_data.iloc[:,-1].unique()        
        self.priors = self.get_priors(labeled_data)

        # get the necessary information depending on the distribution
        if self.distribution == 'gaussian':
            self.gaussian_fit(labeled_data)
        elif self.distribution == 'multinomial':
            self.multinomial_fit(labeled_data)    

    def predict(self, X):
        if self.distribution == 'gaussian':
            return self.gaussian_predict(X)
        elif self.distribution == 'multinomial':
            return self.multinomial_predict(X)

    def get_priors(self, data):
        '''
        Returns a list of classes (represented as indices) and their probabilities
        '''
        priors = np.zeros(len(self.classes))
        for _class in self.classes:
            priors[_class] = len(data[data.iloc[:,-1] == _class]) / len(data)

        return priors

    def gaussian_fit(self, data):
        '''
        Finds means and variances of each class in "data"
        '''
        self.class_means = [0] * len(self.classes)
        self.class_variances = [0] * len(self.classes)
        for _class in self.classes:
                # select all rows for a class without the labels
                X_class = data[data.iloc[:,-1] == _class].iloc[:,:-1]
                self.class_means[_class] = X_class.mean(axis=0)
                self.class_variances[_class] = X_class.var(axis=0)

    def gaussian_predict(self, X):
        '''
        Returns a list of predictions if the dataset is a Gaussian distribution
        '''
        predictions = []
        for sample in X:
            probabilities = [0] * len(self.classes)
            for _class in self.classes:
                # term1 and term2 splits the equation for the likelihood of a Gaussian distribution
                term1 = 1 / np.sqrt(2*np.pi*self.class_variances[_class])
                term2 = np.exp(-(sample - self.class_means[_class])**2 / (2*self.class_variances[_class]))
                probabilities[_class] = self.priors[_class] * np.prod(term1*term2)
            predictions.append(probabilities.index(max(probabilities)))
        return predictions
        
    def multinomial_fit(self, data):
        pass

    def multinomial_predict(self, X):
        pass