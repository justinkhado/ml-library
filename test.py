import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

### LINEAR REGRESSION ###
'''
from sklearn.datasets import load_boston
from linear_regression import LinearRegression

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

scaler = Normalizer()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.mean_squared_error(y_test, predictions) ** (1/2))
'''

### LOGISTIC REGRESSION ###
'''
from sklearn.datasets import load_breast_cancer
from logistic_regression import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = LogisticRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))
'''

### K-NEAREST NEIGHBORS ###
'''
from sklearn.datasets import load_iris
from knn import KNN

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = KNN()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
'''

### NAIVE BAYES CLASSIFIER ###
from sklearn.datasets import load_wine
from naive_bayes import NaiveBayesClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = NaiveBayesClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))