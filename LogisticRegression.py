import numpy as np
from numba import jit
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
"""
    Logistic regression
"""

np.random.seed(113)
# TODO: Add & compare sklearn's logistic regression
# TODO: Minibatches

class LogisticRegression:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.weights = None
        self.X_train = X_train
        self.Y_train = Y_train.reshape(len(Y_train), 1)
        self.X_test = X_test
        self.Y_test = Y_test

    #@jit
    def fit_standard(self, max_iter=1000, learning_rate=1e-7):
        # Initialize weights
        np.random.seed(98)
        self.weights = np.random.randn(np.shape(self.X_train)[1], 1)

        for i in range(max_iter):
            # Compute probabilities
            z = np.dot(self.X_train, self.weights)
            pred = self.sigmoid(z)

            # Compute gradient, note division by data size
            gradient = self.X_train.T.dot((pred-self.Y_train))

            #Update weights
            self.weights -= learning_rate * gradient

            # Print progress
            if (i+1) % 100 == 0:
                print('{}% done'.format(100*(i+1)/max_iter))
                print('Accuracy: ', self.accuracy())

    def fit_stochastic(self, n_epochs=100, t0=5, t1=10000):
        nr_data_points = np.shape(self.Y_train)[0]  # Data points
        M = nr_data_points

        # initiate weights
        self.weights = np.random.randn(np.shape(self.X_train)[1], 1)

        for epoch in range(n_epochs):
            for i in range(M):
                random_index = np.random.randint(M)
                X_i = self.X_train[random_index:random_index+1, :]
                Y_i = self.Y_train[random_index:random_index+1, :]

                # Calculate gradient and update weights
                z = np.dot(X_i, self.weights)
                pred = self.sigmoid(z)
                gradient = X_i.T.dot((pred-Y_i))
                eta = t0/(epoch*M+i + t1)
                self.weights -= eta*gradient

            if (epoch) % 10 == 0:
                print('{}% done'.format(100*(epoch)/n_epochs))
                print('Accuracy: ', self.accuracy())


    def fit_batches(self, n_epochs=100, batch_size=5000, t0=5, t1=500):
        nr_data_points = np.shape(self.Y_train)[0]  # Data points
        M = nr_data_points

        # initiate weights
        self.weights = np.random.normal(size=[np.shape(self.X_train)[1], 1])

        for epoch in range(n_epochs):
            gradient = 0
            for i in range(batch_size):
                random_index = np.random.randint(M)
                X_i = self.X_train[random_index:random_index + 1, :]
                Y_i = self.Y_train[random_index:random_index + 1, :]

                # Calculate gradient and update weights
                z = np.dot(X_i, self.weights)
                pred = self.sigmoid(z)
                gradient += X_i.T.dot((pred - Y_i))

            eta = t0 / (epoch * M + i + t1)
            self.weights -= eta * gradient

            print('Epoch nr {0} of {1}'.format(epoch, n_epochs))
            print('Accuracy: ', self.accuracy())

    def sigmoid(self, z):
        # Sigmoid function
        return 1 / (1 + np.exp(-z))

    def getWeights(self):
        return self.weights

    def loss_function(self, pred, Y):
        # Compute loss function (normalized)
        return (-Y * np.log(pred) - (1 - Y) * np.log(1 - pred)).mean()

    def predict_threshold(self, X, threshold=0.5):
        # Predict
        return self.sigmoid(np.dot(X, self.weights)) >= threshold

    def accuracy(self, X=None, Y=None):
        if X is None:
            X = self.X_test
        if Y is None:
            Y = self.Y_test
        # Compute accuracy using test data
        I = self.predict_threshold(X) == Y.reshape(len(Y), 1)
        return np.sum(I)/np.shape(X)[0]

if __name__ == '__main__':
    # Testing with smaller (iris) data set
    # iris = datasets.load_iris()
    # X = iris.data[:, :2]
    # y = (iris.target != 0) * 1
    # print(np.shape(X), np.shape(y))
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y.reshape(np.shape(X)[0], 1), train_size=0.50)
    # logreg = LogisticRegression(X_train, Y_train, X_test, Y_test)
    # logreg.fit_stochastic()
    # print('Accuracy: ', logreg.accuracy(X_test, Y_test))

    # Test with 30% of data
    data = np.load('test_set.npy')
    print(np.shape(data))
    X = np.c_[data[:, 0:1600], np.ones(np.shape(data)[0])]
    #X = data[:, 0:1600]
    Y = data[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
    print(np.shape(X_train))

    logreg = LogisticRegression(X_train, Y_train, X_test, Y_test)
    logreg.fit_batches()
    print('Accuracy: ', logreg.accuracy())



