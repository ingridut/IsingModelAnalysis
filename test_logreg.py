import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

# TODO: test accuracy
# TODO: test sigmoid

class Test(unittest.TestCase):
    def test_standard_fit(self):
        """
        Test method using a simple data sets (Iris)
        """
        # Load the Iris data set
        iris = load_iris()
        X = iris.data[:, :2]
        y = (iris.target != 0) * 1

        # Use 50% of the data for training, 50% for testing.
        X_train, X_test, Y_train, Y_test = train_test_split(X, y.reshape(np.shape(X)[0], 1), train_size=0.50)
        logreg = LogisticRegression(X_train, Y_train, X_test, Y_test)
        logreg.fit_standard(max_iter=100)

        # Assert accuracy is more than 0.85
        self.assertTrue(logreg.accuracy()>0.85)

    def test_stochastic_fit(self):
        """
        Test method using a simple data sets (Iris)
        """
        # Load the Iris data set
        iris = load_iris()
        X = iris.data[:, :2]
        y = (iris.target != 0) * 1

        # Use 50% of the data for training, 50% for testing.
        X_train, X_test, Y_train, Y_test = train_test_split(X, y.reshape(np.shape(X)[0], 1), train_size=0.50)
        logreg = LogisticRegression(X_train, Y_train, X_test, Y_test)
        logreg.fit_stochastic(n_epochs=50, t0=5, t1=50)

        # Assert accuracy is more than 0.85
        self.assertTrue(logreg.accuracy()>0.85)

    def test_sigmoid(self):

        vector = np.array([0.4, -0.8, 0.5])
        answer = np.array([0.4013, 0.68997, 0.3775])
        #self.assertAlmostEqual()
        pass

    def test_accuracy(self):
        pass

if __name__ == '__main__':
    unittest.TestCase()

