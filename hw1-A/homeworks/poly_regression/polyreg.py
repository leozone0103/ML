"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.mean = None
        self.std = None

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        featured_arr = np.zeros((len(X),degree))
        for i in range(len(X)):
            for j in range(degree):
                featured_arr[i][j] = X[i] ** (j+1)
        return featured_arr

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        X_poly = self.polyfeatures(X, self.degree)
        std = np.std(X_poly, axis=0)
        mean = np.mean(X_poly, axis=0)
        self.mean = mean
        self.std = std
        X_poly_std = ( X_poly - mean ) / std
        X_poly_manipulated = np.c_[np.ones((len(X_poly_std), 1)), X_poly_std]
        n, d = X_poly_manipulated.shape
        reg_matrix = self.reg_lambda * np.eye(d)
        reg_matrix[0,0] = 0      
        self.weight = np.linalg.solve(X_poly_manipulated.T @ X_poly_manipulated + reg_matrix, X_poly_manipulated.T @ y)


    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        poly_X = self.polyfeatures(X,self.degree)
        n,d = poly_X.shape
        poly_X_std = (poly_X - self.mean) / self.std 
        poly_featured = np.c_[np.ones((n,1)), poly_X_std]

        return poly_featured.dot(self.weight)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    return  1.0/(len(a))  * sum((a - b) ** 2)


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    model = PolynomialRegression(degree = degree, reg_lambda = reg_lambda)
    i = 3
    while i <= n:
        Xtrain_set = Xtrain[0:i]
        Ytrain_set = Ytrain[0:i]
        model.fit(Xtrain_set, Ytrain_set)
        predicted_train_set = model.predict(Xtrain_set)
        predicted_test_set = model.predict(Xtest)
        errorTrain[i-1] = mean_squared_error(predicted_train_set,Ytrain_set)
        errorTest[i-1]  = mean_squared_error(predicted_test_set, Ytest)
        i += 1

    return (errorTrain, errorTest)
