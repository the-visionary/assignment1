"""Support Vector Machine (SVM) model."""

import numpy as np
from random import random
from sklearn.utils import shuffle


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        # Size same as X
        self.w = None  # TODO: change this: OK
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def distances(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        N = X_train.shape[0]
        y_train.reshape((N,1))
        # print('X.W ', np.dot(X_train, self.w).shape)
        # print('!! X.W ', np.dot(X_train, self.w).dim)
        # print('Y * X.W ', (y_train.T * np.dot(X_train, self.w) ).shape)
        # print('!! Y * X.W ', (y_train.T * np.dot(X_train, self.w) ).dim)
        dist = 1 - y_train * (np.dot(X_train, self.w))
        dist[dist < 0] = 0
        return dist

    def cost(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        N = X_train.shape[0]
        dist = self.distances(X_train, y_train)
        hinge_loss = reg_const * np.sum(dist) / N
        cost = 1/2 * np.dot(self.w.T, self.w) + hinge_loss
        return cost

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        N = X_train.shape[0]
        cost = self.cost(X_train, y_train)

        dw = np.zeros(self.w.shape)
        # print('dw ', dw.shape)
        dist = self.distances(X_train, y_train)
        # print('dist ', dist.shape)

        for i, d in enumerate(dist):
            d = d.item()
            # print('d ', d.shape)
            # print('w ', self.w.shape)
            # print('y ', y_train[i].shape)
            # print('x ', X_train[i].shape)
            x = X_train[i].reshape(X_train.shape[1], 1)
            y = y_train[i].item()
            if max(0, d) == 0:
                di = self.w
            else:
                di = self.w - (self.reg_const * y) * x
            dw += di

        dw = dw / N
        return dw

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N = X_train.shape[0]
        x_size = X_train.shape[1]
        self.w = np.array([random()] * x_size).reshape(x_size, 1) # Initialize random weights
        # print('W', self.w.shape)

        # grad = self.calc_gradient(X_train, y_train)
        for epoch in range(1, self.epochs + 1):
            # print(f'Epoch #{epoch}')
            X, Y = shuffle(X_train, y_train)
            Y = Y.reshape((y_train.shape[0], 1))
            # print('X ', X.shape)
            # print('Y ', Y.shape)
            grad = self.calc_gradient(X, Y)
            # print('Grad ', grad.shape)
            self.w = self.w - (self.alpha * grad)
            cost = self.cost(X_train, y_train)
            # print(f'Cost: {cost}')
        print(self.w)
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # print(np.sign(X_test @ self.w).shape)
        return np.sign(X_test @ self.w)
