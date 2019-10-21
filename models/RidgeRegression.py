import numpy as np

class RidgeRegression:

    def __init__(self, gamma=None):
        self.gamma = gamma
        self.coeff = None
        self.train_MSE = None

    def fit(self, X_train, y_train):
        intercept = np.ones((X_train.shape[0], 1))
        X = np.append(intercept, X_train, axis=1)

        beta = np.dot(np.linalg.pinv(np.dot(X.T, X) + self.gamma * np.identity(X.shape[1])), np.dot(X.T, y_train))
        #beta = np.dot(X.T, np.linalg.pinv(np.dot(X, X.T) + self.gamma * np.identity(X.shape[0])).dot(y_train))

        self.coeff = beta
        self.train_MSE = np.sqrt(np.linalg.norm(y_train - self.predict(X_train)) ** 2 / y_train.shape[0])

    def predict(self, X_test):
        intercept = np.ones((X_test.shape[0], 1))
        X = np.append(intercept, X_test, axis=1)

        y_test = np.dot(X, self.coeff)

        return y_test
