import numpy as np

class LinearRegression:

    def __init__(self):
        self.coeff = None
        self.train_MSE = None

    def fit(self, X_train, y_train):
        intercept = np.ones((X_train.shape[0], 1))
        X = np.append(intercept, X_train, axis=1)

        beta = np.linalg.pinv(np.dot(X.T, X)).dot(np.dot(X.T, y_train))

        #beta = np.linalg.inv(np.dot(X.T, X)).dot(np.dot(X.T, y_train))
        #beta = scipy.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_train))
        #beta, _, _, _ = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, y_train))

        self.coeff = beta
        self.train_MSE = np.sqrt(np.linalg.norm(y_train - self.predict(X_train)) ** 2 / y_train.shape[0])

    def predict(self, X_test):
        intercept = np.ones((X_test.shape[0], 1))
        X = np.append(intercept, X_test, axis=1)

        y_test = np.dot(X, self.coeff)

        return y_test
