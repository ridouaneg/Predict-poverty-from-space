import numpy as np

class LocallyWeightedLinearRegression:

    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.X = None
        self.y = None
        self.train_MSE = None

    def kernel(self, x, y):
        return np.exp(- 0.5 * np.linalg.norm(x - y)**2 / self.bandwidth**2)

    def fit(self, X_train, y_train):
        intercept = np.ones((X_train.shape[0], 1))
        self.X = np.append(intercept, X_train, axis=1)
        self.y = y_train
        self.train_MSE = np.sqrt(np.linalg.norm(y_train - self.predict(X_train)) ** 2 / y_train.shape[0])

    def predict(self, X_test):
        Y_test = []
        for x_test in X_test:
            x_test = np.resize(x_test, (1, 1))
            W = [self.kernel(x_test, self.X[i]) for i in range(self.X.shape[0])]
            W = np.diag(W)
            beta = scipy.linalg.solve(np.dot(self.X.T, W).dot(self.X), np.dot(self.X.T, W).dot(self.y))
            intercept = np.ones((x_test.shape[0], 1))
            x_test = np.append(intercept, x_test, axis=1)
            y_test = np.dot(x_test, beta)
            Y_test.append(y_test)
        Y_test = np.array(Y_test)
        Y_test = np.resize(Y_test, (Y_test.shape[0], 1))
        return Y_test
