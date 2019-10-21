import numpy as np

class KernelSmoothing:

    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.X = None
        self.y = None
        self.train_MSE = None

    def kernel(self, x, y):
        return (1 / np.sqrt(2*np.pi)) * np.exp(- 0.5 * np.linalg.norm(x - y)**2 / self.bandwidth**2)

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.train_MSE = np.sqrt(np.linalg.norm(y_train - self.predict(X_train)) ** 2 / y_train.shape[0])

    def predict(self, X_test):
        y_test = []
        for x_test in X_test:
            W = [self.kernel(x_test, self.X[i]) for i in range(self.X.shape[0])]
            res = np.dot(W, self.y) / np.sum(W)
            y_test.append(res)
        y_test = np.array(y_test)
        return y_test
