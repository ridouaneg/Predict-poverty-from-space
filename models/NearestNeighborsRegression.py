import numpy as np

class NearestNeighborsRegression:

    def __init__(self, k):
        self.k = int(k)
        self.X = None
        self.y = None
        self.train_MSE = None

    def distance(self, x, y):
        return np.linalg.norm(x - y)

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.train_MSE = np.sqrt(np.linalg.norm(y_train - self.predict(X_train)) ** 2 / y_train.shape[0])

    def predict(self, X_test):
        y_test = []
        for x_test in X_test:
            distance_matrix = [self.distance(x_test, self.X[i]) for i in range(self.X.shape[0])]
            idx = np.argsort(distance_matrix)[:self.k]
            res = np.mean(y_train[idx])
            y_test.append(res)
        y_test = np.array(y_test)
        return y_test
