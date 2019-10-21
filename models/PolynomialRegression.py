import numpy as np
import scipy

class PolynomialRegression:

    def __init__(self, p):
        self.coeff = None
        # p = maximum polynomial degree
        self.p = int(p + 1)
        self.train_MSE = None

    def phi(self, x, j):
        return x**j

    def fit(self, X_train, y_train):
        n = X_train.shape[0]
        psi = [[self.phi(X_train[i], j) for j in range(self.p)] for i in range(n)]
        psi = np.resize(psi, (np.shape(psi)[0], np.shape(psi)[1]))

        alpha = scipy.linalg.solve(np.dot(psi.T, psi), np.dot(psi.T, y_train))
        self.coeff = alpha

        #print("MSE =", np.linalg.norm(y_train - np.dot(psi, self.coeff)))
        self.train_MSE = np.sqrt(np.linalg.norm(y_train - self.predict(X_train)) ** 2 / y_train.shape[0])

    def predict(self, X_test):
        m = X_test.shape[0]
        psi_test = [[self.phi(X_test[i], j) for j in range(self.p)] for i in range(m)]
        psi_test = np.resize(psi_test, (np.shape(psi_test)[0], np.shape(psi_test)[1]))
        y_test = np.dot(psi_test, self.coeff)
        return y_test
