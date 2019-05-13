import numpy as np

class CrossEntropyCost:
    def __init__(self):
        pass

    def compute(self, y_hat, y):
        return np.mean(- y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))

    def final_derivative(self, y_hat, y):
        # returns dL/dy
        return y_hat - y
