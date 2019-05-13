import numpy as np

class sigmoid:
    def __init__(self):
        pass

    def function(self, z):
        return 1. / (1. + np.exp(-z))

    def derivative(self, z):
        return self.function(z) * (1. - self.function(z))

class tanh:
    def __init__(self):
        pass

    def function(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def derivative(self, z):
        return 1 - self.function ** 2

class relu:
    def __init__(self):
        pass

    def function(self, z):
        tmp = (z >= 0)
        return z * tmp

    def derivative(self, z):
        tmp = (z >= 0)
        return 1 * tmp
