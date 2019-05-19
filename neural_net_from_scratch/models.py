import numpy as np
from layers import Dense, Dropout, Flatten

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers
        self.nb_layers = len(layers)

    def initialization(self, X, y):
        # Initialize layer's dimensions and weights
        A = X
        for l in range(self.nb_layers):
            input_dim = A.shape[-1]
            self.layers[l].initialize(input_dim)
            A = self.layers[l].forward(A)

        for l in range(self.nb_layers):
            if(type(self.layers[l]) != Dropout):
                self.layers[l].predict = self.layers[l].forward

    def forward(self, X):
        # Forward propagation
        A = X
        for l in range(self.nb_layers):
            A = self.layers[l].forward(A)
        out = A
        return out

    #def backward

    def train(self, X_train, y_train, opt, nb_epochs, batch_size, learning_rate, cost):
        self.initialization(X_train[:2], y_train[:2])
        opt.train(X_train, y_train, self, nb_epochs, batch_size, learning_rate, cost)

    def predict(self, X_test):
        A = X_test
        for l in range(self.nb_layers):
            A = self.layers[l].predict(A)
        return A
