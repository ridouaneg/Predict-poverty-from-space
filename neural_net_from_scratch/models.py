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
            input_dim = A.shape[0]
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

    def train(self, X_train, y_train, nb_epochs, learning_rate, cost):
        self.initialization(X_train[:2], y_train[:2])

        for i in range(1, nb_epochs + 1):
            print('Step :', i, '/', nb_epochs)

            # Forward propagation
            print('Forward pass')
            out = self.forward(X_train)

            # Compute cost
            c = cost.compute(out, y_train)
            dZ = cost.final_derivative(out, y_train)
            print('Cost = ', c)

            # Backward propagation
            print('Backward pass')
            dZ = cost.final_derivative(out, y_train)
            dA = self.layers[self.nb_layers - 1].backward(dZ=dZ)
            for l in range(1, self.nb_layers):
                dA = self.layers[self.nb_layers - l - 1].backward(dA=dA)

            # Update weights
            print('Updating weights...')
            for l in range(self.nb_layers):
                if self.layers[l].trainable == True:
                    self.layers[l].update_weights(learning_rate)

    def predict(self, X_test):
        A = X_test
        for l in range(self.nb_layers):
            A = self.layers[l].predict(A)
        return A
