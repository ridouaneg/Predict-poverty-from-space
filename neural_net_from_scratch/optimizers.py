import numpy as np

class GradientDescent:

    def __init__(self):
        self.model = None

    def get_model_weights(self):

    def get_model_weights_derivatives(self):

    def set_model_weights(self):

    def fit(self, X_train, y_train, model, nb_epochs, learning_rate, cost):
        self.model = model

        for i in range(1, nb_epochs + 1):
            print('Step :', i, '/', nb_epochs)

            # Forward propagation
            print('Forward pass')
            out = self.model.forward(X_train)

            # Compute cost
            c = cost.compute(out, y_train)
            dZ = cost.final_derivative(out, y_train)
            print('Cost = ', c)

            # Backward propagation
            print('Backward pass')
            dZ = cost.final_derivative(out, y_train)
            dA = self.model.layers[self.model.nb_layers - 1].backward(dZ=dZ)
            for l in range(1, self.nb_layers):
                dA = self.model.layers[self.nb_layers - l - 1].backward(dA=dA)

            # Update weights
            print('Updating weights...')

            # Get parameters
            W, b = self.get_model_weights()
            dW, db = self.get_model_weights_derivatives()

            # Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

            # Set parameters
            W, b = self.set_model_weights()
