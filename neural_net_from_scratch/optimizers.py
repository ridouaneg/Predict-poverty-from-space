import numpy as np

class GradientDescent:

    def __init__(self):
        self.model = None

    def get_model_weights(self):
        W = np.empty((0,))
        b = np.empty((0,))
        for l in range(self.model.nb_layers):
            if self.model.layers[l].trainable == True:
                W = np.concatenate((W, self.model.layers[l].W.ravel()))
                b = np.concatenate((b, self.model.layers[l].b.ravel()))
        return W, b

    def get_model_weights_derivatives(self):
        dW = np.empty((0,))
        db = np.empty((0,))
        for l in range(self.model.nb_layers):
            if self.model.layers[l].trainable == True:
                dW = np.concatenate((dW, self.model.layers[l].dW.ravel()))
                db = np.concatenate((db, self.model.layers[l].db.ravel()))
        return dW, db

    def set_model_weights(self, W, b):
        W_long = 0
        b_long = 0
        for l in range(self.model.nb_layers):
            if self.model.layers[l].trainable == True:
                W_shape = np.shape(self.model.layers[l].W)
                tmp = np.prod(W_shape)
                self.model.layers[l].W = W[W_long:W_long + tmp].reshape(W_shape)
                W_long += tmp

                b_shape = np.shape(self.model.layers[l].b)
                tmp = np.prod(b_shape)
                self.model.layers[l].b = b[b_long:b_long + tmp].reshape(b_shape)
                b_long += tmp

class SGD(GradientDescent):

    def __init__(self):
        super()

    def train(self, X_train, y_train, model, nb_epochs, batch_size, learning_rate, cost, X_val=None, y_val=None):
        self.model = model
        m = X_train.shape[0] // batch_size

        for i in range(1, nb_epochs + 1):

            print('Step :', i, '/', nb_epochs)

            permutation = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[permutation], y_train[permutation]

            for j in range(m):

                batch_X = X_train[j * batch_size:(j+1) * batch_size]
                batch_y = y_train[j * batch_size:(j+1) * batch_size]

                # Forward propagation
                out = self.model.forward(batch_X)

                # Compute cost
                c = cost.compute(out, batch_y)
                dZ = cost.final_derivative(out, batch_y)

                # Backward propagation
                dZ = cost.final_derivative(out, batch_y)
                dA = self.model.layers[self.model.nb_layers - 1].backward(dZ=dZ)
                for l in range(1, self.model.nb_layers):
                    dA = self.model.layers[self.model.nb_layers - l - 1].backward(dA=dA)

                # Update weights
                ## Get parameters
                W, b = self.get_model_weights()
                dW, db = self.get_model_weights_derivatives()
                ## Gradient Descent
                W = W - learning_rate * dW
                b = b - learning_rate * db
                ## Set parameters
                self.set_model_weights(W, b)

            # Display epoch cost
            train_out = self.model.predict(X_train)
            train_cost = cost.compute(train_out, y_train)
            print('     Training cost :', train_cost)
            if X_val is not None:
                val_out = self.model.predict(X_val)
                val_cost = cost.compute(val_out, y_val)
                print('     Validation cost :', val_cost)

class Adam(GradientDescent):

    def __init__(self, beta1=0.9, beta2=0.99):
        super()

        self.beta1 = beta1
        self.beta2 = beta2

        self.VdW, self.Vdb = None, None
        self.SdW, self.Vdb = None, None

    def train(self, X_train, y_train, model, nb_epochs, batch_size, learning_rate, cost, X_val=None, y_val=None):
        self.model = model
        m = X_train.shape[0] // batch_size

        W, b = self.get_model_weights()
        self.VdW = np.zeros_like(W)
        self.Vdb = np.zeros_like(b)
        self.SdW = np.zeros_like(W)
        self.Sdb = np.zeros_like(b)
        t = 0

        for i in range(1, nb_epochs + 1):

            print('Step :', i, '/', nb_epochs)

            permutation = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[permutation], y_train[permutation]

            for j in range(m):

                batch_X = X_train[j * batch_size:(j+1) * batch_size]
                batch_y = y_train[j * batch_size:(j+1) * batch_size]

                # Forward propagation
                out = self.model.forward(batch_X)

                # Compute cost
                c = cost.compute(out, batch_y)
                dZ = cost.final_derivative(out, batch_y)

                # Backward propagation
                dZ = cost.final_derivative(out, batch_y)
                dA = self.model.layers[self.model.nb_layers - 1].backward(dZ=dZ)
                for l in range(1, self.model.nb_layers):
                    dA = self.model.layers[self.model.nb_layers - l - 1].backward(dA=dA)

                # Update weights
                ## Get parameters
                W, b = self.get_model_weights()
                dW, db = self.get_model_weights_derivatives()
                ## Momentum update rule
                self.VdW = self.beta1 * self.VdW + (1 - self.beta1) * dW
                self.Vdb = self.beta1 * self.Vdb + (1 - self.beta1) * db
                ## RMSProp update rule
                self.SdW = self.beta2 * self.SdW + (1 - self.beta2) * (dW ** 2)
                self.Sdb = self.beta2 * self.Sdb + (1 - self.beta2) * (db ** 2)
                ## Correction
                t += 1
                self.VdW /= (1 - self.beta1**t)
                self.Vdb /= (1 - self.beta1**t)
                self.SdW /= (1 - self.beta2**t)
                self.Sdb /= (1 - self.beta2**t)
                ## Gradient descent
                W = W - learning_rate * (self.VdW / np.sqrt(self.SdW + 1e-8))
                b = b - learning_rate * (self.Vdb / np.sqrt(self.Sdb + 1e-8))
                ## Set parameters
                self.set_model_weights(W, b)

            # Display epoch cost
            train_out = self.model.predict(X_train)
            train_cost = cost.compute(train_out, y_train)
            print('     Training cost :', train_cost)
            if X_val is not None:
                val_out = self.model.predict(X_val)
                val_cost = cost.compute(val_out, y_val)
                print('     Validation cost :', val_cost)
