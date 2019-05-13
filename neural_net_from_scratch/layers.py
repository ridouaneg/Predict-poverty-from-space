from activations import sigmoid, tanh, relu, leakyrelu
import numpy as np

class Dense:

    def __init__(self, units, input_dim=None, activation='sigmoid', initialization='random'):
        self.units = units
        self.initialization = initialization
        self.trainable = True

        self.input_dim = input_dim

        # Activation function
        if activation == 'sigmoid':
            self.activation = sigmoid()
        elif activation == 'tanh':
            self.activation = tanh()
        elif activation == 'relu':
            self.activation = relu()
        elif activation == 'leakyrelu':
            self.activation = leakyrelu()

        # Parameters
        self.W = None
        self.b = None

        # Cache
        self.input = None
        self.Z = None
        self.dW = None
        self.db = None

    def initialize(self, input_dim):
        # zero initialization
        if self.initialization == 'zero':
            self.W = np.zeros((self.units, input_dim))
            self.b = np.zeros((self.units, 1))
        # random initialization
        elif self.initialization == 'random':
            self.W = np.random.randn(self.units, input_dim) * 0.01
            self.b = np.zeros((self.units, 1))

    def forward(self, A):
        # input : A[l-1]
        # Z[l] =  W[l] A[l-1] + b[l]
        # A[l] = f(Z[l])
        # cache : Z[l], A[l-1]
        # output : A[l]

        # saved in cache
        self.input = A

        # forward pass
        Z = np.dot(self.W, self.input) + self.b
        A = self.activation.function(Z)

        # saved in cache
        self.Z = Z

        return A

    def backward(self, dA=None, dZ=None):
        # input : dA[l] or dZ[l]
        # dW[l] = (Z[l] * dA[l]) A[l-1]
        # db[l] = (Z[l] * dA[l]) Id
        # cache : dW[l], db[l]
        # output : dA[l-1]

        # batch size
        m = self.input.shape[1]

        # backward pass
        if dZ is None:
            dZ = self.activation.derivative(self.Z) *  dA
        dW = (1 / m) * np.dot(dZ, self.input.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.W.T, dZ)

        # saved in cache
        self.dW = dW
        self.db = db

        return dA


    def update_weights(self, learning_rate):
        # W[l] = W[l] - alpha * dW[l]
        # b[l] = b[l] - alpha * db[l]

        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db


class Dropout():

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.trainable = False
        self.mask = None

    def initialize(self, input_dim):
        pass

    def forward(self, A):
        self.mask = np.random.rand(A.shape[0], A.shape[1]) < self.keep_prob
        A = self.mask * A
        A /= self.keep_prob
        return A

    def predict(self, A):
        return A

    def backward(self, dA):
        return dA * self.mask


class Flatten():

    def __init__(self):
        self.trainable = False
        self.input_shape = None
        self.output_shape = None

    def initialize(self, input_dim):
        pass

    def forward(self, A):
        self.input_shape = np.shape(A)
        self.output_shape = (A.shape[0] * A.shape[1], A.shape[2])
        A = A.reshape(self.output_shape)
        return A

    def backward(self, dA):
        dA = dA.reshape(self.input_shape)
        return dA
