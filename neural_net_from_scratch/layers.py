from activations import sigmoid, tanh, relu
import numpy as np

class Dense:

    def __init__(self, units, activation='sigmoid', initialization='random'):
        self.units = units
        self.initialization = initialization
        self.trainable = True

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
            self.W = np.zeros((input_dim, self.units))
            self.b = np.zeros((1, self.units))
        # random initialization
        elif self.initialization == 'random':
            self.W = np.random.randn(input_dim, self.units) * 0.01
            self.b = np.zeros((1, self.units))

    def forward(self, A):
        # input : A[l-1]
        # Z[l] =  W[l] A[l-1] + b[l]
        # A[l] = f(Z[l])
        # cache : Z[l], A[l-1]
        # output : A[l]

        # saved in cache
        self.input = A

        # forward pass
        Z = np.dot(self.input, self.W) + self.b
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
        m = self.input.shape[0]

        # backward pass
        if dZ is None:
            dZ = self.activation.derivative(self.Z) * dA
        dW = (1 / m) * np.dot(self.input.T, dZ)
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
        dA = np.dot(dZ, self.W.T)

        # saved in cache
        self.dW = dW
        self.db = db

        return dA

    def update_weights(self, learning_rate):
        # W[l] = W[l] - alpha * dW[l]
        # b[l] = b[l] - alpha * db[l]

        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db


class Dropout:

    def __init__(self, keep_prob):
        # Parameters
        self.keep_prob = keep_prob

        self.trainable = False

        # Cache
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


class Flatten:

    def __init__(self):
        self.trainable = False

        # Cache
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

class Conv2D:

    def __init__(self, filters=32, kernel_size=(3, 3), strides=1, padding=1, activation='relu', initialization='random'):
        self.initialization = initialization
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        # Activation function
        if activation == 'sigmoid':
            self.activation = sigmoid()
        elif activation == 'tanh':
            self.activation = tanh()
        elif activation == 'relu':
            self.activation = relu()
        elif activation == 'leakyrelu':
            self.activation = leakyrelu()

        self.trainable = True

        # Parameters
        self.W = None
        self.b = None

        # Cache
        self.input = None

    def initialize(self, input_dim):
        # zero initialization
        if self.initialization == 'zero':
            self.W = np.zeros((self.kernel_size[0], self.kernel_size[1], input_dim, self.filters))
            self.b = np.zeros((1, self.filters))
        # random initialization
        elif self.initialization == 'random':
            self.W = np.random.randn(self.kernel_size[0], self.kernel_size[1], input_dim, self.filters) * 0.01
            self.b = np.zeros((1, self.filters))

    def zero_pad(self, X):
        X_pad = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values = 0)
        return X_pad

    def conv_single_step(self, A, W, b):
        s = np.multiply(A, W)
        Z = np.sum(s)
        Z = float(b) + Z
        return Z

    def forward(self, A):
        self.input = A

        (m, n_H_prev, n_W_prev, n_C_prev) = self.input.shape
        (f, f, n_C_prev, n_C) = self.W.shape
        pad = self.padding
        strides = self.strides

        n_H = int((n_H_prev-f+2*pad)/strides+1)
        n_W = int((n_W_prev-f+2*pad)/strides+1)
        n_C = self.filters

        Z = np.zeros((m, n_H, n_W, n_C))

        input_pad = self.zero_pad(self.input)

        for i in range(m):
            a_prev_pad = input_pad[i, :, :, :]
            for h in range(n_H - f + 1):
                for w in range(n_W - f + 1):
                    for c in range(n_C):
                        a_slice_prev = a_prev_pad[h:h+f, w:w+f, :]
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
        A = self.activation.function(Z)

        return A

class MaxPooling:

    def __init__(self, pool_size=(2, 2), strides=None, padding=0):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        self.trainable = False

        # Cache
        self.input = None

    def initialize(self, input_dim):
        pass

    def forward(self, A):
        self.input = A

        (m, n_H_prev, n_W_prev, n_C_prev) = self.input.shape
        f = self.pool_size[0]
        strides = self.strides

        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        A = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        a_slice_prev = self.input[h*strides:h*strides+f, w*strides:w*strides+f, c]
                        A[i, h, w, c] = np.max(a_prev_slice)

        return A
