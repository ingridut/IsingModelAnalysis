import numpy as np

class Layer():

    """
    Hidden layer l for a multilayer perception model
    """

    def __init__(self, n_input, n_neurons, eta, activation_function='Sigmoid', bias=True):
        self.n_neurons = n_neurons
        self.n_input = n_input

        #self.weights = np.ones(shape = [self.n_input, self.n_neurons])
        self.weights = np.random.randn(self.n_neurons, self.n_input)

        self.bias_bool = bias
        if self.bias_bool:
            self.biases = np.zeros(shape=[1, self.n_neurons]) + 0.01
        self.eta = eta
        self.error = None

        self.error_b = 0

        self.z = None
        self.a_out = None
        self.a_in = None

        self.activation_function = activation_function

        #self.batch_size = batch_size

    def activation(self, u):
        if self.activation_function == 'SELU':
            alpha = 1e-3
            lam = 1e-3
            return np.where(u <= 0, lam * alpha * (np.exp(u) - 1), lam * u)
        elif self.activation_function == 'ELU':
            alpha = 1e-3
            return np.where(u <= 0, alpha * (np.exp(u) - 1), u)
        else:
            return 1/(1+np.exp(-u))

    def a_derivative(self, a_h):
        if self.activation_function == 'SELU':
            alpha = 1e-3
            lam = 1e-3
            return np.where(a_h <= 0, lam * self.activation(a_h) + alpha, lam * 1)
        elif self.activation_function == 'ELU':
            alpha = 1e-3
            return np.where(a_h <= 0, self.activation(a_h) + alpha, 1)
        else:
            return self.activation(a_h)*(1-self.activation(a_h))

    def feed_forward(self, a_in):
        # feed-forward for output
        self.a_in = a_in
        self.z = np.matmul(self.a_in, self.weights) #+ self.biases
        self.a_out = self.activation(self.z)

        return self.a_out

    def calculate_error(self, f_error, f_weights):
        self.error = (np.matmul(f_error, f_weights.T)*self.a_derivative(self.z)) #/(np.shape(self.a_in)[0])

    def backwards_propagation(self):
        # Update weights and bias
        self.weights -= self.eta*np.matmul(self.a_in.T, self.error)
        if self.bias_bool:
            self.biases -= self.eta*np.sum(self.error, axis=0).reshape(1, self.n_neurons)
