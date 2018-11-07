import numpy as np
from hiddenLayer import Layer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numba import jit

class MultilayerNeuralNetwork:
    def __init__(self, X_data, Y_data, n_hidden_layers, n_neurons, eta, batch_size=1):
        self.X_data_full = X_data  # N x M matrix
        self.Y_data_full = Y_data  # N x 1 vector
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_neurons = n_neurons
        self.eta = eta
        self.n_categories = Y_data.shape[1]

        # Initiate hidden layer(s)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layers = []
        if n_hidden_layers > 0:
            self.create_hidden_layers()

        # Initiate outer layer
        #self.output_weights = np.ones(shape=[self.n_neurons, self.n_categories])/5       # initial weights are all = 1
        self.output_weights = np.random.randn(self.n_neurons, self.n_categories)       # Initiate Normal Distribution
        self.output_bias = np.zeros(shape=[self.n_categories, 1]) + 0.01

        # Output
        self.a_in = None
        self.out = None

        # Training data
        self.X_train = X_data
        self.Y_train = Y_data

        self.b_error = 0
        self.batch_size = batch_size

    def update_train(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def create_hidden_layers(self):
        # Create the hidden layers
        self.hidden_layers = [Layer(self.n_features, self.n_neurons, self.eta, activation_function='ELU')]
        for l in range(self.n_hidden_layers-1):
            self.hidden_layers.append(Layer(self.n_neurons, self.n_neurons, self.eta, activation_function='ELU'))

    def train_batch(self, batch_size):
        indices = np.arange(np.shape(self.X_data_full)[0])
        batch = np.random.choice(indices, batch_size)
        self.X_train = self.X_data_full[batch, :]
        self.Y_train = self.Y_data_full[batch, :]

        self.feed_forward()
        self.backpropagation()

    def backpropagation_batch(self):
        # Update weights and bias in outer layer
        self.output_weights -= self.eta * np.matmul(self.a_in.T, self.b_error/self.batch_size)
        self.output_bias -= self.eta * np.sum(self.b_error/self.batch_size)

        # Update the hidden layers, starting from the one closest to the outer layer
        reversed_layers = list(reversed(self.hidden_layers))
        for layer in reversed_layers:
            layer.backwards_propagation_b()

    def activation(self, z):
        return z

    def a_derivative(self, z):
        # Linear regression
        return np.ones(np.shape(z))

    @jit
    def feed_forward(self):
        # Define input
        a_in = self.X_train

        # Go through all hidden layers
        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias

    @jit
    def backpropagation(self):
        # Calculate outer error, number of inputs used to scale the error
        outer_error = (self.out - self.Y_train)/self.n_inputs
        #print('MSE: ', np.mean((self.out-self.Y_train)**2))

        # calculate errors in hidden layers
        reversed_layers = list(reversed(self.hidden_layers))
        f_weights = self.output_weights
        f_error = outer_error
        for layer in reversed_layers:
            layer.calculate_error(f_error, f_weights)
            f_weights = layer.weights
            f_error = layer.error

        # Update outer weights and bias
        self.output_weights -= self.eta * np.matmul(self.a_in.T, outer_error)
        self.output_bias -= self.eta * np.sum(outer_error)
        for layer in reversed_layers:
            layer.backwards_propagation()

        return np.mean((self.out - self.Y_train)**2)

    def save_weights(self):
        # Save weights in order to iterate further later
        np.save('outer_weights.npy', self.output_weights)
        np.save('output_bias.npy', self.output_bias)

        counter = 1
        for layer in self.hidden_layers:
            np.save('weights_{}.npy'.format(counter), layer.weights)
            np.save('bias_{}.npy'.format(counter), layer.biases)
            counter += 1

    def accuracy(self, X, Y):
        # Print MSE and R2 with regards to training data, and new input test data X, Y
        a_in = self.X_train

        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias

        # Accuracy on training data
        #print(np.c_[self.out, self.Y_train])
        MSE = np.mean((self.out - self.Y_train)**2)
        R2 = 1 - np.sum((self.Y_train - self.out)**2)/np.sum((self.Y_train-np.mean(self.Y_train))**2)
        print('MSE Training Data: ', MSE)
        print('R2 Training Data: ', R2)

        # Test data
        a_in = X

        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias

        # Accuracy on test data
        MSE = np.mean((self.out - Y)**2)
        R2 = 1 - np.sum((Y - self.out)**2)/np.sum((Y-np.mean(Y))**2)
        print('MSE Test Data: ', MSE)
        print('R2 Test data: ', R2)
        return MSE

# Functions to generate data
def DesignMatrix(states):
    N = np.size(states, 0)
    size3 = (N, L ** 2)
    X = np.zeros(size3)

    for i in range(0, N):
        X[i] = np.outer(states[i, :], states[i, :]).reshape(1, -1)  # .ravel()
    return X

def ising_energies(states, L):
    J = np.zeros((L, L), )
    for i in range(L):
        J[i, (i + 1) % L] -= 1.0
    E = np.einsum('...i,ij,...j->...', states, J, states)
    # print(J.shape)
    return E

if __name__ == '__main__':
    # Generate data
    # Set random seed
    np.random.seed(62)

    # System size
    L = 40

    # Number of samples
    N = 10000
    states = np.random.choice([-1, 1], size=(N, L))
    energies = ising_energies(states, L).reshape(-1, 1)
    X = DesignMatrix(states)

    # Split into training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, energies, train_size=0.8)

    # Initialise Neural Network
    MLN = MultilayerNeuralNetwork(X_train, np.atleast_2d(Y_train), n_hidden_layers=1, n_neurons=1600, eta=1e-6, batch_size=1)
    epochs = 200
    batch_size = 1000
    iterations = int(np.shape(X)[0]/batch_size)
    indices = np.arange(0, np.shape(X)[0])

    for e in range(epochs):
        for i in range(iterations):
            MLN.train_batch(batch_size)
        MLN.accuracy(X_test, Y_test)


    # Print MSE and R2 for test data and training data
    MLN.accuracy(X_test, Y_test)









