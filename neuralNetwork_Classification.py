import numpy as np
from hiddenLayer import Layer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numba import jit
import winsound
import matplotlib.pyplot as plt

np.random.seed(13)

class MultilayerNeuralNetwork:
    def __init__(self, X_data, Y_data, n_hidden_layers, n_neurons, eta):
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
        #self.output_weights = np.ones(shape=[self.n_neurons, self.n_categories])        # initial weights are all = 1
        self.output_weights = np.random.randn(self.n_neurons, self.n_categories)       # Initiate Normal Distribution

        # Output
        self.a_in = None
        self.out = None

        # Training data
        self.X_train = X_data
        self.Y_train = Y_data

    def create_hidden_layers(self):
        # Create the hidden layers
        self.hidden_layers = [Layer(self.n_features, self.n_neurons, self.eta, activation_function='Sigmoid', bias=False)]
        for l in range(self.n_hidden_layers-1):
            self.hidden_layers.append(Layer(self.n_neurons, self.n_neurons, self.eta, activation_function='Sigmoid', bias=False))

    def activation(self, a):
        # Softmax
        return np.exp(a)/(np.exp(a).sum(axis=1))[:, None]

    def a_derivative(self, z):
        # Linear regression
        return self.activation(z)*(1-self.activation(z))

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
        self.z = np.matmul(self.a_in, self.output_weights)
        self.out = self.activation(self.z)

    def accuracy_train(self):
        Y_predict = (self.out > 0.5)
        a = np.sum(np.all(Y_predict == self.Y_train, axis=1))
        print(a/np.shape(Y_predict)[0])

    def accuracy_test(self, X, Y):
        a_in = X

        # Go through all hidden layers
        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        z = np.matmul(a_in, self.output_weights) #+ self.output_bias
        out = self.activation(z)

        Y_predict = (out > 0.5)
        a = np.sum(np.all(Y_predict == Y, axis=1))
        print('Error test: ', a/np.shape(Y_predict)[0])

    def backpropagation(self):
        # Calculate outer error, number of inputs used to scale the error
        outer_error = (self.out - self.Y_train)

        # calculate errors in hidden layers
        reversed_layers = list(reversed(self.hidden_layers))
        f_weights = self.output_weights
        f_error = outer_error
        for layer in reversed_layers:
            layer.calculate_error(f_error, f_weights)
            f_weights = layer.weights
            f_error = layer.error

        # Update outer weights
        self.output_weights -= self.eta * np.matmul(self.a_in.T, outer_error)
        for layer in reversed_layers:
            layer.backwards_propagation()

    def save_weights(self):
        # Save weights in order to iterate further later
        np.save('outer_weights.npy', self.output_weights)
        np.save('output_bias.npy', self.output_bias)

        counter = 1
        for layer in self.hidden_layers:
            np.save('weights_{}.npy'.format(counter), layer.weights)
            np.save('bias_{}.npy'.format(counter), layer.biases)
            counter += 1

    def train_batch(self, batch_size):
        # Create batch
        indices = np.arange(np.shape(self.X_data_full)[0])
        batch = np.random.choice(indices, batch_size)
        self.X_train = self.X_data_full[batch, :]
        self.Y_train = self.Y_data_full[batch, :]

        # Train on batch
        self.feed_forward()
        self.backpropagation()

if __name__ == '__main__':
    # Load Data
    data = np.load('test_set.npy')
    print(np.shape(data))
    X = data[:, :1600]
    Y = data[:, -1].reshape(np.shape(data)[0], 1)
    Y_binary = np.c_[Y, np.where(Y==1, 0, 1)]

    # Split into test data and train data, initialize network
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_binary, train_size=0.8)
    MLN = MultilayerNeuralNetwork(X_train, Y_train, n_neurons=1600, n_hidden_layers=1, eta=1e-5)

    # Define number of epochs and batch size
    epochs = 5
    batch_size = 1000
    iterations = int(np.shape(X)[0]/batch_size)
    indices = np.arange(0, np.shape(X)[0])
    a = []
    n = 0

    for e in range(epochs):
        for i in range(iterations):
            MLN.train_batch(batch_size)
            a.append(MLN.accuracy_test(X_test, Y_test))
            n += 1

    frequency = 2000  # Set Frequency To 2500 Hertz
    duration = 1500  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

    plt.figure()
    plt.plot(np.arange(n), a)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    x_ticks = np.arange(n, step=iterations)
    plt.xticks(x_ticks)
    plt.grid()
    plt.show()









