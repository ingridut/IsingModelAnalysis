import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.utils import resample

def bootstrap_NN(x, y, estimatorKeras,  n_bootstrap=2):
    # Randomly shuffle data
    data_set = np.c_[x, y]
    np.random.shuffle(data_set)
    x_train , x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=0 )
    y_predict = []
    MSE = []
    R2s = []
       
    
    for i in range(n_bootstrap):
        x_, y_ = resample(x_train, y_train)
        
        print('Round: ', i)
        estimatorKeras.fit( x_, y_ )
        #beta = kerasEstimator.coef_.reshape(1600, )
        #print(kerasEstimator.score(x_test, y_test))

        y_hat = estimatorKeras.predict(x_test)  #.reshape(1600, )
        y_predict.append(y_hat)

        # Calculate MSE adn R2 score
        MSE.append(np.mean((y_test - y_hat)**2))
        R2s.append(r2_score(y_test, y_hat))
        #print('Round: ', i)

    # Calculate MSE, Bias and Variance
    MSE_M = np.mean(MSE)
    R2_M = np.mean(R2s)
    bias = np.mean((y_test - np.mean(y_predict, axis=0, keepdims=True))**2)
    variance = np.mean(np.var(y_predict, axis=0, keepdims=True))
    return MSE_M, R2_M, bias, variance


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

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# define base model
def base_model_1():
	# create model
	model = Sequential()
	model.add(Dense(1600, input_dim=1600, kernel_initializer='normal', activation='elu'))
	model.add(Dense(1, activation="linear")) #kernel_initializer='normal'
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2_keras])
	return model

def base_model():
    model = Sequential()
    model.add(Dense(100,
                input_dim=X.shape[1],
                activation="elu"))
    model.add(Dense(10,
                activation="elu"))
    model.add(Dense(1, activation="linear") )

    # Compile model
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=[r2_keras])
    return model


if __name__ == '__main__':
    np.random.seed(12)
    L = 40
    N = 10000
    states = np.random.choice([-1, 1], size=(N, L))
    energies = ising_energies(states, L).reshape(-1, 1)
    X = DesignMatrix(states)

    trainx, testx, trainy, testy = train_test_split( X, energies, test_size=0.2, random_state=0 )

    estimator = KerasRegressor(build_fn = base_model_1, epochs=5, batch_size=32)

    
    MSE_M, R2_M, bias, variance = bootstrap_NN(X, energies, estimator,  n_bootstrap=10)
    print("MSE: ", MSE_M)
    print("R2 score: ", R2_M)
    print("Bias: ", bias)
    print("Variance: ", variance)