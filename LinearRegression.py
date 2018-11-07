# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.utils import resample
import matplotlib.pyplot as plt


# Comment this to turn on warnings
warnings.filterwarnings('ignore')

# System size
L = 40

def DesignMatrix(states):
    """
    Calculate the design matrix
    """
    N = np.size(states, 0)
    size3 = (N, L ** 2)
    X = np.zeros(size3)

    for i in range(0, N):
        X[i] = np.outer(states[i, :], states[i, :]).reshape(1, -1)  # .ravel()
    return X

def RidgeRegression(x, y, lambda_R=0.5):
    """
    Calculate beta using Ridge regression
    """
    I = np.identity(np.size(x, 1))
    betaRidge = np.linalg.inv(x.T.dot(x) + lambda_R * I).dot(x.T).dot(y)
    return betaRidge

def Lasso(x, y, alpha=0.1):
    """
    Calculate beta using sklearn's lasso model
    """
    lasso = linear_model.Lasso(max_iter=5000, fit_intercept=False, alpha=0.1)
    lasso.fit(x, y) # fit model
    return lasso.coef_

def OLS(x, y):
    """
    Calculate beta using ordinary least squares
    """
    return np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

def beta_model(model, x, y):
    if (model == 'OLS'):
        betaLinear = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
        return betaLinear
    elif (model == 'Ridge'):
        I = np.identity(np.size(x, 1))
        lambda_R = 0.5
        betaRidge = np.linalg.inv(x.T.dot(x) + lambda_R * (I)).dot(x.T).dot(y)
        return betaRidge
    elif (model == 'Lasso'):
        (rows, columns) = np.shape(x)
        poly = PolynomialFeatures(degree=1)

        X = poly.fit_transform(x)
        X_ = X[:, 1:]
        clf = linear_model.Lasso(0.5, max_iter=5000, fit_intercept=False)
        clf.fit(X_, y)
        betaLasso = clf.coef_.reshape(columns, 1)
        return betaLasso

def ising_energies(states, L):
    """
    Calculate ising energies from the states
    """
    J = np.zeros((L, L), )
    for i in range(L):
        J[i, (i + 1) % L] -= 1.0
    E = np.einsum('...i,ij,...j->...', states, J, states)
    return E

def mu(y):
    return np.mean(y)

def MSE(y, y_tilde):
    """
    Calculate MSE for predicted values y_tilde
    """
    n = np.size(y, 0)
    MSE = (1 / n) * (sum(y - y_tilde) ** 2)
    return MSE

def R2(zReal, zPredicted):
    """
    :param zReal: actual z-values, size (n, 1)
    :param zPredicted: predicted z-values, size (n, 1)
    :return: R2-score
    """

    meanValue = np.mean(zReal)
    numerator = np.sum((zReal - zPredicted)**2)
    denominator = np.sum((zReal - meanValue)**2)
    result = 1 - (numerator/denominator)
    return result

def predict(xTest, betaTrained):
    ypredict = xTest.dot(betaTrained)
    return ypredict

def bootstrap(x, y, method, n_bootstrap=20):
    """
    Use bootstrap algorithm to estimate MSE, R2, bias and variance
    """
    # Randomly shuffle data
    data_set = np.c_[x, y]
    np.random.shuffle(data_set)
    set_size = round(len(y)/5)

    # Extract test-set, never used in training. About 1/5 of total data
    x_test = data_set[0:set_size, :-1]
    y_test = data_set[0:set_size, -1]
    test_indices = np.linspace(0, set_size-1, set_size)

    # And define the training set as the rest of the data
    x_train = np.delete(data_set[:, :-1], test_indices, axis=0)
    y_train = np.delete(data_set[:, -1], test_indices, axis=0)

    y_predict = []

    MSE = []
    R2s = []
    for i in range(n_bootstrap):
        x_, y_ = resample(x_train, y_train)

        if method == 'Ridge':
            # Ridge regression, save beta values
            beta = RidgeRegression(x_, y_, lambda_R=10**2)
        elif method == 'Lasso':
            lasso = linear_model.Lasso(max_iter=5000, fit_intercept=False, alpha=0.1)
            lasso.fit(x_, y_)
            beta = lasso.coef_.reshape(1600, )
        elif method == 'OLS':
            beta = OLS(x_, y_)
        else:
            print('ERROR: Cannot recognize method')
            return 0

        y_hat = x_test.dot(beta)
        y_predict.append(y_hat)

        # Calculate MSE
        MSE.append(np.mean((y_test - y_hat)**2))
        R2s.append(R2(y_test, y_hat))
        print('Round: ', i)

    # Calculate MSE, Bias and Variance
    MSE_M = np.mean(MSE)
    R2_M = np.mean(R2s)
    bias = np.mean((y_test - np.mean(y_predict, axis=0, keepdims=True))**2)
    variance = np.mean(np.var(y_predict, axis=0, keepdims=True))
    return MSE_M, R2_M, bias, variance

###############################################################################
# Set random seed
np.random.seed(12)

# Number of samples
N = 10000
states = np.random.choice([-1, 1], size=(N, L))
energies = ising_energies(states, L).reshape(-1, 1)
X = DesignMatrix(states)

# Split into training and test data, y=energies
X_train, X_test, y_train, y_test = train_test_split(X, energies, test_size=0.20)
y_mean = mu(y_test)

# OLS regression
MSE_M, R2_M, bias, variance = bootstrap(X, energies, method='OLS')
print('OLS')
print('MSE: ', MSE_M)
print('R2: ', R2_M)
print('Bias: ', bias)
print('Variance: ', variance)


# Rigde regression - Find a good lambda
lambdas = np.logspace(-5, 5, base=10, num=11)

MSEs = []
R2s = []
for l in lambdas:
    print(l)
    betaRidge = RidgeRegression(X_train, y_train, lambda_R=l)
    energiesRidgepredicted = predict(X_test, betaRidge)
    MSEs.append(MSE(y_test, energiesRidgepredicted))
    R2s.append(R2(y_test, energiesRidgepredicted))

# Make figure
fig, ax1 = plt.subplots()
ax1.plot(np.log10(lambdas), MSEs, 'bo-')
ax1.set_xlabel('Logarithmic lambda')
ax1.set_ylabel('MSE', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(np.log10(lambdas), R2s, 'r*-')
ax2.set_ylabel('R2 score', color='r')
ax2.tick_params('y', colors='r')

plt.grid(True)
plt.title('Influence of lambda on MSE and R2 Score')
fig.tight_layout()
plt.show()

MSE_M, R2_M, bias, variance = bootstrap(X, energies, method='Ridge')
print('RIDGE')
print('MSE: ', MSE_M)
print('R2: ', R2_M)
print('Bias: ', bias)
print('Variance: ', variance)

# The Lasso
betaLasso = beta_model('Lasso', X_train, y_train)
energiesLassopredicted = predict(X_test, betaLasso)
mseLasso = MSE(y_test, energiesLassopredicted)
R_2Lasso = calc_R_2(y_test, energiesLassopredicted, y_mean)
print('--- Lasso Regression ---')
print('MSE: ', mseLasso)
print('R2 score: ', R_2Lasso)

# Find best alpha
alphas = np.logspace(-5, 5, base=10, num=11)

MSEs = []
R2s = []
for a in alphas:
    betaLasso = Lasso(X_train, y_train, alpha=a)
    y_predict = X_test.dot(betaLasso.reshape(len(betaLasso), 1))
    MSEs.append(MSE(y_test, y_predict))
    R2s.append(R2(y_test, y_predict))

print(MSEs)
print(R2s)

# Make figure
fig, ax1 = plt.subplots()
ax1.plot(np.log10(alphas), MSEs, 'bo-')
ax1.set_xlabel('Logarithmic alpha')
ax1.set_ylabel('MSE', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(np.log10(alphas), R2s, 'r*-')
ax2.set_ylabel('R2 score', color='r')
ax2.tick_params('y', colors='r')

plt.grid(True)
plt.title('Influence of alpha on MSE and R2 Score')
fig.tight_layout()
plt.show()

# Regression analysis with optimal alpha
betaLasso = Lasso(X_train, y_train, alpha=10**-1)
print(np.shape(X_train), np.shape(y_train))
y_predict = X_test.dot(betaLasso.reshape(len(betaLasso), 1))
print(MSE(y_test, y_predict))
print(R2(y_test, y_predict))
#

MSE_M, R2_M, bias, variance = bootstrap(X, energies, method='Lasso')
print('LASSO')
print('MSE: ', MSE_M)
print('R2: ', R2_M)
print('Bias: ', bias)
print('Variance: ', variance)