# FYS-STK4155 Project 2

Project 2 in FYS-STK4155 by Polina Dobrovolskaia and Ingrid Utseth. 

In this project, we studied 1D and 2D Ising model data with different machine learning techniques. 

# Data

We use generated data sets and a data set from "A high-bias, low-variance introduction to Machine Learning for physicists" by Mehta et al. Since the latter is rather large, we randomly chose 10% of it in the readData script to use in the classification algorithms (saved as test_set.npy). 

# LinearRegression

This script contains all the functions necessary to estimate the Ising energy from a generated set of spin variables. Ordinary Least Squares, Ridge Regression and the Lasso are used, and with the bootstrap algorithm, one can evaluate the fit using the estimate MSE, R2-score, bias and variance. 


# LogisticRegression

This file contains all the functions necessary for a binary classification of spin lattices using logistic regression and gradient descent methods. 

## ReadData

Reads the data obtained from "A high-bias, low-variance introduction to Machine Learning for physicists". This data is used to train and evaluate the logistic regression classifier. 

## test_logreg

Tests the logistic regression method. 

# NeuralNetwork

neuralNetwork is used for regression analysis. neuralNetwork_Classification is used for binary classification purposes. Both can have an arbitrary number of hidden layers, initialised with hiddenLayer. 
