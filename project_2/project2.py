#!/usr/bin/env python3

# Rishabh Tewari
# R11603985
# Project 2

import numpy as np
import matplotlib.pyplot as plt


# 1 Generate Training Set
training_size_n = 10
uniform_train_X = np.random.uniform(low=0.0, high = 1.0, size = training_size_n)
uniform_train_X = np.reshape(uniform_train_X, (-1, 1)) #Do we have to add a row of ones in the design matrix?
uniform_train_X = np.hstack((uniform_train_X, np.ones(uniform_train_X.shape)))
t_train = (np.sin(2 * np.pi * uniform_train_X)) + (np.random.normal(loc=0.0, scale = 0.3))

# Generate Test Set
test_size_n = 100
uniform_test_X = np.random.uniform(low=0.0, high = 1.0, size = test_size_n)
uniform_test_X = np.reshape(uniform_test_X, (-1, 1))
uniform_test_X = np.hstack((uniform_test_X, np.ones(uniform_test_X.shape)))

t_train = (np.sin(2 * np.pi * uniform_test_X)) + (np.random.normal(loc=0.0, scale = 0.3))

# print(uniform_test_X)

# def phi_function(input, degree):
    # pass

def linear_reg_non_lin(x, t):
    weight = (np.linalg.pinv(x) @ t).T
    prediction_lin = weight * x