#!/usr/bin/env python3
# Rishabh Tewari
# R11603985
# Project 2

import numpy as np
import matplotlib.pyplot as plt

#Generate Training X
n_train = 10
X_train = np.random.uniform(low=0.0, high=1.0, size = n_train)
X_train = np.reshape(X_train, (-1, 1))
X_train = np.hstack((X_train, np.ones(X_train.shape)))
print(X_train.shape)

#Generate Test X
n_test = 100
X_test = np.random.uniform(low=0.0, high=1.0, size=n_test)
X_test = np.reshape(X_test, (-1, 1))
X_test = np.hstack((X_test, np.ones(X_test.shape)))
print(X_test.shape)

# Generate Training t
t_train = (np.sin(2 * (np.pi) * X_train)) + (np.random.normal(loc=0.0, scale = 0.3))
# t_train = t_train[:,0]
# t_train = np.reshape(t_train, (-1, 1))
print(t_train.shape)

#Generate Testing t
t_test = (np.sin(2 * (np.pi) * X_test)) + (np.random.normal(loc=0.0, scale = 0.3))
# t_test = t_test[:,0]
# t_test = np.reshape(t_test, (-1, 1))
print(t_test.shape)

weight = ((np.linalg.pinv(X_train)) @ t_train).T
train_loss = np.square(np.linalg.norm((t_train - (X_train @ weight)), ord=2))
print(train_loss)

def lin_reg_non_lin_map(training_X, training_t, testing_X, testing_t):
    weight = ((np.linalg.pinv(X_train)) @ t_train).T # Weight is trained on the training set and then taken a tranpose of
    train_loss = np.square(np.linalg.norm((training_t - (training_X @ weight)), ord=2))
    test_loss = np.square(np.linalg.norm((testing_t - (testing_X @ weight)), ord=2))
    print("Train Loss:{}".format(train_loss))
    print("Test Loss:{}".format(test_loss))

lin_reg_non_lin_map(X_train, t_train, X_test, t_test)