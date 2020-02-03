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
t_train = t_train[:,0] 
t_train = np.reshape(t_train, (-1, 1))

# Generate Test Set
test_size_n = 100
uniform_test_X = np.random.uniform(low=0.0, high = 1.0, size = test_size_n)
uniform_test_X = np.reshape(uniform_test_X, (-1, 1))
uniform_test_X = np.hstack((uniform_test_X, np.ones(uniform_test_X.shape)))
t_test = (np.sin(2 * np.pi * uniform_test_X)) + (np.random.normal(loc=0.0, scale = 0.3))
t_test = t_test[:,0] 
t_test = np.reshape(t_test, (-1, 1))

def phi(X, degree):
    return (np.polynomial.polynomial.polyval(X, degree))
    

def linear_reg_non_lin(training_X, training_t, test_X, test_t, degree):
    weight = ((np.linalg.pinv(phi(training_X, degree))) @ training_t).T
    train_loss = (np.square(np.linalg.norm(training_t - phi(training_X, degree)@ weight)))
    test_loss = np.square(np.linalg.norm(test_t - phi(test_X, degree) @ weight))
    # return train_loss, test_loss
    print("Train Loss:{}".format(train_loss))
    print("Test Loss:{}".format(test_loss))

def linear_regression(training_X, training_t, test_X, test_t):
    weight = (np.linalg.pinv(training_X) @ training_t).T
    train_loss = np.square(np.linalg.pinv(training_t) - (training_X@weight))
    test_loss = np.square(np.linalg.pinv(training_t) - (test_X@weight))
    print("Train Loss:{}".format(train_loss))
    print("Test Loss:{}".format(test_loss))

def produce_plot(train_loss = 1000, test_loss = 1000):
    plot_vector_train = np.array(t_train.shape)
    plot_vector_test = np.array(t_test.shape)
    M = []
    for i in range(10):
        train_loss, test_loss = linear_reg_non_lin(uniform_train_X, t_train, uniform_test_X, t_test, i)
        train_loss_Erms = np.sqrt((train_loss)/training_size_n)
        test_loss_Erms = np.sqrt((test_loss)/test_size_n)
        np.insert(plot_vector_train, i, train_loss_Erms)
        np.insert(plot_vector_test, i, test_loss_Erms)
        M.append(i)
    plt.title("Project 2\n Rishabh Tewari")
    plt.xlabel('M')
    plt.ylabel('E_RMS')
    plt.plot(M, plot_vector_train, color = "red", label="Training Loss")
    plt.plot(M, plot_vector_test, color = "red", label="Testing Loss")
    plt.legend()
    plt.figure()

# linear_reg_non_lin(uniform_train_X, t_train, uniform_test_X, t_test, 1)
linear_regression(uniform_test_X,t_train, uniform_test_X, t_test) 
print(phi(uniform_train_X, 2))
# produce_plot()
# print(t_test.shape)