# Rishabh Tewari
# Machine Learning ECE 4332/5332
# R11603985

import numpy as np
import matplotlib.pyplot as plt

#To Ensure Results don't change
np.random.seed(seed = 0)
#Training X
X_train = np.random.uniform(low = 0.0, high = 1.0, size = 10)
X_train = np.reshape(X_train, (-1, 1))
# Training t
t_train = (np.sin(2 * (np.pi) * X_train)) + (np.random.normal(loc = 0.0, scale = 0.3))
# Testing X
X_test = np.random.uniform(low = 0.0, high = 1.0, size = 100)
X_test = np.reshape(X_test, (-1, 1))
# Testing t
t_test = (np.sin(2 * (np.pi) * X_test)) + (np.random.normal(loc = 0.0, scale = 0.3))

def phi(x, degree):
    feature_v = np.column_stack((np.ones(x.shape), x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9))
    feature_v = feature_v[:,:degree+1]
    # feature_v = np.fliplr(feature_v)
    return feature_v


def linear_regression_with_non_linear_model(training_X, training_t, testing_X, testing_t, deg):
    graph_X = np.linspace(0, 1, 10)
    graph_X = np.reshape(graph_X, (-1, 1))
    training_design_matrix = phi(training_X, deg)
    print(training_design_matrix.shape)
    weight = (np.linalg.pinv(training_design_matrix) @ training_t)
    print("Shape of Weight" + str(weight.shape))
    loss = np.linalg.norm((t_train-(training_design_matrix @ weight)), 2) # loss will for the design matrix
    error = loss/10
    print("Traning Loss: " + str(error) + " for degree " + str(deg))

linear_regression_with_non_linear_model(X_train, t_train, X_test, t_test, 9)