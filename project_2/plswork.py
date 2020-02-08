# Rishabh Tewari
# Machine Learning ECE 4332/5332
# R11603985

import numpy as np
import matplotlib.pyplot as plt

#To Ensure Results don't change
np.random.seed(seed = 9)
# np.random.seed()
#Training X
X_train = np.random.uniform(low = 0.0, high = 1.0, size = 10)
X_train = np.reshape(X_train, (-1, 1))
X_train_100 = np.random.uniform(low = 0.0, high = 1.0, size = 100)
X_train_100 = np.reshape(X_train_100, (-1, 1))
# Training t
t_train = (np.sin(2 * (np.pi) * X_train)) + (np.random.normal(loc = 0.0, scale = 0.3))
t_train_100 = (np.sin(2 * (np.pi) * X_train_100)) + (np.random.normal(loc = 0.0, scale = 0.3))
# Testing X
X_test = np.random.uniform(low = 0.0, high = 1.0, size = 100)
X_test = np.reshape(X_test, (-1, 1))
# Testing t
t_test = (np.sin(2 * (np.pi) * X_test)) + (np.random.normal(loc = 0.0, scale = 0.3))

def phi(x, degree):
    feature_v = np.column_stack((np.ones(x.shape), x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10))
    feature_v = feature_v[:,:degree+1]
    # feature_v = np.fliplr(feature_v)
    return feature_v


def linear_regression_with_non_linear_model(training_X, training_t, testing_X, testing_t, deg):
    training_design_matrix_X = phi(training_X, deg)
    weight = (np.linalg.pinv(training_design_matrix_X) @ training_t)
    training_loss = (np.linalg.norm((t_train-(training_design_matrix_X @ weight)), 2)/np.sqrt(len(training_X))) # loss will for the design matrix

    testing_design_matrix_X = phi(testing_X, deg)
    # testing_loss = (np.linalg.norm((testing_t-(testing_design_matrix_X @ weight)), 2)/np.sqrt(len(testing_X))) # loss will for the design matrix
    testing_loss = np.linalg.norm((testing_t - (testing_design_matrix_X @ weight)), 2) / np.sqrt(len(testing_X))
    return training_loss, testing_loss



def graph_loss(training_X, training_t, testing_X, testing_t):
    train_error = np.zeros(10)
    test_error = np.zeros(10)

    for i in range(0, 10):
        train_error[i], test_error[i] = linear_regression_with_non_linear_model(X_train, t_train, X_test, t_test, i)

    train_error = np.reshape(train_error, (-1, 1))
    test_error = np.reshape(test_error, (-1, 1))
    graph_dim = np.linspace(0, 10, 10)
    plt.title("10 training samples, Errors")
    plt.plot(graph_dim, train_error, label = "training error")
    plt.plot(graph_dim, test_error, label = "testing error")
    plt.xlabel('M')
    plt.ylabel('E_rms')
    plt.legend()
    plt.show()

graph_loss(X_train, t_train, X_test, t_test)










# def linear_regression_with_non_linear_model(training_X, training_t, testing_X, testing_t, deg):
#     phi_training_X = phi(training_X, deg)
#     weight = np.linalg.pinv(phi_training_X) @ training_t
#     prediction_training = phi_training_X @ weight
#     cost_training = np.square(np.linalg.norm((training_t - prediction_training), ord=2))
#     cost_training_erms = np.sqrt(cost_training/len(training_X))

#     phi_testing_X = phi(testing_X, deg)
#     # prediction_testing = phi_testing_X @ weight
#     testing_loss = (np.linalg.norm((testing_t-(phi_testing_X @ weight)), 2)/np.sqrt(len(testing_X))) # loss will for the design matrix
#     # cost_testing = np.square(np.linalg.norm((testing_t - prediction_testing), ord=2))
#     # cost_testing_erms = np.sqrt(cost_testing/len(testing_X))

#     # return cost_training_erms, cost_testing_erms
#     return cost_training_erms, testing_loss
