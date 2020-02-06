# Rishabh Tewari
# Machine Learning ECE 4332/5332
# R11603985

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as PF

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

def phix(x, degree):
    pix = [1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9]
    pix = pix[:degree]
    print(pix)
    return pix

def phi(x, degree):
    poly = PF(degree)
    Z = poly.fit_transform(x)
    return Z

def fuck_around(training_X, training_t, testing_X, testing_t, degree):
    training_design_matrix_X = phi(training_X, degree)

    pass

fuck_around(X_train, t_train, X_test, testing_t, degree)


def simple_linear_regression_closed_form_solution(x, t):
    """
    Analytical Solution for the weight for Linear Regression
    """
    x = np.hstack((x, np.ones(x.shape)))
    weight = (np.linalg.pinv(x) @ t).T #the transpose of the final weight
    d_closed = np.linspace(0, 1, 10)
    prediction_closed = weight.T [0][0]*d_closed + weight.T[1][0]
    print(t)
    print(np.linalg.norm((prediction_closed - t[:,0]), ord = 2))
    # plt.scatter(X_train[:,0], t_train, label="actual data")
    plt.plot(d_closed, prediction_closed, color = "red", label="Closed Form Solution for Training")
    plt.legend()
    plt.title("Linear Regression with Non Linear Models")
    plt.scatter(X_train, t_train, label="Training data")
    plt.scatter(X_test[:,0], t_test, label="Testing data")
    plt.xlabel('X')
    plt.ylabel('t')
    plt.legend()
    plt.show()


# simple_linear_regression_closed_form_solution(X_train, t_train)
