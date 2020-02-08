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
print(X_train.shape)
# Training t
t_train = (np.sin(2 * (np.pi) * X_train)) + (np.random.normal(loc = 0.0, scale = 0.3))
# Testing X
X_test = np.random.uniform(low = 0.0, high = 1.0, size = 100)
X_test = np.reshape(X_test, (-1, 1))
# Testing t
t_test = (np.sin(2 * (np.pi) * X_test)) + (np.random.normal(loc = 0.0, scale = 0.3))

def phix(x, degree):
    pix = np.array([1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9])
    pix = pix[:degree]
    print(pix.shape)
    # return pix
# phix(X_train, 3)

def phi(x, degree):
    poly = PF(degree)
    Z = poly.fit_transform(x)
    Z - np.fliplr(Z)
    return Z

print(phi(X_train, 2))

# def phi(X, degree):
#     arr = []
#     for x in range(X.len()):
#         for i in range(degree+1):
#             arr.append(X[i]**i)
#     return arr
# print(phi(X_train, 2))


def h(training_X, training_t, testing_X, testing_t, degree):
    # d = np.linspace(0, 1, 10)
    training_design_matrix_X = phi(training_X, degree)
    plot_X = np.arange(0, 1, .1)
    # print(training_design_matrix_X.shape)
    weight_t = (np.linalg.pinv(training_design_matrix_X) @ training_t).T # training_design matrix is a matrix, training t is a vector
    print(weight_t.shape)
    pred_y = (weight_t * training_X)
    print("Prediction" + str(pred_y.shape))
    plt.plot(plot_X, pred_y, color = "red", label="Closed Form Solution for Training")
    plt.legend()
    plt.title("Linear Regression with Non Linear Models")
    plt.scatter(X_train, t_train, label="Training data")
    plt.scatter(X_test[:,0], t_test, label="Testing data")
    plt.xlabel('X')
    plt.ylabel('t')
    plt.legend()
    plt.show()

    pass

h(X_train, t_train, X_test, X_train, 4)


def simple_linear_regression_closed_form_solution(x, t):
    """
    Analytical Solution for the weight for Linear Regression
    """
    x = np.hstack((x, np.ones(x.shape)))
    weight = (np.linalg.pinv(x) @ t).T #the transpose of the final weight
    d_closed = np.linspace(0, 1, 10)
    prediction_closed = weight.T [0][0]*d_closed + weight.T[1][0]
    print(prediction_closed)
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
    feature_v = list(feature_v)
    feature_v = feature_v[:degree]
    feature_v = np.asarray(feature_v)
    print(feature_v.shape)

    # x = np.array([X_train])

def simple_linear_regression_closed_form_solution(x, t, deg):
    """
    Analytical Solution for the weight for Linear Regression
    """
    x = phi(x, deg)
    print(x.shape)
    print(t.shape)
    weight = (np.linalg.pinv(x) @ t).T #the transpose of the final weight
    print(weight.shape)
    d_closed = np.random.uniform(low = 0.0, high = 1.0, size = 10)#np.linspace(0, 1, 10)
    d_closed = phi(d_closed, deg)
    prediction_closed = weight.T [0][0]*d_closed + weight.T[1][0]
    # prediction_closed = weight @ x
    print(prediction_closed)
    print(np.linalg.norm((prediction_closed - t), ord = 2))
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

simple_linear_regression_closed_form_solution(X_train, t_train, 4)

    # plt.plot(graph_X, predicted, color = "red", label="Closed Form Solution for Training")
    # plt.legend()
    # plt.title("Linear Regression with Non Linear Models")
    # plt.scatter(X_train, t_train, label="Training data")
    # plt.scatter(X_test[:,0], t_test, label="Testing data")
    # plt.xlabel('X')
    # plt.ylabel('t')
    # plt.legend()
    # plt.show()

# training_loss, testing_loss = 10000, 10000
#    training_arr = np.empty(training_X.shape)
#    testing_arr = np.empty(testing_X.shape)
#    for deg in range(10):
#        training_loss, testing_loss = linear_regression_with_non_linear_model(training_X, training_t, testing_X, testing_t, deg)
#        np.append(training_arr, training_loss)
#        # np.append(testing_arr, testing_loss)
#    print(training_arr)
#    # print(testing_arr)