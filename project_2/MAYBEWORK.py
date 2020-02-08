import numpy as np
import matplotlib.pyplot as plt

#To Ensure Results don't change
np.random.seed(seed = 147) # 147 140, 138

N_train = 10
#Training X with 10 samples
X_train = np.random.uniform(low = 0.0, high = 1.0, size = N_train)
X_train = np.reshape(X_train, (-1, 1))

# Training t
t_train = (np.sin(2 * (np.pi) * X_train)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X_train.shape))

# Testing X
X_test = np.random.uniform(low = 0.0, high = 1.0, size = 100)
X_test = np.reshape(X_test, (-1, 1))
# Testing t
t_test = (np.sin(2 * (np.pi) * X_test)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X_test.shape))

def phi(x, degree):
    feature_vector = np.column_stack((np.ones(x.shape), x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10))
    feature_vector = feature_vector[:,:degree+1]
    return feature_vector

def linear_regression_with_non_linear_model(training_X, training_t, testing_X, testing_t, deg):
    phi_training_X = phi(training_X, deg)
    phi_training_X = phi_training_X/(phi_training_X.max())
    phi_testing_X  = phi(testing_X, deg)
    phi_testing_X = phi_testing_X/(phi_training_X.max())

    weight = (np.linalg.pinv(phi_training_X)) @ training_t
    training_cost = np.sqrt((np.linalg.norm((training_t - (phi_training_X @ weight))**2)) / N_train)
    testing_cost  = np.sqrt((np.linalg.norm((testing_t - (phi_testing_X @ weight))**2)  ) / N_train)

    return training_cost, testing_cost

for i in range(10):
    print(linear_regression_with_non_linear_model(X_train, t_train, X_test, t_test, i))

def graph_loss(training_X, training_t, testing_X, testing_t):
    train_error = np.zeros(10)
    test_error = np.zeros(10)

    for i in range(0, 10):
        train_error[i], test_error[i] = linear_regression_with_non_linear_model(X_train, t_train, X_test, t_test, i)

    train_error = np.reshape(train_error, (-1, 1))
    test_error = np.reshape(test_error, (-1, 1))
    graph_dim = np.linspace(0, 10, 10)
    plt.title("10 training samples, Errors")
    plt.plot(graph_dim, train_error, label = "training error", marker = "o")
    plt.plot(graph_dim, test_error, label = "testing error", marker = "o")
    plt.ylim(bottom = 0, top = 1)
    plt.xlabel('M')
    plt.ylabel('$E_{rms}$')
    plt.legend()
    plt.show()

graph_loss(X_train, t_train, X_test, t_test)
# linear_regression_with_non_linear_model(X_train, t_train, X_test, t_test, 9)
