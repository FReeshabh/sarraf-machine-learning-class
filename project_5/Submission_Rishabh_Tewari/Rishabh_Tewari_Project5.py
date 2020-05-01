# Rishabh Tewari
# R11603985
# Project 5
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

np.random.seed(seed = 3)  

# Activation Functions
tanh             = lambda z: np.tanh(z)
tanh_backprop    = lambda z: (1.0 - np.square(np.tanh(z)))
sigmoid          = lambda z: (1 / (1 + np.exp(-z)))
sigmoid_backprop = lambda z: (z * (1 - z))

def loss(target, prediction):
    """Calculates the loss"""
    return (1/2) * (np.linalg.norm((np.square(target - prediction)), ord=2))

def grad_loss(target, prediction):
    """Derivative of (1/2) * LSE"""
    return (target-prediction)

def PART1():
    C_INPUT = np.array([[-1, -1, 1, 1],
                         [-1, 1, -1, 1]])  
    T       = np.array([[0, 1,
                         1, 0]])  

    # Hyperparameters
    RHO_LEARN_RATE = 0.2
    EPOCHS = 250
    BETA = 0.9
    costs = []

    INPUT_UNITS, N = C_INPUT.shape
    HIDDEN_UNITS = 2
    OUTPUT_UNITS, _ = T.shape

    # Initialize weights and biases
    weight_hidden = np.random.uniform(size=(HIDDEN_UNITS, INPUT_UNITS))
    weight_output = np.random.uniform(size=(OUTPUT_UNITS, HIDDEN_UNITS))
    bias_hidden   = np.random.uniform(size = (HIDDEN_UNITS, 1))
    bias_output   = np.random.uniform(size = (OUTPUT_UNITS, 1))

    grad_weight_hidden = np.zeros_like(weight_hidden)
    grad_weight_output = np.zeros_like(weight_output)
    grad_bias_hidden   = np.zeros_like(bias_hidden)
    grad_bias_output   = np.zeros_like(bias_output)

    for i in range(EPOCHS):
        input = C_INPUT
        # Forward Pass
        hidden_layer     = tanh((weight_hidden @ input) + np.repeat(bias_hidden, N, axis=1))
        prediction_layer = sigmoid((weight_output @ hidden_layer) + np.repeat(bias_output, N, axis=1))
        costs.append(loss(T, prediction_layer))

        # Backprop
        output_layer_error = grad_loss(prediction_layer , T)
        grad_weight_output    = BETA * grad_weight_output + (1 - BETA) * (output_layer_error @ hidden_layer.T)
        grad_bias_output      = BETA * grad_bias_output + (1 - BETA) * np.array([np.sum(output_layer_error, axis=1)]).T

        hidden_layer_error = np.multiply(weight_output.T@ output_layer_error , tanh_backprop(((weight_hidden @ input) + np.repeat(bias_hidden, N, axis=1))))
        grad_weight_hidden    = BETA * grad_weight_hidden + (1 - BETA) * (hidden_layer_error @ input.T)
        grad_bias_hidden      = BETA * grad_bias_hidden + (1 - BETA) * np.array([np.sum(hidden_layer_error, axis=1)]).T

        # Update using Gradient Descent with momentum
        weight_output   -= np.multiply(RHO_LEARN_RATE, grad_weight_output)
        bias_output     -= np.multiply(RHO_LEARN_RATE, grad_bias_output)
        weight_hidden   -= np.multiply(RHO_LEARN_RATE, grad_weight_hidden)
        bias_hidden     -= np.multiply(RHO_LEARN_RATE, grad_bias_hidden)

    # Graph Costs
    plt.figure()
    plt.title("Rishabh Tewari Error XOR Classification")
    plt.plot(costs, label = "Error - XOR Classification", color = "red")
    plt.xlabel("EPOCHS")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    # Graph Decision Surface
    step = .01
    X_1 = np.arange(-2, 2 + step, step = step)
    X_2 = np.arange(-2, 2 + step, step = step)
    xx, yy = np.meshgrid(X_1, X_2)
    Z = np.sin(xx, yy)

    for i in range(len(X_2)):
        for j in range(len(X_1)):
            input       = np.zeros((2, 1))
            input[0, 0] = X_1[i];
            input[1, 0] = X_2[j]
            hidden_z = tanh((weight_hidden @ input) + bias_hidden)
            prediction_layer_z = tanh((weight_output @ hidden_z) + bias_output)
            Z[i, j] = np.array([prediction_layer_z[0]])

    XX1, XX2 = np.meshgrid(X_1, X_2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(XX1, XX2, Z)  
    plt.title('XOR Classification - Decision Surface, Rishabh Tewari')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('dx')
    plt.legend()
    plt.show()

def PART2():
    read_mat = loadmat('matlab_dataset.mat')
    X = read_mat['X']
    T = read_mat['Y']

    # Hyperparameters
    RHO_LEARN_RATE = 0.02
    EPOCHS = 2500
    BETA = 0.9
    costs = []

    INPUT_UNITS, N = X.shape
    HIDDEN_UNITS = 3
    OUTPUT_UNITS, _ = T.shape

    # Initialize weights and biases
    weight_hidden = np.random.uniform(size=(HIDDEN_UNITS, INPUT_UNITS))
    weight_output = np.random.uniform(size=(OUTPUT_UNITS, HIDDEN_UNITS))
    bias_hidden   = np.random.uniform(size = (HIDDEN_UNITS, 1))
    bias_output   = np.random.uniform(size = (OUTPUT_UNITS, 1))

    grad_weight_hidden = np.zeros_like(weight_hidden)
    grad_weight_output = np.zeros_like(weight_output)
    grad_bias_hidden   = np.zeros_like(bias_hidden)
    grad_bias_output   = np.zeros_like(bias_output)

    for i in range(EPOCHS):
        input = X
        # Forward Pass
        hidden_layer     = tanh((weight_hidden @ input) + np.repeat(bias_hidden, N, axis=1))
        prediction_layer = tanh((weight_output @ hidden_layer) + np.repeat(bias_output, N, axis=1))
        costs.append(loss(T, prediction_layer))

        # Backprop
        output_layer_error    = grad_loss(prediction_layer , T)
        grad_weight_output    = BETA * grad_weight_output + (1 - BETA) * (output_layer_error @ hidden_layer.T)
        grad_bias_output      = BETA * grad_bias_output + (1 - BETA) * np.array([np.sum(output_layer_error, axis=1)]).T

        hidden_layer_error    = np.multiply(weight_output.T@ output_layer_error , tanh_backprop(((weight_hidden @ input) + np.repeat(bias_hidden, N, axis=1))))
        grad_weight_hidden    = BETA * grad_weight_hidden + (1 - BETA) * (hidden_layer_error @ input.T)
        grad_bias_hidden      = BETA * grad_bias_hidden + (1 - BETA) * np.array([np.sum(hidden_layer_error, axis=1)]).T

        # Update using Gradient Descent with momentum
        weight_output   -= np.multiply(RHO_LEARN_RATE , grad_weight_output)
        bias_output     -= np.multiply(RHO_LEARN_RATE , grad_bias_output)
        weight_hidden   -= np.multiply(RHO_LEARN_RATE , grad_weight_hidden)
        bias_hidden     -= np.multiply(RHO_LEARN_RATE , grad_bias_hidden)

    # Graph Costs
    # plt.figure()
    plt.title("Rishabh Tewari Error PART 2")
    plt.plot(costs, label = "Error - 3 Hidden Units", color = "red")
    plt.xlabel("EPOCHS")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    # Plot the graph
    X = np.linspace(-1, 1, 200)
    Y = []
    for i in range(len(X)):
        mock_input = np.zeros((1, 1))
        mock_input[0] = X[i]
        hidden_z = tanh((weight_hidden @ mock_input) + bias_hidden)
        prediction = tanh((weight_output @ hidden_z) + bias_output)
        Y.append(prediction[0][0])

    ax = plt.figure()
    ax = ax.gca()
    ax.plot(X, Y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("3 Units, Prediction Graph - Rishabh Tewari")
    plt.show()

def PART2_2():
    read_mat = loadmat('matlab_dataset.mat')
    X = read_mat['X']
    T = read_mat['Y']

    # Hyperparameters
    RHO_LEARN_RATE = 0.02
    EPOCHS = 2500
    BETA = 0.9
    costs = []

    INPUT_UNITS, N = X.shape
    HIDDEN_UNITS = 20
    OUTPUT_UNITS, _ = T.shape

    # Initialize weights and biases
    weight_hidden = np.random.uniform(size=(HIDDEN_UNITS, INPUT_UNITS))
    weight_output = np.random.uniform(size=(OUTPUT_UNITS, HIDDEN_UNITS))
    bias_hidden   = np.random.uniform(size = (HIDDEN_UNITS, 1))
    bias_output   = np.random.uniform(size = (OUTPUT_UNITS, 1))

    grad_weight_hidden = np.zeros_like(weight_hidden)
    grad_weight_output = np.zeros_like(weight_output)
    grad_bias_hidden   = np.zeros_like(bias_hidden)
    grad_bias_output   = np.zeros_like(bias_output)

    for i in range(EPOCHS):
        input = X
        # Forward Pass
        hidden_layer     = tanh((weight_hidden @ input) + np.repeat(bias_hidden, N, axis=1))
        prediction_layer = tanh((weight_output @ hidden_layer) + np.repeat(bias_output, N, axis=1))
        costs.append(loss(T, prediction_layer))

        # Backprop
        output_layer_error    = grad_loss(prediction_layer , T)
        grad_weight_output    = BETA * grad_weight_output + (1 - BETA) * (output_layer_error @ hidden_layer.T)
        grad_bias_output      = BETA * grad_bias_output + (1 - BETA) * np.array([np.sum(output_layer_error, axis=1)]).T

        hidden_layer_error    = np.multiply(weight_output.T@ output_layer_error , tanh_backprop(((weight_hidden @ input) + np.repeat(bias_hidden, N, axis=1))))
        grad_weight_hidden    = BETA * grad_weight_hidden + (1 - BETA) * (hidden_layer_error @ input.T)
        grad_bias_hidden      = BETA * grad_bias_hidden + (1 - BETA) * np.array([np.sum(hidden_layer_error, axis=1)]).T

        # Update using Gradient Descent with momentum
        weight_output   -= np.multiply(RHO_LEARN_RATE, grad_weight_output)
        bias_output     -= np.multiply(RHO_LEARN_RATE, grad_bias_output)
        weight_hidden   -= np.multiply(RHO_LEARN_RATE, grad_weight_hidden)
        bias_hidden     -= np.multiply(RHO_LEARN_RATE, grad_bias_hidden)

    # Graph Costs
    plt.title("Rishabh Tewari Error PART 2")
    plt.plot(costs, label = "Error - 20 Hidden Units", color = "red")
    plt.xlabel("EPOCHS")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    # Plot the graph
    X = np.linspace(-1, 1, 100)
    Y = []
    for i in range(len(X)):
        mock_input = np.zeros((1, 1))
        mock_input[0] = X[i]
        hidden_z = tanh((weight_hidden @ mock_input) + bias_hidden)
        prediction = tanh((weight_output @ hidden_z) + bias_output)
        Y.append(prediction[0][0])

    ax = plt.figure()
    ax = ax.gca()
    ax.plot(X, Y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("20 Units, Prediction Graph, Rishabh Tewari")
    plt.show()

PART1()
PART2()
PART2_2()