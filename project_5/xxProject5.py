import matplotlib.pyplot as plt 
import numpy as np 

#Calculating loss
def loss(target, prediction):
    """Calculates the loss"""
    return (1/2) * (np.linalg.norm((np.square(target - prediction)), ord=2))

def grad_loss(target, prediction):
    """Derivative of (1/2) * LSE"""
    return (target-prediction)

tanh             = lambda z: np.tanh(z)
tanh_backprop    = lambda z: (1.0 - np.square(np.tanh(z)))
sigmoid          = lambda z: (1 / (1 + np.exp(-z)))
sigmoid_backprop = lambda z: (z * (1 - z))

def PART_1():
    C_INPUT  = np.array([[-1, -1, 1, 1],
                         [-1, 1, -1, 1]])  # XOR Inputs
    C_OUTPUT = np.array([[0, 1,
                         1, 0]])  # XOR Outputs

    INPUT_UNITS, n = C_INPUT.shape
    HIDDEN_UNITS   = 2
    OUTPUT_UNITS, _   = C_OUTPUT.shape

    weight_hidden     = np.random.uniform(size =(HIDDEN_UNITS, INPUT_UNITS))
    weight_prediction = np.random.uniform(size =(OUTPUT_UNITS, HIDDEN_UNITS))
    bias_hidden       = np.random.uniform(size =(HIDDEN_UNITS, 1) )
    bias_prediction   = np.random.uniform(size =(OUTPUT_UNITS, 1) )

    v_weight_hidden     = np.zeros_like(weight_hidden)
    v_weight_prediction = np.zeros_like(weight_prediction)
    v_bias_hidden       = np.zeros_like(bias_hidden)
    v_bias_prediction   = np.zeros_like(bias_prediction)

    EPOCHS = 10000
    RHO_LEARNING_RATE = 0.1
    BETA = 0.9

    costs = []

    for i in range(EPOCHS):
        # Forward Propogation
        hidden_layer_activated     = tanh((weight_hidden @ C_INPUT) + np.repeat(bias_hidden, n, axis=1))
        prediction_layer_activated = sigmoid((weight_prediction @ hidden_layer_activated) + np.repeat(bias_prediction, n, axis=1))

        """BACKPROP"""
        # Prediction Layer
        error_prediction_layer = grad_loss(C_OUTPUT, prediction_layer_activated)
        v_weight_prediction = (BETA * v_weight_prediction) + ((1 - BETA) * (error_prediction_layer @ hidden_layer_activated.T))
        v_bias_prediction   = (BETA * v_bias_prediction) + ((1 - BETA) * (np.sum(error_prediction_layer, axis=1, keepdims=True)).T)

        #Update -- gradient descent with momentum
        weight_prediction -= RHO_LEARNING_RATE * v_weight_prediction
        bias_prediction   -= RHO_LEARNING_RATE * v_bias_prediction

        # Hidden Layer
        error_hidden_layer = np.multiply((weight_prediction.T @ error_prediction_layer), (tanh_backprop((weight_hidden @ C_INPUT) + np.repeat(bias_hidden, n, axis=1))))
        v_weight_hidden = (BETA * v_weight_hidden) + (1 - BETA) * (error_hidden_layer @ C_INPUT.T)
        v_bias_hidden   = (BETA * v_bias_hidden) + ((1 - BETA) * np.sum(error_hidden_layer, axis=1, keepdims=True))

        #Update -- gradient descent with momentum
        weight_hidden -= RHO_LEARNING_RATE * v_weight_hidden
        bias_hidden   -= RHO_LEARNING_RATE * v_bias_hidden

        # Add losses to the array
        costs.append(loss(C_OUTPUT, prediction_layer_activated))

    plt.plot(costs)
    plt.show()
PART_1()