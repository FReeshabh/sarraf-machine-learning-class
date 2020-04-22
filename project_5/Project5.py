import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(42)

# # PART 1 - 2 layered Neural Network with 2 hidden Units
INPUT_UNITS, HIDDEN_UNITS, OUTPUT_UNITS = 2, 2, 1
C_INPUT  = np.array([[0, 0], [0, 1],
                       [1, 0], [1,1]])
# 
C_OUTPUT  = np.array([[0],[1],[1],[0]])
EPOCHS = 10000
RHO_LEARNING_RATE = 0.1

sigmoid_activation = lambda z: (1 / (1 + np.exp(-z)))
sigmoid_backpropagation = lambda z: (z * (1 - z))

weight_hidden     = np.random.uniform(size = (INPUT_UNITS, HIDDEN_UNITS)) # 2, 2
weight_prediction = np.random.uniform(size = (HIDDEN_UNITS, OUTPUT_UNITS)) # 2, 1
bias_hidden       = np.random.uniform(size = (1, HIDDEN_UNITS)) # 1, 2
bias_prediction   = np.random.uniform(size = (1, OUTPUT_UNITS)) # 1, 1


for i in range(EPOCHS):
    """Perform forward propagation"""
    hidden_layer     = sigmoid_activation(np.dot(C_INPUT, weight_hidden) + bias_hidden)
    # print(*hidden_layer)
    prediction_layer = sigmoid_activation(np.dot(hidden_layer, weight_prediction))
    # print(*prediction_layer)

    """Backpropagation"""
    error_prediction_layer = C_OUTPUT - prediction_layer
    error_hidden_layer = np.dot((error_prediction_layer * sigmoid_backpropagation(prediction_layer)), weight_prediction.T)

    # Calculating gradients for the update
    grad_prediction_weight = np.dot(hidden_layer.T, (error_prediction_layer * sigmoid_backpropagation(prediction_layer)))
    grad_prediction_bias   = np.sum(error_prediction_layer * sigmoid_backpropagation(prediction_layer), keepdims=True)
    grad_hidden_weight     = np.dot(C_INPUT.T, (error_hidden_layer * sigmoid_backpropagation(hidden_layer)))
    grad_hidden_bias       = np.sum((error_hidden_layer * sigmoid_backpropagation(hidden_layer)), keepdims=True)

    #Updating Weights and Biases
    weight_prediction += RHO_LEARNING_RATE * grad_prediction_weight
    bias_prediction   += RHO_LEARNING_RATE * grad_prediction_bias
    weight_hidden     += RHO_LEARNING_RATE * grad_hidden_weight
    bias_hidden       += RHO_LEARNING_RATE * grad_hidden_bias

print(*prediction_layer)