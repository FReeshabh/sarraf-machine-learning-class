import numpy as np 
np.random.seed(42)

C_INPUTS         = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
C_TARGET_OUTPUTS = np.array([[-1],[1],[1],[-1]])

# Hyperparameter Selection 
EPOCHS = 2500
RHO_LEARNING_RATE = 0.2

INPUT_UNITS  = 2
HIDDEN_UNITS = 2 #Only one hidden layer, with two units inside it
OUTPUT_UNITS = 1

# Weight Initialization 2 layered network
weight_hidden_1 = np.random.normal(size=(INPUT_UNITS, HIDDEN_UNITS)) #(2, 2)
weight_output_2 = np.random.normal(size=(HIDDEN_UNITS, OUTPUT_UNITS)) #(2, 1)

# Bias Initialization
bias_hidden_1   = np.random.normal(size=(1, HIDDEN_UNITS)) #(1, 2)
bias_output_2   = np.random.normal(size=(1, OUTPUT_UNITS)) #(1, 1)

sigmoid_activation = lambda a: (1 / (1 + np.exp(-a)))
sigmoid_backprop   = lambda a: (a * (1 - a))
error = lambda prediction, target: (1/2)*(np.square(np.linalg.norm(target - prediction)))
# error = lambda prediction, target: 2*(prediction - target)
for i in range(EPOCHS):
    # Forward pass
    hidden_layer_output_1 = sigmoid_activation((np.dot(C_INPUTS, weight_hidden_1) + bias_hidden_1)) # (4,2)
    output_layer_output_2 = sigmoid_activation((np.dot(hidden_layer_output_1, weight_output_2) + bias_output_2)) # (4,1)

    # Backward Propagation
    cost_error = error(output_layer_output_2, C_TARGET_OUTPUTS)
    grad_weight_hidden_1 = np.dot(C_INPUTS.T, (np.dot(cost_error * sigmoid_backprop(C_TARGET_OUTPUTS), weight_output_2.T) * sigmoid_backprop(hidden_layer_output_1))) * RHO_LEARNING_RATE
    grad_weight_output_2 = (np.dot(hidden_layer_output_1.T, (cost_error * sigmoid_activation(output_layer_output_2))))  * RHO_LEARNING_RATE

    weight_hidden_1 += grad_weight_hidden_1
    weight_output_2 += grad_weight_output_2

    print(cost_error)