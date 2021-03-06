import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(137)

# # PART 1 - 2 layered Neural Network with 2 hidden Units
# -1, as inputs, gave funky results so switched it out for 0's
def PART_1():
    INPUT_UNITS  = 2
    HIDDEN_UNITS = 2
    OUTPUT_UNITS = 1
    C_INPUT  = np.array([[0, 0], [0, 1],
                           [1, 0], [1,1]])
    # 
    C_OUTPUT  = np.array([[0], [1],
                          [1], [0]])

    # Hyperparameters
    EPOCHS = 10000
    RHO_LEARNING_RATE = 0.1

    random_initializer = lambda size1, size2: np.random.uniform(size=(size1, size2)) # Found this was the best for initializing weights and biases
    weight_hidden     = random_initializer(INPUT_UNITS, HIDDEN_UNITS) # 2, 2
    weight_prediction = random_initializer(HIDDEN_UNITS, OUTPUT_UNITS) # 2, 1
    bias_hidden       = random_initializer(1, HIDDEN_UNITS) # 1, 2
    bias_prediction   = random_initializer(1, OUTPUT_UNITS) # 1, 1

    # Activation Funcs
    sigmoid_activation = lambda z: (1 / (1 + np.exp(-z)))
    sigmoid_backpropagation = lambda z: (z * (1 - z))

    # Calculating loss
    def loss(target, prediction):
        """Calculates the loss"""
        return (1/2) * (np.linalg.norm((np.square(target - prediction)), ord=2))

    def grad_loss(target, prediction):
        """Derivative of (1/2) * LSE"""
        return (target-prediction)

    costs = []

    for i in range(EPOCHS):
        """Perform forward propagation"""
        hidden_layer     = sigmoid_activation(np.dot(C_INPUT, weight_hidden) + bias_hidden)
        prediction_layer = sigmoid_activation(np.dot(hidden_layer, weight_prediction) + bias_prediction)
        costs.append(loss(C_OUTPUT, prediction_layer))
        # if((costs[i] - costs[i-1]) < 0.00000001):
        #     EPOCHS = i
        #     break
        """Backpropagation"""
        error_prediction_layer = grad_loss(C_OUTPUT, prediction_layer)
        error_hidden_layer = np.dot((error_prediction_layer * sigmoid_backpropagation(prediction_layer)), weight_prediction.T)

        # Calculating gradients for the update
        grad_prediction_weight = np.dot(hidden_layer.T, (error_prediction_layer * sigmoid_backpropagation(prediction_layer)))
        grad_prediction_bias   = np.sum(error_prediction_layer * sigmoid_backpropagation(prediction_layer), keepdims=True)
        grad_hidden_weight     = np.dot(C_INPUT.T, (error_hidden_layer * sigmoid_backpropagation(hidden_layer)))
        grad_hidden_bias       = np.sum((error_hidden_layer * sigmoid_backpropagation(hidden_layer)), keepdims=True)

        #Updating Weights and Biases
        weight_prediction -= (-RHO_LEARNING_RATE * grad_prediction_weight)
        bias_prediction   -= (-RHO_LEARNING_RATE * grad_prediction_bias)
        weight_hidden     -= (-RHO_LEARNING_RATE * grad_hidden_weight)
        bias_hidden       -= (-RHO_LEARNING_RATE * grad_hidden_bias)

    def graph_loss_part_1(loss):
        # plt.plot(loss, label="error", color = 'red')
        # plt.xlabel('Iterations (EPOCHS)')
        # plt.ylabel('Loss')
        # plt.title('Cost graph Rishabh Tewari')
        # plt.legend()

        plt.plot(C_INPUT, prediction_layer, label="prediction", linestyle = None)
        plt.plot(C_INPUT, C_OUTPUT, label = "Target value", linestyle = None)
        plt.show()

    # graph_loss_part_1(costs)
    # print(*prediction_layer)
    plt.plot(costs) # Error Cost
    plt.show()
PART_1()

def PART_2():
    X = np.random.rand(1, 50) - 1
    T = np.sin(2 * np.pi * X) + (0.3 * np.random.randn(1, 50))

    activation_tanh = lambda z: np.tanh(z)
    backprop_tanh   = lambda z: 1.0 - np.square(np.tanh(z))

    INPUT_UNITS  = 1
    HIDDEN_UNITS = 3
    OUTPUT_UNITS = 1

    random_initializer = lambda size1, size2: np.random.uniform(size=(size1, size2)) # Found this was the best for initializing weights and biases
    weight_hidden     = random_initializer(INPUT_UNITS, HIDDEN_UNITS) # 2, 2
    weight_prediction = random_initializer(HIDDEN_UNITS, OUTPUT_UNITS) # 2, 1
    bias_hidden       = random_initializer(1, HIDDEN_UNITS) # 1, 2
    bias_prediction   = random_initializer(1, OUTPUT_UNITS) # 1, 1

    # Hyperparameters
    EPOCHS = 10000
    RHO_LEARNING_RATE = 0.1

    # Calculating loss
    def loss(target, prediction):
        """Calculates the loss"""
        return (1/2) * (np.linalg.norm((np.square(target - prediction)), ord=2))

    def grad_loss(target, prediction):
        """Derivative of (1/2) * LSE"""
        return (target-prediction)

    costs = []

    for i in range(EPOCHS):
        hidden_layer = activation_tanh(np.dot(X.T , weight_hidden) + bias_hidden)
        prediction_layer = backprop_tanh(np.dot(hidden_layer, weight_prediction) + bias_prediction)
        costs.append(loss(T, prediction_layer))
        
        """Backpropagation"""
        error_prediction_layer = grad_loss(T, prediction_layer)

# PART_2()