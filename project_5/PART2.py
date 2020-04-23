import numpy as np 
import matplotlib.pyplot as plt 

np.random.seed (seed = 0)
X = 2 * np.random.rand(1, 50) - 1 #1,50
T = np.sin(2 * np.pi * X) + (0.3 * np.random.randn(1, 50)) #1,50

print(X.shape) #1,50
print(T.shape) #1,50

tanh            = lambda z: np.tanh(z)
tanh_backprop   = lambda z: (1.0 - np.square(np.tanh(z)))

INPUT_UNITS  = 50
HIDDEN_UNITS = 20
OUTPUT_UNITS = 50

random_initializer = lambda size1, size2: np.random.uniform(size=(size1, size2)) # Found this was the best for initializing weights and biases
weight_hidden     = random_initializer(INPUT_UNITS, HIDDEN_UNITS) # 50,3
weight_prediction = random_initializer(HIDDEN_UNITS, OUTPUT_UNITS) # 3,1
bias_hidden       = random_initializer(1, HIDDEN_UNITS) # 1, 3
bias_prediction   = random_initializer(1, OUTPUT_UNITS) # 1, 1

print(bias_prediction.shape)

# Calculating loss
def loss(target, prediction):
    """Calculates the loss"""
    return (1/2) * (np.linalg.norm((np.square(target - prediction)), ord=2))

def grad_loss(target, prediction):
    """Derivative of (1/2) * LSE"""
    return (target-prediction)

# Hyperparameters
EPOCHS = 100
RHO_LEARNING_RATE = 0.05

costs = []

for i in range(EPOCHS):
    hidden_layer     = tanh(np.dot(X, weight_hidden) + bias_hidden)
    prediction_layer = tanh(np.dot(hidden_layer, weight_prediction) + bias_prediction)
    costs.append(loss(T, prediction_layer))

    """Backprop"""
    error_prediction_layer = grad_loss(T, prediction_layer)
    error_hidden_layer = np.dot((error_prediction_layer * tanh_backprop(prediction_layer)), weight_prediction.T)

    # Calculating gradients for the update
    grad_prediction_weight = np.dot(hidden_layer.T, (error_prediction_layer * tanh_backprop(prediction_layer)))
    grad_prediction_bias   = np.sum(error_prediction_layer * tanh_backprop(prediction_layer), keepdims=True)
    grad_hidden_weight     = np.dot(X.T, (error_hidden_layer * tanh_backprop(hidden_layer)))
    grad_hidden_bias       = np.sum((error_hidden_layer * tanh_backprop(hidden_layer)), keepdims=True)

    #Updating Weights and Biases
    weight_prediction -= (-RHO_LEARNING_RATE * grad_prediction_weight)
    bias_prediction   -= (-RHO_LEARNING_RATE * grad_prediction_bias)
    weight_hidden     -= (-RHO_LEARNING_RATE * grad_hidden_weight)
    bias_hidden       -= (-RHO_LEARNING_RATE * grad_hidden_bias)

plt.plot(np.linspace(0, 1, 50), T.T, label = "actual graph")
plt.plot(np.linspace(0, 1, 50), prediction_layer.T, label = "prediction")
costs = plt.plot(costs)
plt.legend()
plt.show()
# print(costs)