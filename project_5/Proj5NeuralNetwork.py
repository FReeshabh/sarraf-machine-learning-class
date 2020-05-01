# Rishabh Tewari 
# R11603985
# Machine Learning Project 5 Class Implementation

import numpy as np 
import matplotlib.pyplot as plt 

# input shape

#Calculating loss
def loss(target, prediction):
    """Calculates the loss"""
    return (1/2) * (np.linalg.norm((np.square(target - prediction)), ord=2))

def grad_loss(target, prediction):
    """Derivative of (1/2) * LSE"""
    return (target-prediction)

class TwoLayerNeuralNet(object):
    def __init__(self, INPUT, OUTPUT, INPUT_UNITS, HIDDEN_UNITS, OUTPUT_UNITS, set_seed, activation):
        np.random.seed(seed=set_seed)
        self.INPUT  = INPUT
        self.OUTPUT = OUTPUT
        self.costs  = []

        random_initializer = lambda size1, size2: np.random.uniform(size=(size1, size2)) # Found this was the best for initializing weights and biases
        self.weight_hidden     = random_initializer(INPUT_UNITS, HIDDEN_UNITS) 
        self.weight_prediction = random_initializer(HIDDEN_UNITS, OUTPUT_UNITS) 
        self.bias_hidden       = random_initializer(1, HIDDEN_UNITS) 
        self.bias_prediction   = random_initializer(1, OUTPUT_UNITS) 

        if (activation == 'tanh'):
            self.activ          = lambda z: np.tanh(z)
            self.activ_backprop = lambda z: (1.0 - np.square(np.tanh(z)))
        elif (activation == 'sigmoid'):
            self.activ = lambda z: (1 / (1 + np.exp(-z)))
            self.activ_backprop = lambda z: (z * (1 - z))
        else:
            print("****INVALID Activation****")

    def train_network(self, EPOCHS, RHO_LEARNING_RATE):
        for i in range(EPOCHS):
            # Forward Prop
            self.hidden_layer     = self.activ(np.dot(self.INPUT, self.weight_hidden) + self.bias_hidden)
            self.prediction_layer = self.activ(np.dot(self.hidden_layer, self.weight_prediction) + self.bias_prediction)
            self.costs.append(loss(self.OUTPUT, self.prediction_layer))

            #Backprop
            self.error_prediction_layer = grad_loss(self.OUTPUT, self.prediction_layer)
            self.error_hidden_layer     = np.dot((self.error_prediction_layer * self.activ_backprop(self.prediction_layer)), self.weight_prediction.T)
            
            # Calculating gradients for the update
            grad_prediction_weight = np.dot(self.hidden_layer.T, (self.error_prediction_layer * self.activ_backprop(self.prediction_layer))) 
            grad_prediction_bias   = np.sum(self.error_prediction_layer * self.activ_backprop(self.prediction_layer), keepdims=True)         
            grad_hidden_weight     = np.dot(self.INPUT.T, (self.error_hidden_layer * self.activ_backprop(self.hidden_layer)))                
            grad_hidden_bias       = np.sum((self.error_hidden_layer * self.activ_backprop(self.hidden_layer)), keepdims=True)               
            
            #Updating Weights and Biases - Gradient Descent - Regular Gradient Descent
            self.weight_prediction -= (-RHO_LEARNING_RATE * grad_prediction_weight)
            self.bias_prediction   -= (-RHO_LEARNING_RATE * grad_prediction_bias)
            self.weight_hidden     -= (-RHO_LEARNING_RATE * grad_hidden_weight)
            self.bias_hidden       -= (-RHO_LEARNING_RATE * grad_hidden_bias)
            self.RHO_LEARNING_RATE = RHO_LEARNING_RATE
            self.prediction         = self.activ(np.dot(self.hidden_layer, self.weight_prediction) + self.bias_prediction)

    def graph_loss(self, XLABEL = 'Epochs', YLABEL = 'Error', set_title = 'Error Graph', set_color = 'red', set_label = 'Error Cost'):
        plt.xlabel(XLABEL)
        plt.ylabel(YLABEL)
        plt.title(set_title + " Learning Rate = " + str(self.RHO_LEARNING_RATE))
        plt.plot(self.costs, color = set_color, label = set_label)
        plt.show()
    
    def predict(self, input):
        layer_1 = self.activ(np.dot(input, self.weight_hidden) + self.bias_hidden)
        layer_2 = self.activ(np.dot(layer_1, self.weight_prediction) + self.bias_prediction)
        return layer_2


        




# Problem 1
C_INPUT  = np.array([[0, 0], [0, 1],
                     [1, 0], [1, 1]])
C_OUTPUT  = np.array([[0], [1],
                     [1], [0]])

Problem1Net = TwoLayerNeuralNet(C_INPUT, C_OUTPUT, 2, 2, 1, 137, 'sigmoid')
Problem1Net.train_network(1000, 2)
print(Problem1Net.prediction)
# Problem1Net.graph_loss()

step = .01
X1 = np.arange(-1.5, 1.5 + step, step)
X2 = np.arange(-1.5, 1.5 + step, step)
xx, yy = np.meshgrid(X1, X2)
Z = np.cos(xx, yy)

fig = plt.figure()
ax = fig.gca(projection='3d')
sigmoid = lambda z: (1 / (1 + np.exp(-z)))
for i in range(len(X1)):
    for j in range(len(X2)):
        # input layer
        i_z = np.zeros((2, 1))
        i_z[0, 0] = X1[i];
        i_z[1, 0] = X2[j]
        # hidden layer
        h_a = np.matmul(Problem1Net.weight_hidden, i_z) + Problem1Net.bias_hidden
        h_z = sigmoid(h_a)
        # o_z = Problem1Net.predict(i_z)
        # output layer
        o_a = np.matmul(Problem1Net.weight_prediction, h_z) + Problem1Net.bias_prediction
        o_z = sigmoid(o_a)
        Z[i, j] = np.array([o_z[0]])


XX1, XX2 = np.meshgrid(X1, X2)
surf = ax.plot_surface(XX1, XX2, Z, cmap=plt.cm.coolwarm)  # cividis
plt.title('XOR Decison Surface')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('d(x)')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
plt.show()

#Problem 2
X = 2 * np.random.rand(1, 50) - 1 #1,50
T = np.sin(2 * np.pi * X) + (0.3 * np.random.randn(1, 50)) #1,50

Problem2Net = TwoLayerNeuralNet(X, T, 50, 3, 50, 137, 'tanh')
# Problem2Net.train_network(1000, 0.1)

# Problem2Net.graph_loss()
# print(T)

# plt.plot(T.T)
# plt.show()