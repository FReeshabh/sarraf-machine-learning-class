# Rishabh Tewari    
# R11603985

import numpy as np 
import matplotlib.pyplot as plt 

#Calculating loss
def loss(target, prediction):
    """Calculates the loss"""
    return (1/2) * (np.linalg.norm((np.square(target - prediction)), ord=2))

def grad_loss(target, prediction):
    """Derivative of (1/2) * LSE"""
    return (target-prediction)

class NeuralNetworkTwo(object):
    def __init__(self, INPUT, OUTPUT, INPUT_UNITS, HIDDEN_UNITS, OUTPUT_UNITS, set_seed, activation):
        self.INPUT = INPUT
        self.OUTPUT = OUTPUT
        self.costs = []
        np.random.seed(seed=set_seed)
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

    def train_network(self, EPOCHS, RHO_LEARNING_RATE, momentum):
        vgpw, vgpb, vghw, vghb = 0, 0, 0, 0
        for i in range(EPOCHS):
            # Forward Prop
            self.hidden_layer     = self.activ(np.dot(self.INPUT, self.weight_hidden) + self.bias_hidden)
            self.prediction_layer = self.activ(np.dot(self.hidden_layer, self.weight_prediction) + self.bias_prediction)
            self.costs.append(loss(self.OUTPUT, self.prediction_layer))
            #Backprop
            self.error_prediction_layer = grad_loss(self.OUTPUT, self.prediction_layer)
            self.error_hidden_layer     = np.dot((self.error_prediction_layer * self.activ_backprop(self.prediction_layer)), self.weight_prediction.T)
            
            # Calculating gradients for the update
            grad_prediction_weight = np.dot(self.hidden_layer.T, (self.error_prediction_layer * self.activ_backprop(self.prediction_layer))) + vgpw
            grad_prediction_bias   = np.sum(self.error_prediction_layer * self.activ_backprop(self.prediction_layer), keepdims=True)         + vgpb
            grad_hidden_weight     = np.dot(self.INPUT.T, (self.error_hidden_layer * self.activ_backprop(self.hidden_layer)))                + vghw
            grad_hidden_bias       = np.sum((self.error_hidden_layer * self.activ_backprop(self.hidden_layer)), keepdims=True)               + vghb
            
            vgpw = momentum * vgpw - grad_prediction_weight
            vgpb = momentum * vgpb - grad_prediction_bias
            vghw = momentum * vghw - grad_hidden_weight
            vghb = momentum * vghb  - grad_hidden_bias

            #Updating Weights and Biases - Gradient Descent
            self.weight_prediction -= (-RHO_LEARNING_RATE * grad_prediction_weight)
            self.bias_prediction   -= (-RHO_LEARNING_RATE * grad_prediction_bias)
            self.weight_hidden     -= (-RHO_LEARNING_RATE * grad_hidden_weight)
            self.bias_hidden       -= (-RHO_LEARNING_RATE * grad_hidden_bias)
            self.prediction = self.activ(np.dot(self.hidden_layer, self.weight_prediction) + self.bias_prediction)


    def graph_loss(self, label_text):
        plt.xlabel("EPOCHS")
        plt.ylabel("Error")
        plt.title("Cost Graph - " + label_text)
        plt.plot(self.costs, color = 'red', label = 'error')
        plt.show()

    def experimental_grapher(self, Xlabel, Ylabel, label_text, graph_x, graph_y, color):
        plt.xlabel(xlabel = Xlabel)
        plt.ylabel(ylabel = Ylabel)
        plt.title(label = label_text)
        plt.plot(graph_x, graph_y, color = color)


C_INPUT  = np.array([[0, 0], [0, 1],
                     [1, 0], [1, 1]])
    # 
C_OUTPUT  = np.array([[0], [1],
                     [1], [0]])
#137
Problem1Net = NeuralNetworkTwo(C_INPUT, C_OUTPUT, 2, 2, 1, 137, activation='sigmoid')
# Problem1Net.train_network(10000, 0.5, 0.1) # Momentum kept at 0.1
# Problem1Net.graph_loss("\nFake  Te")
# print(Problem1Net.prediction)

X = 2 * np.random.rand(1, 50) - 1 #1,50
T = np.sin(2 * np.pi * X) + (0.3 * np.random.randn(1, 50)) #1,50

Problem2Net = NeuralNetworkTwo(X, T, 50, 20, 50, 137, activation='tanh')
Problem2Net.train_network(1000, 0.1, 0.3)
Problem2Net.graph_loss("something")
# Problem2Net.experimental_grapher("bah", "boo", "bee", np.linspace(0, 1, 100) ,Problem2Net.prediction_layer, "red")
