# Rishabh Tewari
# Machine Learning ECE 4332/5332
# R11603985
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread

# Load the data
carbig_filepath = "proj1Dataset.csv"
carbig_data = pd.read_csv(carbig_filepath)

# Assign Data to variables
X = carbig_data['Weight']
carbig_data['Horsepower'] = carbig_data['Horsepower'].fillna(carbig_data['Horsepower'].median()) #Replacing missing values with the median
t_target = carbig_data['Horsepower']

# Reshape and prep X, and X_norm
X = np.reshape(X.values, (-1, 1))
X_norm = np.empty(X.shape)
X_norm = X/X.max()
t_target = np.reshape(t_target.values, (-1, 1))

# Add one as the design matrices to both of them
X = np.hstack((X, np.ones(X.shape)))
X_norm = np.hstack((X_norm, np.ones(X_norm.shape)))

# print(t_target)
def closed_form_solution(x, t):
    """
    Analytical Solution for the weight for Linear Regression
    """
    weight = (np.linalg.pinv(x) @ t).T #the transpose of the final weight
    d_closed = np.linspace(1500, 5500, 3500)
    print(weight.T)
    prediction_closed = weight.T [0][0]*d_closed + weight.T[1][0]
    plt.title("Carbig Dataset, Closed Form\n Rishabh Tewari")
    plt.scatter(X[:,0], t, label="actual data")
    plt.xlabel('Weight')
    plt.ylabel('Horsepower')
    plt.plot(d_closed, prediction_closed, color = "red", label="Closed Form Solution")
    plt.legend()
    plt.figure()


def gradient_descent(iterations, rho, weight, x, t):
    """
    Gradient Descent Solution for the weight for Linear Regression
    """
    for i in range(iterations):
        gradient = ((2* weight.T @ x.T@ x ) - (2* t.T @ x)).T
        weight = weight - (rho * gradient)
        print("Iteration {} 's loss: {}".format(i, (np.square(np.linalg.norm(t - x @ weight)))))
    d_grad = np.linspace(1500, 5500, 3500)
    weight[0][0] = weight[0][0] / X.max()
    # print(weight)
    prediction_grad = weight.T [0][0]*d_grad + weight.T[0][1]
    plt.title("Carbig Dataset, Gradient Descent\n Rishabh Tewari")
    plt.scatter(X[:,0], t, label="actual data")
    plt.xlabel('Weight')
    plt.ylabel('Horsepower')
    plt.plot(d_grad, prediction_grad, color = "green", label="Gradient Descent Solution")
    plt.legend()
    plt.show()

# Initial Assumptions made
init_weight = np.array([0, 0])
init_weight = np.reshape(init_weight, (-1, 1))
max_iterations = 1000 #(epochs)
rho_learning_rate = 0.001

# gradient_descent(max_iterations, rho_learning_rate, init_weight, X_norm, t_target)
# closed_form_solution(X_norm, t_target)

# Run these functions
if __name__ == '__main__':
    Thread(target = closed_form_solution(X, t_target)).start()
    Thread(target = gradient_descent(max_iterations, rho_learning_rate, init_weight, X_norm, t_target))

