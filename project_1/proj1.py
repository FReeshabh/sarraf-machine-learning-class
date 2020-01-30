# Rishabh Tewari
# Machine Learning ECE 4332/5332
# R11603985
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

# Load the data
carbig_filepath = "carbig.csv"
carbig_data = pd.read_csv(carbig_filepath)

# Label the columns
X_predictor = carbig_data['Weight']
median_y = carbig_data['Horsepower'].median()
carbig_data['Horsepower'] = carbig_data['Horsepower'].fillna(median_y) #Replacing missing values with the median
t_target = carbig_data['Horsepower']

#Reshape X and t_target
X_predictor = np.reshape(X_predictor.values, (-1, 1))
ones = np.ones(X_predictor.shape)
X_predictor = np.concatenate([X_predictor, ones], axis=1)
t_target = np.reshape(t_target.values, (-1, 1))
# print(t_target.shape)

# def closed_form_solution():
#     """
#     This function gives the closed form solution(analytical solution)
#     for Linear Regression. The formula used here is 
#     w = pseudo_inverse(X) * t
#     """
#     # weight = (np.dot(np.linalg.pinv(X_predictor), t_target)).T #The pseudinverse of X_predictor, and then the transpose
#     weight = (np.linalg.pinv(X_predictor) @ t_target).T
#     predictor = weight*X_predictor # The prediction

#     plt.title("Carbig Dataset, Closed Form\nMissing Values replaced by median, Rishabh Tewari")
#     # plt.suptitle("Missing Values replaced by median, Rishabh Tewari")
#     plt.scatter(X_predictor[:,0], t_target, label="actual data")
#     plt.xlabel('Weight')
#     plt.ylabel('Horsepower')
#     plt.plot(X_predictor, predictor, color = "red", label="Closed Form Solution")
#     plt.legend()
#     plt.show()

# closed_form_solution()
# print(X_predictor.shape)

# rho_learning_rate = 0.1
# k_max_iter = 500

# weight = np.array([0.1, 0.1])
# weight = np.reshape(weight, (-1, 1))
# print(weight.shape)

# print(t_target.shape)

# def gradient_descent(X_predictor, t_target, k_max_iter, rho_learning_rate):
#     convergence = False
#     while convergence != True:
#         weight = np.zeros_like(X_predictor) #Initalize weight all zeroes
#         weight = np.reshape(weight, (-1, 1))
#         # prediction = np.dot(weight, X_predictor, out = None)
#         prediction = weight * X_predictor
#         np.gradient(mse(t_target, prediction)

#         # grad_loss = np.dot(rho_learning_rate, np.gradient(mse(t_target, prediction)))
#         # weight = weight - rho_learning_rate * grad_loss
#         # print(prediction)
    
    

# gradient_descent(X_predictor, t_target, k_max_iter, rho_learning_rate)

# Calculate the hypothesis h = X * theta
# Calculate the loss = h - y and maybe the squared cost (loss^2)/2m
# Calculate the gradient = X' * loss / m
# Update the parameters theta = theta - alpha * gradien