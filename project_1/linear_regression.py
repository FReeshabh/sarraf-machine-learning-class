# Rishabh Tewari
# Machine Learning ECE 4332/5332
# R11603985
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale as skl
from threading import Thread
# Load the data
carbig_filepath = "carbig.csv"
carbig_data = pd.read_csv(carbig_filepath)

# Label the columns
X_predictor = carbig_data['Weight']
carbig_data['Horsepower'] = carbig_data['Horsepower'].fillna(carbig_data['Horsepower'].median()) #Replacing missing values with the median
t_target = carbig_data['Horsepower']

# Normalize and reshape

# Normalize and reshape X
X_predictor = np.reshape(X_predictor.values, (-1, 1))
X_predictor = np.hstack((X_predictor, np.ones(X_predictor.shape)))
X_predictor_norm = np.empty(X_predictor.shape)
# X_predictor_norm[:,0] = ((X_predictor[:,0] - X_predictor[:,0].mean())/(np.std(X_predictor[:,0])))
# X_predictor_norm[:,1] = X_predictor[:,1]
X_predictor_norm = (X_predictor - X_predictor.mean())/(np.std(X_predictor))
# X_predictor_norm = skl.StandardScaler

# Reshape and normalize y
t_target = np.reshape(t_target.values, (-1, 1))
# t_target_norm = ((t_target - t_target.mean()) / (np.std(t_target)))
# print(t_target_norm.shape)

#closed_form_graph = plt.figure()
#gradient_descent_graph = plt.figure()

def closed_form_solution(X_predictor):
    weight = (np.linalg.pinv(X_predictor) @ t_target).T
    prediction = weight*X_predictor # The prediction
    
    plt.title("Carbig Dataset, Closed Form\nMissing Values replaced by median, Rishabh Tewari")
    plt.scatter(X_predictor[:,0], t_target, label="actual data")
    plt.xlabel('Weight')
    plt.ylabel('Horsepower')
    plt.plot(X_predictor, prediction, color = "red", label="Closed Form Solution")
    plt.legend()
    plt.show()
    
# closed_form_solution()

def gradient_descent(max_iterations, rho_learning_rate, weight, X_predictor, t_target, X_pred_reg): 
    # gradient = 0
    for i in range(max_iterations):
        loss = np.square(np.linalg.norm(t_target - X_predictor @ weight)) 
        gradient = ((2* weight.T @ X_predictor.T@ X_predictor ) - (2* t_target.T @ X_predictor)).T
        weight = weight - (rho_learning_rate)*gradient
        print("Iteration {}'s loss: {}".format(i,loss))

    prediction = weight.T * ((X_predictor - X_predictor.mean())/np.std(X_predictor))
    # X_predictor = (X_predictor * np.std(X_predictor)) + X_predictor.mean()
    plt.title("Carbig Dataset, Gradient Descent\nMissing Values replaced by median, Rishabh Tewari")
    plt.scatter(X_pred_reg[:,0], t_target, label="actual data")
    plt.xlabel('X - Weight')
    plt.ylabel('Y - Horsepower')


    plt.plot(X_pred_reg, prediction, color = "green", label="Gradient Descent")
    plt.legend()
    return plt.show()

init_weight = np.array([0, 0])
init_weight = np.reshape(init_weight, (-1, 1))
max_iterations = 5000 #(epochs)
rho_learning_rate = 0.001
#grad(max_iterations, rho_learning_rate, init_weight, X_predictor, t_target)

if __name__ == '__main__':
    Thread(target = closed_form_solution(X_predictor)).start()
    Thread(target = gradient_descent(max_iterations, rho_learning_rate, init_weight, X_predictor_norm, t_target, X_predictor)).start()