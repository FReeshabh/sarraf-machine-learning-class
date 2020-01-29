import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the adata
carbig_filepath = "carbig.csv"
carbig_data = pd.read_csv(carbig_filepath)

# Label the columns
X_predictor = carbig_data['Weight']
median_y = carbig_data['Horsepower'].median()
carbig_data['Horsepower'] = carbig_data['Horsepower'].fillna(median_y) #Replacing missing values with the median
t_target = carbig_data['Horsepower']

#Reshape X to make it a two dimensional array
X_predictor = np.reshape(X_predictor.values, (-1, 1))
t_target = np.reshape(t_target.values, (-1, 1))


# DEBUG: Check Shape of the predictor, and target variables
# print("t_target shape(Initially):" + str(t_target.shape))
# print("X_predictor shape(Initially):   " + str(X_predictor.shape))

def closed_form_solution():
    weight = np.dot(np.linalg.pinv(X_predictor), t_target)
    weight = weight.T
    predictor = weight*X_predictor

    plt.title("Carbig Dataset, Closed Form\nMissing Values replaced by median, Rishabh Tewari")
    # plt.suptitle("Missing Values replaced by median, Rishabh Tewari")
    plt.scatter(X_predictor, t_target, label="actual data")
    plt.xlabel('Weight')
    plt.ylabel('Horsepower')
    plt.plot(X_predictor, predictor, color = "red", label="Closed Form Solution")
    plt.legend()
    plt.show()

closed_form_solution()
