#Rishabh Tewari
# R11603985
# Machine Learning Project 3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(seed = 200)

# Size of the Datasets
N_train = 25
Big_dataset = []

for i in range(100):
    """
    Create the Datasets and generate csv files from the generated datasets
    """
    X = np.random.uniform(low = 0.0, high = 1.0, size = N_train)
    t_train = (np.sin(2 * (np.pi) * X)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X.shape))
    Dataset = np.stack((X, t_train), axis = -1)
    file_name = "data_" + str(i) + ".csv"
    np.savetxt(file_name, Dataset, delimiter=",", fmt='%s')
    Big_dataset.append(Dataset)

print(Big_dataset)    