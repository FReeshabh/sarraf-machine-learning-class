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
# Big_dataset = np.empty()

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

Big_dataset_np = np.asarray(Big_dataset)
print(Big_dataset_np)

def gauss(X):
    mean = np.mean(X)
    stddev = np.std(X)
    rbf = np.exp((- np.square(X - mean)) / (2 * np.square(stddev)) )
    return rbf

# print(Big_dataset_np[1])

# def phi(X):
#     Xer = np.array(2500)
#     for L in range(100):
#         for N in range(25):
#             XtoChange = X[L, N, 1]
#             mean = np.mean()
#             Xer = np.append(Xer, XtoChange)
#     print(Xer.shape)

# phi(Big_dataset_np)
