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
Big_X = []

for i in range(100):
    """
    Create the Datasets and generate csv files from the generated datasets
    """
    X = np.random.uniform(low = 0.0, high = 1.0, size = N_train)
    Big_X.append(X)
    t_train = (np.sin(2 * (np.pi) * X)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X.shape))
    Dataset = np.stack((X, t_train), axis = -1)
    file_name = "data_" + str(i) + ".csv"
    np.savetxt(file_name, Dataset, delimiter=",", fmt='%s')
    Big_dataset.append(Dataset)

Big_dataset_np = np.asarray(Big_dataset)
Big_X_np = np.asarray(Big_X)
print(Big_X_np.shape)

# for i in range(100):
def gauss(X):
    dataset = X[0, :] #[i, :]
    mean   = np.mean(dataset)
    stddev = np.std(dataset)
    rbf = np.exp((-(np.square(dataset - mean)/(2*np.square(stddev)))))
    rbf = np.reshape(rbf, (-1, 1))
    rbf = np.column_stack((np.ones(rbf.shape), rbf))
    return rbf

print(gauss(Big_X_np))

# def gauss(X):
#     mean = np.mean(X)
#     stddev = np.std(X)
#     rbf = np.exp((- np.square(X - mean)) / (2 * np.square(stddev)) )
#     rbf = np.reshape(rbf, (-1, 1))
#     rbf = np.column_stack(np.ones(rbf.shape))
#     return rbf

# print(gauss(Big_X_np[0,:]).shape)

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