# Rishabh Tewari
# R11603985
# Machine Learning Project 3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed = 200)

"""
a. 𝐿 = 100
b. 𝑁 = 25
c. 𝑋 contains samples from a uniform distribution U(0,1).
d. 𝑡 = sin(2𝜋) + 𝜀, where 𝜀 contains samples from a Gaussian distribution
N(0, 𝜎 =0.3).
"""
L_DATASETS = 100
N_DATAPOINTS = 25

Big_X = []
Big_t = []

# Generate the datasets Required
for i in range(L_DATASETS):
    # generate the X and t
    X = np.random.uniform(low = 0.0, high = 1.0, size = N_DATAPOINTS)
    t = (np.sin(2 * (np.pi) * X)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X.shape))
    # Insert the X and t into lists
    Big_X.append(X)
    Big_t.append(t)

# Convert the lists into numpy array
Big_X = np.asarray(Big_X)
Big_t = np.asarray(Big_t)

def phi_basis(X):
    phier = []
    stddev = 0.1
    for i in range (L_DATASETS):
        input_X = r
        for j in range(N_DATAPOINTS):
    rbf = np.exp((-(np.square(input_X - mean)/(2*np.square(stddev))))) # The radial Basis Function

def linear_regression(x):
    weight = (np.linalg.inv((x.T@x)))
# def phi_rbf(input_X):
#     stddev = 0.1 #np.std (Big_X)
#     mean   = np.
#     rbf = np.exp((-(np.square(input_X - mean)/(2*np.square(stddev))))) # The radial Basis Function
#     rbf = np.reshape(rbf, (-1, 1))
#     rbf = np.column_stack((np.ones(rbf.shape), rbf)) # Add a column of ones
#     return rbf
# 
# RBF = []
# for i in range(L_datasets):
#     RBF.append(phi_rbf(Big_X[i, :])) # Add the phi(x) to RBF
# 
# RBF = np.asarray(RBF) # Convert RBF into a numpy array

# # Size of the Datasets
# N_train = 25
# Big_dataset = []
# Big_X = []
# Big_T = []

# for i in range(100):
#     """
#     Create the Datasets and generate csv files from the generated datasets
#     """
#     X = np.random.uniform(low = 0.0, high = 1.0, size = N_train)
#     Big_X.append(X)
#     t_train = (np.sin(2 * (np.pi) * X)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X.shape))
#     Big_T.append(t_train)
#     Dataset = np.stack((X, t_train), axis = -1)
#     file_name = "data_" + str(i) + ".csv"
#     np.savetxt(file_name, Dataset, delimiter=",", fmt='%s')
#     Big_dataset.append(Dataset)

# Big_dataset_np = np.asarray(Big_dataset)
# Big_X_np = np.asarray(Big_X)
# Big_T_np = np.asarray(Big_T)
# # print(Big_T_np.shape)


# def gauss(X):
#     dataset = X[0, :] #[i, :]
#     mean   = np.mean(dataset)
#     stddev = np.std(dataset)
#     rbf = np.exp((-(np.square(dataset - mean)/(2*np.square(stddev)))))
#     rbf = np.reshape(rbf, (-1, 1))
#     rbf = np.column_stack((np.ones(rbf.shape), rbf))
#     return rbf

# def linear_regression():
#     lambdaX = 0.1
#     weight = ((gauss(Big_X_np)).T @ Big_T_np[0, :]) + lambdaX

# print(gauss(Big_X_np))

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
