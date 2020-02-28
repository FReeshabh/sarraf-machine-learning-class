import numpy as np 
import matplotlib.pyplot as plt

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

def radial_basis_individual(a_dataset):
    variance = np.square(0.1)
    mu_dataset = np.linspace(0, 1, 25)
    gausses_list = []
    for j in range(N_DATAPOINTS):
        datapoint = a_dataset[j]
        gauss = np.exp((-(np.square(datapoint - mu_dataset)) / (2 * variance)))
        gausses_list.append(gauss)
    gausses_list = np.asarray(gausses_list)
    feature_vector = np.column_stack((np.ones(25), gausses_list))
    return feature_vector

# print(radial_basis_individual(Big_X[1]).shape)
# plt.plot(radial_basis_individual(Big_X[2])[2]) ##IMPORTANT: DON'T DELETE, L, N
jox = radial_basis_individual(Big_X[1])
jot = Big_t[1]
# print(jox.shape)
lambda_reg = 0.1
weight =  (jox.T @ jox)   
chock = (np.eye(26) * lambda_reg)
weight = ((np.linalg.inv(weight + chock)) @ jox.T) @ jot
weight = np.reshape(weight, (-1, 1))
print(weight)
    