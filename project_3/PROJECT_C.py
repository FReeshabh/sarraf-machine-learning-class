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

#def gauss_radial_basis(in_x):
#    stddev = np.square(0.1)
#    bigger_chonk = []
#    for i in range(L_DATASETS):
#        curr_set = np.reshape(in_x[i, :], (-1, 1))
#        curr_set = np.linspace(0, 1, 25)
#        chonk = []
#        for j in range(N_DATAPOINTS):
#            curr_datapoint = curr_set[j]
#            curr_datapoint = np.exp((-np.square(curr_datapoint - curr_set)/(stddev * 2)))
#            chonk.append((curr_datapoint))
#        bigger_chonk.append(chonk)
#    
#    chonk = np.asarray(chonk)
#    bigger_chonk = np.asarray(bigger_chonk)
#    return bigger_chonk
print(Big_X.shape)
def individual_gauss(in_x):
    variance = np.square(0.1)
    gauss_basis_list = []
    curr_set = np.linspace(0, 1, 25)
    for i in range(N_DATAPOINTS):
        curr_datapoint = in_x[i]
        curr_datapoint = np.exp((-np.square(curr_datapoint - curr_set)/(variance * 2)))
        gauss_basis_list.append(curr_datapoint)
    gauss_basis_list = np.asarray(gauss_basis_list)
    feature_vector = np.column_stack((np.ones(25), gauss_basis_list))
    return feature_vector
print(individual_gauss(Big_X[1]).shape)

plt.plot(individual_gauss(Big_X[1, :]))
plt.show()


def linear_regression():
    base = gauss_radial_basis(Big_X)
    weights = []
    for i in range(L_DATASETS):
        curr_data = base[i, :, :]
        # calculate weight over here and append it into weights
        pass

# print(gauss_radial_basis(Big_X).shape)
# plt.plot(gauss_radial_basis(Big_X))
# plt.show()























# import numpy as np
# import matplotlib.pyplot as plt

# L_DATASETS = 100
# N_DATAPOINTS = 25

# Big_X = []
# Big_t = []

# # Generate the datasets Required
# for i in range(L_DATASETS):
#     # generate the X and t
#     X = np.random.uniform(low = 0.0, high = 1.0, size = N_DATAPOINTS)
#     t = (np.sin(2 * (np.pi) * X)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X.shape))
#     # Insert the X and t into lists
#     Big_X.append(X)
#     Big_t.append(t)

# # Convert the lists into numpy array
# Big_X = np.asarray(Big_X)
# Big_t = np.asarray(Big_t)

# # print(Big_X.shape)

# rbf = []
# stddev = 0.1
# for i in range(L_DATASETS):
#     gauss = []
#     curr_dataset = Big_X[i,:]
#     curr_dataset = np.reshape(curr_dataset, (-1, 1))
#     for j in range(N_DATAPOINTS):
#         curr_dataset_point = curr_dataset[j]
#         rbf_curr = np.exp((-(np.square(curr_dataset_point - curr_dataset)) / (2 * np.square(stddev))))
#         gauss.append(rbf_curr)
#     rbf.append(gauss)

# rbf = np.asarray(rbf)
# rbf = rbf[:, :, :, 0]
# print(rbf.shape)
# gauss = np.asarray(gauss)
# gauss = gauss[:, :, 0]
# # gauss = np.column_stack((np.ones(gauss[0].shape), gauss))
# plt.plot(gauss)
# plt.show()
# print(gauss.shape) 


# print(radial_basis_individual(Big_X[1]).shape)
# plt.plot(radial_basis_individual(Big_X[2])[2]) ##IMPORTANT: DON'T DELETE, L, N
# plt.show()

# jox = radial_basis_individual(Big_X[1])
# jot = Big_t[1]
# print(Big_X[1].shape)
# lambda_reg = 0.1
# weight =  (jox.T @ jox)   
# chock = (np.eye(26) * lambda_reg)
# weight = ((np.linalg.inv(weight + chock)) @ jox.T) @ jot
# weight = np.reshape(weight, (-1, 1))
# print(weight.shape)
# prediction = ((Big_X[1]) @ weight)
# print(prediction.shape)


















# import numpy as np
# import matplotlib.pyplot as plt

# L_DATASETS = 100
# N_DATAPOINTS = 25

# Big_X = []
# Big_t = []

# # Generate the datasets Required
# for i in range(L_DATASETS):
#     # generate the X and t
#     X = np.random.uniform(low = 0.0, high = 1.0, size = N_DATAPOINTS)
#     t = (np.sin(2 * (np.pi) * X)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X.shape))
#     # Insert the X and t into lists
#     Big_X.append(X)
#     Big_t.append(t)

# # Convert the lists into numpy array
# Big_X = np.asarray(Big_X)
# Big_t = np.asarray(Big_t)
# #print(Big_X.shape)

# rbf_X = []
# for i in range(L_DATASETS):
#     current_dataset = Big_X[i, :]
#     current_dataset = np.reshape(current_dataset, (-1, 1))
#     mu = np.linspace(0, 1, 25)
#     variance = np.square(0.1)
#     gauss = []
#     for j in range(N_DATAPOINTS):
#         current_datapoint = current_dataset[j]
#         rbf = np.exp((-np.square(current_datapoint - mu)/(variance * 2)))
#         gauss.append(rbf)
#     rbf_X.append(gauss)

# plt.plot(rbf_X[0, 1 :])
# plt.show()
        
# def gauss_radial_basis(X_inp):
#     variance = np.square(0.1)
#     mu = np.linspace(0, 1, 25)
#     Big_RBF = []
#     for i in range(L_DATASETS):
#         Big_Gauss = []
#         curr_dataset = X_inp[i, :]
#         for j in range(N_DATAPOINTS):
#             curr_point = curr_dataset[j]
#             rbf = (np.exp((-(np.square(curr_point - mu))/(2*variance))))
#             Big_Gauss.append(rbf)
        
#         Big_Gauss.append(1)
#         Big_RBF.append(Big_Gauss)

#     Big_Gauss = np.asarray(Big_Gauss)
#     Big_RBF = np.asarray(Big_RBF) # The 100 one
#     # Big_RBF = np.column_stack((np.ones(Big_Gauss[0].shape), Big_RBF))

#     return Big_Gauss, Big_RBF


# this, that = gauss_radial_basis(Big_X)
# print(this.shape)
# print(that.shape)


# def linear_regression(x):
    # pinv_1 = (x @ x.T)
    # pinv_2 = np.identity(26) * 0.1
    # weight = (np.linalg.inv(pinv_1 + pinv_2)) @ x.T @ Big_t
    # return weight
# print(linear_regression(gauss_radial_basis(Big_X)))

# print(gauss_radial_basis(Big_X).shape)
# plt.plot(gauss_radial_basis(Big_X))
# plt.show()

# def gauss_radial_basis(in_x):
#     stddev = np.square(0.1)
#     rbf = []
#     gauss = []
#     for i in range(L_DATASETS):
#         # curr_set = np.reshape(in_x[i, :], (-1, 1))
#         curr_set = np.linspace(0, 1, 25)
#         for j in range(N_DATAPOINTS-5):
#             curr_datapoint = curr_set[j]
#             # curr_datapoint = np.exp((-(np.square(curr_datapoint - curr_set)) / (2 * np.square(stddev))))
#             curr_datapoint = np.exp((-np.square(curr_datapoint - curr_set)/(stddev * 2)))
#             gauss.append(curr_datapoint)
#     rbf = rbf.append(gauss)
#     # curr_datapoint = np.column_stack((np.ones(curr_datapoint.shape), curr_datapoint))
#     rbf = np.asarray(rbf)
#     return rbf
# def gauss_radial_basis(in_x):
#     stddev = np.square(0.1)

#     for i in range(L_DATASETS):
#         curr_set_X = np.reshape(in_x[i, :], (-1, 1))
#         curr_set = np.linspace(0, 1, 25)
#         for j in range(N_DATAPOINTS-5):
#             curr_datapoint = curr_set_X[j]
#             # curr_datapoint = np.exp((-(np.square(curr_datapoint - curr_set)) / (2 * np.square(stddev))))
#             curr_datapoint = np.exp((-np.square(curr_datapoint - curr_set)/(stddev * 2)))
#     # curr_datapoint = np.column_stack((np.ones(curr_datapoint.shape), curr_datapoint))

#     return curr_datapoint
# #print(gauss_radial_basis(Big_X).shape)
# plt.plot(gauss_radial_basis(Big_X))
# plt.show()


# print(linear_reg_with_regu(Big_X, Big_t, lambda_regularization).shape)