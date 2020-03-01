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

def radial_basis(a_dataset):
    stddev_2 = np.square(0.1)
    mu_dataset = np.linspace(0, 1, 25)
    gausses_list = []
    for j in range(N_DATAPOINTS):
        datapoint = a_dataset[j]
        gauss = np.exp((-(np.square(datapoint - mu_dataset)) / (2 * stddev_2)))
        gausses_list.append(gauss)
    gausses_list = np.asarray(gausses_list)
    gausses_list = np.column_stack((np.ones(25), gausses_list))
    return gausses_list

# print(radial_basis(Big_X[1]).shape)
# plt.plot(radial_basis(Big_X[1])[1]) # THIS POINTS ONLY ONE POINT FOR A DATASET
# plt.show()

def linear_regression(xvar, tvar):
    lambda_reg = (np.eye(26) * 1)
    weights = []
    for i in range(L_DATASETS):
        gauss_phi = radial_basis(xvar[i])
        target    = tvar[i]
        weight = np.linalg.inv((gauss_phi.T @ gauss_phi) + lambda_reg)
        weight = weight @ gauss_phi.T
        weight = weight @ target
        weight = np.reshape(weight, (-1, 1))
        weights.append(weight)
    weights = np.asarray(weights)
    weights = weights[:, :, 0]
    return weights
    
print(linear_regression(Big_X, Big_t).shape)

def get_predictions():
    Big_Prediction = []
    linspace_X = radial_basis(np.linspace(0, 1, 25))
    for i in range(L_DATASETS):
        current_weight = linear_regression(Big_X, Big_t)[i]
        current_weight = np.reshape(current_weight, (-1, 1))
        prediction = linspace_X @ current_weight
        Big_Prediction.append(prediction)
    Big_Prediction = np.asarray(Big_Prediction)
    return Big_Prediction

def get_single_prediction(current_weight):
   linspace_X = radial_basis(np.linspace(0, 1, 25))
   current_weight = np.reshape(current_weight, (-1, 1))
   prediction = linspace_X @ current_weight
   return prediction
    
def f_x(weights):
    f_x = 0
    for i in range(L_DATASETS):
        f_x += weights[i]
    f_x = f_x / 100
    # f_x = (np.asarray(f_x)).reshape(-1, 1)
    return f_x

# all_the_predictions = (get_predictions())[:, :, 0]
# print(f_x(linear_regression(Big_X, Big_t)).shape)
def bias_squared():
    fbar = f_x(linear_regression(Big_X, Big_t))
    bias = 0
    for sox in range(26):
        h = Big_t[sox]
        bias += np.square(fbar[sox] - h)
    bias = bias/26
    return bias

print(bias_squared().shape)

# # # def xoc():
# # #     yy = all_the_predictions
# # #     ooo = []
# # #     for ii in range(L_DATASETS):
# # #         hh = np.sin(2*np.pi*Big_X[ii])
# # #         oo = np.mean(np.square(yy-hh))
# # #         ooo.append(oo)
# # #     ooo = np.asarray(ooo)
# # #     # print(ooo.shape)
# # # 
# # # def variance():
# # #     sumy = np.sum(np.square(all_the_predictions))
# # #     sumy = sumy/100
# # #     sumo =  (sumy)/N_DATAPOINTS
# # #     return sumo
# # # print(variance())

# print(f_x(linear_regression(Big_X, Big_t)).shape)

# print(bias_2(linear_regression(Big_X, Big_t)[1], Big_X[1]).shape)

# print(get_predictions().shape)


# print(all_the_predictions.shape)

# bars = []
# for iter in range (L_DATASETS):
#     f_bar = np.mean(all_the_predictions[iter])
#     bars.append(f_bar)
# bars = np.asarray(bars)
# # print(bars.shape)
