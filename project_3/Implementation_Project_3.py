#Rishabh Tewari
# R11603985

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(33)
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

def calculate_weight_individual(XData, tData, lamba):
    lambda_reg = (np.eye(26) * lamba)
    gauss_phi = radial_basis(XData)
    weight = (np.linalg.inv((gauss_phi.T @ gauss_phi) + lambda_reg)) @ (gauss_phi.T) @ tData
    return weight

def get_weights(lamba):
    Weights = []
    for datasets in range(L_DATASETS):
        temp = calculate_weight_individual(Big_X[datasets], Big_t[datasets], lamba)
        Weights.append(temp)
    return Weights

def f_bar_of_x(XData, tData, lamba):
    fbar = 0
    for i in range(L_DATASETS):
        current_weight = calculate_weight_individual(XData[i], tData[i], lamba)
        # current_weight = X @ current_weight
        fbar += current_weight
    return (fbar/L_DATASETS) # average of all the weights

def bias_2(XData, tData, lamba):
    basisX = radial_basis(XData) #((np.linspace(0, 1, 25)))
    true_h = np.sin(2*np.pi*XData)
    prediction = basisX @ f_bar_of_x(XData, tData, lamba)
    # prediction = f_bar_of_x(XData, tData, lamba)
    bias2 = np.mean(np.square(prediction - true_h))
    return bias2

def variance(XData, tData, lamba):
    Fbar = f_bar_of_x(XData, tData, lamba)
    basisX = radial_basis(XData)
    Weights = get_weights(lamba)
    variance = 0
    for weight in Weights:
        variance += np.square(((basisX @ weight) - (basisX @ Fbar)))
    variance = variance / L_DATASETS
    variance = np.sum(variance) / N_DATAPOINTS
    return variance

def cost_function(prediction, target):
    loss = np.sum(np.square(prediction - target))
    return loss/1000

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

Big_X = np.asarray(Big_X)
Big_t = np.asarray(Big_t)

# Test Dataset
X_test_dataset = np.random.uniform(low = 0.0, high = 1.0, size = 1000)
t_test_dataset = (np.sin(2 * (np.pi) * X_test_dataset)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X_test_dataset.shape))

radial_basis_X_test = radial_basis(X_test_dataset)

ALL_BIASES = []
ALL_VARIANCES = []
ALL_BIAS_PLUS_VARIANCES = []
ALL_TEST_ERRORS = []
ALL_ERRORS = []
range_of_lambas = np.exp(np.linspace(-3, 2, 60)) # np.arange(np.exp(-3), np.exp(2), step = 0.1)
for Lambda in range_of_lambas:
    BIAS = bias_2(Big_X, Big_t, Lambda)
    ALL_BIASES.append(BIAS)

    VARIANCE = variance(Big_X, Big_t, Lambda)
    ALL_VARIANCES.append(VARIANCE)
# 
    BIAS_PLUS_VARIANCES = BIAS + VARIANCE
    ALL_BIAS_PLUS_VARIANCES.append(BIAS_PLUS_VARIANCES)
    # cost_function((np.linalg.inv(radial_basis(X_test_dataset).T @ radial_basis(X_test_dataset))), t_test_dataset)

print(ALL_BIASES)
plt.ylim([0,0.6])
plt.xlabel('ln(lambda)')
plt.title('Bias-Variance Decomposition')
plt.plot(np.log(range_of_lambas), ALL_BIASES, label="$(BIAS)^2$", color = "blue")
plt.plot(np.log(range_of_lambas), ALL_VARIANCES, label="VARIANCE", color="red")
plt.plot(np.log(range_of_lambas), ALL_BIAS_PLUS_VARIANCES, label="$(BIASES)^2$ + VARIANCE", color="green")
plt.legend()
plt.show()