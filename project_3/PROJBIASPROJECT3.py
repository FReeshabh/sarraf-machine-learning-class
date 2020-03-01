import numpy as np
import matplotlib.pyplot as plt

L_DATASETS = 100
N_DATAPOINTS = 25

Big_X = []
Big_t = []
orig_Lambda = 1
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

def calculate_weight_individual(XData, tData, new_lamba):
    lambda_reg = (np.eye(26) * new_lamba)
    gauss_phi = radial_basis(XData)
    weight = (np.linalg.inv((gauss_phi.T @ gauss_phi) + lambda_reg)) @ (gauss_phi.T) @ tData
    return weight

def FbarOfX(XData, tData, new_lamba):
    fbar = 0
    for i in range(L_DATASETS):
        current_weight = calculate_weight_individual(XData[i], tData[i], new_lamba)
        fbar += current_weight
    return (fbar/L_DATASETS) # average of all the weights
# print(FbarOfX(Big_X, Big_t, ).shape)


Weights = []
def get_weights(lamba):
    for datasets in range(L_DATASETS):
        temp = calculate_weight_individual(Big_X[datasets], Big_t[datasets], lamba)
        Weights.append(temp)

new_x = np.linspace(0, 1, 25)
new_x_basis = radial_basis(new_x)

def bias_2(Fbar, regular_x, new_x_basis):
    pred = new_x_basis @ Fbar
    true_h = np.sin(2*np.pi*regular_x)
    bias2 = np.mean(np.square(pred - true_h))
    return bias2 

def variance(weights,Fbar, basis_x, regular_x):
    var = 0
    for weight in weights:
        var += np.square(((basis_x @ weight) - (basis_x @ Fbar)))
    var = var/100
    variance = np.sum(var) / N_DATAPOINTS
    return variance

# print(variance(Weights, bar_f,new_x_basis, new_x))

# Test Dataset
X_test_dataset = np.random.uniform(low = 0.0, high = 1.0, size = 1000)
t_test_dataset = (np.sin(2 * (np.pi) * X_test_dataset)) + (np.random.normal(loc = 0.0, scale = 0.3, size = X_test_dataset.shape))
radial_basis_X_test = radial_basis(X_test_dataset)

biases = []
variances = []
bplusvs = []
orig_Lambda = np.arange(np.exp(-3), np.exp(2), 0.1)

for i in orig_Lambda:
    bar_f = FbarOfX(Big_X, Big_t, i) 
    get_weights(i)
    BIAS = bias_2(bar_f, new_x, new_x_basis)
    biases.append(BIAS)
plt.plot(biases)
plt.show()
# logger = np.log(plt_X)