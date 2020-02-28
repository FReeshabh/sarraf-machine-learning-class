import numpy as np 
import matplotlib.pyplot as plt

L_DATASETS = 100
N_DATAPOINTS = 25

Big_X = []
Big_t = []
lambda_regularization = 1

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

def linear_reg_with_regu(Xdataset, tDataset, reg_constant):
    lambda_reg = (np.eye(26) * reg_constant)
    predictions = []
    weights = []
    for i in range(L_DATASETS):
        # Xdataset = np.reshape(Xdataset, (-1, 1))
        gauss_phi = radial_basis_individual(Xdataset[i])
        target = tDataset[i]
        weight = np.linalg.inv((gauss_phi.T @ gauss_phi) + lambda_reg)
        weight = weight @ gauss_phi.T 
        weight = weight @ target
        # weight = ((np.linalg.inv((gauss_phi.T @ gauss_phi) + lambda_reg)) @ gauss_phi.T) @ target
        weight = np.reshape(weight, (-1, 1))
        prediction = ((gauss_phi @ weight))
        predictions.append(prediction)

    predictions = np.asarray(predictions)
    predictions = predictions[:, :, 0] #Strip the array of the last dimension
    return predictions

f_bar = (linear_reg_with_regu(Big_X, Big_t, lambda_regularization))
f_bars = []
for iteration in range(L_DATASETS):
    bar = np.mean(f_bar[i])
    f_bars.append(bar)
f_bars = np.asarray(f_bars)
f_bars = np.reshape(f_bars, (-1, 1))
print(f_bars.shape)

