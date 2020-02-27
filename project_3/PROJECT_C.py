import numpy as np

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

print(Big_X.shape)

rbf = []
stddev = 0.1
for i in range(L_DATASETS):
    gauss = []
    curr_dataset = Big_X[i,:]
    curr_dataset = np.reshape(curr_dataset, (-1, 1))
    for j in range(N_DATAPOINTS):
        curr_dataset_point = curr_dataset[j]
        rbf_curr = np.exp((-(np.square(curr_dataset_point - curr_dataset)) / (2 * np.square(stddev))))
        gauss.append(rbf_curr)
    rbf.append(gauss)

gauss = np.asarray(gauss)
gauss = gauss[:, :, 0]
gauss = np.column_stack((np.ones(gauss[0].shape), gauss))
print(gauss)