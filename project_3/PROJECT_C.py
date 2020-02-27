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

def gauss_radial_basis(in_x):
    stddev = np.square(0.1)

    for i in range(L_DATASETS):
        curr_set = np.reshape(in_x[i, :], (-1, 1))
        curr_set = np.linspace(0, 1, 25)
        for j in range(N_DATAPOINTS-5):
            curr_datapoint = curr_set[j]
            # curr_datapoint = np.exp((-(np.square(curr_datapoint - curr_set)) / (2 * np.square(stddev))))
            curr_datapoint = np.exp((-np.square(curr_datapoint - curr_set)/(stddev * 2)))
    # curr_datapoint = np.column_stack((np.ones(curr_datapoint.shape), curr_datapoint))

    return curr_datapoint
print(gauss_radial_basis(Big_X).shape)
plt.plot(gauss_radial_basis(Big_X))
plt.show()