# Rishabh Tewari
# Machine Learning ECE 4332/5332
# R11603985

import numpy as np
import matplotlib.pyplot as plt

#To Ensure Results don't change
np.random.seed(seed = 0)
#Training X
X_train = np.random.uniform(low = 0.0, high = 1.0, size = 10)
X_train = np.reshape(X_train, (-1, 1))
# Training t
t_train = (np.sin(2 * (np.pi) * X_train)) + (np.random.normal(loc = 0.0, scale = 0.3))
# Testing X
X_test = np.random.uniform(low = 0.0, high = 1.0, size = 100)
X_test = np.reshape(X_test, (-1, 1))
# Testing t
t_test = (np.sin(2 * (np.pi) * X_test)) + (np.random.normal(loc = 0.0, scale = 0.3))

def phi(x, degree):
    feature_v = np.column_stack((np.ones(x.shape), x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9))
    feature_v = feature_v[:,:degree+1]
    print(feature_v)

dummy = np.array([[1, 2, 3], [4,5,6]])
phi(X_train, 4)

