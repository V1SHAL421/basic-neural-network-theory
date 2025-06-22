"""Understanding the inner workings of backpropagation"""

# Part 1
"""
A neural network trained with backpropagation attempting to use input to predict output

Inputs: 0, 0, 1
Output: 0

Inputs: 1, 1, 1
Output: 1

Inputs: 1, 0, 1
Output: 1

Inputs: 0, 1, 1
Outputs: 0

To try to predict the output column given the inputs, we can solve this by measuring
statistics between the input and output values. Backpropagation measures statistics like
this to make a model.
"""
import numpy as np

learning_rate = 0.1

# Sigmoid function (An activation function): 1/(1 + e^-x)
def nonlin(x, deriv=False):
    sig = 1 / (1 + np.exp(-x))
    if not deriv:
        return sig
    return sig * (1 - sig)

# Input dataset
X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

# Output dataset
y = np.array([[0, 0, 1, 1]]).T

"""
The following line is a best practice to ensure code provides consistent results every time
it is ran especially when using randomness.
It sets the seed for the NumPy's RNG to a fixed value (1). This makes random operations e.g
np.random.rand() deterministic.
This is useful for debugging and testing.
"""
np.random.seed(1)

bias = np.zeros((1, 1))

"""The first layer of weights, Synapse 0 connecting l0 (layer 0) to (layer 1)"""
syn0 = 2 * np.random.random((3, 1)) - 1

for iter in range(1000):
    
    # Forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0) + bias)

    # Error
    l1_error = y - l1

    # Delta calculated by multiplying the error by the slope of the sigmoid
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn0 += learning_rate * np.dot(l0.T, l1_delta)

    bias += learning_rate * np.sum(l1_delta, axis=0, keepdims=True)

print(f"Output after Training: {l1}")





# Part 2
"""This will include a hidden layer of 4 neurons"""
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

bias1 = np.zeros((1, 4))
bias2 = np.zeros((1, 1))

syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

for j in range(6000):
    # Forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0) + bias1)
    l2 = nonlin(np.dot(l1, syn1) + bias2)

    l2_error = y - l2

    if (j % 1000) == 0:
        print(f"Error: {np.mean(np.abs(l2_error))}")
    
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # How much did each l1 value contribute to the l2 error?
    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += learning_rate * l1.T.dot(l2_delta)
    syn0 += learning_rate * l0.T.dot(l1_delta)

    bias2 += learning_rate * np.sum(l2_delta, axis=0, keepdims=True)
    bias1 += learning_rate * np.sum(l1_delta, axis=0, keepdims=True)
