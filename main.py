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

# Sigmoid function (An activation function): 1/(1 + e^-x)
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))